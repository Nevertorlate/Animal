from __future__ import annotations

from pathlib import Path

import torch

from ._env import PROJECT_ROOT  # noqa: F401
from .data import MotionSequence
from .pipeline import PlayerOnePipeline
from .rendering import BasePointMapRenderer, SceneCondition, normalize_scene_condition

from diffsynth.diffusion.training_module import DiffusionTrainingModule


class PlayerOneTrainingHarness(DiffusionTrainingModule):
    def __init__(self, pipeline: PlayerOnePipeline):
        super().__init__()
        self.pipeline = pipeline

    def apply_stage1_lora(
        self,
        lora_rank: int = 128,
        lora_alpha: int = 4,
        target_modules: tuple[str, ...] = ("q", "k", "v", "o", "ffn.0", "ffn.2"),
    ) -> None:
        self.pipeline.requires_grad_(False)
        self.pipeline.backbone.wan_dit = self.add_lora_to_model(
            self.pipeline.backbone.wan_dit,
            target_modules=list(target_modules),
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            upcast_dtype=self.pipeline.torch_dtype,
        )

    def apply_stage2_finetune(self, last_blocks: int = 6) -> None:
        self.pipeline.requires_grad_(False)
        trainable_modules = [
            self.pipeline.backbone.input_adapter,
            self.pipeline.backbone.motion_encoder,
            self.pipeline.backbone.motion_context_encoder,
            self.pipeline.backbone.point_map_encoder,
            self.pipeline.backbone.scene_memory_encoder,
            self.pipeline.backbone.scene_context_encoder,
        ]
        if self.pipeline.backbone.wan_dit.control_adapter is not None:
            trainable_modules.append(self.pipeline.backbone.wan_dit.control_adapter)
        for module in trainable_modules:
            module.train()
            module.requires_grad_(True)
        for block in self.pipeline.backbone.wan_dit.blocks[-last_blocks:]:
            block.train()
            block.requires_grad_(True)

    def _compute_single_clip_flow_matching_loss(
        self,
        video: list,
        motion: MotionSequence,
        *,
        context: torch.Tensor | None = None,
        prompt: str | None = None,
        scene_condition: SceneCondition | torch.Tensor | None = None,
        sigma_shift: float = 5.0,
    ) -> torch.Tensor:
        scene_condition = normalize_scene_condition(scene_condition)
        height = video[0].size[1]
        width = video[0].size[0]
        target_num_frames, motion, point_maps, video = self.pipeline._normalize_total_num_frames(
            motion,
            len(video),
            point_maps=scene_condition.point_maps,
            video=video,
        )
        scene_condition = SceneCondition(
            point_maps=point_maps,
            memory_latents=scene_condition.memory_latents,
            context_tokens=scene_condition.context_tokens,
            metadata=dict(scene_condition.metadata),
        )
        height, width, num_frames = self.pipeline.check_resize_height_width(height, width, target_num_frames, verbose=0)
        self.pipeline.scheduler.set_timesteps(1000, training=True, shift=sigma_shift)

        input_latents = self.pipeline.encode_video(video, height=height, width=width)
        first_frame_latents = input_latents[:, :, 0:1]
        if context is None:
            if prompt is None:
                raise ValueError("Pass either prompt or precomputed context when computing flow-matching loss.")
            context = self.pipeline.encode_prompt(prompt)
        motion = motion.to(device=self.pipeline.device, dtype=self.pipeline.torch_dtype)
        if scene_condition.point_maps is not None:
            scene_condition.point_maps = torch.as_tensor(scene_condition.point_maps).to(
                device=self.pipeline.device,
                dtype=self.pipeline.torch_dtype,
            )
        if scene_condition.memory_latents is not None:
            scene_condition.memory_latents = torch.as_tensor(scene_condition.memory_latents).to(
                device=self.pipeline.device,
                dtype=self.pipeline.torch_dtype,
            )
        if scene_condition.context_tokens is not None:
            scene_condition.context_tokens = torch.as_tensor(scene_condition.context_tokens).to(
                device=self.pipeline.device,
                dtype=self.pipeline.torch_dtype,
            )

        timestep_id = torch.randint(0, len(self.pipeline.scheduler.timesteps), (1,))
        timestep = self.pipeline.scheduler.timesteps[timestep_id].to(dtype=self.pipeline.torch_dtype, device=self.pipeline.device)
        noise = torch.randn_like(input_latents)
        latents = self.pipeline.scheduler.add_noise(input_latents, noise, timestep)
        latents[:, :, 0:1] = first_frame_latents
        target = self.pipeline.scheduler.training_target(input_latents, noise, timestep)

        noise_pred = self.pipeline.backbone(
            latents=latents,
            timestep=timestep,
            context=context,
            first_frame_latents=first_frame_latents,
            motion=motion,
            scene_condition=scene_condition,
            original_height=height,
            original_width=width,
            use_gradient_checkpointing=True,
        )
        noise_pred = noise_pred[:, :, 1:]
        target = target[:, :, 1:]
        loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float())
        return loss * self.pipeline.scheduler.training_weight(timestep)

    @torch.no_grad()
    def _generate_rollout_chunk(
        self,
        *,
        context: torch.Tensor,
        negative_context: torch.Tensor,
        input_image,
        motion: MotionSequence,
        scene_condition: SceneCondition | torch.Tensor | None,
        height: int,
        width: int,
        num_frames: int,
        seed: int | None,
        cfg_scale: float,
        num_inference_steps: int,
        sigma_shift: float,
    ) -> list:
        was_training = self.pipeline.training
        self.pipeline.eval()
        try:
            return self.pipeline._generate_clip_from_embeddings(
                context=context,
                negative_context=negative_context,
                input_image=input_image,
                motion=motion,
                height=height,
                width=width,
                num_frames=num_frames,
                scene_condition=scene_condition,
                seed=seed,
                cfg_scale=cfg_scale,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
            )
        finally:
            if was_training:
                self.pipeline.train()

    def compute_flow_matching_loss(
        self,
        video: list,
        prompt: str,
        motion: MotionSequence | str | Path,
        point_maps: torch.Tensor | str | Path | None = None,
        point_map_renderer: BasePointMapRenderer | None = None,
        scene_conditioner: BasePointMapRenderer | None = None,
        chunk_num_frames: int = 49,
        chunk_overlap: int = 1,
        sigma_shift: float = 5.0,
        scene_update_mode: str = "generated",
        teacher_forced_warmup_chunks: int = 0,
        rollout_negative_prompt: str = "",
        rollout_cfg_scale: float = 1.0,
        rollout_num_inference_steps: int = 12,
        rollout_seed: int | None = None,
    ) -> torch.Tensor:
        if scene_update_mode not in {"teacher_forced", "generated", "mixed"}:
            raise ValueError(f"Unsupported scene_update_mode={scene_update_mode!r}")

        loop_context = self.pipeline.prepare_autoregressive_loop(
            input_image=video[0],
            motion=motion,
            point_maps=point_maps,
            point_map_renderer=point_map_renderer,
            scene_conditioner=scene_conditioner,
            height=video[0].size[1],
            width=video[0].size[0],
            num_frames=len(video),
            chunk_num_frames=chunk_num_frames,
            chunk_overlap=chunk_overlap,
            video=video,
        )
        if loop_context.video is None:
            raise ValueError("Expected aligned training video after autoregressive loop preparation.")

        context = self.pipeline.encode_prompt(prompt)
        rollout_negative_context = self.pipeline.encode_prompt(rollout_negative_prompt)
        carry_image = loop_context.carry_image
        scene_state = loop_context.scene_state

        weighted_losses: list[torch.Tensor] = []
        loss_weights: list[int] = []
        for chunk_index, (start_frame, stop_frame) in enumerate(loop_context.windows):
            video_chunk = loop_context.video[start_frame:stop_frame]
            motion_chunk = loop_context.motion_sequence.slice_frames(start_frame, stop_frame)
            rendered_condition = loop_context.renderer.render_condition(
                scene_state,
                motion=motion_chunk,
                start_frame=start_frame,
                stop_frame=stop_frame,
                height=loop_context.height,
                width=loop_context.width,
                input_image=carry_image,
            )
            scene_condition = self.pipeline.compose_scene_condition(scene_state, rendered_condition)
            loss = self._compute_single_clip_flow_matching_loss(
                video=video_chunk,
                motion=motion_chunk,
                context=context,
                scene_condition=scene_condition,
                sigma_shift=sigma_shift,
            )
            weight = max(1, stop_frame - start_frame - 1)
            weighted_losses.append(loss * weight)
            loss_weights.append(weight)

            use_teacher_forced_update = scene_update_mode == "teacher_forced"
            if scene_update_mode == "mixed" and chunk_index < teacher_forced_warmup_chunks:
                use_teacher_forced_update = True
            if use_teacher_forced_update:
                scene_update_video = video_chunk
            else:
                chunk_seed = None if rollout_seed is None else rollout_seed + chunk_index
                scene_update_video = self._generate_rollout_chunk(
                    context=context,
                    negative_context=rollout_negative_context,
                    input_image=carry_image,
                    motion=motion_chunk,
                    scene_condition=scene_condition,
                    height=loop_context.height,
                    width=loop_context.width,
                    num_frames=stop_frame - start_frame,
                    seed=chunk_seed,
                    cfg_scale=rollout_cfg_scale,
                    num_inference_steps=rollout_num_inference_steps,
                    sigma_shift=sigma_shift,
                )

            scene_state = loop_context.renderer.update(
                scene_state,
                video=scene_update_video,
                motion=motion_chunk,
                start_frame=start_frame,
                stop_frame=stop_frame,
                height=loop_context.height,
                width=loop_context.width,
            )
            scene_state = self.pipeline.update_scene_memory(
                scene_state,
                video=scene_update_video,
                height=loop_context.height,
                width=loop_context.width,
                chunk_overlap=loop_context.chunk_overlap,
            )
            carry_image = scene_update_video[-1]

        total_weight = sum(loss_weights)
        return torch.stack(weighted_losses).sum() / total_weight
