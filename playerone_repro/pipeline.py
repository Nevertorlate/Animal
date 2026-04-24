from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from ._env import PROJECT_ROOT  # noqa: F401
from .data import (
    MotionSequence,
    build_temporal_windows,
    get_temporal_length,
    load_motion_sequence,
    load_point_maps,
    pad_temporal_tensor_to_frames,
    pad_video_frames_to_length,
    slice_temporal_tensor,
)
from .modules import PlayerOneBackbone
from .rendering import (
    BasePointMapRenderer,
    NullPointMapRenderer,
    PrecomputedPointMapRenderer,
    SceneCondition,
    ScenePointCloudState,
    normalize_scene_condition,
)

from diffsynth.diffusion.base_pipeline import BasePipeline
from diffsynth.diffusion.flow_match import FlowMatchScheduler
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline


DEFAULT_WAN_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
DEFAULT_WAN_MODEL_FILE_PATTERNS = (
    "models_t5_umt5-xxl-enc-bf16.pth",
    "diffusion_pytorch_model*.safetensors",
    "Wan2.2_VAE.pth",
)


def build_wan_model_configs(model_id_override: str | None = None) -> list[ModelConfig]:
    model_id = str(model_id_override or DEFAULT_WAN_MODEL_ID)
    return [
        ModelConfig(model_id=model_id, origin_file_pattern=origin_file_pattern)
        for origin_file_pattern in DEFAULT_WAN_MODEL_FILE_PATTERNS
    ]


@dataclass
class AutoregressiveChunkResult:
    index: int
    start_frame: int
    stop_frame: int
    seed: int | None
    video: list[Image.Image]
    point_maps: torch.Tensor | None
    scene_memory_latents: torch.Tensor | None = None


@dataclass
class AutoregressiveGenerationResult:
    video: list[Image.Image]
    chunks: list[AutoregressiveChunkResult]
    scene_state: ScenePointCloudState | None


@dataclass
class AutoregressiveLoopContext:
    height: int
    width: int
    requested_num_frames: int
    total_num_frames: int
    chunk_num_frames: int
    chunk_overlap: int
    motion_sequence: MotionSequence
    renderer: BasePointMapRenderer
    scene_state: ScenePointCloudState | None
    carry_image: Image.Image
    windows: list[tuple[int, int]]
    video: list[Image.Image] | None = None


class PlayerOnePipeline(BasePipeline):
    def __init__(
        self,
        tokenizer,
        text_encoder,
        vae,
        backbone: PlayerOneBackbone,
        device: str | torch.device = "cpu",
        torch_dtype: torch.dtype = torch.bfloat16,
        use_scene_memory: bool = True,
        scene_memory_max_latent_frames: int | None = 32,
    ):
        super().__init__(
            device=device,
            torch_dtype=torch_dtype,
            height_division_factor=vae.upsampling_factor * 2,
            width_division_factor=vae.upsampling_factor * 2,
            time_division_factor=4,
            time_division_remainder=1,
        )
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.vae = vae
        self.backbone = backbone
        self.scheduler = FlowMatchScheduler("Wan")
        self.in_iteration_models = ("backbone",)
        self.use_scene_memory = use_scene_memory
        self.scene_memory_max_latent_frames = scene_memory_max_latent_frames
        self.to(device=device, dtype=torch_dtype)

    @classmethod
    def from_wan_pretrained(
        cls,
        model_configs: list[ModelConfig] | None = None,
        tokenizer_config: ModelConfig | None = None,
        model_id_override: str | None = None,
        device: str | torch.device = "cpu",
        torch_dtype: torch.dtype = torch.bfloat16,
        use_scene_memory: bool = True,
        scene_memory_max_latent_frames: int | None = 32,
    ) -> "PlayerOnePipeline":
        if model_configs is None:
            model_configs = build_wan_model_configs(model_id_override=model_id_override)
        load_kwargs = {
            "torch_dtype": torch_dtype,
            "device": device,
            "model_configs": model_configs,
        }
        if tokenizer_config is not None:
            load_kwargs["tokenizer_config"] = tokenizer_config
        base_pipe = WanVideoPipeline.from_pretrained(**load_kwargs)
        if getattr(base_pipe, "dit2", None) is not None:
            raise NotImplementedError(
                "PlayerOnePipeline currently supports single-DiT Wan 2.2 variants only. "
                "This wrapper does not yet support dual-DiT Wan 2.2 models such as the A14B family."
            )
        latent_channels = getattr(base_pipe.vae, "z_dim", None)
        if latent_channels is None:
            latent_channels = getattr(getattr(base_pipe.vae, "model", None), "z_dim", None)
        if latent_channels is None:
            raise ValueError("Unable to infer Wan latent channels from the loaded VAE.")
        backbone = PlayerOneBackbone(base_pipe.dit, latent_channels=latent_channels)
        return cls(
            tokenizer=base_pipe.tokenizer,
            text_encoder=base_pipe.text_encoder,
            vae=base_pipe.vae,
            backbone=backbone,
            device=device,
            torch_dtype=torch_dtype,
            use_scene_memory=use_scene_memory,
            scene_memory_max_latent_frames=scene_memory_max_latent_frames,
        )

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        prompt_emb = self.text_encoder(ids, mask)
        seq_lens = mask.gt(0).sum(dim=1).long()
        for index, valid_tokens in enumerate(seq_lens):
            prompt_emb[index, valid_tokens:] = 0
        return prompt_emb.to(dtype=self.torch_dtype, device=self.device)

    def encode_first_frame(
        self,
        image: Image.Image,
        height: int,
        width: int,
        tiled: bool = False,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
    ) -> torch.Tensor:
        video = self.preprocess_video([image.resize((width, height))])
        return self.vae.encode(video, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(
            dtype=self.torch_dtype,
            device=self.device,
        )

    def encode_video(
        self,
        video: list[Image.Image],
        height: int,
        width: int,
        tiled: bool = False,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
    ) -> torch.Tensor:
        resized = [frame.resize((width, height)) for frame in video]
        video_tensor = self.preprocess_video(resized)
        return self.vae.encode(video_tensor, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(
            dtype=self.torch_dtype,
            device=self.device,
        )

    def decode_latents(
        self,
        latents: torch.Tensor,
        tiled: bool = False,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
    ) -> list[Image.Image]:
        video = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return self.vae_output_to_video(video)

    def _prepare_motion(self, motion: MotionSequence | str | Path) -> MotionSequence:
        if isinstance(motion, MotionSequence):
            sequence = motion
        else:
            sequence = load_motion_sequence(motion)
        return sequence.to(device=self.device, dtype=self.torch_dtype)

    def _load_motion_sequence(self, motion: MotionSequence | str | Path) -> MotionSequence:
        if isinstance(motion, MotionSequence):
            return motion
        return load_motion_sequence(motion)

    def _prepare_point_maps(self, point_maps: torch.Tensor | str | Path | None) -> torch.Tensor | None:
        if point_maps is None:
            return None
        if isinstance(point_maps, (str, Path)):
            point_maps = load_point_maps(point_maps)
        return torch.as_tensor(point_maps).to(device=self.device, dtype=self.torch_dtype)

    def _resolve_renderer(
        self,
        point_maps: torch.Tensor | str | Path | None,
        point_map_renderer: BasePointMapRenderer | None,
        scene_conditioner: BasePointMapRenderer | None = None,
    ) -> BasePointMapRenderer:
        if point_map_renderer is not None and scene_conditioner is not None:
            raise ValueError("Pass either point_map_renderer or scene_conditioner, not both.")
        renderer = scene_conditioner or point_map_renderer
        if point_maps is not None and renderer is not None:
            raise ValueError("Pass either point_maps or a scene-conditioner for autoregressive generation, not both.")
        if renderer is not None:
            return renderer
        if point_maps is not None:
            return PrecomputedPointMapRenderer(point_maps)
        return NullPointMapRenderer()

    def _normalize_total_num_frames(
        self,
        motion: MotionSequence,
        num_frames: int | None,
        point_maps: torch.Tensor | None = None,
        video: list[Image.Image] | None = None,
    ) -> tuple[int, MotionSequence, torch.Tensor | None, list[Image.Image] | None]:
        desired_num_frames = motion.num_frames if num_frames is None else num_frames
        if desired_num_frames < 2:
            raise ValueError(f"Need at least 2 frames for I2V continuation, got {desired_num_frames}")
        if motion.num_frames < desired_num_frames:
            motion = motion.pad_to_num_frames(desired_num_frames)
        else:
            motion = motion.slice_frames(0, desired_num_frames)

        if point_maps is not None:
            point_maps = slice_temporal_tensor(point_maps, 0, min(desired_num_frames, get_temporal_length(point_maps)))
            point_maps = pad_temporal_tensor_to_frames(point_maps, desired_num_frames)
        if video is not None:
            video = pad_video_frames_to_length(video[:desired_num_frames], desired_num_frames)

        _, _, aligned_num_frames = self.check_resize_height_width(32, 32, desired_num_frames, verbose=0)
        if aligned_num_frames != desired_num_frames:
            motion = motion.pad_to_num_frames(aligned_num_frames)
            if point_maps is not None:
                point_maps = pad_temporal_tensor_to_frames(point_maps, aligned_num_frames)
            if video is not None:
                video = pad_video_frames_to_length(video, aligned_num_frames)
        return aligned_num_frames, motion, point_maps, video

    def latent_shape(self, height: int, width: int, num_frames: int) -> tuple[int, int, int]:
        latent_frames = (num_frames - 1) // 4 + 1
        latent_height = height // self.vae.upsampling_factor
        latent_width = width // self.vae.upsampling_factor
        return latent_frames, latent_height, latent_width

    def compose_scene_condition(
        self,
        scene_state: ScenePointCloudState | None,
        rendered_condition: SceneCondition | torch.Tensor | str | Path | None,
    ) -> SceneCondition:
        scene_condition = normalize_scene_condition(rendered_condition)
        if not self.use_scene_memory or scene_state is None or scene_state.latent_history is None:
            return scene_condition
        if scene_condition.memory_latents is None:
            scene_condition.memory_latents = scene_state.latent_history
            return scene_condition
        history = torch.as_tensor(scene_state.latent_history)
        memory_latents = torch.as_tensor(scene_condition.memory_latents).to(device=history.device, dtype=history.dtype)
        if history.shape[1] != memory_latents.shape[1]:
            raise ValueError(
                "Explicit scene memory latents must use the same channel count as the internal scene-memory state."
            )
        scene_condition.memory_latents = torch.cat([history, memory_latents], dim=2)
        if (
            self.scene_memory_max_latent_frames is not None
            and scene_condition.memory_latents.shape[2] > self.scene_memory_max_latent_frames
        ):
            scene_condition.memory_latents = scene_condition.memory_latents[:, :, -self.scene_memory_max_latent_frames:]
        return scene_condition

    def update_scene_memory(
        self,
        scene_state: ScenePointCloudState | None,
        *,
        video: list[Image.Image],
        height: int,
        width: int,
        chunk_overlap: int = 1,
        tiled: bool = False,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
    ) -> ScenePointCloudState | None:
        if not self.use_scene_memory:
            return scene_state
        state = ScenePointCloudState() if scene_state is None else scene_state
        with torch.no_grad():
            chunk_latents = self.encode_video(
                video,
                height=height,
                width=width,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            ).detach()
        if state.latent_history is not None and chunk_overlap >= 1 and chunk_latents.shape[2] > 1:
            chunk_latents = chunk_latents[:, :, 1:]
        if state.latent_history is None:
            state.latent_history = chunk_latents
        else:
            state.latent_history = torch.cat([state.latent_history, chunk_latents], dim=2)
        if (
            self.scene_memory_max_latent_frames is not None
            and state.latent_history.shape[2] > self.scene_memory_max_latent_frames
        ):
            state.latent_history = state.latent_history[:, :, -self.scene_memory_max_latent_frames:]
        state.latent_history = state.latent_history.detach()
        return state

    def prepare_autoregressive_loop(
        self,
        *,
        input_image: Image.Image,
        motion: MotionSequence | str | Path,
        point_maps: torch.Tensor | str | Path | None = None,
        point_map_renderer: BasePointMapRenderer | None = None,
        scene_conditioner: BasePointMapRenderer | None = None,
        height: int = 480,
        width: int = 480,
        num_frames: int | None = None,
        chunk_num_frames: int = 49,
        chunk_overlap: int = 1,
        video: list[Image.Image] | None = None,
    ) -> AutoregressiveLoopContext:
        if chunk_overlap != 1:
            raise NotImplementedError("The I2V continuation pipeline only supports a 1-frame overlap between chunks.")

        height, width, _ = self.check_resize_height_width(height, width, 1, verbose=0)
        _, _, chunk_num_frames = self.check_resize_height_width(height, width, chunk_num_frames, verbose=0)

        motion_sequence = self._load_motion_sequence(motion)
        requested_num_frames = motion_sequence.num_frames if num_frames is None else num_frames

        raw_point_maps = None
        explicit_scene_conditioner = scene_conditioner or point_map_renderer
        if point_maps is not None and explicit_scene_conditioner is None:
            raw_point_maps = load_point_maps(point_maps) if isinstance(point_maps, (str, Path)) else torch.as_tensor(point_maps)

        prepared_video = None if video is None else list(video)
        total_num_frames, motion_sequence, raw_point_maps, prepared_video = self._normalize_total_num_frames(
            motion_sequence,
            num_frames,
            point_maps=raw_point_maps,
            video=prepared_video,
        )
        if prepared_video is not None:
            prepared_video = [frame.resize((width, height)) for frame in prepared_video]

        renderer = self._resolve_renderer(raw_point_maps, point_map_renderer, scene_conditioner)
        windows = build_temporal_windows(total_num_frames, chunk_num_frames, overlap=chunk_overlap)
        carry_image = input_image.resize((width, height))
        scene_state = renderer.initialize(input_image=carry_image, height=height, width=width)

        return AutoregressiveLoopContext(
            height=height,
            width=width,
            requested_num_frames=requested_num_frames,
            total_num_frames=total_num_frames,
            chunk_num_frames=chunk_num_frames,
            chunk_overlap=chunk_overlap,
            motion_sequence=motion_sequence,
            renderer=renderer,
            scene_state=scene_state,
            carry_image=carry_image,
            windows=windows,
            video=prepared_video,
        )

    @torch.no_grad()
    def _generate_clip_from_embeddings(
        self,
        *,
        context: torch.Tensor,
        negative_context: torch.Tensor,
        input_image: Image.Image,
        motion: MotionSequence | str | Path,
        height: int,
        width: int,
        num_frames: int,
        scene_condition: SceneCondition | torch.Tensor | str | Path | None = None,
        seed: int | None = None,
        cfg_scale: float = 7.5,
        num_inference_steps: int = 50,
        sigma_shift: float = 5.0,
        tiled: bool = False,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
    ) -> list[Image.Image]:
        height, width, num_frames = self.check_resize_height_width(height, width, num_frames)
        motion = self._prepare_motion(motion)
        scene_condition = normalize_scene_condition(scene_condition)
        if scene_condition.point_maps is not None:
            scene_condition.point_maps = self._prepare_point_maps(scene_condition.point_maps)
        if scene_condition.memory_latents is not None:
            scene_condition.memory_latents = torch.as_tensor(scene_condition.memory_latents).to(
                device=self.device,
                dtype=self.torch_dtype,
            )
        if scene_condition.context_tokens is not None:
            scene_condition.context_tokens = torch.as_tensor(scene_condition.context_tokens).to(
                device=self.device,
                dtype=self.torch_dtype,
            )
        self.scheduler.set_timesteps(num_inference_steps, shift=sigma_shift)

        first_frame_latents = self.encode_first_frame(
            image=input_image,
            height=height,
            width=width,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )

        latent_frames, latent_height, latent_width = self.latent_shape(height, width, num_frames)
        latents = self.generate_noise(
            shape=(1, self.backbone.latent_channels, latent_frames, latent_height, latent_width),
            seed=seed,
            rand_device="cpu",
        )
        latents[:, :, 0:1] = first_frame_latents

        for timestep in self.scheduler.timesteps:
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            noise_pred_posi = self.backbone(
                latents=latents,
                timestep=timestep,
                context=context,
                first_frame_latents=first_frame_latents,
                motion=motion,
                scene_condition=scene_condition,
                original_height=height,
                original_width=width,
            )
            if cfg_scale != 1.0:
                noise_pred_nega = self.backbone(
                    latents=latents,
                    timestep=timestep,
                    context=negative_context,
                    first_frame_latents=first_frame_latents,
                    motion=motion,
                    scene_condition=scene_condition,
                    original_height=height,
                    original_width=width,
                )
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi
            latents = self.scheduler.step(noise_pred, timestep, latents)
            latents[:, :, 0:1] = first_frame_latents

        return self.decode_latents(latents, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        input_image: Image.Image,
        motion: MotionSequence | str | Path,
        negative_prompt: str = "",
        height: int = 480,
        width: int = 480,
        num_frames: int = 49,
        point_maps: torch.Tensor | str | Path | None = None,
        seed: int | None = None,
        cfg_scale: float = 7.5,
        num_inference_steps: int = 50,
        sigma_shift: float = 5.0,
        tiled: bool = False,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
    ) -> list[Image.Image]:
        context = self.encode_prompt(prompt)
        negative_context = self.encode_prompt(negative_prompt)
        return self._generate_clip_from_embeddings(
            context=context,
            negative_context=negative_context,
            input_image=input_image,
            motion=motion,
            height=height,
            width=width,
            num_frames=num_frames,
            scene_condition=point_maps,
            seed=seed,
            cfg_scale=cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )

    @torch.no_grad()
    def generate_long_video(
        self,
        prompt: str,
        input_image: Image.Image,
        motion: MotionSequence | str | Path,
        negative_prompt: str = "",
        point_maps: torch.Tensor | str | Path | None = None,
        point_map_renderer: BasePointMapRenderer | None = None,
        scene_conditioner: BasePointMapRenderer | None = None,
        height: int = 480,
        width: int = 480,
        num_frames: int | None = None,
        chunk_num_frames: int = 49,
        chunk_overlap: int = 1,
        seed: int | None = None,
        cfg_scale: float = 7.5,
        num_inference_steps: int = 50,
        sigma_shift: float = 5.0,
        tiled: bool = False,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
    ) -> AutoregressiveGenerationResult:
        loop_context = self.prepare_autoregressive_loop(
            input_image=input_image,
            motion=motion,
            point_maps=point_maps,
            point_map_renderer=point_map_renderer,
            scene_conditioner=scene_conditioner,
            height=height,
            width=width,
            num_frames=num_frames,
            chunk_num_frames=chunk_num_frames,
            chunk_overlap=chunk_overlap,
        )

        context = self.encode_prompt(prompt)
        negative_context = self.encode_prompt(negative_prompt)
        carry_image = loop_context.carry_image
        scene_state = loop_context.scene_state
        stitched_video: list[Image.Image] = []
        chunk_results: list[AutoregressiveChunkResult] = []

        for chunk_index, (start_frame, stop_frame) in enumerate(loop_context.windows):
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
            scene_condition = self.compose_scene_condition(scene_state, rendered_condition)
            chunk_seed = None if seed is None else seed + chunk_index
            chunk_video = self._generate_clip_from_embeddings(
                context=context,
                negative_context=negative_context,
                input_image=carry_image,
                motion=motion_chunk,
                height=loop_context.height,
                width=loop_context.width,
                num_frames=stop_frame - start_frame,
                scene_condition=scene_condition,
                seed=chunk_seed,
                cfg_scale=cfg_scale,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
            scene_state = loop_context.renderer.update(
                scene_state,
                video=chunk_video,
                motion=motion_chunk,
                start_frame=start_frame,
                stop_frame=stop_frame,
                height=loop_context.height,
                width=loop_context.width,
            )
            scene_state = self.update_scene_memory(
                scene_state,
                video=chunk_video,
                height=loop_context.height,
                width=loop_context.width,
                chunk_overlap=loop_context.chunk_overlap,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
            stored_point_maps = (
                None if scene_condition.point_maps is None else torch.as_tensor(scene_condition.point_maps).detach().cpu()
            )
            stored_scene_memory_latents = (
                None
                if scene_condition.memory_latents is None
                else torch.as_tensor(scene_condition.memory_latents).detach().cpu()
            )
            chunk_results.append(
                AutoregressiveChunkResult(
                    index=chunk_index,
                    start_frame=start_frame,
                    stop_frame=stop_frame,
                    seed=chunk_seed,
                    video=chunk_video,
                    point_maps=stored_point_maps,
                    scene_memory_latents=stored_scene_memory_latents,
                )
            )
            if chunk_index == 0:
                stitched_video.extend(chunk_video)
            else:
                stitched_video.extend(chunk_video[loop_context.chunk_overlap:])
            carry_image = chunk_video[-1]

        stitched_video = stitched_video[:loop_context.requested_num_frames]
        return AutoregressiveGenerationResult(video=stitched_video, chunks=chunk_results, scene_state=scene_state)
