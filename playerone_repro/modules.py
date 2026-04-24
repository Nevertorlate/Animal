from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ._env import PROJECT_ROOT  # noqa: F401
from .data import MotionSequence, BODY_FEET_DIM, HAND_DIM, HEAD_DIM
from .rendering import SceneCondition, normalize_scene_condition

from diffsynth.core import gradient_checkpoint_forward
from diffsynth.models.wan_video_camera_controller import SimpleAdapter, process_pose_file
from diffsynth.models.wan_video_dit import WanModel, sinusoidal_embedding_1d


class ResidualBlock3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(1, channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class VectorToVolumeEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_channels: int,
        hidden_channels: int = 64,
        base_grid: tuple[int, int] = (4, 4),
        num_layers: int = 8,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.base_grid = base_grid
        self.input_projection = nn.Linear(input_dim, hidden_channels * base_grid[0] * base_grid[1])
        self.conv_in = nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([ResidualBlock3D(hidden_channels) for _ in range(num_layers)])
        self.conv_out = nn.Conv3d(hidden_channels, output_channels, kernel_size=3, padding=1)

    def forward(self, sequence: torch.Tensor, target_shape: tuple[int, int, int]) -> torch.Tensor:
        batch, frames, _ = sequence.shape
        x = self.input_projection(sequence)
        x = x.view(batch, frames, self.hidden_channels, self.base_grid[0], self.base_grid[1])
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        if x.shape[2:] != target_shape:
            x = F.interpolate(x, size=target_shape, mode="trilinear", align_corners=False)
        x = self.conv_in(x)
        for block in self.blocks:
            x = block(x)
        return self.conv_out(x)


class PartDisentangledMotionEncoder(nn.Module):
    def __init__(
        self,
        motion_channels_per_part: int = 1,
        hidden_channels: int = 64,
        base_grid: tuple[int, int] = (4, 4),
        num_layers: int = 8,
    ):
        super().__init__()
        self.body_feet_encoder = VectorToVolumeEncoder(
            input_dim=BODY_FEET_DIM,
            output_channels=motion_channels_per_part,
            hidden_channels=hidden_channels,
            base_grid=base_grid,
            num_layers=num_layers,
        )
        self.hands_encoder = VectorToVolumeEncoder(
            input_dim=HAND_DIM * 2,
            output_channels=motion_channels_per_part,
            hidden_channels=hidden_channels,
            base_grid=base_grid,
            num_layers=num_layers,
        )
        self.head_encoder = VectorToVolumeEncoder(
            input_dim=HEAD_DIM,
            output_channels=motion_channels_per_part,
            hidden_channels=hidden_channels,
            base_grid=base_grid,
            num_layers=num_layers,
        )

    @property
    def output_channels(self) -> int:
        return int(self.body_feet_encoder.conv_out.out_channels * 3)

    def forward(self, motion: MotionSequence, target_shape: tuple[int, int, int]) -> torch.Tensor:
        body_feet = self.body_feet_encoder(motion.body_feet, target_shape)
        hands = self.hands_encoder(motion.hands, target_shape)
        head = self.head_encoder(motion.head, target_shape)
        return torch.cat([body_feet, hands, head], dim=1)


def normalize_condition_layout(volume: torch.Tensor) -> torch.Tensor:
    if volume.ndim == 4:
        volume = volume.unsqueeze(0)
    if volume.ndim != 5:
        raise ValueError(f"Condition tensor must be 5D, got {tuple(volume.shape)}")
    if volume.shape[-1] in (3, 6):
        volume = rearrange(volume, "b t h w c -> b c t h w")
    return volume


class VolumeConditionEncoderWithAdapter(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        hidden_channels: int = 64,
        output_channels: int = 64,
        num_layers: int = 5,
    ):
        super().__init__()
        self.conv_in = nn.Conv3d(input_channels, hidden_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([ResidualBlock3D(hidden_channels) for _ in range(num_layers)])
        self.adapter = nn.Sequential(
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden_channels, output_channels, kernel_size=3, padding=1),
        )

    def _normalize_layout(self, volume: torch.Tensor) -> torch.Tensor:
        volume = normalize_condition_layout(volume)
        expected_channels = self.conv_in.in_channels
        actual_channels = volume.shape[1]
        if actual_channels == expected_channels:
            return volume
        if actual_channels > expected_channels:
            return volume[:, :expected_channels]
        pad = volume.new_zeros(volume.shape[0], expected_channels - actual_channels, *volume.shape[2:])
        return torch.cat([volume, pad], dim=1)

    def forward(self, volume: torch.Tensor, target_shape: tuple[int, int, int]) -> torch.Tensor:
        volume = self._normalize_layout(volume)
        if volume.shape[2:] != target_shape:
            volume = F.interpolate(volume, size=target_shape, mode="trilinear", align_corners=False)
        x = self.conv_in(volume)
        for block in self.blocks:
            x = block(x)
        return self.adapter(x)


class PointMapEncoderWithAdapter(VolumeConditionEncoderWithAdapter):
    pass


class MotionTokenEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_tokens: int = 4,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.shape[1] != self.num_tokens:
            sequence = sequence.transpose(1, 2)
            sequence = F.interpolate(sequence, size=self.num_tokens, mode="linear", align_corners=False)
            sequence = sequence.transpose(1, 2)
        return self.proj(sequence)


class MotionContextEncoder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        tokens_per_part: int = 4,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.body_feet_encoder = MotionTokenEncoder(
            BODY_FEET_DIM,
            output_dim=output_dim,
            num_tokens=tokens_per_part,
            hidden_dim=hidden_dim,
        )
        self.hands_encoder = MotionTokenEncoder(
            HAND_DIM * 2,
            output_dim=output_dim,
            num_tokens=tokens_per_part,
            hidden_dim=hidden_dim,
        )
        self.head_encoder = MotionTokenEncoder(
            HEAD_DIM,
            output_dim=output_dim,
            num_tokens=tokens_per_part,
            hidden_dim=hidden_dim,
        )
        self.summary = nn.Sequential(
            nn.LayerNorm(output_dim * 3),
            nn.Linear(output_dim * 3, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, motion: MotionSequence) -> torch.Tensor:
        body_tokens = self.body_feet_encoder(motion.body_feet)
        hands_tokens = self.hands_encoder(motion.hands)
        head_tokens = self.head_encoder(motion.head)
        summary = self.summary(
            torch.cat(
                [
                    body_tokens.mean(dim=1),
                    hands_tokens.mean(dim=1),
                    head_tokens.mean(dim=1),
                ],
                dim=-1,
            )
        ).unsqueeze(1)
        return torch.cat([summary, body_tokens, hands_tokens, head_tokens], dim=1)


class SceneLatentContextEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_tokens: int = 4,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.token_encoder = MotionTokenEncoder(
            input_dim=input_dim,
            output_dim=output_dim,
            num_tokens=num_tokens,
            hidden_dim=hidden_dim,
        )

    def forward(self, scene_memory_latents: torch.Tensor) -> torch.Tensor:
        scene_memory_latents = normalize_condition_layout(scene_memory_latents)
        sequence = scene_memory_latents.mean(dim=(-1, -2)).transpose(1, 2)
        return self.token_encoder(sequence)


def resample_sequence(sequence: torch.Tensor, target_frames: int) -> torch.Tensor:
    if sequence.shape[1] == target_frames:
        return sequence
    sequence = sequence.transpose(1, 2)
    sequence = F.interpolate(sequence, size=target_frames, mode="linear", align_corners=False)
    return sequence.transpose(1, 2)


def axis_angle_to_rotation_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    safe_theta = theta.clamp_min(1e-8)
    axis = axis_angle / safe_theta
    x, y, z = axis.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    cross = torch.stack(
        [
            zeros, -z, y,
            z, zeros, -x,
            -y, x, zeros,
        ],
        dim=-1,
    ).view(*axis.shape[:-1], 3, 3)
    eye = torch.eye(3, device=axis.device, dtype=axis.dtype).expand(*axis.shape[:-1], 3, 3)
    sin_theta = torch.sin(theta)[..., None]
    cos_theta = torch.cos(theta)[..., None]
    return eye + sin_theta * cross + (1 - cos_theta) * (cross @ cross)


def head_rotations_to_camera_coordinates(head_rotations: torch.Tensor) -> list[list[float]]:
    rotations = axis_angle_to_rotation_matrix(head_rotations)
    coordinates: list[list[float]] = []
    for rotation in rotations:
        entry = [
            0.0,
            0.532139961,
            0.946026558,
            0.5,
            0.5,
            0.0,
            0.0,
            float(rotation[0, 0]),
            float(rotation[0, 1]),
            float(rotation[0, 2]),
            0.0,
            float(rotation[1, 0]),
            float(rotation[1, 1]),
            float(rotation[1, 2]),
            0.0,
            float(rotation[2, 0]),
            float(rotation[2, 1]),
            float(rotation[2, 2]),
            0.0,
        ]
        coordinates.append(entry)
    return coordinates


def head_rotations_to_control_latents(
    head_rotations: torch.Tensor,
    num_frames: int,
    height: int,
    width: int,
    device: str | torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    latent_outputs = []
    resampled = resample_sequence(head_rotations.float(), num_frames)
    for sample in resampled:
        coordinates = head_rotations_to_camera_coordinates(sample.cpu())
        plucker = process_pose_file(coordinates, width=width, height=height, device="cpu")
        control_camera_video = plucker[:num_frames].permute(3, 0, 1, 2).unsqueeze(0)
        packed = torch.cat(
            [
                torch.repeat_interleave(control_camera_video[:, :, 0:1], repeats=4, dim=2),
                control_camera_video[:, :, 1:],
            ],
            dim=2,
        ).transpose(1, 2)
        batch, frames, channels, image_h, image_w = packed.shape
        packed = packed.contiguous().view(batch, frames // 4, 4, channels, image_h, image_w).transpose(2, 3)
        packed = packed.contiguous().view(batch, frames // 4, channels * 4, image_h, image_w).transpose(1, 2)
        latent_outputs.append(packed.squeeze(0))
    return torch.stack(latent_outputs, dim=0).to(device=device, dtype=dtype)


class PlayerOneBackbone(nn.Module):
    def __init__(
        self,
        wan_dit: WanModel,
        latent_channels: int | None = None,
        motion_channels_per_part: int = 1,
        motion_hidden_channels: int = 64,
        motion_num_layers: int = 8,
        point_input_channels: int = 9,
        point_hidden_channels: int = 64,
        point_latent_channels: int = 64,
        point_num_layers: int = 5,
        scene_hidden_channels: int = 64,
        scene_latent_channels: int = 64,
        scene_num_layers: int = 5,
        motion_context_tokens_per_part: int = 4,
        motion_context_hidden_dim: int = 128,
        scene_context_tokens: int = 4,
        scene_context_hidden_dim: int = 128,
    ):
        super().__init__()
        self.wan_dit = wan_dit
        self.latent_channels = int(self.wan_dit.in_dim if latent_channels is None else latent_channels)
        if self.wan_dit.control_adapter is None:
            self.wan_dit.control_adapter = SimpleAdapter(
                in_dim=24,
                out_dim=self.wan_dit.dim,
                kernel_size=self.wan_dit.patch_size[1:],
                stride=self.wan_dit.patch_size[1:],
            )
        self.motion_encoder = PartDisentangledMotionEncoder(
            motion_channels_per_part=motion_channels_per_part,
            hidden_channels=motion_hidden_channels,
            num_layers=motion_num_layers,
        )
        self.point_map_encoder = PointMapEncoderWithAdapter(
            input_channels=point_input_channels,
            hidden_channels=point_hidden_channels,
            output_channels=point_latent_channels,
            num_layers=point_num_layers,
        )
        self.scene_memory_encoder = VolumeConditionEncoderWithAdapter(
            input_channels=self.latent_channels,
            hidden_channels=scene_hidden_channels,
            output_channels=scene_latent_channels,
            num_layers=scene_num_layers,
        )
        self.motion_context_encoder = MotionContextEncoder(
            output_dim=self.wan_dit.dim,
            tokens_per_part=motion_context_tokens_per_part,
            hidden_dim=motion_context_hidden_dim,
        )
        self.scene_context_encoder = SceneLatentContextEncoder(
            input_dim=self.latent_channels,
            output_dim=self.wan_dit.dim,
            num_tokens=scene_context_tokens,
            hidden_dim=scene_context_hidden_dim,
        )
        self.point_latent_channels = point_latent_channels
        self.scene_latent_channels = scene_latent_channels
        fused_channels = (
            self.latent_channels
            + self.latent_channels
            + self.motion_encoder.output_channels
            + self.point_latent_channels
            + self.scene_latent_channels
        )
        self.input_adapter = nn.Sequential(
            nn.Conv3d(fused_channels, self.wan_dit.in_dim, kernel_size=1),
            nn.SiLU(),
            ResidualBlock3D(self.wan_dit.in_dim),
        )

    def expected_frame_count(self, latent_frames: int) -> int:
        return 4 * (latent_frames - 1) + 1

    def build_condition_latents(
        self,
        latents: torch.Tensor,
        first_frame_latents: torch.Tensor,
        motion: MotionSequence,
        scene_condition: SceneCondition | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latents.shape[1] != self.latent_channels:
            raise ValueError(
                f"Expected noisy latents with {self.latent_channels} channels, got {latents.shape[1]}. "
                "This usually means the Wan DiT input channels and VAE latent channels were mixed up."
            )
        if first_frame_latents.shape[1] != self.latent_channels:
            raise ValueError(
                f"Expected first-frame latents with {self.latent_channels} channels, got {first_frame_latents.shape[1]}."
            )
        scene_condition = normalize_scene_condition(scene_condition)
        target_shape = latents.shape[2:]
        first_frame = first_frame_latents.expand(-1, -1, target_shape[0], -1, -1)
        motion_latents = self.motion_encoder(motion, target_shape)
        point_maps = scene_condition.point_maps
        scene_memory_latents = scene_condition.memory_latents
        if point_maps is None:
            point_latents = torch.zeros(
                latents.shape[0],
                self.point_latent_channels,
                *target_shape,
                device=latents.device,
                dtype=latents.dtype,
            )
        else:
            point_latents = self.point_map_encoder(point_maps.to(device=latents.device, dtype=latents.dtype), target_shape)
        if scene_memory_latents is None:
            scene_memory = torch.zeros(
                latents.shape[0],
                self.scene_latent_channels,
                *target_shape,
                device=latents.device,
                dtype=latents.dtype,
            )
        else:
            scene_memory = self.scene_memory_encoder(
                torch.as_tensor(scene_memory_latents).to(device=latents.device, dtype=latents.dtype),
                target_shape,
            )
        fused = torch.cat([latents, first_frame, motion_latents, point_latents, scene_memory], dim=1)
        return self.input_adapter(fused)

    def build_condition_context(
        self,
        context: torch.Tensor,
        motion: MotionSequence,
        scene_condition: SceneCondition | torch.Tensor | None = None,
    ) -> torch.Tensor:
        scene_condition = normalize_scene_condition(scene_condition)
        text_context = self.wan_dit.text_embedding(context)
        extra_tokens = [
            self.motion_context_encoder(motion).to(device=text_context.device, dtype=text_context.dtype),
        ]
        if scene_condition.memory_latents is not None:
            scene_tokens = self.scene_context_encoder(
                torch.as_tensor(scene_condition.memory_latents).to(device=text_context.device, dtype=text_context.dtype)
            )
            extra_tokens.append(scene_tokens)
        if scene_condition.context_tokens is not None:
            context_tokens = torch.as_tensor(scene_condition.context_tokens).to(device=text_context.device, dtype=text_context.dtype)
            if context_tokens.ndim == 2:
                context_tokens = context_tokens.unsqueeze(0)
            if context_tokens.shape[-1] != text_context.shape[-1]:
                raise ValueError(
                    f"Scene context tokens must have dim={text_context.shape[-1]}, got {context_tokens.shape[-1]}."
                )
            if context_tokens.shape[0] != text_context.shape[0]:
                if context_tokens.shape[0] == 1:
                    context_tokens = context_tokens.expand(text_context.shape[0], -1, -1)
                else:
                    raise ValueError(
                        f"Scene context tokens batch size must be 1 or {text_context.shape[0]}, got {context_tokens.shape[0]}."
                    )
            extra_tokens.append(context_tokens)
        return torch.cat([text_context, *extra_tokens], dim=1)

    def build_time_embeddings(self, latents: torch.Tensor, timestep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.wan_dit.seperated_timestep and self.wan_dit.fuse_vae_embedding_in_latents:
            patch_h, patch_w = self.wan_dit.patch_size[1:]
            spatial_tokens = latents.shape[3] * latents.shape[4] // (patch_h * patch_w)
            timestep = torch.cat(
                [
                    torch.zeros((1, spatial_tokens), dtype=timestep.dtype, device=timestep.device),
                    torch.ones((latents.shape[2] - 1, spatial_tokens), dtype=timestep.dtype, device=timestep.device) * timestep.view(1, 1),
                ],
                dim=0,
            ).flatten()
            t = self.wan_dit.time_embedding(sinusoidal_embedding_1d(self.wan_dit.freq_dim, timestep).unsqueeze(0))
            t_mod = self.wan_dit.time_projection(t).unflatten(2, (6, self.wan_dit.dim))
        else:
            t = self.wan_dit.time_embedding(sinusoidal_embedding_1d(self.wan_dit.freq_dim, timestep))
            t_mod = self.wan_dit.time_projection(t).unflatten(1, (6, self.wan_dit.dim))
        return t, t_mod

    def build_camera_condition(
        self,
        motion: MotionSequence,
        latent_frames: int,
        height: int,
        width: int,
        device: str | torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if self.wan_dit.control_adapter is None:
            return None
        return head_rotations_to_control_latents(
            head_rotations=motion.head,
            num_frames=self.expected_frame_count(latent_frames),
            height=height,
            width=width,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        first_frame_latents: torch.Tensor,
        motion: MotionSequence,
        scene_condition: SceneCondition | torch.Tensor | None = None,
        original_height: int | None = None,
        original_width: int | None = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ) -> torch.Tensor:
        if original_height is None:
            original_height = latents.shape[3] * 8
        if original_width is None:
            original_width = latents.shape[4] * 8

        x = self.build_condition_latents(latents, first_frame_latents, motion, scene_condition)
        camera_latents = self.build_camera_condition(
            motion=motion,
            latent_frames=latents.shape[2],
            height=original_height,
            width=original_width,
            device=latents.device,
            dtype=latents.dtype,
        )

        t, t_mod = self.build_time_embeddings(latents, timestep)
        context = self.build_condition_context(context, motion, scene_condition)

        x = self.wan_dit.patchify(x, camera_latents)
        frames, height_tokens, width_tokens = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        freqs = torch.cat(
            [
                self.wan_dit.freqs[0][:frames].view(frames, 1, 1, -1).expand(frames, height_tokens, width_tokens, -1),
                self.wan_dit.freqs[1][:height_tokens].view(1, height_tokens, 1, -1).expand(frames, height_tokens, width_tokens, -1),
                self.wan_dit.freqs[2][:width_tokens].view(1, 1, width_tokens, -1).expand(frames, height_tokens, width_tokens, -1),
            ],
            dim=-1,
        ).reshape(frames * height_tokens * width_tokens, 1, -1).to(x.device)

        for block in self.wan_dit.blocks:
            if self.training:
                x = gradient_checkpoint_forward(
                    block,
                    use_gradient_checkpointing,
                    use_gradient_checkpointing_offload,
                    x,
                    context,
                    t_mod,
                    freqs,
                )
            else:
                x = block(x, context, t_mod, freqs)

        x = self.wan_dit.head(x, t)
        return self.wan_dit.unpatchify(x, (frames, height_tokens, width_tokens))
