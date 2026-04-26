from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffsynth.models.wan_video_camera_controller import SimpleAdapter
from diffsynth.models.wan_video_dit import WanModel

from .modules import PlayerOneBackbone
from .pipeline import PlayerOnePipeline


class TinyTokenizer:
    def __init__(self, max_length: int = 16, vocab_size: int = 2048):
        self.max_length = max_length
        self.vocab_size = vocab_size

    def _token_id(self, token: str) -> int:
        value = sum(ord(char) * (index + 1) for index, char in enumerate(token))
        return 2 + (value % max(1, self.vocab_size - 2))

    def __call__(self, prompt: str, return_mask: bool = True, add_special_tokens: bool = True):
        words = prompt.strip().split()
        token_ids: list[int] = []
        if add_special_tokens:
            token_ids.append(1)
        token_ids.extend(self._token_id(word) for word in words[: self.max_length - 2])
        if add_special_tokens:
            token_ids.append(1)
        if not token_ids:
            token_ids = [1]
        token_ids = token_ids[: self.max_length]
        ids = torch.zeros(1, self.max_length, dtype=torch.long)
        mask = torch.zeros(1, self.max_length, dtype=torch.long)
        ids[0, : len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
        mask[0, : len(token_ids)] = 1
        if return_mask:
            return ids, mask
        return ids


class TinyTextEncoder(nn.Module):
    def __init__(self, vocab_size: int = 2048, embedding_dim: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        embeddings = self.embedding(ids)
        if mask is not None:
            embeddings = embeddings * mask.unsqueeze(-1)
        return embeddings


class TinyVideoVAE(nn.Module):
    def __init__(self, latent_channels: int = 16, input_channels: int = 3, upsampling_factor: int = 8):
        super().__init__()
        self.z_dim = latent_channels
        self.upsampling_factor = upsampling_factor
        self.encoder_projection = nn.Conv2d(input_channels, latent_channels, kernel_size=1)
        self.decoder_projection = nn.Conv2d(latent_channels, input_channels, kernel_size=1)

    def _compress_time(self, video_latents: torch.Tensor) -> torch.Tensor:
        if video_latents.shape[2] == 1:
            return video_latents
        if (video_latents.shape[2] - 1) % 4 != 0:
            raise ValueError(
                "TinyVideoVAE expects video frames aligned to Wan's 1 mod 4 temporal layout, "
                f"got {video_latents.shape[2]} frames."
            )
        first_frame = video_latents[:, :, :1]
        residual = video_latents[:, :, 1:]
        batch, channels, _, height, width = residual.shape
        residual = residual.reshape(batch, channels, -1, 4, height, width).mean(dim=3)
        return torch.cat([first_frame, residual], dim=2)

    def encode(
        self,
        video: torch.Tensor,
        device: str | torch.device | None = None,
        tiled: bool = False,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
    ) -> torch.Tensor:
        del tiled, tile_size, tile_stride
        if device is not None:
            video = video.to(device)
        batch, channels, frames, height, width = video.shape
        video = video.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        video = F.avg_pool2d(video, kernel_size=self.upsampling_factor, stride=self.upsampling_factor)
        video = self.encoder_projection(video)
        latent_height, latent_width = video.shape[-2:]
        video = video.reshape(batch, frames, self.z_dim, latent_height, latent_width).permute(0, 2, 1, 3, 4).contiguous()
        return self._compress_time(video)

    def decode(
        self,
        latents: torch.Tensor,
        device: str | torch.device | None = None,
        tiled: bool = False,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
    ) -> torch.Tensor:
        del tiled, tile_size, tile_stride
        if device is not None:
            latents = latents.to(device)
        batch, channels, latent_frames, latent_height, latent_width = latents.shape
        if latent_frames == 1:
            expanded = latents
        else:
            repeated = latents[:, :, 1:].unsqueeze(3).expand(-1, -1, -1, 4, -1, -1)
            expanded = torch.cat(
                [
                    latents[:, :, :1],
                    repeated.reshape(batch, channels, (latent_frames - 1) * 4, latent_height, latent_width),
                ],
                dim=2,
            )
        expanded = expanded.permute(0, 2, 1, 3, 4).reshape(batch * expanded.shape[2], channels, latent_height, latent_width)
        expanded = self.decoder_projection(expanded)
        expanded = F.interpolate(
            expanded,
            scale_factor=self.upsampling_factor,
            mode="bilinear",
            align_corners=False,
        )
        expanded = torch.tanh(expanded)
        video_height, video_width = expanded.shape[-2:]
        expanded = expanded.reshape(batch, -1, 3, video_height, video_width).permute(0, 2, 1, 3, 4).contiguous()
        return expanded


def build_tiny_playerone_pipeline(
    *,
    device: str | torch.device = "cpu",
    torch_dtype: torch.dtype = torch.float32,
    latent_channels: int = 16,
    text_dim: int = 32,
) -> PlayerOnePipeline:
    tokenizer = TinyTokenizer(max_length=16, vocab_size=2048)
    text_encoder = TinyTextEncoder(vocab_size=2048, embedding_dim=text_dim)
    vae = TinyVideoVAE(latent_channels=latent_channels, input_channels=3, upsampling_factor=8)

    dit = WanModel(
        dim=64,
        in_dim=latent_channels,
        ffn_dim=128,
        out_dim=latent_channels,
        text_dim=text_dim,
        freq_dim=32,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=4,
        num_layers=2,
        has_image_input=False,
        seperated_timestep=True,
        require_vae_embedding=False,
        require_clip_embedding=False,
        fuse_vae_embedding_in_latents=True,
    )
    dit.control_adapter = SimpleAdapter(
        in_dim=24,
        out_dim=dit.dim,
        kernel_size=dit.patch_size[1:],
        stride=dit.patch_size[1:],
    )
    backbone = PlayerOneBackbone(
        wan_dit=dit,
        latent_channels=latent_channels,
        motion_channels_per_part=1,
        motion_hidden_channels=16,
        motion_num_layers=2,
        point_input_channels=3,
        point_hidden_channels=16,
        point_latent_channels=8,
        point_num_layers=2,
        scene_hidden_channels=16,
        scene_latent_channels=8,
        scene_num_layers=2,
        motion_context_tokens_per_part=2,
        motion_context_hidden_dim=32,
        scene_context_tokens=2,
        scene_context_hidden_dim=32,
    )
    return PlayerOnePipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        backbone=backbone,
        device=device,
        torch_dtype=torch_dtype,
        use_scene_memory=True,
        scene_memory_max_latent_frames=16,
    )
