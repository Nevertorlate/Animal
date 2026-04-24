from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "diffsynth-studio") not in sys.path:
    sys.path.insert(0, str(ROOT / "diffsynth-studio"))

from diffsynth.models.wan_video_dit import WanModel
from diffsynth.models.wan_video_camera_controller import SimpleAdapter

from playerone_repro import MotionSequence, PlayerOneBackbone, SceneCondition


def main() -> None:
    torch.manual_seed(0)

    dit = WanModel(
        dim=64,
        in_dim=16,
        ffn_dim=128,
        out_dim=16,
        text_dim=32,
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
        latent_channels=16,
        motion_channels_per_part=1,
        motion_hidden_channels=16,
        motion_num_layers=2,
        point_input_channels=3,
        point_hidden_channels=16,
        point_latent_channels=8,
        point_num_layers=2,
    )

    latents = torch.randn(1, 16, 5, 8, 8)
    first_frame = latents[:, :, :1].clone()
    context = torch.randn(1, 12, 32)
    timestep = torch.tensor([100.0])
    motion = MotionSequence(
        body_feet=torch.randn(1, 17, 66),
        head=torch.randn(1, 17, 3) * 0.1,
        hands=torch.randn(1, 17, 90),
    )
    point_maps = torch.randn(1, 3, 5, 8, 8)
    scene_memory_latents = torch.randn(1, 16, 5, 8, 8)

    output = backbone(
        latents=latents,
        timestep=timestep,
        context=context,
        first_frame_latents=first_frame,
        motion=motion,
        scene_condition=SceneCondition(
            point_maps=point_maps,
            memory_latents=scene_memory_latents,
        ),
        original_height=64,
        original_width=64,
    )
    assert output.shape == latents.shape, (output.shape, latents.shape)
    print("smoke test passed:", tuple(output.shape))


if __name__ == "__main__":
    main()
