from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "diffsynth-studio") not in sys.path:
    sys.path.insert(0, str(ROOT / "diffsynth-studio"))

from diffsynth.utils.data import save_video

from playerone_repro import PlayerOnePipeline, load_motion_sequence, load_point_maps


def load_renderer(factory_spec: str | None, config_path: str | None):
    if factory_spec is None:
        return None
    module_name, sep, attr_name = factory_spec.partition(":")
    if not sep:
        raise ValueError("Renderer factories must be provided as 'module.path:callable'.")
    module = importlib.import_module(module_name)
    factory = getattr(module, attr_name)
    kwargs = {}
    if config_path is not None:
        loaded = json.loads(Path(config_path).read_text())
        if not isinstance(loaded, dict):
            raise ValueError("Renderer config JSON must be an object mapping keyword names to values.")
        kwargs = loaded
    return factory(**kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PlayerOne-style DiffSynth reproduction.")
    parser.add_argument("--first-frame", required=True, help="Path to the first frame image.")
    parser.add_argument("--motion", required=True, help="Path to a motion .npz/.npy/.json file.")
    parser.add_argument("--prompt", required=True, help="Text prompt for the scene.")
    parser.add_argument("--negative-prompt", default="", help="Negative prompt.")
    parser.add_argument("--point-maps", default=None, help="Optional precomputed point-map .npy/.npz path for debugging the autoregressive scene-conditioning loop.")
    parser.add_argument("--renderer-factory", default=None, help="Optional custom scene-conditioner factory in 'module.path:callable' form.")
    parser.add_argument("--renderer-config", default=None, help="Optional JSON file passed as keyword args to --renderer-factory.")
    parser.add_argument("--scene-conditioner-factory", default=None, help="Alias of --renderer-factory.")
    parser.add_argument("--scene-conditioner-config", default=None, help="Alias of --renderer-config.")
    parser.add_argument("--output", default="playerone_repro.mp4", help="Output video path.")
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--num-frames", type=int, default=None, help="Optional total output frames. Defaults to the motion sequence length.")
    parser.add_argument("--chunk-frames", type=int, default=49, help="Frames per autoregressive I2V chunk. Must satisfy Wan's 1 mod 4 constraint after auto-alignment.")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--disable-scene-memory", action="store_true", help="Disable latent scene-memory conditioning from previously generated chunks.")
    parser.add_argument("--scene-memory-max-latent-frames", type=int, default=32, help="Maximum latent-history frames retained for autoregressive scene memory.")
    parser.add_argument(
        "--wan-model-id",
        default=None,
        help="Optional Wan 2.2 repo override. Useful if you have another single-DiT checkpoint with the same file layout as Wan2.2-TI2V-5B.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipe = PlayerOnePipeline.from_wan_pretrained(
        device=args.device,
        model_id_override=args.wan_model_id,
        use_scene_memory=not args.disable_scene_memory,
        scene_memory_max_latent_frames=args.scene_memory_max_latent_frames,
    )
    image = Image.open(args.first_frame).convert("RGB")
    motion = load_motion_sequence(args.motion)
    point_maps = None if args.point_maps is None else load_point_maps(args.point_maps)
    renderer_factory = args.scene_conditioner_factory or args.renderer_factory
    renderer_config = args.scene_conditioner_config or args.renderer_config
    scene_conditioner = load_renderer(renderer_factory, renderer_config)
    result = pipe.generate_long_video(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        input_image=image,
        motion=motion,
        point_maps=point_maps,
        scene_conditioner=scene_conditioner,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        chunk_num_frames=args.chunk_frames,
        num_inference_steps=args.steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
    )
    save_video(result.video, args.output, fps=args.fps, quality=5)


if __name__ == "__main__":
    main()
