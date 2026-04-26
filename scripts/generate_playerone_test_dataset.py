from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a small synthetic dataset for PlayerOne training smoke tests.")
    parser.add_argument("--output-dir", default="testdata/playerone_synth", help="Directory to write the synthetic dataset into.")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of clips to generate.")
    parser.add_argument("--num-frames", type=int, default=17, help="Frames per clip. Use 1 mod 4 to match Wan temporal alignment.")
    parser.add_argument("--height", type=int, default=64, help="Frame height.")
    parser.add_argument("--width", type=int, default=64, help="Frame width.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--without-point-maps", action="store_true", help="Skip synthetic point-map generation.")
    return parser.parse_args()


def build_palette(sample_index: int) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    hue = ((sample_index * 37) % 360) / 360.0
    bg = colorsys.hsv_to_rgb(hue, 0.45, 0.92)
    fg = colorsys.hsv_to_rgb((hue + 0.18) % 1.0, 0.55, 0.88)
    accent = colorsys.hsv_to_rgb((hue + 0.45) % 1.0, 0.7, 0.96)

    def to_rgb(color: tuple[float, float, float]) -> tuple[int, int, int]:
        return tuple(int(channel * 255) for channel in color)

    return to_rgb(bg), to_rgb(fg), to_rgb(accent)


def render_frame(
    *,
    sample_index: int,
    frame_index: int,
    num_frames: int,
    width: int,
    height: int,
    palette: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
) -> Image.Image:
    bg, fg, accent = palette
    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)

    t = frame_index / max(1, num_frames - 1)
    horizon = int(height * (0.52 + 0.04 * np.sin(2 * np.pi * t + sample_index * 0.3)))
    corridor_shift = int(width * 0.14 * np.sin(2 * np.pi * t * 1.5 + sample_index))
    orb_x = width * (0.5 + 0.22 * np.cos(2 * np.pi * t + sample_index * 0.5))
    orb_y = height * (0.42 + 0.12 * np.sin(2 * np.pi * t * 2.0 + sample_index * 0.25))

    for row in range(horizon):
        blend = row / max(1, horizon)
        color = tuple(
            int(bg[channel] * (1.0 - 0.3 * blend) + fg[channel] * 0.3 * blend)
            for channel in range(3)
        )
        draw.line((0, row, width, row), fill=color)

    floor_color = tuple(int(bg[channel] * 0.55 + 25) for channel in range(3))
    draw.rectangle((0, horizon, width, height), fill=floor_color)

    vanishing_left = width * 0.18 + corridor_shift
    vanishing_right = width * 0.82 + corridor_shift
    draw.polygon(
        [(0, height), (vanishing_left, horizon), (vanishing_right, horizon), (width, height)],
        fill=tuple(int(fg[channel] * 0.65) for channel in range(3)),
    )

    doorway_width = width * 0.18
    doorway_x = width * (0.5 + 0.1 * np.sin(2 * np.pi * t * 1.2 + sample_index * 0.7))
    draw.rectangle(
        (
            doorway_x - doorway_width / 2,
            horizon - height * 0.28,
            doorway_x + doorway_width / 2,
            horizon + height * 0.02,
        ),
        outline=accent,
        width=2,
    )

    orb_radius = width * (0.06 + 0.015 * np.sin(2 * np.pi * t * 1.7 + sample_index))
    draw.ellipse(
        (
            orb_x - orb_radius,
            orb_y - orb_radius,
            orb_x + orb_radius,
            orb_y + orb_radius,
        ),
        fill=accent,
    )

    hand_span = width * 0.16
    hand_y = height * (0.82 + 0.04 * np.cos(2 * np.pi * t * 2.3 + sample_index))
    left_hand_x = width * 0.22 + hand_span * np.sin(2 * np.pi * t * 1.8 + sample_index)
    right_hand_x = width * 0.78 + hand_span * np.cos(2 * np.pi * t * 1.6 + sample_index)
    hand_color = tuple(int(channel * 0.85) for channel in accent)
    draw.rounded_rectangle((left_hand_x - 6, hand_y - 4, left_hand_x + 14, hand_y + 10), radius=4, fill=hand_color)
    draw.rounded_rectangle((right_hand_x - 14, hand_y - 4, right_hand_x + 6, hand_y + 10), radius=4, fill=hand_color)
    return image


def build_motion_arrays(num_frames: int, sample_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)
    phase = sample_index * 0.37

    def wave(channels: int, frequency: float, amplitude: float) -> np.ndarray:
        values = []
        for channel in range(channels):
            channel_phase = phase + channel * 0.07
            values.append(amplitude * np.sin(2 * np.pi * frequency * t + channel_phase))
        return np.stack(values, axis=1).astype(np.float32)

    body_feet = wave(66, frequency=1.0, amplitude=0.8)
    body_feet[:, 0] += np.linspace(0.0, 0.35, num_frames, dtype=np.float32)
    head = np.stack(
        [
            0.08 * np.sin(2 * np.pi * 1.1 * t + phase),
            0.06 * np.sin(2 * np.pi * 0.7 * t + phase * 1.3),
            0.05 * np.cos(2 * np.pi * 0.9 * t + phase * 0.8),
        ],
        axis=1,
    ).astype(np.float32)
    left_hand = wave(45, frequency=1.8, amplitude=0.55)
    right_hand = wave(45, frequency=1.6, amplitude=0.55) * np.float32(-1.0)
    return body_feet, head, left_hand, right_hand


def build_point_maps(num_frames: int, height: int, width: int, sample_index: int) -> np.ndarray:
    grid_x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    grid_y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(grid_x, grid_y)
    frames = []
    phase = sample_index * 0.23
    for frame_index in range(num_frames):
        t = frame_index / max(1, num_frames - 1)
        offset_x = 0.2 * np.sin(2 * np.pi * t + phase)
        offset_y = 0.16 * np.cos(2 * np.pi * t * 1.3 + phase)
        depth = 1.0 - np.sqrt((xx - offset_x) ** 2 + (yy - offset_y) ** 2)
        frame = np.stack(
            [
                np.clip(xx + offset_x, -1.0, 1.0),
                np.clip(yy + offset_y, -1.0, 1.0),
                np.clip(depth, -1.0, 1.0),
            ],
            axis=-1,
        ).astype(np.float32)
        frames.append(frame)
    return np.stack(frames, axis=0)


def main() -> None:
    args = parse_args()
    if args.num_frames < 2:
        raise ValueError("--num-frames must be at least 2.")
    if args.num_frames % 4 != 1:
        raise ValueError("--num-frames must satisfy 1 mod 4 so it can pass Wan temporal alignment.")

    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    manifest_path = output_dir / "manifest.jsonl"
    manifest_entries: list[str] = []

    for sample_index in range(args.num_samples):
        sample_id = f"sample_{sample_index:04d}"
        sample_dir = samples_dir / sample_id
        frames_dir = sample_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        palette = build_palette(sample_index)
        for frame_index in range(args.num_frames):
            image = render_frame(
                sample_index=sample_index,
                frame_index=frame_index,
                num_frames=args.num_frames,
                width=args.width,
                height=args.height,
                palette=palette,
            )
            image.save(frames_dir / f"{frame_index:05d}.png")

        body_feet, head, left_hand, right_hand = build_motion_arrays(args.num_frames, sample_index)
        motion_path = sample_dir / "motion.npz"
        np.savez(
            motion_path,
            body_feet=body_feet,
            head=head,
            left_hand=left_hand,
            right_hand=right_hand,
        )

        point_maps_path = None
        if not args.without_point_maps:
            point_maps = build_point_maps(args.num_frames, args.height, args.width, sample_index)
            point_maps += rng.normal(scale=0.01, size=point_maps.shape).astype(np.float32)
            point_maps = np.clip(point_maps, -1.0, 1.0)
            point_maps_path = sample_dir / "point_maps.npy"
            np.save(point_maps_path, point_maps)

        prompt = f"synthetic egocentric room traversal sample {sample_index:02d}"
        entry = {
            "sample_id": sample_id,
            "prompt": prompt,
            "frames": str(frames_dir.relative_to(output_dir)),
            "motion": str(motion_path.relative_to(output_dir)),
            "metadata": {
                "height": args.height,
                "width": args.width,
                "num_frames": args.num_frames,
            },
        }
        if point_maps_path is not None:
            entry["point_maps"] = str(point_maps_path.relative_to(output_dir))
        manifest_entries.append(json.dumps(entry, ensure_ascii=True))

    manifest_path.write_text("\n".join(manifest_entries) + "\n")
    print(f"synthetic dataset written to {output_dir}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()

