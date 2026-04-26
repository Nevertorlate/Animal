from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

from .data import MotionSequence, load_motion_sequence, load_point_maps


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _resolve_manifest_path(value: str | Path | None, manifest_dir: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = manifest_dir / path
    return path.resolve()


def load_video_frames(frames_dir: str | Path) -> list[Image.Image]:
    frames_dir = Path(frames_dir)
    if not frames_dir.exists():
        raise FileNotFoundError(frames_dir)
    if not frames_dir.is_dir():
        raise ValueError(f"Expected a frame directory, got {frames_dir}")
    frame_paths = sorted(path for path in frames_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not frame_paths:
        raise ValueError(f"No image frames found under {frames_dir}")
    frames: list[Image.Image] = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as image:
            frames.append(image.convert("RGB"))
    return frames


@dataclass(frozen=True)
class TrainingClipRecord:
    sample_id: str
    prompt: str
    frames_dir: Path
    motion_path: Path
    point_maps_path: Path | None = None
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_json_entry(cls, entry: dict[str, Any], manifest_dir: Path, index: int) -> "TrainingClipRecord":
        prompt = entry.get("prompt", entry.get("caption"))
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Manifest row {index} is missing a non-empty 'prompt' string.")

        frames_value = entry.get("frames", entry.get("frames_dir", entry.get("video")))
        if frames_value is None:
            raise ValueError(f"Manifest row {index} is missing 'frames' or 'frames_dir'.")
        motion_value = entry.get("motion")
        if motion_value is None:
            raise ValueError(f"Manifest row {index} is missing 'motion'.")

        sample_id = str(entry.get("sample_id", f"sample_{index:04d}"))
        metadata = entry.get("metadata")
        if metadata is None:
            metadata = {
                key: value
                for key, value in entry.items()
                if key not in {"sample_id", "prompt", "caption", "frames", "frames_dir", "video", "motion", "point_maps", "metadata"}
            }
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError(f"Manifest row {index} has a non-object 'metadata' field.")

        return cls(
            sample_id=sample_id,
            prompt=prompt,
            frames_dir=_resolve_manifest_path(frames_value, manifest_dir),
            motion_path=_resolve_manifest_path(motion_value, manifest_dir),
            point_maps_path=_resolve_manifest_path(entry.get("point_maps"), manifest_dir),
            metadata=metadata,
        )


def load_training_manifest(manifest_path: str | Path) -> list[TrainingClipRecord]:
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    records: list[TrainingClipRecord] = []
    for index, line in enumerate(manifest_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        entry = json.loads(line)
        if not isinstance(entry, dict):
            raise ValueError(f"Manifest row {index} must be a JSON object.")
        records.append(TrainingClipRecord.from_json_entry(entry, manifest_path.parent, index))
    if not records:
        raise ValueError(f"No samples found in manifest {manifest_path}")
    return records


class TrainingClipDataset(Dataset):
    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path).resolve()
        self.records = load_training_manifest(self.manifest_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        point_maps = None if record.point_maps_path is None else load_point_maps(record.point_maps_path)
        return {
            "sample_id": record.sample_id,
            "prompt": record.prompt,
            "video": load_video_frames(record.frames_dir),
            "motion": load_motion_sequence(record.motion_path),
            "point_maps": point_maps,
            "metadata": dict(record.metadata or {}),
        }

