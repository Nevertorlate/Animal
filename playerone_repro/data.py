from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import torch


BODY_FEET_DIM = 66
HEAD_DIM = 3
HAND_DIM = 45
TOTAL_SMPL_DIM = BODY_FEET_DIM + HEAD_DIM + HAND_DIM * 2


@dataclass
class MotionSequence:
    body_feet: torch.Tensor
    head: torch.Tensor
    hands: torch.Tensor

    def __post_init__(self) -> None:
        for name, value in {
            "body_feet": self.body_feet,
            "head": self.head,
            "hands": self.hands,
        }.items():
            if value.ndim != 3:
                raise ValueError(f"{name} must be [B, T, C], got {tuple(value.shape)}")
        bt = self.body_feet.shape[:2]
        if self.head.shape[:2] != bt or self.hands.shape[:2] != bt:
            raise ValueError("body_feet, head, hands must share the same batch/time shape")

    @property
    def batch_size(self) -> int:
        return int(self.body_feet.shape[0])

    @property
    def num_frames(self) -> int:
        return int(self.body_feet.shape[1])

    def to(self, device: str | torch.device | None = None, dtype: torch.dtype | None = None) -> "MotionSequence":
        kwargs = {}
        if device is not None:
            kwargs["device"] = device
        if dtype is not None:
            kwargs["dtype"] = dtype
        return MotionSequence(
            body_feet=self.body_feet.to(**kwargs),
            head=self.head.to(**kwargs),
            hands=self.hands.to(**kwargs),
        )

    def slice_frames(self, start: int, stop: int) -> "MotionSequence":
        if not (0 <= start < stop <= self.num_frames):
            raise ValueError(f"Invalid motion slice [{start}, {stop}) for sequence length {self.num_frames}")
        return MotionSequence(
            body_feet=self.body_feet[:, start:stop],
            head=self.head[:, start:stop],
            hands=self.hands[:, start:stop],
        )

    def pad_to_num_frames(self, target_num_frames: int) -> "MotionSequence":
        if target_num_frames < self.num_frames:
            raise ValueError(f"Cannot pad motion to a shorter length: {target_num_frames} < {self.num_frames}")
        if target_num_frames == self.num_frames:
            return self
        pad = target_num_frames - self.num_frames
        body_feet = torch.cat([self.body_feet, self.body_feet[:, -1:].expand(-1, pad, -1)], dim=1)
        head = torch.cat([self.head, self.head[:, -1:].expand(-1, pad, -1)], dim=1)
        hands = torch.cat([self.hands, self.hands[:, -1:].expand(-1, pad, -1)], dim=1)
        return MotionSequence(body_feet=body_feet, head=head, hands=hands)

    @classmethod
    def from_smpl_array(cls, array: np.ndarray | torch.Tensor) -> "MotionSequence":
        tensor = torch.as_tensor(array)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 3:
            raise ValueError(f"Expected motion array with shape [T, C] or [B, T, C], got {tuple(tensor.shape)}")
        if tensor.shape[-1] != TOTAL_SMPL_DIM:
            raise ValueError(
                f"Expected last dim={TOTAL_SMPL_DIM} for concatenated SMPL motion, got {tensor.shape[-1]}"
            )
        body_feet = tensor[..., :BODY_FEET_DIM]
        head = tensor[..., BODY_FEET_DIM: BODY_FEET_DIM + HEAD_DIM]
        hands = tensor[..., BODY_FEET_DIM + HEAD_DIM:]
        return cls(body_feet=body_feet.float(), head=head.float(), hands=hands.float())

    @classmethod
    def from_mapping(cls, mapping: dict) -> "MotionSequence":
        if {"body_feet", "head", "hands"} <= set(mapping):
            body_feet = torch.as_tensor(mapping["body_feet"]).float()
            head = torch.as_tensor(mapping["head"]).float()
            hands = torch.as_tensor(mapping["hands"]).float()
        elif {"body_feet", "head", "left_hand", "right_hand"} <= set(mapping):
            body_feet = torch.as_tensor(mapping["body_feet"]).float()
            head = torch.as_tensor(mapping["head"]).float()
            hands = torch.cat(
                [
                    torch.as_tensor(mapping["left_hand"]).float(),
                    torch.as_tensor(mapping["right_hand"]).float(),
                ],
                dim=-1,
            )
        elif "motion" in mapping:
            return cls.from_smpl_array(mapping["motion"])
        elif "smpl" in mapping:
            return cls.from_smpl_array(mapping["smpl"])
        else:
            raise ValueError("Motion mapping must contain either body_feet/head/hands or a single motion/smpl array")
        if body_feet.ndim == 2:
            body_feet = body_feet.unsqueeze(0)
        if head.ndim == 2:
            head = head.unsqueeze(0)
        if hands.ndim == 2:
            hands = hands.unsqueeze(0)
        return cls(body_feet=body_feet, head=head, hands=hands)


def load_motion_sequence(path: str | Path) -> MotionSequence:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".json":
        return MotionSequence.from_mapping(json.loads(path.read_text()))
    if path.suffix == ".npy":
        return MotionSequence.from_smpl_array(np.load(path))
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        return MotionSequence.from_mapping({key: data[key] for key in data.files})
    raise ValueError(f"Unsupported motion format: {path.suffix}")


def load_point_maps(path: str | Path) -> torch.Tensor:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".npy":
        tensor = torch.as_tensor(np.load(path)).float()
    elif path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        first_key = data.files[0]
        tensor = torch.as_tensor(data[first_key]).float()
    else:
        raise ValueError(f"Unsupported point map format: {path.suffix}")
    if tensor.ndim == 4:
        tensor = tensor.unsqueeze(0)
    return tensor


def build_temporal_windows(total_frames: int, chunk_num_frames: int, overlap: int = 1) -> list[tuple[int, int]]:
    if total_frames < 2:
        raise ValueError(f"Need at least 2 frames for I2V continuation, got {total_frames}")
    if chunk_num_frames < 2:
        raise ValueError(f"chunk_num_frames must be at least 2, got {chunk_num_frames}")
    if overlap < 1:
        raise ValueError(f"overlap must be at least 1, got {overlap}")
    if overlap >= chunk_num_frames:
        raise ValueError(f"overlap must be smaller than chunk_num_frames, got overlap={overlap}, chunk_num_frames={chunk_num_frames}")

    windows: list[tuple[int, int]] = []
    start = 0
    while True:
        stop = min(start + chunk_num_frames, total_frames)
        if stop - start < 2 and windows:
            prev_start, _ = windows.pop()
            windows.append((prev_start, total_frames))
            break
        windows.append((start, stop))
        if stop >= total_frames:
            break
        start = stop - overlap
    return windows


def slice_temporal_tensor(tensor: torch.Tensor | np.ndarray, start: int, stop: int) -> torch.Tensor:
    tensor = torch.as_tensor(tensor)
    if tensor.ndim == 5:
        if tensor.shape[-1] in (3, 6):
            return tensor[:, start:stop]
        return tensor[:, :, start:stop]
    if tensor.ndim == 4:
        if tensor.shape[-1] in (3, 6):
            return tensor[start:stop]
        if tensor.shape[0] in (3, 6):
            return tensor[:, start:stop]
        return tensor[start:stop]
    if tensor.ndim == 3:
        return tensor[start:stop]
    raise ValueError(f"Unsupported temporal tensor rank: {tensor.ndim}")


def get_temporal_length(tensor: torch.Tensor | np.ndarray) -> int:
    tensor = torch.as_tensor(tensor)
    if tensor.ndim == 5:
        return int(tensor.shape[1] if tensor.shape[-1] in (3, 6) else tensor.shape[2])
    if tensor.ndim == 4:
        if tensor.shape[-1] in (3, 6):
            return int(tensor.shape[0])
        if tensor.shape[0] in (3, 6):
            return int(tensor.shape[1])
        return int(tensor.shape[0])
    if tensor.ndim == 3:
        return int(tensor.shape[0])
    raise ValueError(f"Unsupported temporal tensor rank: {tensor.ndim}")


def pad_temporal_tensor_to_frames(tensor: torch.Tensor | np.ndarray, target_num_frames: int) -> torch.Tensor:
    tensor = torch.as_tensor(tensor)
    current_num_frames = get_temporal_length(tensor)
    if target_num_frames < current_num_frames:
        raise ValueError(f"Cannot pad temporal tensor to a shorter length: {target_num_frames} < {current_num_frames}")
    if target_num_frames == current_num_frames:
        return tensor
    pad = target_num_frames - current_num_frames

    if tensor.ndim == 5:
        if tensor.shape[-1] in (3, 6):
            return torch.cat([tensor, tensor[:, -1:].expand(-1, pad, -1, -1, -1)], dim=1)
        return torch.cat([tensor, tensor[:, :, -1:].expand(-1, -1, pad, -1, -1)], dim=2)
    if tensor.ndim == 4:
        if tensor.shape[-1] in (3, 6):
            return torch.cat([tensor, tensor[-1:].expand(pad, -1, -1, -1)], dim=0)
        if tensor.shape[0] in (3, 6):
            return torch.cat([tensor, tensor[:, -1:].expand(-1, pad, -1, -1)], dim=1)
        return torch.cat([tensor, tensor[-1:].expand(pad, -1, -1, -1)], dim=0)
    if tensor.ndim == 3:
        return torch.cat([tensor, tensor[-1:].expand(pad, -1, -1)], dim=0)
    raise ValueError(f"Unsupported temporal tensor rank: {tensor.ndim}")


def pad_video_frames_to_length(video: list, target_num_frames: int) -> list:
    if target_num_frames < len(video):
        raise ValueError(f"Cannot pad video to a shorter length: {target_num_frames} < {len(video)}")
    if target_num_frames == len(video):
        return list(video)
    if not video:
        raise ValueError("Cannot pad an empty video")
    return list(video) + [video[-1]] * (target_num_frames - len(video))
