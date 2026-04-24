from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from .data import MotionSequence, load_point_maps, slice_temporal_tensor


@dataclass
class SceneCondition:
    point_maps: torch.Tensor | None = None
    memory_latents: torch.Tensor | None = None
    context_tokens: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenePointCloudState:
    renderer_state: Any = None
    latent_history: torch.Tensor | None = None
    history: list[dict[str, Any]] = field(default_factory=list)


def normalize_scene_condition(
    scene_condition: SceneCondition | torch.Tensor | str | Path | None,
) -> SceneCondition:
    if scene_condition is None:
        return SceneCondition()
    if isinstance(scene_condition, SceneCondition):
        return scene_condition
    if isinstance(scene_condition, (str, Path)):
        scene_condition = load_point_maps(scene_condition)
    return SceneCondition(point_maps=torch.as_tensor(scene_condition))


class BasePointMapRenderer:
    def initialize(
        self,
        *,
        input_image: Image.Image,
        height: int,
        width: int,
    ) -> ScenePointCloudState | None:
        return None

    def render_condition(
        self,
        scene_state: ScenePointCloudState | None,
        *,
        motion: MotionSequence,
        start_frame: int,
        stop_frame: int,
        height: int,
        width: int,
        input_image: Image.Image | None = None,
    ) -> SceneCondition | torch.Tensor | None:
        return None

    def update(
        self,
        scene_state: ScenePointCloudState | None,
        *,
        video: list[Image.Image],
        motion: MotionSequence,
        start_frame: int,
        stop_frame: int,
        height: int,
        width: int,
    ) -> ScenePointCloudState | None:
        return scene_state


class NullPointMapRenderer(BasePointMapRenderer):
    pass


class PrecomputedPointMapRenderer(BasePointMapRenderer):
    """
    Replay a pre-rendered cumulative point-map sequence chunk by chunk.

    This is primarily useful for debugging the autoregressive training/inference loop
    before wiring in a real point-cloud renderer such as CUT3R.
    """

    def __init__(
        self,
        point_maps: torch.Tensor | str | Path,
        *,
        use_first_chunk_condition: bool = False,
    ):
        if isinstance(point_maps, (str, Path)):
            point_maps = load_point_maps(point_maps)
        self.point_maps = torch.as_tensor(point_maps).float()
        self.use_first_chunk_condition = use_first_chunk_condition

    def initialize(
        self,
        *,
        input_image: Image.Image,
        height: int,
        width: int,
    ) -> ScenePointCloudState:
        return ScenePointCloudState(renderer_state={"initialized": True})

    def render_condition(
        self,
        scene_state: ScenePointCloudState | None,
        *,
        motion: MotionSequence,
        start_frame: int,
        stop_frame: int,
        height: int,
        width: int,
        input_image: Image.Image | None = None,
    ) -> SceneCondition | None:
        if start_frame == 0 and not self.use_first_chunk_condition:
            return None
        return SceneCondition(point_maps=slice_temporal_tensor(self.point_maps, start_frame, stop_frame))

    def update(
        self,
        scene_state: ScenePointCloudState | None,
        *,
        video: list[Image.Image],
        motion: MotionSequence,
        start_frame: int,
        stop_frame: int,
        height: int,
        width: int,
    ) -> ScenePointCloudState:
        state = ScenePointCloudState() if scene_state is None else scene_state
        state.history.append(
            {
                "start_frame": start_frame,
                "stop_frame": stop_frame,
                "num_frames": len(video),
            }
        )
        return state


BaseSceneConditioner = BasePointMapRenderer
NullSceneConditioner = NullPointMapRenderer
PrecomputedSceneConditioner = PrecomputedPointMapRenderer
