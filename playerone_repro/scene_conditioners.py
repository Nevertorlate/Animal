from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ._env import PROJECT_ROOT
from .rendering import BaseSceneConditioner, SceneCondition, ScenePointCloudState


def _ensure_diffsynth_path() -> None:
    diffsynth_root = PROJECT_ROOT / "diffsynth-studio"
    if diffsynth_root.exists() and str(diffsynth_root) not in sys.path:
        sys.path.insert(0, str(diffsynth_root))


def _to_rgb_image(image: Image.Image | torch.Tensor | np.ndarray) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if torch.is_tensor(image):
        array = image.detach().cpu().numpy()
    else:
        array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] > 3:
        array = array[..., :3]
    if array.dtype != np.uint8:
        max_value = float(array.max()) if array.size > 0 else 0.0
        if max_value <= 1.0:
            array = np.clip(array * 255.0, 0.0, 255.0)
        array = array.astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


class HistoryStructureSceneConditioner(BaseSceneConditioner):
    """
    Build a structural scene condition from previously generated frames.

    This is a practical fallback when a real 3D renderer (CUT3R / DUSt3R / custom point-cloud
    backend) is unavailable. The conditioner extracts structure cues such as depth/normal/canny
    from generated chunks, caches the recent history, and resamples that history as the condition
    for the next autoregressive chunk.
    """

    def __init__(
        self,
        processor_ids: Sequence[str] = ("depth", "normal", "canny"),
        *,
        annotator_model_path: str = "models/Annotators",
        annotator_device: str | None = None,
        detect_resolution: int | None = None,
        use_first_chunk_condition: bool = True,
        bootstrap_from_input: bool = True,
        drop_first_overlap_frame: bool = True,
        max_history_frames: int = 49,
    ):
        if not processor_ids:
            raise ValueError("processor_ids must not be empty.")
        self.processor_ids = tuple(processor_ids)
        self.annotator_model_path = annotator_model_path
        self.annotator_device = "cuda" if annotator_device is None and torch.cuda.is_available() else (annotator_device or "cpu")
        self.detect_resolution = detect_resolution
        self.use_first_chunk_condition = use_first_chunk_condition
        self.bootstrap_from_input = bootstrap_from_input
        self.drop_first_overlap_frame = drop_first_overlap_frame
        self.max_history_frames = max_history_frames
        self._annotators = None

    def _lazy_init_annotators(self):
        if self._annotators is not None:
            return self._annotators
        _ensure_diffsynth_path()
        try:
            from diffsynth.utils.controlnet.annotator import Annotator
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Unable to import DiffSynth annotators. Install DiffSynth dependencies and controlnet_aux, "
                "or use a different scene-conditioner."
            ) from exc

        annotators = []
        for processor_id in self.processor_ids:
            annotators.append(
                Annotator(
                    processor_id=processor_id,
                    model_path=self.annotator_model_path,
                    detect_resolution=self.detect_resolution,
                    device=self.annotator_device,
                )
            )
        self._annotators = annotators
        return annotators

    def _get_scene_maps(self, scene_state: ScenePointCloudState | None) -> list[torch.Tensor]:
        state = ScenePointCloudState() if scene_state is None else scene_state
        if state.renderer_state is None:
            state.renderer_state = {}
        return state.renderer_state.setdefault("scene_maps", [])

    def _encode_image(self, image: Image.Image, *, height: int, width: int) -> torch.Tensor:
        image = _to_rgb_image(image).resize((width, height))
        features = []
        for annotator in self._lazy_init_annotators():
            annotated = _to_rgb_image(annotator(image)).resize((width, height))
            feature = torch.from_numpy(np.asarray(annotated, dtype=np.float32) / 255.0).permute(2, 0, 1)
            features.append(feature)
        return torch.cat(features, dim=0)

    def _resample_history(self, history: list[torch.Tensor], *, target_frames: int, height: int, width: int) -> torch.Tensor:
        stacked = torch.stack(history[-self.max_history_frames:], dim=1).unsqueeze(0)
        if stacked.shape[2:] != (target_frames, height, width):
            stacked = F.interpolate(stacked.float(), size=(target_frames, height, width), mode="trilinear", align_corners=False)
        return stacked

    def initialize(
        self,
        *,
        input_image: Image.Image,
        height: int,
        width: int,
    ) -> ScenePointCloudState:
        scene_state = ScenePointCloudState(renderer_state={})
        if self.bootstrap_from_input:
            self._get_scene_maps(scene_state).append(self._encode_image(input_image, height=height, width=width))
        return scene_state

    def render_condition(
        self,
        scene_state: ScenePointCloudState | None,
        *,
        motion,
        start_frame: int,
        stop_frame: int,
        height: int,
        width: int,
        input_image: Image.Image | None = None,
    ) -> SceneCondition | None:
        if start_frame == 0 and not self.use_first_chunk_condition:
            return None

        state = ScenePointCloudState() if scene_state is None else scene_state
        history = self._get_scene_maps(state)
        if not history and self.bootstrap_from_input and input_image is not None:
            history.append(self._encode_image(input_image, height=height, width=width))
        if not history:
            return None

        target_frames = max(1, stop_frame - start_frame)
        scene_maps = self._resample_history(history, target_frames=target_frames, height=height, width=width)
        return SceneCondition(
            point_maps=scene_maps,
            metadata={
                "processor_ids": list(self.processor_ids),
                "history_frames": len(history),
            },
        )

    def update(
        self,
        scene_state: ScenePointCloudState | None,
        *,
        video: list[Image.Image],
        motion,
        start_frame: int,
        stop_frame: int,
        height: int,
        width: int,
    ) -> ScenePointCloudState:
        state = ScenePointCloudState() if scene_state is None else scene_state
        history = self._get_scene_maps(state)
        frames = video
        if self.drop_first_overlap_frame and history and len(video) > 1:
            frames = video[1:]

        for frame in frames:
            history.append(self._encode_image(frame, height=height, width=width))
        if len(history) > self.max_history_frames:
            del history[:-self.max_history_frames]

        state.history.append(
            {
                "start_frame": start_frame,
                "stop_frame": stop_frame,
                "num_frames": len(video),
                "cached_scene_frames": len(history),
                "processor_ids": list(self.processor_ids),
            }
        )
        return state


def build_structural_scene_conditioner(**kwargs) -> HistoryStructureSceneConditioner:
    return HistoryStructureSceneConditioner(**kwargs)
