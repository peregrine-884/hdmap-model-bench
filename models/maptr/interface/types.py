"""MapTR adapter input and output data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union


DEFAULT_MAP_CLASSES = ("divider", "ped_crossing", "boundary")

ArrayLike = Any
PathLike = Union[str, Path]


@dataclass(frozen=True)
class CameraSensor:
    """One camera image and its calibration for a single timestamp."""

    name: str
    image_path: PathLike
    lidar2img: Optional[ArrayLike] = None
    camera_intrinsic: Optional[ArrayLike] = None
    camera2ego: Optional[ArrayLike] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def resolved_image_path(self, base_dir: Optional[PathLike] = None) -> str:
        path = Path(self.image_path)
        if not path.is_absolute() and base_dir is not None:
            path = Path(base_dir) / path
        return str(path)


@dataclass(frozen=True)
class MapTRSensorInput:
    """One inference sample before MapTR test-pipeline formatting.

    The coordinate frame follows MapTR/nuScenes convention: predicted map
    points are in the ego/lidar BEV frame, with x forward and y left.
    """

    sample_id: str
    cameras: Sequence[CameraSensor]
    can_bus: Sequence[float] = field(default_factory=lambda: [0.0] * 18)
    timestamp: Union[int, float] = 0
    lidar_path: Optional[PathLike] = None
    lidar2ego: Optional[ArrayLike] = None
    lidar2global: Optional[ArrayLike] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_maptr_info(self, base_dir: Optional[PathLike] = None) -> Dict[str, Any]:
        """Return the single-sample dictionary consumed by the MapTR test pipeline."""

        info = dict(self.metadata)
        info.update(
            {
                "sample_idx": self.sample_id,
                "scene_token": info.get("scene_token", self.sample_id),
                "pts_filename": str(self.lidar_path or self.sample_id),
                "timestamp": self.timestamp,
                "can_bus": _as_float32_array(self.can_bus),
                "img_filename": [camera.resolved_image_path(base_dir) for camera in self.cameras],
            }
        )

        lidar2img = _camera_values_or_none(self.cameras, "lidar2img")
        if lidar2img is not None:
            info["lidar2img"] = _as_float32_array(lidar2img)

        camera_intrinsics = _camera_values_or_none(self.cameras, "camera_intrinsic")
        if camera_intrinsics is not None:
            info["camera_intrinsics"] = _as_float32_array(camera_intrinsics)

        camera2ego = _camera_values_or_none(self.cameras, "camera2ego")
        if camera2ego is not None:
            info["camera2ego"] = _as_float32_array(camera2ego)

        if self.lidar2ego is not None:
            info["lidar2ego"] = _as_float32_array(self.lidar2ego)
        if self.lidar2global is not None:
            info["lidar2global"] = _as_float32_array(self.lidar2global)

        return info


@dataclass(frozen=True)
class MapVector:
    """One predicted HD map vector in ego/lidar BEV coordinates."""

    class_id: int
    class_name: str
    score: float
    points: Tuple[Tuple[float, float], ...]

    @property
    def pts_num(self) -> int:
        return len(self.points)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pts": [list(point) for point in self.points],
            "pts_num": self.pts_num,
            "cls_name": self.class_name,
            "type": self.class_id,
            "confidence_level": self.score,
        }


@dataclass(frozen=True)
class MapTRPrediction:
    """MapTR output normalized for notebooks and external evaluation."""

    sample_id: str
    vectors: Tuple[MapVector, ...]
    map_classes: Tuple[str, ...] = DEFAULT_MAP_CLASSES
    raw_output: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_token": self.sample_id,
            "vectors": [vector.to_dict() for vector in self.vectors],
        }

    def by_class(self) -> Dict[str, List[MapVector]]:
        grouped = {class_name: [] for class_name in self.map_classes}
        for vector in self.vectors:
            grouped.setdefault(vector.class_name, []).append(vector)
        return grouped


def _camera_values_or_none(cameras: Sequence[CameraSensor], field_name: str) -> Optional[List[Any]]:
    values = [getattr(camera, field_name) for camera in cameras]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        missing = [camera.name for camera, value in zip(cameras, values) if value is None]
        raise ValueError(f"Missing {field_name} for cameras: {', '.join(missing)}")
    return values


def _as_float32_array(value: Any) -> Any:
    try:
        import numpy as np
    except ImportError:
        return value
    return np.asarray(value, dtype=np.float32)


__all__ = [
    "ArrayLike",
    "CameraSensor",
    "DEFAULT_MAP_CLASSES",
    "MapTRPrediction",
    "MapTRSensorInput",
    "MapVector",
    "PathLike",
]
