"""Input adapters from shared evaluation schemas to MapTR input objects."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence

try:
    from models.maptr.interface.types import CameraSensor, MapTRSensorInput
except ImportError:
    module_dir = Path(__file__).resolve().parent
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    import importlib.util

    spec = importlib.util.spec_from_file_location("maptr_local_types", module_dir / "types.py")
    if spec is None or spec.loader is None:
        raise
    maptr_types = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = maptr_types
    spec.loader.exec_module(maptr_types)
    CameraSensor = maptr_types.CameraSensor
    MapTRSensorInput = maptr_types.MapTRSensorInput


DEFAULT_CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


def _add_workspace_repo_to_path(repo_name: str) -> None:
    repo_root = Path(__file__).resolve().parents[4] / repo_name
    if repo_root.is_dir() and str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_add_workspace_repo_to_path("hdmap-driving-eval")

try:
    from schemas.carla_sensor_sample import CarlaCameraFrame, CarlaLidarFrame, CarlaSensorSample
except ImportError as exc:
    raise ImportError(
        "CARLA input schemas are missing. Add repos/hdmap-driving-eval to PYTHONPATH "
        "or keep it next to hdmap-model-bench in the workspace."
    ) from exc


def maptr_input_from_carla(
    sample: CarlaSensorSample,
    *,
    camera_order: Sequence[str] = DEFAULT_CAMERA_ORDER,
    require_all_cameras: bool = True,
) -> MapTRSensorInput:
    """Convert a common CARLA sensor sample into MapTR input.

    MapTR uses camera image paths, camera intrinsics/extrinsics, ego
    pose/motion, timestamp, and an optional top-lidar transform. Extra CARLA
    sample fields are preserved only in ``metadata``.
    """

    if not isinstance(sample, CarlaSensorSample):
        raise TypeError("sample must be a schemas.carla_sensor_sample.CarlaSensorSample instance.")

    transforms = _load_carla_to_nuscenes_transforms()
    cameras = list(sample.cameras)
    ordered_cameras = _order_cameras(cameras, camera_order, require_all_cameras)
    ego_to_world = transforms["carla_pose_to_world_matrix"](sample.ego_pose, "ego_pose")
    lidar = sample.lidar
    if lidar is not None and not isinstance(lidar, CarlaLidarFrame):
        raise TypeError("sample.lidar must be a CarlaLidarFrame instance when provided.")
    lidar_to_ego = transforms["carla_sensor_to_ego_matrix"](lidar, ego_to_world, default_identity=True)

    maptr_cameras = []
    for camera in ordered_cameras:
        if not isinstance(camera, CarlaCameraFrame):
            raise TypeError("sample.cameras must contain CarlaCameraFrame instances.")

        camera_to_ego = transforms["carla_sensor_to_ego_matrix"](camera, ego_to_world, default_identity=False)
        intrinsic = camera.intrinsic
        camera_intrinsic = _camera_intrinsic4(intrinsic)
        lidar2img = transforms["carla_lidar_to_image_matrix"](intrinsic, camera_to_ego, lidar_to_ego)
        maptr_cameras.append(
            CameraSensor(
                name=camera.name,
                image_path=camera.image_path,
                lidar2img=lidar2img,
                camera_intrinsic=camera_intrinsic,
                camera2ego=camera_to_ego,
                metadata=dict(camera.metadata),
            )
        )

    return MapTRSensorInput(
        sample_id=sample.sample_id,
        cameras=tuple(maptr_cameras),
        can_bus=transforms["carla_sample_to_can_bus"](sample),
        timestamp=sample.timestamp,
        lidar_path=lidar.point_path if lidar is not None else None,
        lidar2ego=lidar_to_ego,
        metadata={
            "source": "carla",
            "map_context": sample.map_context,
            **dict(sample.metadata),
        },
    )


def _order_cameras(
    cameras: Sequence[CarlaCameraFrame],
    camera_order: Sequence[str],
    require_all: bool,
) -> List[CarlaCameraFrame]:
    by_name = {camera.name: camera for camera in cameras}
    ordered = []
    missing = []

    for camera_name in camera_order:
        camera = by_name.get(camera_name)
        if camera is None:
            missing.append(camera_name)
            continue
        ordered.append(camera)

    if require_all and missing:
        raise ValueError(f"Missing required CARLA cameras for MapTR: {', '.join(missing)}")
    if ordered:
        return ordered
    return list(cameras)


def _load_carla_to_nuscenes_transforms() -> Dict[str, Any]:
    try:
        from carla_nuscenes.coordinates import (
            carla_lidar_to_image_matrix,
            carla_pose_to_nuscenes_matrix,
            carla_sample_to_can_bus,
            carla_sensor_to_ego_matrix,
        )
    except ImportError:
        builder_root = Path(__file__).resolve().parents[4] / "carla-nuscenes-dataset-builder"
        if builder_root.is_dir() and str(builder_root) not in sys.path:
            sys.path.insert(0, str(builder_root))
        try:
            from carla_nuscenes.coordinates import (
                carla_lidar_to_image_matrix,
                carla_pose_to_nuscenes_matrix,
                carla_sample_to_can_bus,
                carla_sensor_to_ego_matrix,
            )
        except ImportError as exc:
            raise ImportError(
                "CARLA coordinate transforms are missing. Add repos/carla-nuscenes-dataset-builder "
                "to PYTHONPATH or keep it next to hdmap-model-bench in the workspace."
            ) from exc

    return {
        "carla_lidar_to_image_matrix": carla_lidar_to_image_matrix,
        "carla_pose_to_world_matrix": carla_pose_to_nuscenes_matrix,
        "carla_sample_to_can_bus": carla_sample_to_can_bus,
        "carla_sensor_to_ego_matrix": carla_sensor_to_ego_matrix,
    }


def _camera_intrinsic4(intrinsic: Any) -> Any:
    try:
        import numpy as np
    except ImportError:
        return intrinsic

    matrix = np.asarray(intrinsic, dtype=np.float32)
    if matrix.shape == (4, 4):
        return matrix
    if matrix.shape != (3, 3):
        raise ValueError(f"camera intrinsic must be 3x3 or 4x4, got {matrix.shape}")

    viewpad = np.eye(4, dtype=np.float32)
    viewpad[:3, :3] = matrix
    return viewpad


__all__ = ["DEFAULT_CAMERA_ORDER", "maptr_input_from_carla"]
