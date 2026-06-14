"""Single-sample MapTR inference wrapper."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple, Union


try:
    from models.maptr.interface.runtime import DEFAULT_BUNDLED_MAPTR_ROOT, import_maptr_plugins
    from models.maptr.interface.types import (
        DEFAULT_MAP_CLASSES,
        CameraSensor,
        MapTRPrediction,
        MapTRSensorInput,
        MapVector,
        PathLike,
    )
except ImportError:
    module_dir = Path(__file__).resolve().parent
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    from runtime import DEFAULT_BUNDLED_MAPTR_ROOT, import_maptr_plugins
    spec = importlib.util.spec_from_file_location("maptr_local_types", module_dir / "types.py")
    if spec is None or spec.loader is None:
        raise
    maptr_types = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = maptr_types
    spec.loader.exec_module(maptr_types)
    DEFAULT_MAP_CLASSES = maptr_types.DEFAULT_MAP_CLASSES
    CameraSensor = maptr_types.CameraSensor
    MapTRPrediction = maptr_types.MapTRPrediction
    MapTRSensorInput = maptr_types.MapTRSensorInput
    MapVector = maptr_types.MapVector
    PathLike = maptr_types.PathLike


class MapTRPredictor:
    """Load a MapTR checkpoint and run single-sample inference."""

    def __init__(
        self,
        config_path: PathLike,
        checkpoint_path: PathLike,
        *,
        maptr_root: Optional[PathLike] = None,
        device: Optional[str] = None,
        cfg_options: Optional[Mapping[str, Any]] = None,
        score_threshold: float = 0.0,
    ) -> None:
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.maptr_root = Path(maptr_root) if maptr_root is not None else self._infer_maptr_root(self.config_path)
        self.score_threshold = float(score_threshold)

        self._ensure_maptr_importable()
        deps = self._load_runtime_dependencies()

        self.torch = deps["torch"]
        self.Config = deps["Config"]
        self.Compose = deps["Compose"]
        self.get_box_type = deps["get_box_type"]
        self.import_modules_from_strings = deps["import_modules_from_strings"]
        self.load_checkpoint = deps["load_checkpoint"]
        self.wrap_fp16_model = deps["wrap_fp16_model"]
        self.build_model = deps["build_model"]
        self.device = device or "cuda:0"
        if not str(self.device).startswith("cuda"):
            raise ValueError("MapTRPredictor requires a CUDA device. CPU inference is not supported.")
        if not self.torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Run MapTRPredictor in a GPU-enabled MapTR environment.")

        self.cfg = self.Config.fromfile(str(self.config_path))
        if cfg_options:
            self.cfg.merge_from_dict(dict(cfg_options))

        self._import_config_plugins(self.cfg)
        self.map_classes = self._get_map_classes(self.cfg)
        self.pipeline = self.Compose(self.cfg.data.test.pipeline)
        self.box_type_3d, self.box_mode_3d = self.get_box_type("LiDAR")

        self.cfg.model.pretrained = None
        self.cfg.model.train_cfg = None
        self.model = self.build_model(self.cfg.model, test_cfg=self.cfg.get("test_cfg"))
        if self.cfg.get("fp16", None) is not None:
            self.wrap_fp16_model(self.model)

        self.checkpoint = self.load_checkpoint(self.model, str(self.checkpoint_path), map_location="cpu")
        self._set_checkpoint_metadata()

        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        sample: Union[MapTRSensorInput, Mapping[str, Any]],
        *,
        score_threshold: Optional[float] = None,
    ) -> MapTRPrediction:
        """Predict HD map vectors for one input sample."""

        raw = self.predict_raw(sample)
        sample_id = sample.sample_id if isinstance(sample, MapTRSensorInput) else str(_mapping_get(sample, "sample_idx", "input_sample"))
        threshold = self.score_threshold if score_threshold is None else float(score_threshold)
        return self.format_output(raw, sample_id=sample_id, score_threshold=threshold)

    def predict_raw(self, sample: Union[MapTRSensorInput, Mapping[str, Any]], **kwargs: Any) -> Any:
        """Run the official MapTR model and return its raw output."""

        data = self._format_sample_for_pipeline(sample)
        data.update(kwargs)

        rescale = bool(data.pop("rescale", True))
        data = self._unwrap_data_containers(data)
        data = self._normalize_inference_input(data)
        data = self._move_to_device(data)

        with self.torch.no_grad():
            return self.model(return_loss=False, rescale=rescale, **data)

    def format_output(
        self,
        raw_output: Any,
        *,
        sample_id: str = "input_sample",
        score_threshold: float = 0.0,
        raw_output_field: bool = True,
    ) -> MapTRPrediction:
        """Convert official MapTR raw output into ``MapTRPrediction``."""

        detection = self._extract_pts_bbox(raw_output)
        points = _to_numpy(detection["pts_3d"])
        scores = _to_numpy(detection["scores_3d"])
        labels = _to_numpy(detection["labels_3d"])

        vectors = []
        for idx in range(len(scores)):
            score = float(scores[idx])
            if score < score_threshold:
                continue

            class_id = int(labels[idx])
            class_name = self.map_classes[class_id] if class_id < len(self.map_classes) else str(class_id)
            vectors.append(
                MapVector(
                    class_id=class_id,
                    class_name=class_name,
                    score=score,
                    points=_points_to_tuple(points[idx]),
                )
            )

        raw = raw_output if raw_output_field else None
        return MapTRPrediction(sample_id=sample_id, vectors=tuple(vectors), map_classes=self.map_classes, raw_output=raw)

    def _format_sample_for_pipeline(self, sample: Union[MapTRSensorInput, Mapping[str, Any]]) -> Dict[str, Any]:
        """Format one input sample with MapTR's test pipeline."""

        if isinstance(sample, MapTRSensorInput):
            info = sample.to_maptr_info()
            info = self._add_pipeline_fields(info)
            return self.pipeline(info)
        return dict(sample)

    def _add_pipeline_fields(self, info: Dict[str, Any]) -> Dict[str, Any]:
        info = dict(info)
        info.setdefault("img_fields", [])
        info.setdefault("bbox3d_fields", [])
        info.setdefault("pts_mask_fields", [])
        info.setdefault("pts_seg_fields", [])
        info.setdefault("bbox_fields", [])
        info.setdefault("mask_fields", [])
        info.setdefault("seg_fields", [])
        info.setdefault("box_type_3d", self.box_type_3d)
        info.setdefault("box_mode_3d", self.box_mode_3d)
        return info

    def _ensure_maptr_importable(self) -> None:
        root = str(self.maptr_root.resolve())
        if root not in sys.path:
            sys.path.insert(0, root)

    def _import_config_plugins(self, cfg: Any) -> None:
        custom_imports = cfg.get("custom_imports", None)
        if custom_imports:
            self.import_modules_from_strings(**custom_imports)

        if not cfg.get("plugin", False):
            return

        plugin_dir = cfg.get("plugin_dir", None)
        if plugin_dir is None:
            return
        plugin_path = Path(plugin_dir)
        if not plugin_path.is_absolute():
            plugin_path = self.maptr_root / plugin_path
        import_maptr_plugins(self.maptr_root, plugin_path)

    def _set_checkpoint_metadata(self) -> None:
        meta = self.checkpoint.get("meta", {}) if isinstance(self.checkpoint, Mapping) else {}
        if "CLASSES" in meta:
            self.model.CLASSES = meta["CLASSES"]
        if "PALETTE" in meta:
            self.model.PALETTE = meta["PALETTE"]

    def _move_to_device(self, value: Any) -> Any:
        if hasattr(value, "to") and callable(value.to):
            try:
                return value.to(self.device)
            except TypeError:
                return value
        if isinstance(value, MutableMapping):
            return {key: self._move_to_device(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(self._move_to_device(item) for item in value)
        if isinstance(value, list):
            return [self._move_to_device(item) for item in value]
        return value

    @staticmethod
    def _normalize_inference_input(data: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(data)

        if "img_metas" in normalized:
            metas = normalized["img_metas"]
            if isinstance(metas, Mapping):
                normalized["img_metas"] = [[metas]]
            elif isinstance(metas, list) and metas and isinstance(metas[0], Mapping):
                normalized["img_metas"] = [metas]

        if "img" in normalized and not isinstance(normalized["img"], list):
            normalized["img"] = [normalized["img"]]
        if "img" in normalized and isinstance(normalized["img"], list):
            normalized["img"] = [
                item.unsqueeze(0) if hasattr(item, "dim") and item.dim() == 4 else item
                for item in normalized["img"]
            ]
        if "points" in normalized and normalized["points"] is not None and not isinstance(normalized["points"], list):
            normalized["points"] = [normalized["points"]]

        return normalized

    @classmethod
    def _unwrap_data_containers(cls, value: Any) -> Any:
        if hasattr(value, "_data"):
            return cls._unwrap_data_containers(value._data)
        if hasattr(value, "data") and value.__class__.__name__ == "DataContainer":
            return cls._unwrap_data_containers(value.data)
        if isinstance(value, MutableMapping):
            return {key: cls._unwrap_data_containers(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(cls._unwrap_data_containers(item) for item in value)
        if isinstance(value, list):
            return [cls._unwrap_data_containers(item) for item in value]
        return value

    @staticmethod
    def _infer_maptr_root(config_path: Path) -> Path:
        for parent in config_path.resolve().parents:
            if (parent / "projects").is_dir() and (parent / "mmdetection3d").is_dir():
                return parent
        return DEFAULT_BUNDLED_MAPTR_ROOT

    @staticmethod
    def _get_map_classes(cfg: Any) -> Tuple[str, ...]:
        if hasattr(cfg, "map_classes"):
            return tuple(cfg.map_classes)
        if cfg.get("map_classes", None) is not None:
            return tuple(cfg.get("map_classes"))
        return DEFAULT_MAP_CLASSES

    @staticmethod
    def _extract_pts_bbox(raw_output: Any) -> Mapping[str, Any]:
        output = raw_output[0] if isinstance(raw_output, (list, tuple)) else raw_output
        if isinstance(output, Mapping) and "pts_bbox" in output:
            return output["pts_bbox"]
        if isinstance(output, Mapping):
            return output
        raise TypeError("MapTR raw output must be a mapping or a sequence containing a mapping.")

    @staticmethod
    def _load_runtime_dependencies() -> Dict[str, Any]:
        try:
            import torch
            from mmcv import Config
            from mmcv.runner import load_checkpoint, wrap_fp16_model
            from mmcv.utils import import_modules_from_strings
            from mmdet.datasets.pipelines import Compose
            from mmdet3d.core.bbox import get_box_type
            from mmdet3d.models import build_model
        except ImportError as exc:
            raise ImportError(
                "MapTR runtime dependencies are missing. Activate the MapTR environment "
                "before constructing MapTRPredictor."
            ) from exc

        return {
            "torch": torch,
            "Config": Config,
            "Compose": Compose,
            "get_box_type": get_box_type,
            "import_modules_from_strings": import_modules_from_strings,
            "load_checkpoint": load_checkpoint,
            "wrap_fp16_model": wrap_fp16_model,
            "build_model": build_model,
        }


def _mapping_get(mapping: Mapping[str, Any], key: str, default: Any) -> Any:
    if key in mapping:
        return mapping[key]
    if "img_metas" in mapping:
        metas = mapping["img_metas"]
        while isinstance(metas, (list, tuple)) and metas:
            metas = metas[0]
        if isinstance(metas, Mapping):
            return metas.get(key, default)
    return default


def _to_numpy(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    try:
        import numpy as np
    except ImportError:
        return value
    return np.asarray(value)


def _points_to_tuple(points: Any) -> Tuple[Tuple[float, float], ...]:
    result = []
    for point in points:
        result.append((float(point[0]), float(point[1])))
    return tuple(result)


__all__ = [
    "CameraSensor",
    "MapTRSensorInput",
    "MapVector",
    "MapTRPrediction",
    "MapTRPredictor",
    "DEFAULT_MAP_CLASSES",
]
