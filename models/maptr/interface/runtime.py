"""Runtime helpers for the vendored MapTR package."""

from __future__ import annotations

from pathlib import Path
import importlib
import sys
import types
from typing import Any, Mapping, Optional, Union


PathLike = Union[str, Path]
DEFAULT_BUNDLED_MAPTR_ROOT = Path(__file__).resolve().parent.parent / "external" / "MapTR"


def import_maptr_plugins(maptr_root: PathLike = DEFAULT_BUNDLED_MAPTR_ROOT, plugin_path: Optional[PathLike] = None) -> None:
    """Import MapTR plugin modules needed for nuScenes/MapTR inference.

    The official plugin package imports AV2 modules from its top-level
    ``__init__``. Some MapTR environments have an AV2/numpy combination that
    fails during that optional import, even for nuScenes-only inference. In
    that case we register the modules used by the nuScenes MapTR config
    directly.
    """

    maptr_root = Path(maptr_root)
    plugin_path = Path(plugin_path) if plugin_path is not None else maptr_root / "projects" / "mmdet3d_plugin"
    if str(maptr_root.resolve()) not in sys.path:
        sys.path.insert(0, str(maptr_root.resolve()))

    rel = plugin_path.resolve().relative_to(maptr_root.resolve())
    plugin_module = ".".join(rel.parts)
    try:
        importlib.import_module(plugin_module)
    except Exception:
        _import_minimal_maptr_plugins(plugin_module, plugin_path)
    _patch_maptr_plugin_compat(plugin_module)


def _ensure_namespace_package(module_name: str, package_path: Path) -> None:
    module = sys.modules.get(module_name)
    if module is None:
        module = types.ModuleType(module_name)
        module.__path__ = [str(package_path)]
        sys.modules[module_name] = module


def _import_minimal_maptr_plugins(plugin_module: str, plugin_path: Path) -> None:
    _ensure_namespace_package(plugin_module, plugin_path)
    _ensure_namespace_package(f"{plugin_module}.datasets", plugin_path / "datasets")

    modules = [
        f"{plugin_module}.core.bbox.assigners.hungarian_assigner_3d",
        f"{plugin_module}.core.bbox.coders.nms_free_coder",
        f"{plugin_module}.core.bbox.match_costs.match_cost",
        f"{plugin_module}.datasets.pipelines",
        f"{plugin_module}.datasets.nuscenes_dataset",
        f"{plugin_module}.datasets.nuscenes_map_dataset",
        f"{plugin_module}.models.utils",
        f"{plugin_module}.models.opt.adamw",
        f"{plugin_module}.bevformer.modules",
        f"{plugin_module}.maptr",
    ]
    for module in modules:
        importlib.import_module(module)


def _patch_maptr_plugin_compat(plugin_module: str) -> None:
    """Keep compatibility fixes local to hdmap-model-bench.

    The vendored MapTR checkout can mix modules whose encoder/transformer
    return contracts differ slightly. We adapt those contracts at runtime so
    the official source tree can stay untouched.
    """

    transformer_module = importlib.import_module(f"{plugin_module}.maptr.modules.transformer")
    transformer_cls = transformer_module.MapTRPerceptionTransformer
    if getattr(transformer_cls, "_hdmap_compat_patched", False):
        return

    original_get_bev_features = transformer_cls.get_bev_features
    original_attn_bev_encode = transformer_cls.attn_bev_encode
    original_forward = transformer_cls.forward

    def attn_bev_encode(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_attn_bev_encode(self, *args, **kwargs)
        if isinstance(result, Mapping):
            return result
        return {"bev": result, "depth": None}

    def get_bev_features(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_get_bev_features(self, *args, **kwargs)
        if isinstance(result, Mapping):
            return result
        return {"bev": result, "depth": None}

    def forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_forward(self, *args, **kwargs)
        if isinstance(result, tuple) and len(result) == 5:
            bev_embed, _depth, inter_states, init_reference, inter_references = result
            return bev_embed, inter_states, init_reference, inter_references
        return result

    transformer_cls.attn_bev_encode = attn_bev_encode
    transformer_cls.get_bev_features = get_bev_features
    transformer_cls.forward = forward
    transformer_cls._hdmap_compat_patched = True


__all__ = ["DEFAULT_BUNDLED_MAPTR_ROOT", "import_maptr_plugins"]
