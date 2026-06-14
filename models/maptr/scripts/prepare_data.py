#!/usr/bin/env python3
"""Interactive launcher for MapTR nuScenes data preparation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Iterable


THIS_FILE = Path(__file__).resolve()
MAPTR_MODEL_DIR = THIS_FILE.parents[1]
HDMAP_MODEL_BENCH_ROOT = MAPTR_MODEL_DIR.parents[1]
WORKSPACE_ROOT = HDMAP_MODEL_BENCH_ROOT.parents[1]
OFFICIAL_MAPTR_ROOT = MAPTR_MODEL_DIR / "external" / "MapTR"
OFFICIAL_CONVERTER = OFFICIAL_MAPTR_ROOT / "tools" / "maptrv2" / "custom_nusc_map_converter.py"


DATASET_DEFAULTS = {
    "nuscenes": {
        "root_path": WORKSPACE_ROOT / "dataset" / "nuscenes" / "nuscenes",
        "out_dir": WORKSPACE_ROOT / "dataset" / "nuscenes" / "nuscenes",
        "canbus": WORKSPACE_ROOT / "dataset" / "nuscenes",
        "version": "v1.0-mini",
        "extra_tag": "nuscenes",
        "patched": False,
    },
    "carla_nuscenes": {
        "root_path": WORKSPACE_ROOT / "dataset" / "carla_nuscenes" / "nuscenes" / "nuscenes",
        "out_dir": WORKSPACE_ROOT / "dataset" / "carla_nuscenes" / "nuscenes" / "nuscenes",
        "canbus": WORKSPACE_ROOT / "dataset" / "carla_nuscenes" / "nuscenes",
        "version": "v1.0-carla",
        "extra_tag": "carla_nuscenes",
        "patched": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MapTR custom nuScenes map data preparation."
    )
    parser.add_argument(
        "--dataset",
        choices=["interactive", "nuscenes", "carla_nuscenes"],
        default="interactive",
        help="Dataset to prepare. Default opens an interactive prompt.",
    )
    parser.add_argument("--root-path", type=Path, help="nuScenes-format dataset root.")
    parser.add_argument("--out-dir", type=Path, help="Output directory for generated pkl files.")
    parser.add_argument(
        "--canbus",
        type=Path,
        help="Directory that contains can_bus, or the dataset root for CARLA nuScenes.",
    )
    parser.add_argument(
        "--version",
        help=(
            "Dataset version. For nuscenes, use v1.0-trainval or v1.0-mini. "
            "For carla_nuscenes, use v1.0-carla."
        ),
    )
    parser.add_argument(
        "--extra-tag",
        help="Output filename prefix. Example: nuscenes or carla_nuscenes.",
    )
    parser.add_argument("--max-sweeps", type=int, default=10)
    parser.add_argument("--python", default=sys.executable, help="Python executable to run.")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running it.")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt.")
    parser.add_argument(
        "--skip-dependency-check",
        action="store_true",
        help="Skip import checks before running the official converter.",
    )
    return parser.parse_args()


def choose_dataset() -> str:
    print("Select dataset to prepare:")
    print("  1. nuscenes")
    print("  2. carla_nuscenes")
    while True:
        answer = input("dataset [1/2]: ").strip()
        if answer in {"1", "nuscenes"}:
            return "nuscenes"
        if answer in {"2", "carla_nuscenes"}:
            return "carla_nuscenes"
        print("Please enter 1 or 2.")


def apply_overrides(args: argparse.Namespace, dataset_name: str) -> dict:
    config = dict(DATASET_DEFAULTS[dataset_name])
    for key in ("root_path", "out_dir", "canbus", "version", "extra_tag"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    return config


def normalize_version(dataset_name: str, version: str) -> str:
    if dataset_name == "nuscenes":
        aliases = {
            "v1.0": "v1.0-trainval",
            "v1.0-train": "v1.0-trainval",
            "trainval": "v1.0-trainval",
            "mini": "v1.0-mini",
        }
        version = aliases.get(version, version)
        allowed = {"v1.0-trainval", "v1.0-mini"}
        if version not in allowed:
            raise ValueError(
                f"Unsupported nuscenes version: {version}. "
                f"Choose one of: {', '.join(sorted(allowed))}"
            )
    return version


def should_use_patched_converter(config: dict) -> bool:
    return bool(config["patched"]) or str(config["version"]) != "v1.0"


def list_map_names(root_path: Path) -> list[str]:
    expansion_dir = root_path / "maps" / "expansion"
    if not expansion_dir.exists():
        return []
    return sorted(path.stem for path in expansion_dir.glob("*.json"))


def list_scene_names(root_path: Path, version: str) -> list[str]:
    scene_path = root_path / version / "scene.json"
    if not scene_path.exists():
        return []
    with scene_path.open("r", encoding="utf-8") as f:
        scenes = json.load(f)
    return sorted(scene["name"] for scene in scenes)


def replace_required(source: str, old: str, new: str, label: str) -> str:
    if old not in source:
        raise RuntimeError(f"Could not patch official converter block: {label}")
    return source.replace(old, new, 1)


def validate_paths(config: dict) -> None:
    root_path = Path(config["root_path"])
    canbus = Path(config["canbus"])
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root_path}")
    if not canbus.exists():
        raise FileNotFoundError(f"CAN bus root does not exist: {canbus}")
    if not OFFICIAL_CONVERTER.exists():
        raise FileNotFoundError(f"Official MapTR converter does not exist: {OFFICIAL_CONVERTER}")


def patched_converter_source(source: str, root_path: Path, version: str) -> str:
    map_names = list_map_names(root_path)
    if not map_names:
        raise RuntimeError(f"No map expansion json files found under {root_path / 'maps' / 'expansion'}")

    scene_names = list_scene_names(root_path, version)
    if not scene_names:
        raise RuntimeError(f"No scenes found in {root_path / version / 'scene.json'}")

    map_names_repr = repr(map_names)
    map_api_patch = f"""from nuscenes.map_expansion import map_api as _nusc_map_api
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
for _map_name in {map_names_repr}:
    if _map_name not in _nusc_map_api.locations:
        _nusc_map_api.locations.append(_map_name)
"""
    split_block = """    from nuscenes.utils import splits
    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    if version in available_vers:
        if version == 'v1.0-trainval':
            train_scenes = splits.train
            val_scenes = splits.val
        elif version == 'v1.0-test':
            train_scenes = splits.test
            val_scenes = []
        elif version == 'v1.0-mini':
            train_scenes = splits.mini_train
            val_scenes = splits.mini_val
        else:
            raise ValueError('unknown')
    else:
        scene_names = sorted([scene['name'] for scene in nusc.scene])
        split_index = max(1, int(len(scene_names) * 0.8)) if len(scene_names) > 1 else len(scene_names)
        train_scenes = scene_names[:split_index]
        val_scenes = scene_names[split_index:]
"""
    old_split_block = """    from nuscenes.utils import splits
    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    assert version in available_vers
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError('unknown')
"""
    old_main_block = """if __name__ == '__main__':
    train_version = f'{args.version}-trainval'
    nuscenes_data_prep(
        root_path=args.root_path,
        can_bus_root_path=args.canbus,
        info_prefix=args.extra_tag,
        version=train_version,
        dataset_name='NuScenesDataset',
        out_dir=args.out_dir,
        max_sweeps=args.max_sweeps)
    test_version = f'{args.version}-test'
    nuscenes_data_prep(
        root_path=args.root_path,
        can_bus_root_path=args.canbus,
        info_prefix=args.extra_tag,
        version=test_version,
        dataset_name='NuScenesDataset',
        out_dir=args.out_dir,
        max_sweeps=args.max_sweeps)
"""
    old_networkx_block = """        roots = (v for v, d in pts_G.in_degree() if d == 0)
        leaves = [v for v, d in pts_G.out_degree() if d == 0]
        all_paths = []
        for root in roots:
            paths = nx.all_simple_paths(pts_G, root, leaves)
            all_paths.extend(paths)
"""
    new_networkx_block = """        roots = [v for v, d in pts_G.in_degree() if d == 0]
        leaves = [v for v, d in pts_G.out_degree() if d == 0]
        all_paths = []
        for root in roots:
            for leaf in leaves:
                if root == leaf:
                    continue
                if root not in pts_G or leaf not in pts_G:
                    continue
                paths = nx.all_simple_paths(pts_G, root, leaf)
                all_paths.extend(paths)
"""
    new_main_block = """if __name__ == '__main__':
    if args.version == 'v1.0':
        versions = [f'{args.version}-trainval', f'{args.version}-test']
    else:
        versions = [args.version]
    for data_version in versions:
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=data_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
"""

    source = replace_required(
        source,
        "    MAPS = ['boston-seaport', 'singapore-hollandvillage',\n"
        "                     'singapore-onenorth', 'singapore-queenstown']",
        f"    MAPS = {map_names_repr}",
        "map names",
    )
    source = replace_required(
        source,
        "from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer",
        map_api_patch.rstrip(),
        "nuScenes map api locations",
    )
    source = replace_required(source, old_split_block, split_block, "scene split")
    source = replace_required(source, old_networkx_block, new_networkx_block, "networkx all_simple_paths")
    source = replace_required(source, old_main_block.rstrip(), new_main_block.rstrip(), "main entrypoint")
    compile(source, str(OFFICIAL_CONVERTER), "exec")
    return source


def build_command(python: str, converter: Path, config: dict) -> list[str]:
    return [
        python,
        str(converter),
        "--root-path",
        str(Path(config["root_path"])),
        "--out-dir",
        str(Path(config["out_dir"])),
        "--extra-tag",
        str(config["extra_tag"]),
        "--version",
        str(config["version"]),
        "--canbus",
        str(Path(config["canbus"])),
        "--max-sweeps",
        str(config["max_sweeps"]),
    ]


def print_summary(dataset_name: str, config: dict, command: Iterable[str], patched: bool) -> None:
    print("")
    print("MapTR nuScenes data preparation")
    print(f"  dataset   : {dataset_name}")
    print(f"  root_path : {config['root_path']}")
    print(f"  out_dir   : {config['out_dir']}")
    print(f"  canbus    : {config['canbus']}")
    print(f"  version   : {config['version']}")
    print(f"  extra_tag : {config['extra_tag']}")
    print(f"  converter : {'temporary patched official converter' if patched else OFFICIAL_CONVERTER}")
    print("")
    print("Command:")
    print("  " + " ".join(str(part) for part in command))
    print("")


def run_command(command: list[str]) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(OFFICIAL_MAPTR_ROOT)
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    completed = subprocess.run(command, cwd=OFFICIAL_MAPTR_ROOT, env=env, check=False)
    return completed.returncode


def check_converter_dependencies(python: str) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(OFFICIAL_MAPTR_ROOT)
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    imports = [
        "mmcv",
        "numpy",
        "nuscenes.nuscenes",
        "nuscenes.utils.geometry_utils",
        "nuscenes.map_expansion.map_api",
        "nuscenes.eval.common.utils",
        "nuscenes.map_expansion.bitmap",
        "shapely.geometry",
        "matplotlib.patches",
        "networkx",
        "mmdet3d.core.bbox.box_np_ops",
        "mmdet3d.datasets",
    ]
    code = (
        "import importlib, sys\n"
        f"imports = {imports!r}\n"
        "missing = []\n"
        "for name in imports:\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "    except ModuleNotFoundError as exc:\n"
        "        missing.append((name, exc.name))\n"
        "if missing:\n"
        "    for requested, missing_name in missing:\n"
        "        print(f'{requested}: missing {missing_name}')\n"
        "    sys.exit(1)\n"
    )
    completed = subprocess.run(
        [python, "-c", code],
        cwd=OFFICIAL_MAPTR_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode == 0:
        return

    details = completed.stdout.strip() or completed.stderr.strip()
    message = [
        "MapTR converter dependencies are not available in this Python environment.",
        f"python: {python}",
    ]
    if details:
        message.extend(["missing imports:", details])
    message.extend(
        [
            "",
            "At minimum, the official converter needs nuscenes-devkit and the MapTR/MMCV environment.",
            "For the current error, install nuscenes-devkit into the same environment used to run this script.",
            "Example:",
            f"  {python} -m pip install nuscenes-devkit",
            "",
            "If later imports fail, install the package reported in 'missing imports'.",
            "Use --skip-dependency-check only if you intentionally want to run the converter anyway.",
        ]
    )
    raise RuntimeError("\n".join(message))


def main() -> int:
    args = parse_args()
    dataset_name = choose_dataset() if args.dataset == "interactive" else args.dataset
    config = apply_overrides(args, dataset_name)
    config["version"] = normalize_version(dataset_name, str(config["version"]))
    config["max_sweeps"] = args.max_sweeps
    validate_paths(config)
    if not args.dry_run and not args.skip_dependency_check:
        check_converter_dependencies(args.python)

    use_patched_converter = should_use_patched_converter(config)

    if use_patched_converter:
        with OFFICIAL_CONVERTER.open("r", encoding="utf-8") as f:
            source = f.read()
        source = patched_converter_source(source, Path(config["root_path"]), str(config["version"]))
        with tempfile.TemporaryDirectory(prefix="maptr_prepare_") as tmp_dir:
            patched_converter = Path(tmp_dir) / OFFICIAL_CONVERTER.name
            patched_converter.write_text(source, encoding="utf-8")
            command = build_command(args.python, patched_converter, config)
            print_summary(dataset_name, config, command, patched=True)
            if args.dry_run:
                return 0
            if not args.yes and input("Run this command? [y/N]: ").strip().lower() != "y":
                print("Canceled.")
                return 1
            return run_command(command)

    command = build_command(args.python, OFFICIAL_CONVERTER, config)
    print_summary(dataset_name, config, command, patched=False)
    if args.dry_run:
        return 0
    if not args.yes and input("Run this command? [y/N]: ").strip().lower() != "y":
        print("Canceled.")
        return 1
    return run_command(command)


if __name__ == "__main__":
    raise SystemExit(main())
