#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CALVIN_ROOT = ROOT / "data" / "calvin_raw" / "calvin"


def patch_numpy_legacy_aliases() -> None:
    aliases = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
    }
    for name, value in aliases.items():
        if not hasattr(np, name):
            setattr(np, name, value)


def add_calvin_paths() -> None:
    paths = [
        CALVIN_ROOT / "calvin_env" / "tacto",
        CALVIN_ROOT / "calvin_models",
        CALVIN_ROOT / "calvin_env",
    ]
    for path in paths:
        sys.path.insert(0, str(path))


def patch_calvin_git_hash_for_tarball(play_table_env_module) -> None:
    original_get_git_commit_hash = play_table_env_module.get_git_commit_hash

    def safe_get_git_commit_hash(repo_path: Path) -> str:
        try:
            return original_get_git_commit_hash(repo_path)
        except Exception:
            return "tarball-no-git-metadata"

    play_table_env_module.get_git_commit_hash = safe_get_git_commit_hash


def summarize_obs(obs: dict) -> dict:
    summary = {}
    for key, value in obs.items():
        if hasattr(value, "shape"):
            arr = np.asarray(value)
            summary[key] = {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "min": float(np.nanmin(arr)) if arr.size else None,
                "max": float(np.nanmax(arr)) if arr.size else None,
            }
        else:
            summary[key] = str(type(value))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        default=str(ROOT / "data" / "calvin_raw" / "datasets" / "calvin_debug_dataset"),
    )
    parser.add_argument("--out", default=str(ROOT / "results" / "calvin_env_smoke.json"))
    args = parser.parse_args()

    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    patch_numpy_legacy_aliases()
    add_calvin_paths()

    from calvin_env.envs import play_table_env

    patch_calvin_git_hash_for_tarball(play_table_env)
    get_env = play_table_env.get_env

    dataset_path = Path(args.dataset_path).resolve()
    val_folder = dataset_path / "validation"
    if not val_folder.exists():
        raise FileNotFoundError(f"Missing validation folder: {val_folder}")

    env = get_env(val_folder, show_gui=False)
    obs = env.get_obs()
    action = np.zeros(7, dtype=np.float32)
    action[-1] = 1.0
    next_obs, reward, done, info = env.step(action)
    image = env.render(mode="rgb_array")

    result = {
        "dataset_path": str(dataset_path),
        "validation_path": str(val_folder),
        "obs": summarize_obs(obs),
        "next_obs": summarize_obs(next_obs),
        "reward": float(reward),
        "done": bool(done),
        "info_keys": sorted(list(info.keys())) if isinstance(info, dict) else str(type(info)),
        "render_shape": list(np.asarray(image).shape),
        "render_dtype": str(np.asarray(image).dtype),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
