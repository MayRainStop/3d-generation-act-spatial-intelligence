#!/usr/bin/env python3
"""Train task-conditioned ACT on official CALVIN source-scene episodes."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from lerobot.scripts import lerobot_train


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ID = "fywang/calvin-task-ABC-D-lerobot"


def load_source_module() -> Any:
    path = ROOT / "scripts" / "26_train_source_split_task_conditioned_act.py"
    spec = importlib.util.spec_from_file_location("hw3_source_split_task_conditioned", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SOURCE = load_source_module()


def selected_episodes_from_official_map(split_map: Path, env_label: str) -> list[int]:
    df = pd.read_csv(split_map)
    if env_label == "ABC":
        return [int(v) for v in df["episode_index"].tolist()]
    selected = df[df["official_env"] == env_label]["episode_index"].tolist()
    if not selected:
        raise ValueError(f"No episodes found for official_env={env_label} in {split_map}")
    return [int(v) for v in selected]


def write_selection_artifact(dataset_root: Path, split_map: Path, env_label: str, selected: list[int]) -> None:
    df = pd.read_csv(split_map)
    sdf = df[df["episode_index"].isin(selected)]
    payload = {
        "source_split_mode": "official_calvin_scene_info",
        "official_env": env_label,
        "dataset_root": str(dataset_root),
        "split_map": str(split_map),
        "num_episodes": int(len(sdf)),
        "num_frames": int(sdf["length"].sum()),
        "episode_start": int(sdf["episode_index"].min()),
        "episode_end": int(sdf["episode_index"].max()),
        "unique_tasks": int(sdf["task"].nunique()),
        "all_fully_within_single_scene": bool(sdf["fully_within_single_scene"].all()),
        "selected_episodes_head": [int(v) for v in selected[:10]],
        "selected_episodes_tail": [int(v) for v in selected[-10:]],
    }
    out = ROOT / "results" / f"task2_official_{env_label}_selection_summary.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


def make_official_scene_task_conditioned_dataset(cfg):
    repo_id = str(cfg.dataset.repo_id)
    if repo_id != DEFAULT_REPO_ID:
        raise ValueError(f"Expected {DEFAULT_REPO_ID}, got {repo_id}")
    env_label = os.environ.get("HW3_OFFICIAL_SOURCE_SPLIT", "A").upper()
    split_map = Path(os.environ.get("HW3_OFFICIAL_SCENE_SPLIT_MAP", str(ROOT / "results" / "task2_official_scene_split_map.csv")))
    dataset_root = SOURCE._dataset_root_for_repo(repo_id, getattr(cfg.dataset, "root", None))
    selected = selected_episodes_from_official_map(split_map, env_label)
    selected_set = None if env_label == "ABC" else set(selected)
    if env_label != "ABC":
        cfg.dataset.episodes = selected
    write_selection_artifact(dataset_root, split_map, env_label, selected)
    dataset = SOURCE._ORIGINAL_MAKE_DATASET(cfg)
    wrapped = SOURCE.TaskConditionedDataset(dataset, selected_set, f"official_scene_{env_label}")
    print(
        "Official-scene task-conditioned dataset:",
        f"split={env_label}",
        f"selected_episodes={'all' if selected_set is None else len(selected_set)}",
        f"len={len(wrapped)}",
        f"num_frames={getattr(wrapped, 'num_frames', 'unknown')}",
        f"num_episodes={getattr(wrapped, 'num_episodes', 'unknown')}",
        f"tasks={wrapped.num_tasks}",
        f"state_shape={wrapped.meta.features['observation.state']['shape']}",
        flush=True,
    )
    return wrapped


def main() -> int:
    lerobot_train.make_dataset = make_official_scene_task_conditioned_dataset
    lerobot_train.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
