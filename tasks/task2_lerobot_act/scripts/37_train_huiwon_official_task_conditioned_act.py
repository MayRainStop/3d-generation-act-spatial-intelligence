#!/usr/bin/env python3
"""Train ACT on huiwon CALVIN with official A/B/C labels from original_frame_idx."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from lerobot.scripts import lerobot_train


ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "huiwon/calvin_task_ABC_D"
DEFAULT_ROOT = ROOT / "data" / "huiwon_calvin_task_ABC_D" / "calvin_task_ABC_D_lerobot_0_4"


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


def selected_episodes(split_map: Path, env_label: str) -> list[int]:
    df = pd.read_csv(split_map)
    if env_label == "ABC":
        selected = df["episode_index"].tolist()
    else:
        selected = df[df["official_env"] == env_label]["episode_index"].tolist()
    if not selected:
        raise ValueError(f"No episodes found for {env_label} in {split_map}")
    return [int(v) for v in selected]


def rename_feature_key(mapping: dict, old: str, new: str) -> None:
    if old in mapping:
        mapping[new] = mapping.pop(old)


class HuiwonOfficialMetadata:
    def __init__(self, base_meta: Any, num_tasks: int, selected: set[int] | None):
        self._base_meta = base_meta
        self.info = copy.deepcopy(base_meta.info)
        self.stats = copy.deepcopy(base_meta.stats)
        self.num_tasks = num_tasks
        self.selected = selected

        features = self.info["features"]
        rename_feature_key(features, "observation.images.image", "observation.images.top")
        rename_feature_key(features, "observation.images.wrist_image", "observation.images.wrist")
        rename_feature_key(self.stats, "observation.images.image", "observation.images.top")
        rename_feature_key(self.stats, "observation.images.wrist_image", "observation.images.wrist")

        state_feature = features["observation.state"]
        old_shape = tuple(state_feature["shape"])
        state_feature["shape"] = (old_shape[0] + num_tasks,)

        state_stats = self.stats["observation.state"]
        counts = self.task_frame_counts(base_meta, num_tasks)
        probs = (counts / counts.sum()).numpy()
        std = np.sqrt(np.maximum(probs * (1.0 - probs), 1e-12))
        self.stats["observation.state"] = {
            "min": np.concatenate([state_stats["min"], np.zeros(num_tasks, dtype=state_stats["min"].dtype)]),
            "max": np.concatenate([state_stats["max"], np.ones(num_tasks, dtype=state_stats["max"].dtype)]),
            "mean": np.concatenate([state_stats["mean"], probs.astype(state_stats["mean"].dtype)]),
            "std": np.concatenate([state_stats["std"], std.astype(state_stats["std"].dtype)]),
            "count": state_stats["count"],
        }

    def task_frame_counts(self, base_meta: Any, num_tasks: int) -> torch.Tensor:
        task_to_idx = {str(name): int(row.task_index) for name, row in base_meta.tasks.iterrows()}
        counts = torch.zeros(num_tasks, dtype=torch.float64)
        for ep in SOURCE._episode_iter(base_meta.episodes):
            episode_index = int(SOURCE._episode_get(ep, "episode_index"))
            if self.selected is not None and episode_index not in self.selected:
                continue
            task = SOURCE._first_task(SOURCE._episode_get(ep, "tasks"))
            length = int(SOURCE._episode_get(ep, "length"))
            if task in task_to_idx:
                counts[task_to_idx[task]] += length
        if int(counts.sum().item()) <= 0:
            raise ValueError("No task frame counts for selected huiwon split")
        return counts

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base_meta, name)

    @property
    def features(self) -> dict[str, dict]:
        return self.info["features"]


class HuiwonOfficialDataset:
    def __init__(self, base_dataset: Any, selected: set[int] | None, split_label: str):
        self.base_dataset = base_dataset
        self.num_tasks = len(base_dataset.meta.tasks)
        self.meta = HuiwonOfficialMetadata(base_dataset.meta, self.num_tasks, selected)
        self.split_label = split_label

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.base_dataset[idx]
        if "observation.images.image" in item:
            item["observation.images.top"] = item.pop("observation.images.image")
        if "observation.images.wrist_image" in item:
            item["observation.images.wrist"] = item.pop("observation.images.wrist_image")

        task_index = SOURCE._as_int(item["task_index"])
        state = item["observation.state"]
        if not torch.is_tensor(state):
            state = torch.as_tensor(state)
        one_hot = torch.zeros(self.num_tasks, dtype=state.dtype, device=state.device)
        one_hot[task_index] = 1
        if state.ndim == 1:
            item["observation.state"] = torch.cat([state, one_hot], dim=0)
        elif state.ndim == 2:
            item["observation.state"] = torch.cat([state, one_hot.unsqueeze(0).expand(state.shape[0], -1)], dim=-1)
        else:
            raise ValueError(f"Unsupported state shape {tuple(state.shape)}")
        item["task_condition_index"] = torch.tensor(task_index, dtype=torch.long)
        return item

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_dataset, name)


def make_dataset(cfg):
    split = os.environ.get("HW3_HUIWON_SOURCE_SPLIT", "A").upper()
    split_map = Path(os.environ.get("HW3_HUIWON_SPLIT_MAP", str(ROOT / "results" / "task2_huiwon_official_scene_split_map.csv")))
    root = Path(getattr(cfg.dataset, "root", None) or DEFAULT_ROOT).expanduser()
    cfg.dataset.root = str(root)
    cfg.dataset.repo_id = REPO_ID
    cfg.dataset.video_backend = os.environ.get("HW3_HUIWON_VIDEO_BACKEND", "pyav")
    selected = selected_episodes(split_map, split)
    selected_set = None if split == "ABC" else set(selected)
    if split != "ABC":
        cfg.dataset.episodes = selected
    dataset = SOURCE._ORIGINAL_MAKE_DATASET(cfg)
    wrapped = HuiwonOfficialDataset(dataset, selected_set, f"huiwon_official_{split}")
    out = ROOT / "results" / f"task2_huiwon_{split}_selection_summary.json"
    df = pd.read_csv(split_map)
    sdf = df[df["episode_index"].isin(selected)]
    payload = {
        "split": split,
        "repo_id": REPO_ID,
        "dataset_root": str(root),
        "num_episodes": int(len(sdf)),
        "num_frames": int(sdf["length"].sum()),
        "unique_tasks": int(sdf["task_index"].nunique()),
        "official_source": "task_ABC_D scene_info + original_frame_idx",
        "state_shape": wrapped.meta.features["observation.state"]["shape"],
        "image_keys": ["observation.images.top", "observation.images.wrist"],
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    return wrapped


def main() -> int:
    lerobot_train.make_dataset = make_dataset
    lerobot_train.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
