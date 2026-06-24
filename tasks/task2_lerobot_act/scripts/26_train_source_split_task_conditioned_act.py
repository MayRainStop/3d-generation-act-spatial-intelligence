#!/usr/bin/env python
from __future__ import annotations

import copy
import csv
import json
import os
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from lerobot.scripts import lerobot_train

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ID = "fywang/calvin-task-ABC-D-lerobot"


def _as_int(value: Any) -> int:
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def _episode_iter(episodes: Any):
    if hasattr(episodes, "iterrows"):
        for _, episode in episodes.iterrows():
            yield episode
    else:
        yield from episodes


def _episode_get(episode: Any, key: str):
    if isinstance(episode, Mapping):
        return episode[key]
    return episode[key]


def _first_task(tasks: Any) -> str:
    if isinstance(tasks, np.ndarray):
        return str(tasks[0])
    if isinstance(tasks, (list, tuple)):
        return str(tasks[0])
    return str(tasks)


def _dataset_root_for_repo(repo_id: str, cfg_root: Any = None) -> Path:
    if cfg_root:
        return Path(str(cfg_root)).expanduser()
    hf_home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))).expanduser()
    return hf_home / "lerobot" / repo_id


def _read_episode_metadata(dataset_root: Path) -> pd.DataFrame:
    episode_dir = dataset_root / "meta" / "episodes"
    paths = sorted(episode_dir.rglob("*.parquet"))
    if not paths:
        alt = dataset_root / "meta" / "episodes.parquet"
        if alt.exists():
            paths = [alt]
    if not paths:
        raise FileNotFoundError(f"No episode metadata parquet found under {episode_dir}")
    frames = [pd.read_parquet(path) for path in paths]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("episode_index").reset_index(drop=True)
    return df


def _build_split_frames(episodes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    n = len(episodes)
    first = n // 3
    second = (2 * n) // 3
    return {
        "A": episodes.iloc[:first].copy(),
        "B": episodes.iloc[first:second].copy(),
        "C": episodes.iloc[second:].copy(),
        "ABC": episodes.copy(),
    }


def _split_summary_row(label: str, frame: pd.DataFrame, mode: str) -> dict[str, Any]:
    tasks = set()
    for value in frame["tasks"]:
        tasks.add(_first_task(value))
    return {
        "split_label": label,
        "split_mode": mode,
        "proxy_source_split": True,
        "episode_start": int(frame["episode_index"].min()) if len(frame) else None,
        "episode_end": int(frame["episode_index"].max()) if len(frame) else None,
        "num_episodes": int(len(frame)),
        "num_frames": int(frame["length"].sum()) if "length" in frame else None,
        "unique_tasks": int(len(tasks)),
        "note": "episode-order third of the public ABC LeRobot dataset; used because raw A/B/C labels are not exposed in this converted dataset",
    }


def _write_split_artifacts(dataset_root: Path, split_frames: dict[str, pd.DataFrame], mode: str) -> None:
    results = ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    rows = [_split_summary_row(label, frame, mode) for label, frame in split_frames.items()]
    csv_path = results / "task2_source_split_map.csv"
    json_path = results / "task2_source_split_map.json"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "updated_at": datetime.now().astimezone().isoformat(),
        "dataset_root": str(dataset_root),
        "split_mode": mode,
        "proxy_source_split": True,
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class TaskConditionedMetadata:
    """Metadata proxy that exposes task one-hot as part of observation.state."""

    def __init__(self, base_meta: Any, num_tasks: int, selected_episodes: set[int] | None):
        self._base_meta = base_meta
        self.info = copy.deepcopy(base_meta.info)
        self.stats = copy.deepcopy(base_meta.stats)
        self.num_tasks = num_tasks
        self.selected_episodes = selected_episodes

        state_feature = self.info["features"]["observation.state"]
        old_shape = tuple(state_feature["shape"])
        if len(old_shape) != 1:
            raise ValueError(f"Expected 1D observation.state, got shape={old_shape}")
        state_feature["shape"] = (old_shape[0] + num_tasks,)

        state_stats = self.stats["observation.state"]
        counts = self._task_frame_counts(base_meta, num_tasks)
        probs = (counts / counts.sum()).numpy()
        std = np.sqrt(np.maximum(probs * (1.0 - probs), 1e-12))

        self.stats["observation.state"] = {
            "min": np.concatenate([state_stats["min"], np.zeros(num_tasks, dtype=state_stats["min"].dtype)]),
            "max": np.concatenate([state_stats["max"], np.ones(num_tasks, dtype=state_stats["max"].dtype)]),
            "mean": np.concatenate([state_stats["mean"], probs.astype(state_stats["mean"].dtype)]),
            "std": np.concatenate([state_stats["std"], std.astype(state_stats["std"].dtype)]),
            "count": state_stats["count"],
        }

    def _task_frame_counts(self, base_meta: Any, num_tasks: int) -> torch.Tensor:
        task_to_idx = {str(name): int(row.task_index) for name, row in base_meta.tasks.iterrows()}
        counts = torch.zeros(num_tasks, dtype=torch.float64)
        selected = self.selected_episodes
        for episode in _episode_iter(base_meta.episodes):
            episode_index = int(_episode_get(episode, "episode_index"))
            if selected is not None and episode_index not in selected:
                continue
            task = _first_task(_episode_get(episode, "tasks"))
            length = int(_episode_get(episode, "length"))
            counts[task_to_idx[task]] += length
        if int(counts.sum().item()) <= 0:
            raise ValueError("No task frame counts found for task-conditioned metadata")
        return counts

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base_meta, name)

    @property
    def features(self) -> dict[str, dict]:
        return self.info["features"]


class TaskConditionedDataset:
    """Dataset wrapper that appends task one-hot to observation.state."""

    def __init__(self, base_dataset: Any, selected_episodes: set[int] | None, split_label: str):
        self.base_dataset = base_dataset
        self.num_tasks = len(base_dataset.meta.tasks)
        self.meta = TaskConditionedMetadata(base_dataset.meta, self.num_tasks, selected_episodes)
        self.split_label = split_label

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.base_dataset[idx]
        task_index = _as_int(item["task_index"])
        state = item["observation.state"]

        if not torch.is_tensor(state):
            state = torch.as_tensor(state)
        one_hot = torch.zeros(self.num_tasks, dtype=state.dtype, device=state.device)
        one_hot[task_index] = 1

        if state.ndim == 1:
            conditioned_state = torch.cat([state, one_hot], dim=0)
        elif state.ndim == 2:
            expanded = one_hot.unsqueeze(0).expand(state.shape[0], -1)
            conditioned_state = torch.cat([state, expanded], dim=-1)
        else:
            raise ValueError(f"Unsupported observation.state shape: {tuple(state.shape)}")

        item["observation.state"] = conditioned_state
        item["task_condition_index"] = torch.tensor(task_index, dtype=torch.long)
        return item

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_dataset, name)


_ORIGINAL_MAKE_DATASET = lerobot_train.make_dataset


def make_source_split_task_conditioned_dataset(cfg):
    split_label = os.environ.get("HW3_SOURCE_SPLIT", "ABC").upper()
    split_mode = os.environ.get("HW3_SOURCE_SPLIT_MODE", "episode_order_thirds")
    repo_id = str(cfg.dataset.repo_id)
    dataset_root = _dataset_root_for_repo(repo_id, getattr(cfg.dataset, "root", None))

    selected_set: set[int] | None = None
    if repo_id == DEFAULT_REPO_ID and split_mode == "episode_order_thirds":
        episodes = _read_episode_metadata(dataset_root)
        split_frames = _build_split_frames(episodes)
        _write_split_artifacts(dataset_root, split_frames, split_mode)
        if split_label not in split_frames:
            raise ValueError(f"Unknown HW3_SOURCE_SPLIT={split_label}; expected A, B, C, or ABC")
        if split_label != "ABC":
            selected = [int(v) for v in split_frames[split_label]["episode_index"].tolist()]
            cfg.dataset.episodes = selected
            selected_set = set(selected)
    else:
        print(
            f"Source split disabled: repo_id={repo_id}, split_mode={split_mode}. Using full dataset.",
            flush=True,
        )
        split_label = "ABC"

    dataset = _ORIGINAL_MAKE_DATASET(cfg)
    wrapped = TaskConditionedDataset(dataset, selected_set, split_label)
    print(
        "Source-split task-conditioned dataset:",
        f"split={split_label}",
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
    lerobot_train.make_dataset = make_source_split_task_conditioned_dataset
    lerobot_train.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
