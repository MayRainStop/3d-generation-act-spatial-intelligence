#!/usr/bin/env python
from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from lerobot.scripts import lerobot_train


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


class TaskConditionedMetadata:
    """Metadata proxy that exposes task one-hot as part of observation.state."""

    def __init__(self, base_meta: Any, num_tasks: int):
        self._base_meta = base_meta
        self.info = copy.deepcopy(base_meta.info)
        self.stats = copy.deepcopy(base_meta.stats)
        self.num_tasks = num_tasks

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
        for episode in _episode_iter(base_meta.episodes):
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

    def __init__(self, base_dataset: Any):
        self.base_dataset = base_dataset
        self.num_tasks = len(base_dataset.meta.tasks)
        self.meta = TaskConditionedMetadata(base_dataset.meta, self.num_tasks)

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


def make_task_conditioned_dataset(cfg):
    dataset = _ORIGINAL_MAKE_DATASET(cfg)
    wrapped = TaskConditionedDataset(dataset)
    print(
        "Task-conditioned dataset:",
        f"tasks={wrapped.num_tasks}",
        f"state_shape={wrapped.meta.features['observation.state']['shape']}",
        flush=True,
    )
    return wrapped


def main() -> int:
    lerobot_train.make_dataset = make_task_conditioned_dataset
    lerobot_train.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
