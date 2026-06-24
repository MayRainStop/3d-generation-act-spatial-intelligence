#!/usr/bin/env python3
"""Train ACT on the TA-provided official CALVIN splitA/B/C data.

The TA dataset is stored as four LeRobot v2.1 subfolders:
xiaoma26/calvin-lerobot/splitA, splitB, splitC, splitD.
This wrapper trains either on one official source scene splitA/splitB/splitC
or on a concatenation of splitA+B+C, and appends either task one-hot vectors
or SBERT language embeddings to observation.state.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
from bisect import bisect_right
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from lerobot.scripts import lerobot_train


ROOT = Path(__file__).resolve().parents[1]
TA_REPO_PREFIX = "xiaoma26/calvin_lerobot_split"
TA_ROOT_BASE = ROOT / "data" / "xiaoma_calvin_lerobot_v30" / "xiaoma26"


def load_module(name: str, rel_path: str) -> Any:
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SOURCE = load_module("hw3_source_taskcond", "scripts/26_train_source_split_task_conditioned_act.py")
LANG = load_module("hw3_lang_taskcond", "scripts/17_train_language_conditioned_act.py")


def split_repo_id(label: str) -> str:
    return f"{TA_REPO_PREFIX}{label}"


def split_root(label: str) -> Path:
    return TA_ROOT_BASE / f"calvin_lerobot_split{label}"


def set_dataset_cfg(cfg: Any, label: str) -> Any:
    cfg = copy.deepcopy(cfg)
    cfg.dataset.repo_id = split_repo_id(label)
    cfg.dataset.root = str(split_root(label))
    return cfg


def _as_array(value: Any) -> np.ndarray:
    return np.asarray(value)


def combine_feature_stats(items: list[dict[str, Any]]) -> dict[str, Any]:
    first = copy.deepcopy(items[0])
    if not all(isinstance(x, dict) for x in items):
        return first
    required = {"min", "max", "mean", "std", "count"}
    if not required.issubset(first):
        return first

    counts = [_as_array(x["count"]).astype(np.float64) for x in items]
    means = [_as_array(x["mean"]).astype(np.float64) for x in items]
    stds = [_as_array(x["std"]).astype(np.float64) for x in items]
    total = sum(counts)
    total = np.maximum(total, 1.0)
    mean = sum(m * c for m, c in zip(means, counts)) / total
    second = sum((s * s + m * m) * c for m, s, c in zip(means, stds, counts)) / total
    var = np.maximum(second - mean * mean, 0.0)

    first["min"] = np.minimum.reduce([_as_array(x["min"]) for x in items])
    first["max"] = np.maximum.reduce([_as_array(x["max"]) for x in items])
    first["mean"] = mean.astype(_as_array(items[0]["mean"]).dtype)
    first["std"] = np.sqrt(var).astype(_as_array(items[0]["std"]).dtype)
    first["count"] = total.astype(_as_array(items[0]["count"]).dtype)
    return first


def combine_stats(metas: list[Any]) -> dict[str, Any]:
    keys = sorted(set().union(*(set(meta.stats) for meta in metas)))
    combined: dict[str, Any] = {}
    for key in keys:
        values = [meta.stats[key] for meta in metas if key in meta.stats]
        combined[key] = combine_feature_stats(values)
    return combined


def episode_frame(meta: Any) -> pd.DataFrame:
    episodes = meta.episodes
    if hasattr(episodes, "copy"):
        return episodes.copy()
    return pd.DataFrame(list(episodes))


CANONICAL_KEY_MAP = {
    "state": "observation.state",
    "actions": "action",
    "image": "observation.images.image",
    "wrist_image": "observation.images.wrist_image",
}


def canonicalize_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(mapping)
    for old_key, new_key in CANONICAL_KEY_MAP.items():
        if old_key in out and new_key not in out:
            out[new_key] = out.pop(old_key)
        elif old_key in out:
            out.pop(old_key)
    return out


class CanonicalMetadata:
    def __init__(self, base_meta: Any):
        self._base_meta = base_meta
        self.info = copy.deepcopy(base_meta.info)
        self.info["features"] = canonicalize_mapping(self.info.get("features", {}))
        self.stats = canonicalize_mapping(getattr(base_meta, "stats", {}) or {})
        self.tasks = base_meta.tasks
        self.episodes = base_meta.episodes

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base_meta, name)

    @property
    def features(self) -> dict[str, dict]:
        return self.info["features"]

    @property
    def camera_keys(self) -> list[str]:
        keys = getattr(self._base_meta, "camera_keys", [])
        return [CANONICAL_KEY_MAP.get(key, key) for key in keys]


class CanonicalKeyDataset:
    def __init__(self, dataset: Any):
        self.dataset = dataset
        self.meta = CanonicalMetadata(dataset.meta)
        self.num_frames = getattr(dataset, "num_frames", 0)
        self.num_episodes = getattr(dataset, "num_episodes", 0)
        self.action_chunk_size = int(os.environ.get("HW3_TA_CHUNK_SIZE", "50"))

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def _to_int(value: Any) -> int:
        if hasattr(value, "item"):
            return int(value.item())
        return int(value)

    def _raw_row(self, idx: int) -> dict[str, Any]:
        hf_dataset = getattr(self.dataset, "hf_dataset", None)
        if hf_dataset is not None:
            return hf_dataset[int(idx)]
        return self.dataset[int(idx)]

    def _action_sequence(self, idx: int, episode_index: int, current_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        actions: list[torch.Tensor] = []
        pad: list[bool] = []
        for offset in range(self.action_chunk_size):
            src_idx = int(idx) + offset
            if src_idx < len(self.dataset):
                row = self._raw_row(src_idx)
                row_episode = self._to_int(row["episode_index"])
                if row_episode == episode_index:
                    action_value = row.get("actions", row.get("action"))
                    actions.append(torch.as_tensor(action_value, dtype=current_action.dtype).reshape(-1))
                    pad.append(False)
                    continue
            actions.append(torch.zeros_like(current_action))
            pad.append(True)
        return torch.stack(actions, dim=0), torch.tensor(pad, dtype=torch.bool)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = dict(self.dataset[idx])
        for old_key, new_key in CANONICAL_KEY_MAP.items():
            if old_key in item and new_key not in item:
                item[new_key] = item.pop(old_key)
            elif old_key in item:
                item.pop(old_key)

        state = item.get("observation.state")
        if state is not None:
            if not torch.is_tensor(state):
                state = torch.as_tensor(state)
            item["observation.state"] = state

        action = item.get("action")
        if action is not None:
            if not torch.is_tensor(action):
                action = torch.as_tensor(action)
            episode_index = self._to_int(item["episode_index"])
            action = action.reshape(-1)
            action_seq, action_is_pad = self._action_sequence(int(idx), episode_index, action)
            item["action"] = action_seq
            item["action_is_pad"] = action_is_pad
        return item

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataset, name)


class CombinedMetadata:
    def __init__(self, metas: list[Any], labels: list[str]):
        self._metas = metas
        self.info = copy.deepcopy(metas[0].info)
        self.stats = combine_stats(metas)
        self.tasks = metas[0].tasks
        self.labels = labels
        self.episodes = pd.concat(
            [episode_frame(meta).assign(ta_scene=label) for meta, label in zip(metas, labels)],
            ignore_index=True,
        )

        total_episodes = int(sum(getattr(meta, "num_episodes", len(episode_frame(meta))) for meta in metas))
        total_frames = int(sum(getattr(meta, "num_frames", int(episode_frame(meta)["length"].sum())) for meta in metas))
        self.info["total_episodes"] = total_episodes
        self.info["total_frames"] = total_frames
        self.info["scene"] = "".join(labels)
        self.info["source_dataset"] = "xiaoma26/calvin-lerobot official splitA/B/C"

    def __getattr__(self, name: str) -> Any:
        return getattr(self._metas[0], name)

    @property
    def features(self) -> dict[str, dict]:
        return self.info["features"]


class CombinedLeRobotDataset:
    def __init__(self, datasets: list[Any], labels: list[str]):
        self.datasets = datasets
        self.labels = labels
        self.cumulative: list[int] = []
        total = 0
        for dataset in datasets:
            total += len(dataset)
            self.cumulative.append(total)
        self.meta = CombinedMetadata([dataset.meta for dataset in datasets], labels)
        self.num_frames = sum(getattr(dataset, "num_frames", 0) for dataset in datasets)
        self.num_episodes = sum(getattr(dataset, "num_episodes", 0) for dataset in datasets)

    def __len__(self) -> int:
        return self.cumulative[-1]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        idx = int(idx)
        ds_idx = bisect_right(self.cumulative, idx)
        prev = 0 if ds_idx == 0 else self.cumulative[ds_idx - 1]
        return self.datasets[ds_idx][idx - prev]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.datasets[0], name)


class TaskAlignedMetadata:
    def __init__(self, meta: Any, tasks: pd.DataFrame):
        self._meta = meta
        self.tasks = tasks

    def __getattr__(self, name: str) -> Any:
        return getattr(self._meta, name)


class TaskIndexAlignedDataset:
    """Remap per-split task indices to the canonical splitA task table."""

    def __init__(self, dataset: Any, canonical_tasks: pd.DataFrame):
        self.dataset = dataset
        self.meta = TaskAlignedMetadata(dataset.meta, canonical_tasks)
        self.num_frames = getattr(dataset, "num_frames", 0)
        self.num_episodes = getattr(dataset, "num_episodes", 0)
        canonical_by_text = {str(task_text): int(row["task_index"]) for task_text, row in canonical_tasks.iterrows()}
        self.old_to_new: dict[int, int] = {}
        for task_text, row in dataset.meta.tasks.iterrows():
            task_text = str(task_text)
            if task_text not in canonical_by_text:
                raise ValueError(f"Task missing from canonical splitA table: {task_text!r}")
            self.old_to_new[int(row["task_index"])] = canonical_by_text[task_text]
        if len(self.old_to_new) != len(canonical_tasks):
            raise ValueError(
                f"Task remap incomplete: mapped={len(self.old_to_new)} canonical={len(canonical_tasks)}"
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = dict(self.dataset[idx])
        old_index = CanonicalKeyDataset._to_int(item["task_index"])
        item["task_index"] = self.old_to_new[old_index]
        return item

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataset, name)


def assert_task_sets_compatible(datasets: list[Any]) -> None:
    ref = set(map(str, datasets[0].meta.tasks.index))
    for dataset in datasets[1:]:
        cur = set(map(str, dataset.meta.tasks.index))
        if cur != ref:
            raise ValueError(
                "TA split task descriptions differ; cannot concatenate splitA/B/C safely"
            )


def make_base_dataset(cfg: Any) -> tuple[Any, list[str]]:
    split = os.environ.get("HW3_TA_SOURCE_SPLIT", "ABC").upper()
    if split not in {"A", "B", "C", "ABC"}:
        raise ValueError(f"Unsupported HW3_TA_SOURCE_SPLIT={split!r}")
    if split == "ABC":
        labels = ["A", "B", "C"]
        datasets = [CanonicalKeyDataset(SOURCE._ORIGINAL_MAKE_DATASET(set_dataset_cfg(cfg, label))) for label in labels]
        assert_task_sets_compatible(datasets)
        canonical_tasks = datasets[0].meta.tasks.copy()
        datasets = [TaskIndexAlignedDataset(dataset, canonical_tasks) for dataset in datasets]
        return CombinedLeRobotDataset(datasets, labels), labels
    dataset = CanonicalKeyDataset(SOURCE._ORIGINAL_MAKE_DATASET(set_dataset_cfg(cfg, split)))
    return dataset, [split]


def write_selection_summary(base_dataset: Any, labels: list[str], condition_type: str) -> None:
    results = ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "TA-provided official split dataset: xiaoma26/calvin-lerobot",
        "labels": labels,
        "condition_type": condition_type,
        "repo_ids": [split_repo_id(label) for label in labels],
        "roots": [str(split_root(label)) for label in labels],
        "num_frames": int(getattr(base_dataset, "num_frames", 0)),
        "num_episodes": int(getattr(base_dataset, "num_episodes", 0)),
        "num_tasks": int(len(base_dataset.meta.tasks)),
        "state_shape_before_conditioning": base_dataset.meta.features["observation.state"]["shape"],
    }
    out = results / f"task2_ta_official_{''.join(labels)}_{condition_type}_selection_summary.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


def make_ta_official_dataset(cfg: Any) -> Any:
    condition_type = os.environ.get("HW3_TA_CONDITION_TYPE", "onehot").strip().lower()
    base_dataset, labels = make_base_dataset(cfg)
    write_selection_summary(base_dataset, labels, condition_type)
    if condition_type == "onehot":
        wrapped = SOURCE.TaskConditionedDataset(base_dataset, None, f"ta_official_{''.join(labels)}")
    elif condition_type == "sbert":
        wrapped = LANG.LanguageConditionedDataset(base_dataset)
    else:
        raise ValueError(f"Unsupported HW3_TA_CONDITION_TYPE={condition_type!r}")
    print(
        "TA official split conditioned dataset:",
        f"labels={''.join(labels)}",
        f"condition_type={condition_type}",
        f"len={len(wrapped)}",
        f"num_frames={getattr(wrapped, 'num_frames', 'unknown')}",
        f"num_episodes={getattr(wrapped, 'num_episodes', 'unknown')}",
        f"state_shape={wrapped.meta.features['observation.state']['shape']}",
        flush=True,
    )
    return wrapped

_ORIGINAL_UPDATE_POLICY = lerobot_train.update_policy


def debug_update_policy(train_metrics: Any, policy: Any, batch: Any, optimizer: Any, grad_clip_norm: float, accelerator: Any, lr_scheduler: Any = None, lock: Any = None, rabc_weights_provider: Any = None) -> tuple[Any, dict]:
    debug_path = ROOT / "results" / "task2_ta_debug_batch_shapes.json"
    if os.environ.get("HW3_TA_DEBUG_SHAPES") == "1" and not debug_path.exists():
        payload = {}
        for key, value in batch.items():
            payload[key] = {
                "type": type(value).__name__,
                "shape": list(value.shape) if hasattr(value, "shape") else None,
                "dtype": str(getattr(value, "dtype", "")),
            }
        debug_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return _ORIGINAL_UPDATE_POLICY(train_metrics, policy, batch, optimizer, grad_clip_norm, accelerator, lr_scheduler, lock, rabc_weights_provider)


def main() -> int:
    lerobot_train.make_dataset = make_ta_official_dataset
    if os.environ.get("HW3_TA_DEBUG_SHAPES") == "1":
        lerobot_train.update_policy = debug_update_policy
    lerobot_train.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
