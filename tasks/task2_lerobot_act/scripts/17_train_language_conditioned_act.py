#!/usr/bin/env python
from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any
import os
import zlib

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


def normalize_text(text: str) -> str:
    return " ".join(str(text).lower().strip().replace("_", " ").replace(":", " ").split())


def language_from_full_task(full_task: str) -> str:
    if ":" in str(full_task):
        subtask, language = str(full_task).split(":", 1)
        return f"{subtask.strip()}: {language.strip()}"
    return str(full_task)


def hash_embed(text: str, dim: int = 384) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for token in normalize_text(text).split():
        digest = zlib.crc32(token.encode("utf-8"))
        idx = digest % dim
        sign = 1.0 if ((digest >> 8) & 1) else -1.0
        vec[idx] += sign
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


def build_embedding_table(tasks: Any) -> tuple[np.ndarray, dict[int, str], dict[str, Any]]:
    backend = os.environ.get("HW3_LANG_EMBED_BACKEND", "sbert").strip().lower()
    model_name = os.environ.get("HW3_LANG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    hash_dim = int(os.environ.get("HW3_LANG_HASH_DIM", "384"))
    task_rows = []
    for full_task, row in tasks.sort_values("task_index").iterrows():
        task_rows.append((int(row["task_index"]), str(full_task), language_from_full_task(str(full_task))))
    if not task_rows:
        raise ValueError("No task rows available for language-conditioned metadata")
    max_idx = max(idx for idx, _, _ in task_rows)
    task_text_by_idx = {idx: text for idx, _, text in task_rows}
    if backend == "sbert":
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise RuntimeError(
                "sentence-transformers is required for HW3_LANG_EMBED_BACKEND=sbert. "
                "Install it first or set HW3_LANG_EMBED_BACKEND=hash for a fallback ablation."
            ) from exc
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        cache_folder = os.environ.get("SENTENCE_TRANSFORMERS_HOME") or os.environ.get("HF_HOME")
        model = SentenceTransformer(model_name, cache_folder=cache_folder)
        texts = [task_text_by_idx.get(i, "") for i in range(max_idx + 1)]
        embeddings = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        return embeddings, task_text_by_idx, {"backend": backend, "model_name": model_name, "dim": int(embeddings.shape[1])}
    if backend == "hash":
        embeddings = np.stack([hash_embed(task_text_by_idx.get(i, ""), hash_dim) for i in range(max_idx + 1)])
        return embeddings.astype(np.float32), task_text_by_idx, {"backend": backend, "model_name": "hash-bow-crc32", "dim": int(embeddings.shape[1])}
    raise ValueError(f"Unsupported HW3_LANG_EMBED_BACKEND={backend!r}")


class LanguageConditionedMetadata:
    """Metadata proxy that exposes language embedding as part of observation.state."""

    def __init__(self, base_meta: Any, embeddings: np.ndarray, task_text_by_idx: dict[int, str], embed_info: dict[str, Any]):
        self._base_meta = base_meta
        self.info = copy.deepcopy(base_meta.info)
        self.stats = copy.deepcopy(base_meta.stats)
        self.embeddings = embeddings.astype(np.float32)
        self.task_text_by_idx = task_text_by_idx
        self.embed_info = embed_info
        state_feature = self.info["features"]["observation.state"]
        old_shape = tuple(state_feature["shape"])
        if len(old_shape) != 1:
            raise ValueError(f"Expected 1D observation.state, got shape={old_shape}")
        state_feature["shape"] = (old_shape[0] + int(self.embeddings.shape[1]),)
        state_stats = self.stats["observation.state"]
        counts = self._task_frame_counts(base_meta, self.embeddings.shape[0]).numpy().astype(np.float64)
        weights = counts / counts.sum()
        emb = self.embeddings.astype(np.float64)
        mean = (weights[:, None] * emb).sum(axis=0)
        var = (weights[:, None] * np.square(emb - mean[None, :])).sum(axis=0)
        std = np.sqrt(np.maximum(var, 1e-12))
        base_min = np.asarray(state_stats["min"])
        base_max = np.asarray(state_stats["max"])
        base_mean = np.asarray(state_stats["mean"])
        base_std = np.asarray(state_stats["std"])
        dtype = base_mean.dtype
        self.stats["observation.state"] = {
            "min": np.concatenate([base_min, emb.min(axis=0).astype(dtype)]),
            "max": np.concatenate([base_max, emb.max(axis=0).astype(dtype)]),
            "mean": np.concatenate([base_mean, mean.astype(dtype)]),
            "std": np.concatenate([base_std, std.astype(dtype)]),
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
            raise ValueError("No task frame counts found for language-conditioned metadata")
        return counts

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base_meta, name)

    @property
    def features(self) -> dict[str, dict]:
        return self.info["features"]


class LanguageConditionedDataset:
    """Dataset wrapper that appends language embeddings to observation.state."""

    def __init__(self, base_dataset: Any):
        self.base_dataset = base_dataset
        embeddings, task_text_by_idx, embed_info = build_embedding_table(base_dataset.meta.tasks)
        self.embeddings = torch.from_numpy(embeddings)
        self.meta = LanguageConditionedMetadata(base_dataset.meta, embeddings, task_text_by_idx, embed_info)
        self.embedding_dim = int(embeddings.shape[1])
        self.embed_info = embed_info

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.base_dataset[idx]
        task_index = _as_int(item["task_index"])
        state = item["observation.state"]
        if not torch.is_tensor(state):
            state = torch.as_tensor(state)
        embedding = self.embeddings[task_index].to(dtype=state.dtype, device=state.device)
        if state.ndim == 1:
            conditioned_state = torch.cat([state, embedding], dim=0)
        elif state.ndim == 2:
            expanded = embedding.unsqueeze(0).expand(state.shape[0], -1)
            conditioned_state = torch.cat([state, expanded], dim=-1)
        else:
            raise ValueError(f"Unsupported observation.state shape: {tuple(state.shape)}")
        item["observation.state"] = conditioned_state
        item["language_condition_index"] = torch.tensor(task_index, dtype=torch.long)
        return item

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_dataset, name)


_ORIGINAL_MAKE_DATASET = lerobot_train.make_dataset


def make_language_conditioned_dataset(cfg):
    dataset = _ORIGINAL_MAKE_DATASET(cfg)
    wrapped = LanguageConditionedDataset(dataset)
    print(
        "Language-conditioned dataset:",
        f"backend={wrapped.embed_info['backend']}",
        f"model={wrapped.embed_info['model_name']}",
        f"embedding_dim={wrapped.embedding_dim}",
        f"state_shape={wrapped.meta.features['observation.state']['shape']}",
        flush=True,
    )
    return wrapped


def main() -> int:
    lerobot_train.make_dataset = make_language_conditioned_dataset
    lerobot_train.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
