#!/usr/bin/env python
from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATASET_ROOT = Path(os.environ.get("HW3_ABC_LEROBOT_ROOT", str(Path.home() / ".cache/huggingface/lerobot/fywang/calvin-task-ABC-D-lerobot")))
D_RAW_ROOT = ROOT / "data/calvin_raw/datasets/task_D_D"


def _first_task(tasks: Any) -> str:
    if isinstance(tasks, np.ndarray):
        return str(tasks[0])
    if isinstance(tasks, (list, tuple)):
        return str(tasks[0])
    return str(tasks)


def _read_episodes(root: Path) -> pd.DataFrame:
    paths = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episode parquet files under {root / 'meta' / 'episodes'}")
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    return df.sort_values("episode_index").reset_index(drop=True)


def _split_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    n = len(df)
    a = n // 3
    b = (2 * n) // 3
    return {"A": df.iloc[:a].copy(), "B": df.iloc[a:b].copy(), "C": df.iloc[b:].copy(), "ABC": df.copy()}


def _flatten3(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        arr = np.pad(arr, (0, 3 - arr.size), constant_values=np.nan)
    return arr[:3]


def _weighted_rgb(df: pd.DataFrame, key: str) -> list[float]:
    col = f"stats/{key}/mean"
    if col not in df:
        return [math.nan, math.nan, math.nan]
    weights = df["length"].to_numpy(dtype=np.float64)
    vals = np.vstack([_flatten3(v) for v in df[col]])
    return np.average(vals, axis=0, weights=weights).astype(float).tolist()


def _episode_summary(label: str, df: pd.DataFrame) -> dict[str, Any]:
    tasks = {_first_task(v) for v in df["tasks"]}
    row = {
        "split": label,
        "episodes": int(len(df)),
        "frames": int(df["length"].sum()),
        "unique_tasks": int(len(tasks)),
        "episode_start": int(df["episode_index"].min()),
        "episode_end": int(df["episode_index"].max()),
        "top_mean_r": None,
        "top_mean_g": None,
        "top_mean_b": None,
        "wrist_mean_r": None,
        "wrist_mean_g": None,
        "wrist_mean_b": None,
    }
    top = _weighted_rgb(df, "observation.images.top")
    wrist = _weighted_rgb(df, "observation.images.wrist")
    for name, vals in [("top", top), ("wrist", wrist)]:
        for channel, value in zip(["r", "g", "b"], vals, strict=True):
            row[f"{name}_mean_{channel}"] = value
    return row


def _sample_d_rgb(max_files: int = 768) -> dict[str, Any]:
    files = sorted((D_RAW_ROOT / "training").glob("episode_*.npz"))
    if not files:
        return {"available": False, "sampled_files": 0}
    if len(files) > max_files:
        idx = np.linspace(0, len(files) - 1, max_files, dtype=int)
        files = [files[int(i)] for i in idx]
    sums = {"rgb_static": np.zeros(3, dtype=np.float64), "rgb_gripper": np.zeros(3, dtype=np.float64)}
    counts = defaultdict(int)
    errors = 0
    for path in files:
        try:
            data = np.load(path, allow_pickle=False)
            for key in list(sums):
                if key not in data:
                    continue
                img = np.asarray(data[key], dtype=np.float64)
                if img.ndim < 3:
                    continue
                max_value = float(np.max(img)) if img.size else 0.0
                if max_value > 1.5:
                    img = img / 255.0
                sums[key] += img.reshape(-1, img.shape[-1])[:, :3].mean(axis=0)
                counts[key] += 1
        except Exception:
            errors += 1
    out = {"available": True, "sampled_files": len(files), "errors": errors}
    for key in list(sums):
        if counts[key]:
            out[f"{key}_mean"] = (sums[key] / counts[key]).astype(float).tolist()
            out[f"{key}_count"] = counts[key]
    return out


def _compute_action_smoothness(split_frames: dict[str, pd.DataFrame]) -> dict[str, dict[str, Any]]:
    episode_to_split = {}
    for label, df in split_frames.items():
        if label == "ABC":
            continue
        for ep in df["episode_index"].tolist():
            episode_to_split[int(ep)] = label
    values: dict[str, list[float]] = {"A": [], "B": [], "C": [], "ABC": []}
    data_paths = sorted(DATASET_ROOT.glob("data/**/*.parquet"))
    for path in data_paths:
        df = pd.read_parquet(path, columns=["episode_index", "action"])
        last_ep = None
        last_action = None
        for ep_raw, action_raw in zip(df["episode_index"].tolist(), df["action"].tolist(), strict=True):
            ep = int(ep_raw)
            action = np.asarray(action_raw, dtype=np.float64).reshape(-1)
            if last_ep == ep and last_action is not None:
                diff = float(np.linalg.norm(action - last_action))
                label = episode_to_split.get(ep)
                if label:
                    values[label].append(diff)
                values["ABC"].append(diff)
            last_ep = ep
            last_action = action
    rows = {}
    for label, vals in values.items():
        arr = np.asarray(vals, dtype=np.float64)
        rows[label] = {
            "num_action_diffs": int(arr.size),
            "mean_l2_delta": float(arr.mean()) if arr.size else math.nan,
            "median_l2_delta": float(np.median(arr)) if arr.size else math.nan,
            "p95_l2_delta": float(np.quantile(arr, 0.95)) if arr.size else math.nan,
        }
    return rows


def _load_eval_rows() -> list[dict[str, Any]]:
    summary_csv = RESULTS / "task2_extension_eval_summary.csv"
    if not summary_csv.exists():
        return []
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    episodes = _read_episodes(DATASET_ROOT)
    split_frames = _split_frames(episodes)
    split_rows = [_episode_summary(label, df) for label, df in split_frames.items()]
    d_rgb = _sample_d_rgb()
    smooth = _compute_action_smoothness(split_frames)

    d_static = np.asarray(d_rgb.get("rgb_static_mean", [math.nan, math.nan, math.nan]), dtype=np.float64)
    d_gripper = np.asarray(d_rgb.get("rgb_gripper_mean", [math.nan, math.nan, math.nan]), dtype=np.float64)
    shift_rows = []
    for row in split_rows:
        top = np.asarray([row["top_mean_r"], row["top_mean_g"], row["top_mean_b"]], dtype=np.float64)
        wrist = np.asarray([row["wrist_mean_r"], row["wrist_mean_g"], row["wrist_mean_b"]], dtype=np.float64)
        action_stats = smooth.get(row["split"], {})
        shift_rows.append({
            **row,
            "top_to_D_static_rgb_l2": float(np.linalg.norm(top - d_static)) if np.isfinite(d_static).all() else math.nan,
            "wrist_to_D_gripper_rgb_l2": float(np.linalg.norm(wrist - d_gripper)) if np.isfinite(d_gripper).all() else math.nan,
            **action_stats,
        })

    csv_path = RESULTS / "task2_source_split_visual_action_analysis.csv"
    fieldnames = list(shift_rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(shift_rows)

    payload = {
        "run_id": "task2_source_split_visual_action_analysis",
        "updated_at": datetime.now().astimezone().isoformat(),
        "abc_lerobot_root": str(DATASET_ROOT),
        "d_raw_root": str(D_RAW_ROOT),
        "split_mode": "episode_order_thirds_proxy",
        "d_rgb_sample": d_rgb,
        "rows": shift_rows,
        "eval_rows": _load_eval_rows(),
        "notes": [
            "A/B/C labels are proxy episode-order thirds because the public converted LeRobot ABC dataset exposes ABC combined episodes without raw environment labels.",
            "Visual shift is measured by mean RGB distance from each split to the sampled raw CALVIN D domain.",
            "Action smoothness is measured as per-step L2 delta of dataset actions; chunk-size performance is reported separately in task2_extension_eval_summary.csv.",
        ],
    }
    json_path = RESULTS / "task2_source_split_visual_action_analysis.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_path = RESULTS / "task2_final_completion_summary.md"
    lines = [
        "# HW3 Task 2 Final Completion Summary",
        "",
        f"Updated: {payload['updated_at']}",
        "",
        "## Source Split Caveat",
        "",
        "The available public LeRobot dataset is `fywang/calvin-task-ABC-D-lerobot`, which exposes ABC as one converted dataset without raw A/B/C environment tags. For the required A/B/C comparison, this workspace uses deterministic episode-order thirds as proxy source domains and reports visual statistics to make the limitation explicit.",
        "",
        "## Visual Shift and Action Smoothness",
        "",
        "| Split | Episodes | Frames | RGB L2 top->D | RGB L2 wrist->D | Mean action delta | P95 action delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in shift_rows:
        lines.append(
            f"| {row['split']} | {row['episodes']} | {row['frames']} | "
            f"{row['top_to_D_static_rgb_l2']:.4f} | {row['wrist_to_D_gripper_rgb_l2']:.4f} | "
            f"{row['mean_l2_delta']:.4f} | {row['p95_l2_delta']:.4f} |"
        )
    eval_rows = _load_eval_rows()
    if eval_rows:
        lines += ["", "## CALVIN D Online Evaluation Rows", "", "| Run | Condition | Avg Len | SR@1 | SR@5 |", "|---|---|---:|---:|---:|"]
        interesting = [r for r in eval_rows if r.get("run_id", "").endswith("_D_eval")]
        for r in sorted(interesting, key=lambda x: x.get("run_id", "")):
            lines.append(
                f"| {r.get('run_id','')} | {r.get('condition_type','')} | "
                f"{r.get('avg_successful_sequence_len','')} | {r.get('success_len1','')} | {r.get('success_len5','')} |"
            )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "md": str(md_path), "rows": len(shift_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
