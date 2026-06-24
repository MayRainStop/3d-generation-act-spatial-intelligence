#!/usr/bin/env python3
"""Map LeRobot CALVIN ABC episodes to official CALVIN A/B/C scenes."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def first_task(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return str(value[0])
    if isinstance(value, (list, tuple)):
        return str(value[0])
    return str(value)


def read_episode_metadata(dataset_root: Path) -> pd.DataFrame:
    paths = sorted((dataset_root / "meta" / "episodes").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episode parquet files under {dataset_root / 'meta' / 'episodes'}")
    df = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    return df.sort_values("episode_index").reset_index(drop=True)


def load_scene_intervals(scene_info_json: Path) -> list[dict[str, Any]]:
    payload = json.loads(scene_info_json.read_text(encoding="utf-8"))
    training_items = [(name, info) for name, info in payload.items() if "/training/" in name or name.endswith("training/scene_info.npy")]
    if not training_items:
        training_items = list(payload.items())
    if not training_items:
        raise ValueError(f"No scene_info entries in {scene_info_json}")
    _name, scene_info = training_items[0]
    rows = []
    for scene_name, bounds in scene_info.items():
        match = re.search(r"calvin_scene_([A-D])", scene_name)
        if not match:
            continue
        start, end = int(bounds[0]), int(bounds[1])
        rows.append({"official_env": match.group(1), "scene_name": scene_name, "start_index": start, "end_index": end})
    if not rows:
        raise ValueError(f"No calvin_scene_A/B/C/D intervals found in {scene_info_json}")
    return sorted(rows, key=lambda row: row["start_index"])


def assign_env(start: int, end_exclusive: int, intervals: list[dict[str, Any]]) -> tuple[str, bool, dict[str, int]]:
    end = end_exclusive - 1
    overlaps: dict[str, int] = {}
    full_within = False
    for interval in intervals:
        lo = max(start, interval["start_index"])
        hi = min(end, interval["end_index"])
        if hi >= lo:
            overlaps[interval["official_env"]] = overlaps.get(interval["official_env"], 0) + hi - lo + 1
            if start >= interval["start_index"] and end <= interval["end_index"]:
                full_within = True
    if not overlaps:
        return "unknown", False, {}
    env = max(overlaps.items(), key=lambda item: item[1])[0]
    return env, full_within and len(overlaps) == 1, overlaps


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--scene-info-json", required=True)
    parser.add_argument("--out-csv", default="results/task2_official_scene_split_map.csv")
    parser.add_argument("--out-json", default="results/task2_official_scene_split_map.json")
    parser.add_argument("--a-episodes-out", default="results/task2_official_A_episode_indices.txt")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser()
    scene_info_json = Path(args.scene_info_json)
    intervals = load_scene_intervals(scene_info_json)
    episodes = read_episode_metadata(dataset_root)
    rows = []
    for row in episodes.to_dict("records"):
        start = int(row["dataset_from_index"])
        end_exclusive = int(row["dataset_to_index"])
        official_env, fully_within, overlaps = assign_env(start, end_exclusive, intervals)
        rows.append(
            {
                "episode_index": int(row["episode_index"]),
                "dataset_from_index": start,
                "dataset_to_index": end_exclusive,
                "length": int(row["length"]),
                "task": first_task(row["tasks"]),
                "official_env": official_env,
                "fully_within_single_scene": bool(fully_within),
                "overlap_frames": json.dumps(overlaps, sort_keys=True),
            }
        )

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    counts = {}
    for env in sorted({row["official_env"] for row in rows}):
        env_rows = [row for row in rows if row["official_env"] == env]
        counts[env] = {
            "episodes": len(env_rows),
            "frames": int(sum(row["length"] for row in env_rows)),
            "unique_tasks": len({row["task"] for row in env_rows}),
            "episode_start": min(row["episode_index"] for row in env_rows) if env_rows else None,
            "episode_end": max(row["episode_index"] for row in env_rows) if env_rows else None,
            "all_fully_within_single_scene": all(row["fully_within_single_scene"] for row in env_rows),
        }
    payload = {
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset_root": str(dataset_root),
        "scene_info_json": str(scene_info_json),
        "source": "official CALVIN task_ABC_D.zip training/scene_info.npy via HTTP Range extraction",
        "scene_intervals": intervals,
        "counts": counts,
        "num_rows": len(rows),
        "unknown_episodes": [row["episode_index"] for row in rows if row["official_env"] == "unknown"],
        "cross_scene_episodes": [row["episode_index"] for row in rows if not row["fully_within_single_scene"]],
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    a_episodes = [str(row["episode_index"]) for row in rows if row["official_env"] == "A"]
    Path(args.a_episodes_out).write_text("\n".join(a_episodes) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
