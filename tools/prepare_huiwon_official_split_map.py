#!/usr/bin/env python3
"""Download huiwon CALVIN metadata/parquets and map episodes by original_frame_idx."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download


REPO_ID = "huiwon/calvin_task_ABC_D"
SUBDIR = "calvin_task_ABC_D_lerobot_0_4"


def load_intervals(scene_info_json: Path) -> list[dict[str, Any]]:
    payload = json.loads(scene_info_json.read_text(encoding="utf-8"))
    scene_info = next(iter(payload.values()))
    rows = []
    for name, bounds in scene_info.items():
        env = name.rsplit("_", 1)[-1]
        rows.append({"env": env, "scene_name": name, "start": int(bounds[0]), "end": int(bounds[1])})
    return sorted(rows, key=lambda r: r["start"])


def env_for_frame(idx: int, intervals: list[dict[str, Any]]) -> str:
    for row in intervals:
        if row["start"] <= idx <= row["end"]:
            return row["env"]
    return "unknown"


def download_file(filename: str, local_dir: Path) -> str:
    path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=filename,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        etag_timeout=30,
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-json", default="results/api_huiwon_calvin_task_ABC_D.json")
    parser.add_argument("--scene-info-json", default="results/official_task_ABC_D_scene_info.json")
    parser.add_argument("--local-dir", default="data/huiwon_calvin_task_ABC_D")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out-csv", default="results/task2_huiwon_official_scene_split_map.csv")
    parser.add_argument("--out-json", default="results/task2_huiwon_official_scene_split_map.json")
    args = parser.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    api = json.loads(Path(args.api_json).read_text(encoding="utf-8"))
    files = [s["rfilename"] for s in api["siblings"]]
    wanted = [f for f in files if f.startswith(f"{SUBDIR}/meta/")]
    wanted += [f for f in files if f.startswith(f"{SUBDIR}/data/") and f.endswith(".parquet")]
    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"downloading {len(wanted)} meta/parquet files from {REPO_ID}", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_file, f, local_dir): f for f in wanted}
        for fut in as_completed(futures):
            fut.result()
            done += 1
            if done % 250 == 0 or done == len(wanted):
                print(f"downloaded {done}/{len(wanted)}", flush=True)

    intervals = load_intervals(Path(args.scene_info_json))
    parquet_paths = sorted((local_dir / SUBDIR / "data").rglob("*.parquet"))
    rows = []
    for path in parquet_paths:
        df = pd.read_parquet(path, columns=["episode_index", "index", "frame_index", "task_index", "original_frame_idx"])
        env_counts = Counter(env_for_frame(int(v), intervals) for v in df["original_frame_idx"].tolist())
        env, frames = env_counts.most_common(1)[0]
        rows.append(
            {
                "episode_index": int(df["episode_index"].iloc[0]),
                "length": int(len(df)),
                "dataset_from_index": int(df["index"].min()),
                "dataset_to_index": int(df["index"].max()) + 1,
                "original_frame_min": int(df["original_frame_idx"].min()),
                "original_frame_max": int(df["original_frame_idx"].max()),
                "official_env": env,
                "fully_within_single_scene": len(env_counts) == 1,
                "env_counts": json.dumps(dict(sorted(env_counts.items())), sort_keys=True),
                "task_index": int(df["task_index"].iloc[0]),
            }
        )
    rows.sort(key=lambda r: r["episode_index"])
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    counts = {}
    for env in sorted({r["official_env"] for r in rows}):
        ers = [r for r in rows if r["official_env"] == env]
        counts[env] = {
            "episodes": len(ers),
            "frames": sum(r["length"] for r in ers),
            "episode_start": min(r["episode_index"] for r in ers),
            "episode_end": max(r["episode_index"] for r in ers),
            "unique_tasks": len({r["task_index"] for r in ers}),
            "all_fully_within_single_scene": all(r["fully_within_single_scene"] for r in ers),
        }
    payload = {
        "repo_id": REPO_ID,
        "subdir": SUBDIR,
        "local_dataset_root": str(local_dir / SUBDIR),
        "source": "official CALVIN task_ABC_D scene_info + huiwon original_frame_idx",
        "scene_intervals": intervals,
        "counts": counts,
        "num_episodes": len(rows),
        "num_frames": sum(r["length"] for r in rows),
        "csv": str(out_csv),
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
