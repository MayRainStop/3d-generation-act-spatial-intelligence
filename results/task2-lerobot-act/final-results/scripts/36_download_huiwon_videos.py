#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


REPO_ID = "huiwon/calvin_task_ABC_D"
SUBDIR = "calvin_task_ABC_D_lerobot_0_4"


def download(filename: str, local_dir: Path) -> str:
    return hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=filename,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        etag_timeout=30,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-json", default="results/api_huiwon_calvin_task_ABC_D.json")
    parser.add_argument("--split-map", default="results/task2_huiwon_official_scene_split_map.csv")
    parser.add_argument("--env", default="ABC")
    parser.add_argument("--local-dir", default="data/huiwon_calvin_task_ABC_D")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    api = json.loads(Path(args.api_json).read_text(encoding="utf-8"))
    files = [s["rfilename"] for s in api["siblings"]]
    videos = [f for f in files if f.startswith(f"{SUBDIR}/videos/") and f.endswith(".mp4")]
    split = args.env.upper()
    if split != "ABC":
        df = pd.read_csv(args.split_map)
        selected = set(int(v) for v in df[df["official_env"] == split]["episode_index"].tolist())
        videos = [f for f in videos if int(Path(f).stem.split("_")[-1]) in selected]
    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"downloading {len(videos)} videos for split={split}", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download, f, local_dir): f for f in videos}
        for fut in as_completed(futures):
            fut.result()
            done += 1
            if done % 250 == 0 or done == len(videos):
                print(f"downloaded_videos {done}/{len(videos)}", flush=True)
    print("video download complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
