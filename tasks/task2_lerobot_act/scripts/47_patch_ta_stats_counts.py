#!/usr/bin/env python3
"""Add LeRobot v3-compatible count fields to TA split episodes_stats.jsonl.

The TA-provided xiaoma26/calvin-lerobot split metadata is v2.1-style and
contains per-episode min/max/mean/std only. Newer LeRobot v2.1->v3.0
conversion aggregates stats with a required per-feature count field. The
count is the episode frame length, available in meta/episodes.jsonl.
"""
import argparse
import json
from pathlib import Path


def load_episode_lengths(split_dir: Path) -> dict[int, int]:
    lengths: dict[int, int] = {}
    episodes_path = split_dir / "meta" / "episodes.jsonl"
    with episodes_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            lengths[int(row["episode_index"])] = int(row["length"])
    return lengths


def patch_split(split_dir: Path) -> dict[str, int | str]:
    stats_path = split_dir / "meta" / "episodes_stats.jsonl"
    tmp_path = stats_path.with_suffix(stats_path.suffix + ".tmp")
    lengths = load_episode_lengths(split_dir)
    rows = 0
    features_seen = 0
    fields_set = 0
    missing_lengths: list[int] = []

    with stats_path.open("r", encoding="utf-8") as src, tmp_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            episode_index = int(row["episode_index"])
            length = lengths.get(episode_index)
            if length is None:
                missing_lengths.append(episode_index)
                length = 0
            for feature_stats in row.get("stats", {}).values():
                if isinstance(feature_stats, dict):
                    features_seen += 1
                    desired_count = [length]
                    if feature_stats.get("count") != desired_count:
                        feature_stats["count"] = desired_count
                        fields_set += 1
            dst.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            rows += 1

    if missing_lengths:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"{split_dir}: missing episode lengths for {missing_lengths[:10]}")

    tmp_path.replace(stats_path)
    return {
        "split_dir": str(split_dir),
        "episodes": rows,
        "features_seen": features_seen,
        "count_fields_set": fields_set,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/xiaoma_calvin_lerobot")
    parser.add_argument("--summary", default="results/task2_ta_stats_count_patch.json")
    args = parser.parse_args()

    root = Path(args.root)
    results = []
    for label in "ABCD":
        split_dir = root / f"split{label}"
        if not split_dir.exists():
            raise FileNotFoundError(split_dir)
        results.append(patch_split(split_dir))

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({"root": str(root), "splits": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "splits": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
