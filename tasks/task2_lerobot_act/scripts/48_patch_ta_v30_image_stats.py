#!/usr/bin/env python3
"""Patch TA-converted LeRobot v3 metadata for image stats containers.

The converted stats.json contains state/actions only. LeRobot's training
factory fills ImageNet visual stats when use_imagenet_stats=True, but it
expects a stats dict to already exist for each camera key.
"""
import argparse
import json
from pathlib import Path


def patch_one(root: Path) -> dict[str, object]:
    info_path = root / "meta" / "info.json"
    stats_path = root / "meta" / "stats.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    image_keys = [k for k, v in info.get("features", {}).items() if isinstance(v, dict) and v.get("dtype") == "image"]
    added = []
    imagenet_stats = {
        "mean": [[[0.485]], [[0.456]], [[0.406]]],
        "std": [[[0.229]], [[0.224]], [[0.225]]],
    }
    for key in image_keys:
        current = stats.get(key)
        if not isinstance(current, dict) or not current:
            stats[key] = dict(imagenet_stats)
            added.append(key)
        else:
            changed = False
            for stat_key, stat_value in imagenet_stats.items():
                if stat_key not in current:
                    current[stat_key] = stat_value
                    changed = True
            if changed:
                added.append(key)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"root": str(root), "image_keys": image_keys, "added_or_reset": added}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/xiaoma_calvin_lerobot_v30/xiaoma26")
    parser.add_argument("--summary", default="results/task2_ta_v30_image_stats_patch.json")
    args = parser.parse_args()
    base = Path(args.root)
    results = []
    for label in "ABCD":
        root = base / f"calvin_lerobot_split{label}"
        if not root.exists():
            raise FileNotFoundError(root)
        results.append(patch_one(root))
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({"root": str(base), "splits": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "splits": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
