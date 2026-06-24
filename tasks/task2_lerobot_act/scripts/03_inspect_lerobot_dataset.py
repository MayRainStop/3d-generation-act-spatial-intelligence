from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "results" / "dataset_inspection"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl_head(path: Path, limit: int = 5) -> list[object]:
    rows: list[object] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= limit:
                break
            rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--local-dir", default="")
    args = parser.parse_args()

    safe_name = args.repo_id.replace("/", "__")
    local_dir = Path(args.local_dir) if args.local_dir else ROOT / "data" / "hf_meta" / safe_name
    local_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.revision,
            local_dir=str(local_dir),
            allow_patterns=["README.md", ".gitattributes", "meta/*"],
        )
    )

    report: dict[str, object] = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "snapshot_path": str(snapshot_path),
        "files": [],
        "info": None,
        "episodes_head": [],
        "tasks_head": [],
        "feature_names": [],
        "source_env_candidates": [],
    }

    for file in sorted(snapshot_path.rglob("*")):
        if file.is_file():
            report["files"].append(str(file.relative_to(snapshot_path)))

    info_path = snapshot_path / "meta" / "info.json"
    if info_path.exists():
        info = read_json(info_path)
        report["info"] = info
        if isinstance(info, dict):
            features = info.get("features", {})
            if isinstance(features, dict):
                report["feature_names"] = sorted(features.keys())

    episodes_path = snapshot_path / "meta" / "episodes.jsonl"
    if episodes_path.exists():
        episodes_head = read_jsonl_head(episodes_path, 10)
        report["episodes_head"] = episodes_head
        for row in episodes_head:
            if isinstance(row, dict):
                for key in row.keys():
                    low = key.lower()
                    if "env" in low or "scene" in low or "split" in low:
                        report["source_env_candidates"].append(key)

    tasks_path = snapshot_path / "meta" / "tasks.jsonl"
    if tasks_path.exists():
        report["tasks_head"] = read_jsonl_head(tasks_path, 10)

    out_path = OUT_ROOT / f"{safe_name}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
