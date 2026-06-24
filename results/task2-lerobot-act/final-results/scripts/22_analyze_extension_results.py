#!/usr/bin/env python
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results"


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": repr(exc), "path": str(path)}


def row_from_summary(path: Path) -> dict:
    data = load_json(path)
    success = data.get("success_rates") or {}
    return {
        "run_id": data.get("run_id", path.stem.replace("_summary", "")),
        "model_dir": str(path.parent),
        "condition_type": data.get("condition_type", "unknown"),
        "avg_successful_sequence_len": data.get("avg_successful_sequence_len", ""),
        "success_len1": success.get("1", ""),
        "success_len2": success.get("2", ""),
        "success_len3": success.get("3", ""),
        "success_len4": success.get("4", ""),
        "success_len5": success.get("5", ""),
        "num_sequences": data.get("num_sequences", ""),
        "elapsed_sec": data.get("elapsed_sec", ""),
        "summary_file": str(path),
    }


summary_paths: list[Path] = []
for root in [
    OUT_DIR / "calvin_eval_extensions",
    OUT_DIR / "calvin_eval_taskcond",
    OUT_DIR / "calvin_eval_taskcond" / "D",
]:
    if root.exists():
        summary_paths.extend(root.rglob("*_summary.json"))

rows = [row_from_summary(path) for path in sorted(set(summary_paths))]
csv_path = OUT_DIR / "task2_extension_eval_summary.csv"
fieldnames = [
    "run_id",
    "model_dir",
    "condition_type",
    "avg_successful_sequence_len",
    "success_len1",
    "success_len2",
    "success_len3",
    "success_len4",
    "success_len5",
    "num_sequences",
    "elapsed_sec",
    "summary_file",
]
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

json_path = OUT_DIR / "task2_extension_eval_summary.json"
json_path.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")

per_subtask_rows = []
for detail_path in sorted((OUT_DIR / "calvin_eval_extensions").rglob("*_details.json")) if (OUT_DIR / "calvin_eval_extensions").exists() else []:
    details = load_json(detail_path)
    if not isinstance(details, list):
        continue
    for seq in details:
        for sub in seq.get("subtasks", []):
            mapping = sub.get("mapping") or {}
            per_subtask_rows.append(
                {
                    "detail_file": str(detail_path),
                    "sequence_index": seq.get("sequence_index"),
                    "subtask": sub.get("subtask"),
                    "success": sub.get("success"),
                    "steps": sub.get("steps"),
                    "condition_type": mapping.get("condition_type", ""),
                    "task_index": mapping.get("task_index", ""),
                    "match_score": mapping.get("match_score", ""),
                    "language": sub.get("language", ""),
                    "matched_training_task": mapping.get("matched_training_task", ""),
                }
            )
per_subtask_csv = OUT_DIR / "task2_extension_subtask_breakdown.csv"
with per_subtask_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "detail_file",
            "sequence_index",
            "subtask",
            "success",
            "steps",
            "condition_type",
            "task_index",
            "match_score",
            "language",
            "matched_training_task",
        ],
    )
    writer.writeheader()
    writer.writerows(per_subtask_rows)

print(
    json.dumps(
        {
            "summary_csv": str(csv_path),
            "summary_json": str(json_path),
            "subtask_csv": str(per_subtask_csv),
            "num_eval_rows": len(rows),
            "num_subtask_rows": len(per_subtask_rows),
        },
        indent=2,
    )
)
