#!/usr/bin/env python3
"""Create a CALVIN-compatible task metadata table for huiwon CALVIN data."""

from __future__ import annotations

import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd


def normalize(text: str) -> str:
    return " ".join(str(text).lower().strip().replace("_", " ").replace(":", " ").split())


def load_reference(path: Path) -> list[dict]:
    table = pd.read_parquet(path)
    entries: list[dict] = []
    for full_task, row in table.sort_values("task_index").iterrows():
        text = str(full_task)
        if ":" in text:
            subtask, language = text.split(":", 1)
        else:
            subtask, language = text, text
        entries.append(
            {
                "full_task": text,
                "subtask": subtask.strip(),
                "language": language.strip(),
                "reference_task_index": int(row["task_index"]),
                "norm_language": normalize(language),
            }
        )
    return entries


def load_huiwon_tasks(path: Path) -> list[dict]:
    tasks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            tasks.append({"task_index": int(row["task_index"]), "task": str(row["task"])})
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--huiwon-tasks", default="data/huiwon_calvin_task_ABC_D/calvin_task_ABC_D_lerobot_0_4/meta/tasks.jsonl")
    parser.add_argument(
        "--reference-task-meta",
        default=str(Path.home() / ".cache/huggingface/lerobot/fywang/calvin-task-ABC-D-lerobot/meta/tasks.parquet"),
    )
    parser.add_argument("--output", default="results/huiwon_tasks.parquet")
    parser.add_argument("--summary", default="results/huiwon_tasks_summary.json")
    args = parser.parse_args()

    huiwon_tasks = load_huiwon_tasks(Path(args.huiwon_tasks))
    reference = load_reference(Path(args.reference_task_meta))

    records = []
    audit = []
    for task in huiwon_tasks:
        norm_task = normalize(task["task"])
        exact = [entry for entry in reference if entry["norm_language"] == norm_task]
        if exact:
            match = exact[0]
            score = 1.0
        else:
            match = max(reference, key=lambda entry: SequenceMatcher(None, norm_task, entry["norm_language"]).ratio())
            score = SequenceMatcher(None, norm_task, match["norm_language"]).ratio()
        full_task = f"{match['subtask']}: {task['task']}"
        records.append({"full_task": full_task, "task_index": int(task["task_index"])})
        audit.append(
            {
                "task_index": int(task["task_index"]),
                "huiwon_task": task["task"],
                "subtask": match["subtask"],
                "reference_full_task": match["full_task"],
                "reference_task_index": match["reference_task_index"],
                "match_score": float(score),
            }
        )

    table = pd.DataFrame(records).sort_values("task_index").set_index("full_task")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output)

    summary = {
        "huiwon_tasks": len(huiwon_tasks),
        "output": str(output),
        "min_match_score": min(item["match_score"] for item in audit),
        "num_exact_matches": sum(1 for item in audit if item["match_score"] == 1.0),
        "audit": audit,
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "audit"}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
