from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_run(run_id: str) -> dict[str, str]:
    path = ROOT / "configs" / "runs.csv"
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["run_id"] == run_id:
                return row
    raise SystemExit(f"Unknown run_id: {run_id}")


def help_text() -> str:
    completed = subprocess.run(["lerobot-train", "--help"], text=True, capture_output=True, check=False)
    return completed.stdout + completed.stderr


def has_flag(help_output: str, flag: str) -> bool:
    return flag in help_output


def build_command(row: dict[str, str], help_output: str) -> list[str]:
    cmd = ["lerobot-train"]

    if has_flag(help_output, "--policy.type"):
        cmd += ["--policy.type=act"]
    else:
        cmd += ["--policy=act"]

    cmd += [f"--dataset.repo_id={row['dataset_repo']}"]

    if has_flag(help_output, "--output_dir"):
        cmd += [f"--output_dir={row['output_dir']}"]
    elif has_flag(help_output, "--training.output_dir"):
        cmd += [f"--training.output_dir={row['output_dir']}"]

    if has_flag(help_output, "--seed"):
        cmd += [f"--seed={row['seed']}"]
    elif has_flag(help_output, "--training.seed"):
        cmd += [f"--training.seed={row['seed']}"]

    if has_flag(help_output, "--steps"):
        cmd += [f"--steps={row['steps']}"]
    elif has_flag(help_output, "--training.steps"):
        cmd += [f"--training.steps={row['steps']}"]

    if has_flag(help_output, "--batch_size"):
        cmd += [f"--batch_size={row['batch_size']}"]
    elif has_flag(help_output, "--training.batch_size"):
        cmd += [f"--training.batch_size={row['batch_size']}"]

    if has_flag(help_output, "--lr"):
        cmd += [f"--lr={row['learning_rate']}"]
    elif has_flag(help_output, "--training.lr"):
        cmd += [f"--training.lr={row['learning_rate']}"]
    elif has_flag(help_output, "--optimizer.lr"):
        cmd += [f"--optimizer.lr={row['learning_rate']}"]

    if has_flag(help_output, "--policy.chunk_size"):
        cmd += [f"--policy.chunk_size={row['chunk_size']}"]
    if has_flag(help_output, "--policy.n_action_steps"):
        cmd += [f"--policy.n_action_steps={row['n_action_steps']}"]
    if has_flag(help_output, "--policy.push_to_hub"):
        cmd += ["--policy.push_to_hub=false"]
    if has_flag(help_output, "--wandb.mode"):
        cmd += ["--wandb.mode=offline"]
    elif has_flag(help_output, "--wandb.enable"):
        cmd += ["--wandb.enable=false"]

    return cmd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--print-shell", action="store_true")
    args = parser.parse_args()

    row = load_run(args.run_id)
    command = build_command(row, help_text())
    if args.print_shell:
        print(" ".join(shlex.quote(part) for part in command))
    else:
        for part in command:
            print(part)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
