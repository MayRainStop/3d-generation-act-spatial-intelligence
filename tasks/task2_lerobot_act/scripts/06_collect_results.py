from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


LOSS_PATTERNS = [
    re.compile(r"(?:action[_ ]?l1|l1[_ ]?loss|loss)[=: ]+([0-9]+(?:\.[0-9]+)?(?:e[-+]?[0-9]+)?)", re.IGNORECASE),
]


def parse_losses(log_path: Path) -> list[float]:
    values: list[float] = []
    text = log_path.read_text(errors="ignore")
    for line in text.splitlines():
        for pattern in LOSS_PATTERNS:
            match = pattern.search(line)
            if match:
                values.append(float(match.group(1)))
                break
    return values


def summarize_run(run_id: str, log_path: Path, status_path: Path) -> dict[str, object]:
    losses = parse_losses(log_path) if log_path.exists() else []
    status = {}
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
    return {
        "run_id": run_id,
        "log_path": str(log_path),
        "status": status.get("status", "missing"),
        "exit_code": status.get("exit_code", ""),
        "num_loss_values": len(losses),
        "first_loss": losses[0] if losses else "",
        "last_loss": losses[-1] if losses else "",
        "best_loss": min(losses) if losses else "",
    }


def collect() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for log_path in sorted((ROOT / "logs").glob("*.train.log")):
        run_id = log_path.name.removesuffix(".train.log")
        status_path = ROOT / "results" / f"{run_id}.status.json"
        rows.append(summarize_run(run_id, log_path, status_path))
    return pd.DataFrame(rows)


def write_figures(df: pd.DataFrame) -> None:
    fig_dir = ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if df.empty or "best_loss" not in df:
        return
    plot_df = df[df["best_loss"] != ""].copy()
    if plot_df.empty:
        return
    plot_df["best_loss"] = plot_df["best_loss"].astype(float)
    plot_df = plot_df.sort_values("best_loss")
    plt.figure(figsize=(10, 5))
    plt.bar(plot_df["run_id"], plot_df["best_loss"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Best parsed Action/L1 loss")
    plt.tight_layout()
    plt.savefig(fig_dir / "best_loss_by_run.png", dpi=200)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/metrics_summary.csv")
    args = parser.parse_args()
    df = collect()
    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)
    write_figures(df)
    print(df.to_string(index=False) if not df.empty else "No train logs found.")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
