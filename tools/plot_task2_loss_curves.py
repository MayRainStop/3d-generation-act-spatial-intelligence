#!/usr/bin/env python3
"""Extract LeRobot training loss from logs and plot Task 2 curves."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "results" / "task2-lerobot-act" / "final-results"
LOG_DIRS = [
    ARTIFACT_DIR / "logs-full",
    ARTIFACT_DIR / "logs",
]
OUT_CSV = ARTIFACT_DIR / "results" / "task2_loss_curves.csv"
OUT_PNG = ROOT / "docs" / "report_assets" / "task2_loss_curves.png"

RUNS = [
    ("TA splitA A-only 100k", "act_ta_a_taskcond_100k_seed0.ta_official.train.log"),
    ("TA splitABC one-hot 100k", "act_ta_abc_taskcond_100k_seed0.ta_official.train.log"),
    ("TA splitABC SBERT 100k", "act_ta_abc_lang_sbert_100k_seed0.ta_official.train.log"),
    ("TA splitABC chunk100 100k", "act_ta_abc_taskcond_chunk100_100k_seed0.ta_official.train.log"),
    ("Public ABC high-score", "act_abc_taskcond_200k_seed0.high_score.train.log"),
    ("TA splitA A-only seed0 160k", "act_ta_a_taskcond_100k_seed0.ta_sweep_seed1.train.log"),
    ("TA splitABC SBERT seed0 160k", "act_ta_abc_lang_sbert_100k_seed0.ta_sweep_seed1.train.log"),
    ("TA splitA A-only seed1 100k", "act_ta_a_taskcond_100k_seed1.ta_sweep_seed1.train.log"),
    ("TA splitABC SBERT seed1 100k", "act_ta_abc_lang_sbert_100k_seed1.ta_sweep_seed1.train.log"),
]

LOSS_RE = re.compile(r"step:(?P<step>[0-9]+(?:K)?)\b.*?\bloss:(?P<loss>[0-9.]+)")


def parse_step(value: str) -> int:
    if value.endswith("K"):
        return int(value[:-1]) * 1000
    return int(value)


def parse_log(path: Path) -> list[tuple[int, float]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    points: list[tuple[int, float]] = []
    for match in LOSS_RE.finditer(text):
        points.append((parse_step(match.group("step")), float(match.group("loss"))))
    dedup: dict[int, float] = {}
    for step, loss in points:
        dedup[step] = loss
    return sorted(dedup.items())


def find_log(filename: str) -> Path | None:
    for log_dir in LOG_DIRS:
        path = log_dir / filename
        if path.exists():
            return path
    return None


def moving_average(values: list[float], window: int = 5) -> list[float]:
    if window <= 1 or len(values) < window:
        return values
    out: list[float] = []
    for idx in range(len(values)):
        left = max(0, idx - window + 1)
        chunk = values[left : idx + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    parsed: list[tuple[str, list[tuple[int, float]]]] = []

    for label, filename in RUNS:
        path = find_log(filename)
        if path is None:
            continue
        points = parse_log(path)
        if not points:
            continue
        parsed.append((label, points))
        for step, loss in points:
            all_rows.append({
                "run": label,
                "step": step,
                "loss": loss,
                "source_log": str(path.relative_to(ARTIFACT_DIR)),
            })

    if not parsed:
        searched = ", ".join(str(path) for path in LOG_DIRS)
        raise SystemExit(f"No loss points found under {searched}")

    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["run", "step", "loss", "source_log"])
        writer.writeheader()
        writer.writerows(all_rows)

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.size": 9.5,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 7.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    colors = [
        "#3568A7", "#2F8F5B", "#D77A2D", "#7B61A8", "#B94747",
        "#2A9D8F", "#8A5A44", "#6E7781", "#9A7D0A",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.8), dpi=190)
    ax_full, ax_zoom = axes

    for idx, (label, points) in enumerate(parsed):
        steps = [step for step, _ in points]
        losses = moving_average([loss for _, loss in points], window=5)
        color = colors[idx % len(colors)]
        ax_full.plot(steps, losses, linewidth=1.75, label=label, color=color, alpha=0.94)
        zoom_pairs = [(s, l) for s, l in zip(steps, losses) if s >= 20000]
        if zoom_pairs:
            ax_zoom.plot(
                [s for s, _ in zoom_pairs],
                [l for _, l in zoom_pairs],
                linewidth=1.9,
                label=label,
                color=color,
                alpha=0.96,
            )

    ax_full.set_title("Full Training Range")
    ax_full.set_xlabel("Training step")
    ax_full.set_ylabel("Training loss, 5-point moving average")
    ax_full.set_xlim(left=0)
    ax_full.set_yscale("log")
    ax_full.grid(True, linewidth=0.45, alpha=0.42)

    ax_zoom.set_title("After 20k Steps")
    ax_zoom.set_xlabel("Training step")
    ax_zoom.set_ylabel("Loss zoom")
    ax_zoom.set_xlim(left=20000)
    ax_zoom.set_ylim(bottom=0)
    ax_zoom.grid(True, linewidth=0.45, alpha=0.42)

    handles, labels = ax_zoom.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Task 2 ACT Training Loss Curves", y=0.995, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.24, wspace=0.24)
    fig.savefig(OUT_PNG, bbox_inches="tight")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
