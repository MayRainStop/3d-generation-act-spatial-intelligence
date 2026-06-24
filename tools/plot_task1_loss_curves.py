#!/usr/bin/env python3
"""Plot Task 1 Object A 3DGS training loss from TensorBoard logs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[1]
EVENT_FILE = (
    ROOT
    / "tasks"
    / "task1_3dgs_aigc"
    / "object_a_reconstruction"
    / "outputs"
    / "object_a_3dgs"
    / "events.out.tfevents.1782272958.9bd3310986fc.2093.0"
)
OUT_FIG = ROOT / "docs" / "report_assets" / "task1_object_a_loss_curve.png"
OUT_CSV = (
    ROOT
    / "results"
    / "task1-3dgs-aigc"
    / "main-results"
    / "results"
    / "task1_object_a_3dgs_loss_curve.csv"
)

TAGS = {
    "train_loss_patches/total_loss": "Total loss",
    "train_loss_patches/l1_loss": "L1 loss",
}


def load_scalars() -> pd.DataFrame:
    if not EVENT_FILE.exists():
        raise FileNotFoundError(f"TensorBoard event file not found: {EVENT_FILE}")

    accumulator = EventAccumulator(str(EVENT_FILE))
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", []))
    missing = [tag for tag in TAGS if tag not in available]
    if missing:
        raise ValueError(f"Missing scalar tags in event file: {missing}")

    rows = []
    for tag, label in TAGS.items():
        for event in accumulator.Scalars(tag):
            rows.append(
                {
                    "tag": tag,
                    "label": label,
                    "step": int(event.step),
                    "value": float(event.value),
                }
            )
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame) -> None:
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    df = df.sort_values(["tag", "step"]).copy()
    df["smooth"] = (
        df.groupby("tag", group_keys=False)["value"]
        .rolling(window=101, center=True, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df.to_csv(OUT_CSV, index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colors = {"Total loss": "#1f77b4", "L1 loss": "#d62728"}

    for label, group in df.groupby("label", sort=False):
        ax.plot(
            group["step"],
            group["value"],
            color=colors[label],
            alpha=0.16,
            linewidth=0.75,
        )
        ax.plot(
            group["step"],
            group["smooth"],
            label=label,
            color=colors[label],
            linewidth=2.0,
        )

    ax.set_title("Object A 3DGS Training Loss", fontsize=13, pad=10)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_xlim(1, int(df["step"].max()))
    ax.set_ylim(0, 0.22)
    ax.legend(frameon=True, loc="upper right")
    ax.grid(True, linewidth=0.55, alpha=0.55)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_scalars()
    plot(df)
    summary = (
        df.groupby("label")
        .agg(first_step=("step", "first"), first_value=("value", "first"), last_step=("step", "last"), last_value=("value", "last"))
        .reset_index()
    )
    print(summary.to_string(index=False))
    print(f"Wrote {OUT_FIG}")
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
