#!/usr/bin/env python3
"""Generate Task 2 report figures from downloaded CALVIN ACT results."""

from __future__ import annotations

import re
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "results" / "task2-lerobot-act" / "final-results"
RESULT_DIR = ARTIFACT_DIR / "results"
ASSET_DIR = ROOT / "docs" / "report_assets"

OFFICIAL_SUMMARY = RESULT_DIR / "task2_ta_official_split_summary.csv"
SWEEP_SUMMARY = RESULT_DIR / "task2_ta_sweep_seed1_summary.csv"
HIGH_SCORE_SUMMARY = RESULT_DIR / "task2_high_score_summary.csv"
SHIFT_ANALYSIS = RESULT_DIR / "task2_source_split_visual_action_analysis.csv"
SUBTASK_BREAKDOWN = RESULT_DIR / "task2_extension_subtask_breakdown.csv"
TA_AUDIT = RESULT_DIR / "task2_ta_official_split_audit.json"

OUT_INDEX = RESULT_DIR / "task2_report_figures_index.csv"
OUT_SUBTASK = RESULT_DIR / "task2_subtask_breakdown_summary.csv"


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 9.2,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})
COLORS = {
    "blue": "#3568A7",
    "green": "#2F8F5B",
    "orange": "#D77A2D",
    "red": "#B94747",
    "purple": "#7B61A8",
    "gray": "#6E7781",
    "teal": "#2A9D8F",
}


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def clean_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def savefig(fig: plt.Figure, filename: str) -> Path:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    path = ASSET_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def label_bars(ax: plt.Axes, fmt: str = "{:.3f}", pad: float = 0.01) -> None:
    for patch in ax.patches:
        height = patch.get_height()
        if pd.isna(height):
            continue
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height + pad,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=0,
        )


def get_row(df: pd.DataFrame, family: str) -> pd.Series:
    rows = df[df["family"] == family]
    if rows.empty:
        raise KeyError(family)
    return rows.iloc[0]


def plot_main_results(official: pd.DataFrame, sweep: pd.DataFrame, high: pd.DataFrame) -> Path:
    rows = [
        ("splitA A-only", get_row(official, "ta_A_taskcond_100k")),
        ("splitB only", get_row(official, "ta_B_taskcond_100k")),
        ("splitC only", get_row(official, "ta_C_taskcond_100k")),
        ("splitABC one-hot", get_row(official, "ta_ABC_taskcond_100k")),
        ("splitABC SBERT", get_row(official, "ta_ABC_sbert_100k")),
        ("splitABC chunk100", get_row(official, "ta_ABC_chunk100_100k")),
        ("splitA A-only seed1", get_row(sweep, "ta_A_seed1_100k")),
        ("splitABC SBERT seed1", get_row(sweep, "ta_ABC_sbert_seed1_100k")),
    ]
    high_120 = high[high["run_id"].str.contains("200k_seed0_ckpt120k", na=False)]
    if not high_120.empty:
        rows.append(("ABC extended 120k", high_120.iloc[0]))

    data = pd.DataFrame(
        {
            "label": [label for label, _ in rows],
            "avg_len": [float(row["avg_successful_sequence_len"]) for _, row in rows],
            "sr1": [float(row["success_len1"]) for _, row in rows],
            "sr5": [float(row["success_len5"]) for _, row in rows],
        }
    )
    data = data.iloc[::-1].reset_index(drop=True)
    y = range(len(data))
    fig, ax = plt.subplots(figsize=(9.8, 6.3))
    bars = ax.barh(y, data["avg_len"], color=COLORS["blue"], alpha=0.90, label="AvgLen")
    ax.scatter(data["sr1"], y, color=COLORS["orange"], marker="o", s=34, label="SR@1", zorder=3)
    ax.scatter(data["sr5"], y, color=COLORS["green"], marker="s", s=30, label="SR@5", zorder=3)
    for patch, value in zip(bars, data["avg_len"]):
        ax.text(value + 0.018, patch.get_y() + patch.get_height() / 2, f"{value:.3f}", va="center", fontsize=8)
    ax.set_title("Task 2 CALVIN D Online Evaluation")
    ax.set_xlabel("Metric value")
    ax.set_yticks(list(y))
    ax.set_yticklabels(data["label"])
    ax.set_xlim(0, max(1.12, data["avg_len"].max() + 0.16))
    ax.legend(loc="lower right", ncol=3, fontsize=8)
    ax.grid(axis="x", linewidth=0.45, alpha=0.45)
    return savefig(fig, "task2_main_results_bar.png")


def plot_success_horizon(official: pd.DataFrame, sweep: pd.DataFrame, high: pd.DataFrame) -> Path:
    selected = [
        ("splitA A-only", get_row(official, "ta_A_taskcond_100k"), COLORS["blue"]),
        ("splitABC one-hot", get_row(official, "ta_ABC_taskcond_100k"), COLORS["gray"]),
        ("splitABC SBERT", get_row(official, "ta_ABC_sbert_100k"), COLORS["purple"]),
        ("splitABC SBERT seed1", get_row(sweep, "ta_ABC_sbert_seed1_100k"), COLORS["green"]),
    ]
    high_120 = high[high["run_id"].str.contains("200k_seed0_ckpt120k", na=False)]
    if not high_120.empty:
        selected.append(("ABC extended 120k", high_120.iloc[0], COLORS["orange"]))

    xs = [1, 2, 3, 4, 5]
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    for label, row, color in selected:
        ys = [float(row[f"success_len{i}"]) for i in xs]
        ax.plot(xs, ys, marker="o", linewidth=2.2, label=label, color=color)
    ax.set_title("Task 2 Success Horizon on CALVIN D")
    ax.set_xlabel("Required consecutive subtasks")
    ax.set_ylabel("Success rate")
    ax.set_xticks(xs)
    ax.set_ylim(0, 0.62)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linewidth=0.5, alpha=0.55)
    return savefig(fig, "task2_success_horizon.png")


def parse_step_from_family(family: str) -> int | None:
    match = re.search(r"_(\d+)k$", family)
    if match:
        return int(match.group(1)) * 1000
    return None


def plot_checkpoint_sweep(sweep: pd.DataFrame) -> Path:
    rows = []
    for prefix, label in [
        ("ta_A_seed0_", "splitA A-only seed0"),
        ("ta_ABC_sbert_seed0_", "splitABC SBERT seed0"),
    ]:
        part = sweep[sweep["family"].str.startswith(prefix, na=False)].copy()
        part["step"] = part["family"].map(parse_step_from_family)
        part = part.dropna(subset=["step", "avg_successful_sequence_len"])
        for _, row in part.iterrows():
            rows.append(
                {
                    "label": label,
                    "step": int(row["step"]),
                    "avg_len": float(row["avg_successful_sequence_len"]),
                    "sr5": float(row["success_len5"]),
                }
            )
    data = pd.DataFrame(rows).sort_values(["label", "step"])
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    for label, color in [("splitA A-only seed0", COLORS["blue"]), ("splitABC SBERT seed0", COLORS["green"])]:
        part = data[data["label"] == label]
        ax.plot(part["step"], part["avg_len"], marker="o", linewidth=2.2, color=color, label=label)
        for _, row in part.iterrows():
            ax.text(row["step"], row["avg_len"] + 0.018, f"{row['avg_len']:.3f}", fontsize=7, ha="center")
    ax.set_title("Checkpoint Sweep on CALVIN Explicit Split")
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel("AvgLen on CALVIN D")
    ax.set_ylim(0.62, 0.98)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, linewidth=0.5, alpha=0.55)
    return savefig(fig, "task2_checkpoint_sweep.png")


def plot_seed_robustness() -> Path:
    data = pd.DataFrame(
        [
            {"policy": "splitA A-only", "seed0": 0.855, "seed1": 0.846, "mean": 0.8505, "std": 0.0064},
            {"policy": "splitABC SBERT", "seed0": 0.787, "seed1": 0.923, "mean": 0.8550, "std": 0.0962},
        ]
    )
    x = range(len(data))
    width = 0.28
    fig, ax = plt.subplots(figsize=(7.8, 5.1))
    ax.bar([v - width / 2 for v in x], data["seed0"], width=width, color=COLORS["blue"], label="seed0")
    ax.bar([v + width / 2 for v in x], data["seed1"], width=width, color=COLORS["green"], label="seed1")
    ax.errorbar(x, data["mean"], yerr=data["std"], fmt="o", color=COLORS["red"], capsize=5, label="mean +/- std")
    ax.set_title("Task 2 Seed Robustness at 100k Steps")
    ax.set_ylabel("AvgLen on CALVIN D")
    ax.set_xticks(list(x))
    ax.set_xticklabels(data["policy"])
    ax.set_ylim(0.72, 0.96)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", linewidth=0.5, alpha=0.55)
    return savefig(fig, "task2_seed_robustness.png")


def plot_split_size() -> Path:
    audit = pd.read_json(TA_AUDIT)
    rows = pd.DataFrame(audit["rows"].tolist())
    x = range(len(rows))
    fig, ax1 = plt.subplots(figsize=(8.6, 5.2))
    ax2 = ax1.twinx()
    bars = ax1.bar([v - 0.18 for v in x], rows["total_episodes"], width=0.36, color=COLORS["blue"], label="episodes")
    ax2.bar([v + 0.18 for v in x], rows["total_frames"] / 1000, width=0.36, color=COLORS["orange"], label="frames (k)")
    ax1.set_title("CALVIN Explicit Split Size")
    ax1.set_ylabel("Episodes")
    ax2.set_ylabel("Frames (thousand)")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(rows["split"])
    ax1.set_ylim(0, rows["total_episodes"].max() * 1.18)
    ax2.set_ylim(0, (rows["total_frames"] / 1000).max() * 1.18)
    ax1.grid(axis="y", linewidth=0.5, alpha=0.4)
    for idx, patch in enumerate(bars):
        ax1.text(patch.get_x() + patch.get_width() / 2, patch.get_height() + 70, str(int(rows.iloc[idx]["total_episodes"])), ha="center", fontsize=8)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    return savefig(fig, "task2_split_size.png")


def plot_visual_action_shift(shift: pd.DataFrame, official: pd.DataFrame) -> Path:
    rows = shift[shift["split"].isin(["A", "B", "C", "ABC"])].copy()
    perf_map = {
        "A": float(get_row(official, "ta_A_taskcond_100k")["avg_successful_sequence_len"]),
        "B": float(get_row(official, "ta_B_taskcond_100k")["avg_successful_sequence_len"]),
        "C": float(get_row(official, "ta_C_taskcond_100k")["avg_successful_sequence_len"]),
        "ABC": float(get_row(official, "ta_ABC_taskcond_100k")["avg_successful_sequence_len"]),
    }
    rows["avg_len"] = rows["split"].map(perf_map)
    x = range(len(rows))
    fig, ax1 = plt.subplots(figsize=(8.8, 5.4))
    ax2 = ax1.twinx()
    ax1.bar([v - 0.18 for v in x], rows["top_to_D_static_rgb_l2"], width=0.36, color=COLORS["gray"], label="top RGB L2 to D")
    ax1.bar([v + 0.18 for v in x], rows["wrist_to_D_gripper_rgb_l2"], width=0.36, color=COLORS["purple"], label="wrist RGB L2 to D")
    ax2.plot(x, rows["avg_len"], color=COLORS["green"], marker="o", linewidth=2.2, label="AvgLen")
    ax1.set_title("Visual Distance to D Does Not Fully Predict Policy Success")
    ax1.set_ylabel("RGB mean L2 distance to D")
    ax2.set_ylabel("AvgLen on CALVIN D")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(rows["split"])
    ax1.set_ylim(0, 0.033)
    ax2.set_ylim(0, 0.92)
    ax1.grid(axis="y", linewidth=0.5, alpha=0.45)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    return savefig(fig, "task2_visual_action_shift.png")


def detail_key(path: str) -> str:
    name = Path(path).name
    name = name.replace("calvin_", "").replace("_D_eval_details.json", "")
    return name


def plot_subtask_breakdown(breakdown: pd.DataFrame) -> Path:
    # Focus on the strongest explicit-split policy so the plot describes final behavior.
    detail_path = (
        RESULT_DIR
        / "calvin_eval_extensions"
        / "act_ta_abc_lang_sbert_100k_seed1"
        / "D"
        / "calvin_act_ta_abc_lang_sbert_100k_seed1_sbert_D_eval_details.json"
    )
    if detail_path.exists():
        raw = json.loads(detail_path.read_text(encoding="utf-8"))
        detail_rows = []
        for seq in raw:
            for item in seq.get("subtasks", []):
                detail_rows.append(
                    {
                        "subtask": item.get("subtask", "unknown"),
                        "success": bool(item.get("success")),
                        "steps": item.get("steps"),
                    }
                )
        rows = pd.DataFrame(detail_rows)
    else:
        target = "act_ta_abc_lang_sbert_100k_seed1"
        rows = breakdown[breakdown["detail_file"].str.contains(target, na=False)].copy()
        if rows.empty:
            rows = breakdown[breakdown["detail_file"].str.contains("act_ta_abc_lang_sbert_100k_seed0", na=False)].copy()
        rows["success"] = rows["success"].astype(bool)

    if rows.empty:
        raise RuntimeError("No subtask details found for final Task 2 policy")

    agg = (
        rows.groupby("subtask")
        .agg(attempts=("success", "size"), successes=("success", "sum"), mean_steps=("steps", "mean"))
        .reset_index()
    )
    agg["success_rate"] = agg["successes"] / agg["attempts"]
    agg = agg[agg["attempts"] >= 8].sort_values(["success_rate", "attempts"], ascending=[False, False])
    top_success = agg.head(8).copy()
    demanding = agg.sort_values(["success_rate", "attempts"], ascending=[True, False]).head(8).copy()
    summary = pd.concat(
        [
            top_success.assign(group="highest_success"),
            demanding.assign(group="precision_demanding"),
        ],
        ignore_index=True,
    )
    summary.to_csv(OUT_SUBTASK, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.8), sharex=True)
    for ax, data, title, color in [
        (axes[0], top_success.sort_values("success_rate"), "High-confidence subtasks", COLORS["green"]),
        (axes[1], demanding.sort_values("success_rate"), "Precision-demanding subtasks", COLORS["orange"]),
    ]:
        ax.barh(data["subtask"], data["success_rate"], color=color)
        ax.set_title(title)
        ax.set_xlabel("Success rate")
        ax.set_xlim(0, 1.0)
        for idx, (_, row) in enumerate(data.iterrows()):
            ax.text(row["success_rate"] + 0.015, idx, f"{row['success_rate']:.2f} ({int(row['attempts'])})", va="center", fontsize=7)
    fig.suptitle("Subtask-Level Behavior of splitABC SBERT seed1")
    return savefig(fig, "task2_subtask_breakdown.png")


def main() -> None:
    official = clean_numeric(load_csv(OFFICIAL_SUMMARY), ["avg_successful_sequence_len", "success_len1", "success_len2", "success_len3", "success_len4", "success_len5"])
    sweep = clean_numeric(load_csv(SWEEP_SUMMARY), ["avg_successful_sequence_len", "success_len1", "success_len2", "success_len3", "success_len4", "success_len5"])
    high = clean_numeric(load_csv(HIGH_SCORE_SUMMARY), ["avg_successful_sequence_len", "success_len1", "success_len2", "success_len3", "success_len4", "success_len5"])
    shift = clean_numeric(load_csv(SHIFT_ANALYSIS), ["top_to_D_static_rgb_l2", "wrist_to_D_gripper_rgb_l2", "mean_l2_delta"])
    breakdown = load_csv(SUBTASK_BREAKDOWN)

    outputs = [
        ("main_results", plot_main_results(official, sweep, high), "AvgLen bar plot with SR@1/SR@5 overlays for main Task 2 policies."),
        ("success_horizon", plot_success_horizon(official, sweep, high), "SR@1--SR@5 curves showing long-horizon degradation."),
        ("checkpoint_sweep", plot_checkpoint_sweep(sweep), "A-only and ABC SBERT checkpoint sensitivity on CALVIN explicit split."),
        ("seed_robustness", plot_seed_robustness(), "Seed0/seed1 100k comparison with mean and sample standard deviation."),
        ("split_size", plot_split_size(), "CALVIN explicit splitA/B/C/D episode and frame counts."),
        ("visual_action_shift", plot_visual_action_shift(shift, official), "RGB distance to D compared with policy AvgLen."),
        ("subtask_breakdown", plot_subtask_breakdown(breakdown), "Final-policy subtask-level success-rate breakdown."),
    ]

    index = pd.DataFrame(
        [
            {
                "figure_id": figure_id,
                "path": str(path.relative_to(ROOT)),
                "description": description,
            }
            for figure_id, path, description in outputs
        ]
    )
    index.to_csv(OUT_INDEX, index=False)
    print(f"Wrote {OUT_INDEX}")
    for _, path, _ in outputs:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
