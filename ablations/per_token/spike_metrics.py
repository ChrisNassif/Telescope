#!/usr/bin/env python3
"""Per-token perplexity spike analysis.

Merges the two previously-duplicated ``main()`` implementations from the original
``per_token_metrics.py`` into a single parameterized pipeline. Runs, in order:

1. Per-label spike statistics at each supplied prominence threshold (with plots).
2. Bidirectional (up/down) spike analysis at the last threshold.
3. Distribution analysis of per-token perplexity values by label.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks

from ablations._common import (
    add_common_args,
    get_logger,
    is_valid_perplexity_data,
    process_array_string,
    resolve_common_args,
    save_plot,
)


def analyze_perplexity_spikes(values: np.ndarray, prominence: float):
    stats = {
        "sequence_length": len(values),
        "mean_perplexity": float(np.mean(values)),
        "median_perplexity": float(np.median(values)),
        "std_perplexity": float(np.std(values)),
        "max_perplexity": float(np.max(values)),
        "min_perplexity": float(np.min(values)),
        "num_spikes": 0,
        "spike_density": 0.0,
        "mean_spike_height": 0.0,
        "mean_prominence": 0.0,
        "early_spikes": 0,
        "middle_spikes": 0,
        "late_spikes": 0,
        "mean_spike_distance": 0.0,
        "std_spike_distance": 0.0,
    }
    peaks, properties = find_peaks(values, prominence=prominence)
    if len(peaks) > 0:
        stats["num_spikes"] = len(peaks)
        stats["spike_density"] = len(peaks) / len(values)
        stats["mean_spike_height"] = float(np.mean(values[peaks]))
        stats["mean_prominence"] = float(np.mean(properties["prominences"]))
        rel = peaks / len(values)
        stats["early_spikes"] = int(np.sum(rel < 0.33))
        stats["middle_spikes"] = int(np.sum((rel >= 0.33) & (rel < 0.66)))
        stats["late_spikes"] = int(np.sum(rel >= 0.66))
        if len(peaks) > 1:
            d = np.diff(peaks)
            stats["mean_spike_distance"] = float(np.mean(d))
            stats["std_spike_distance"] = float(np.std(d))
    return stats, peaks, properties


def plot_perplexity_analysis(values, peaks, label, threshold, plots_dir: Path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    ax1.plot(values, label="Perplexity", alpha=0.7)
    ax1.scatter(peaks, values[peaks], color="red", marker="x", s=100, label="Spikes")
    ax1.set_title(f"Perplexity Scores with Detected Spikes — Label {label}")
    ax1.set_xlabel("Token Position")
    ax1.set_ylabel("Perplexity")
    ax1.legend()
    if len(peaks) > 0:
        sns.histplot(values[peaks], ax=ax2, bins=20)
        ax2.set_title(f"Distribution of Spike Heights — Label {label}")
        ax2.set_xlabel("Perplexity at Spike")
        ax2.set_ylabel("Count")
    fig.tight_layout()
    save_plot(fig, f"perplexity_spikes_label_{label}_threshold_{threshold}.png", plots_dir)


def analyze_bidirectional_spikes(
    valid_data: pd.DataFrame, threshold: float, signal_column: str, plots_dir: Path, logger
):
    label_data = {
        0: {"up_spikes": [], "down_spikes": [], "sequences": []},
        1: {"up_spikes": [], "down_spikes": [], "sequences": []},
    }
    for label in (0, 1):
        for _, row in valid_data[valid_data["y_labels"] == label].iterrows():
            values = process_array_string(row[signal_column])
            if values.size == 0:
                continue
            up_peaks, _ = find_peaks(values, prominence=threshold)
            down_peaks, _ = find_peaks(-values, prominence=threshold)
            if len(up_peaks) > 0:
                label_data[label]["up_spikes"].extend(values[up_peaks])
            if len(down_peaks) > 0:
                label_data[label]["down_spikes"].extend(values[down_peaks])
            label_data[label]["sequences"].extend(values)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    plot_data = {"Label": [], "Magnitude": [], "Direction": []}
    for label in (0, 1):
        up = np.array(label_data[label]["up_spikes"])
        down = np.array(label_data[label]["down_spikes"])
        seqs = label_data[label]["sequences"]
        if not seqs:
            continue
        median_perp = np.median(seqs)
        if up.size:
            m = up - median_perp
            plot_data["Label"].extend([f"Label {label}"] * len(m))
            plot_data["Magnitude"].extend(m)
            plot_data["Direction"].extend(["Up"] * len(m))
        if down.size:
            m = median_perp - down
            plot_data["Label"].extend([f"Label {label}"] * len(m))
            plot_data["Magnitude"].extend(m)
            plot_data["Direction"].extend(["Down"] * len(m))

    df = pd.DataFrame(plot_data)
    up_df = df[df["Direction"] == "Up"]
    down_df = df[df["Direction"] == "Down"]

    if not up_df.empty:
        sns.boxplot(data=up_df, x="Label", y="Magnitude", ax=ax1, showfliers=False)
        ax1.set_title("Upward Spike Magnitudes (relative to median)")
    if not down_df.empty:
        sns.boxplot(data=down_df, x="Label", y="Magnitude", ax=ax2, showfliers=False)
        ax2.set_title("Downward Spike Magnitudes (relative to median)")
    if not up_df.empty:
        for label in (0, 1):
            sub = up_df[up_df["Label"] == f"Label {label}"]
            if not sub.empty:
                sns.histplot(data=sub, x="Magnitude", bins=50, alpha=0.5, label=f"Label {label}", ax=ax3)
        ax3.set_yscale("log")
        ax3.set_title("Distribution of Upward Spike Magnitudes")
        ax3.legend()
    if not down_df.empty:
        for label in (0, 1):
            sub = down_df[down_df["Label"] == f"Label {label}"]
            if not sub.empty:
                sns.histplot(data=sub, x="Magnitude", bins=50, alpha=0.5, label=f"Label {label}", ax=ax4)
        ax4.set_yscale("log")
        ax4.set_title("Distribution of Downward Spike Magnitudes")
        ax4.legend()

    fig.tight_layout()
    save_plot(fig, f"bidirectional_spike_analysis_threshold_{threshold}.png", plots_dir)

    for label in (0, 1):
        seqs = np.array(label_data[label]["sequences"])
        if seqs.size == 0:
            continue
        up = np.array(label_data[label]["up_spikes"])
        down = np.array(label_data[label]["down_spikes"])
        median_perp = np.median(seqs)
        logger.info(f"Label {label} — median={median_perp:.3f} n_tokens={len(seqs)}")
        if up.size:
            logger.info(f"  up spikes: n={len(up)} mean_above_median={np.mean(up - median_perp):.3f}")
        if down.size:
            logger.info(f"  down spikes: n={len(down)} mean_below_median={np.mean(median_perp - down):.3f}")


def analyze_perplexity_distributions(
    valid_data: pd.DataFrame, signal_column: str, plots_dir: Path, logger
):
    label_perp: dict[int, list[float]] = {0: [], 1: []}
    for label in (0, 1):
        for _, row in valid_data[valid_data["y_labels"] == label].iterrows():
            v = process_array_string(row[signal_column])
            if v.size:
                label_perp[label].extend(v)

    perp_0 = np.array(label_perp[0])
    perp_1 = np.array(label_perp[1])
    if perp_0.size == 0 or perp_1.size == 0:
        logger.warning("Not enough per-label data for distribution analysis; skipping.")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    labels_all = [f"Label 0"] * len(perp_0) + [f"Label 1"] * len(perp_1)
    values_all = list(perp_0) + list(perp_1)
    sns.boxplot(x=labels_all, y=values_all, ax=ax1, showfliers=False)
    ax1.set_title("Main Distribution of Perplexity Values (excl. outliers)")

    sns.histplot(perp_0, bins=100, alpha=0.5, label="Label 0", ax=ax2)
    sns.histplot(perp_1, bins=100, alpha=0.5, label="Label 1", ax=ax2)
    ax2.set_yscale("log")
    ax2.set_title("Histogram of Perplexity Values (log scale)")
    ax2.legend()

    s0 = np.sort(perp_0)
    s1 = np.sort(perp_1)
    c0 = np.arange(1, len(s0) + 1) / len(s0)
    c1 = np.arange(1, len(s1) + 1) / len(s1)
    ax3.plot(s0, c0, label="Label 0", alpha=0.7)
    ax3.plot(s1, c1, label="Label 1", alpha=0.7)
    ax3.set_title("Cumulative Distribution")
    ax3.legend()

    i0 = int(0.9 * len(s0))
    i1 = int(0.9 * len(s1))
    ax4.plot(s0[i0:], c0[i0:], label="Label 0", alpha=0.7)
    ax4.plot(s1[i1:], c1[i1:], label="Label 1", alpha=0.7)
    ax4.set_title("Cumulative Distribution (Top 10%)")
    ax4.legend()

    fig.tight_layout()
    save_plot(fig, "perplexity_distribution_analysis.png", plots_dir)

    for label, perp in [(0, perp_0), (1, perp_1)]:
        logger.info(
            f"Label {label} — n={len(perp)} mean={np.mean(perp):.3f} "
            f"median={np.median(perp):.3f} p90={np.percentile(perp, 90):.3f} "
            f"p99={np.percentile(perp, 99):.3f} max={np.max(perp):.3f}"
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", type=str, required=True)
    p.add_argument("--signal-column", type=str, default="telescope_perplexity_per_token")
    p.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0, 10.0, 12.5, 15.0, 20.0],
        help="Prominence thresholds to sweep for spike detection.",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on the number of rows loaded from the CSV.",
    )
    p.add_argument(
        "--skip-bidirectional",
        action="store_true",
        help="Skip the bidirectional spike analysis stage.",
    )
    p.add_argument(
        "--skip-distributions",
        action="store_true",
        help="Skip the perplexity distribution analysis stage.",
    )
    add_common_args(p, "per_token/spike_metrics")
    return p


def main() -> None:
    args = build_parser().parse_args()
    out_dir = resolve_common_args(args, "per_token/spike_metrics")
    logger = get_logger("spike_metrics", args.log_level)

    max_rows = args.max_rows
    thresholds = args.thresholds
    if args.quick:
        max_rows = min(max_rows or 200, 200)
        thresholds = thresholds[:2]

    logger.info(f"Loading {args.input_csv}")
    df = pd.read_csv(args.input_csv, nrows=max_rows)
    valid = df[df[args.signal_column].apply(is_valid_perplexity_data)].copy()
    logger.info(
        f"Total={len(df)}, valid={len(valid)} ({len(valid) / max(len(df),1) * 100:.1f}%)"
    )
    logger.info(f"Label distribution: {valid['y_labels'].value_counts().to_dict()}")

    labels_present = valid["y_labels"].unique()

    for threshold in thresholds:
        logger.info(f"\n=== Prominence threshold {threshold} ===")
        label_stats: dict = {}
        for label in labels_present:
            sub = valid[valid["y_labels"] == label]
            per_seq = []
            plotted = False
            for _, row in sub.iterrows():
                values = process_array_string(row[args.signal_column])
                if values.size == 0:
                    continue
                stats, peaks, _ = analyze_perplexity_spikes(values, threshold)
                per_seq.append(stats)
                if not plotted:
                    plot_perplexity_analysis(values, peaks, label, threshold, out_dir)
                    plotted = True
            if not per_seq:
                continue
            avg = {k: float(np.mean([s[k] for s in per_seq])) for k in per_seq[0]}
            label_stats[label] = avg
            logger.info(
                f"Label {label} n={len(per_seq)} "
                f"mean_perp={avg['mean_perplexity']:.3f} "
                f"spike_density={avg['spike_density']:.3f} "
                f"mean_spike_height={avg['mean_spike_height']:.3f}"
            )

        if len(label_stats) > 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            metrics = ["mean_perplexity", "spike_density", "mean_spike_height"]
            x = np.arange(len(metrics))
            width = 0.35
            for i, label in enumerate(label_stats.keys()):
                vals = [label_stats[label][m] for m in metrics]
                ax.bar(x + i * width, vals, width, label=f"Label {label}")
            ax.set_ylabel("Value")
            ax.set_title(f"Comparison of Metrics (Threshold {threshold})")
            ax.set_xticks(x + width / 2)
            ax.set_xticklabels(metrics, rotation=45)
            ax.legend()
            fig.tight_layout()
            save_plot(fig, f"label_comparison_threshold_{threshold}.png", out_dir)

    if not args.skip_bidirectional and thresholds:
        logger.info("\n=== Bidirectional spike analysis ===")
        analyze_bidirectional_spikes(valid, thresholds[-1], args.signal_column, out_dir, logger)

    if not args.skip_distributions:
        logger.info("\n=== Perplexity distribution analysis ===")
        analyze_perplexity_distributions(valid, args.signal_column, out_dir, logger)

    logger.info(f"\nPlots saved to {out_dir}")


if __name__ == "__main__":
    main()
