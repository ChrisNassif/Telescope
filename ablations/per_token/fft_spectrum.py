#!/usr/bin/env python3
"""FFT-based frequency-band analysis of per-token telescope perplexity signals.

Compares the average frequency spectrum of Label 0 vs Label 1 sequences.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ablations._common import (
    add_common_args,
    get_logger,
    is_valid_perplexity_data,
    process_array_string,
    resolve_common_args,
    save_plot,
)


def analyze_signal_frequency(
    valid_data: pd.DataFrame,
    plots_dir: Path,
    signal_column: str,
    low_freq_cutoff: float,
    logger,
):
    """Compute per-label average FFT spectra and produce the frequency-band plot."""
    label_data: dict[int, list[np.ndarray]] = {0: [], 1: []}
    sequence_lengths: list[int] = []
    all_perplexities: list[float] = []

    for _, row in valid_data.iterrows():
        perp_data = process_array_string(row[signal_column])
        if perp_data.size == 0:
            continue
        sequence_lengths.append(len(perp_data))
        all_perplexities.extend(perp_data)

    all_perplexities = np.array(all_perplexities)
    logger.info(
        f"Perplexity stats — max={np.max(all_perplexities):.2f}, "
        f"mean={np.mean(all_perplexities):.2f}, std={np.std(all_perplexities):.2f}"
    )

    median_length = int(np.median(sequence_lengths))
    sample_rate = 1.0

    for label in (0, 1):
        for _, row in valid_data[valid_data["y_labels"] == label].iterrows():
            perp_data = process_array_string(row[signal_column])
            if perp_data.size == 0:
                continue

            if len(perp_data) < median_length:
                perp_data = np.pad(
                    perp_data,
                    (0, median_length - len(perp_data)),
                    "constant",
                    constant_values=np.mean(perp_data),
                )
            elif len(perp_data) > median_length:
                perp_data = perp_data[:median_length]

            windowed = perp_data * np.hanning(len(perp_data))
            fft_mag = np.abs(np.fft.rfft(windowed)) / (median_length / 2)
            label_data[label].append(fft_mag)

    avg_spectrum = {
        label: np.mean(np.array(spectra), axis=0)
        for label, spectra in label_data.items()
        if spectra
    }
    freqs = np.fft.rfftfreq(median_length, d=1 / sample_rate)

    low_idx = freqs <= low_freq_cutoff
    high_idx = freqs > low_freq_cutoff

    fig = plt.figure(figsize=(15, 12))

    ax1 = plt.subplot(2, 2, 1)
    for label, spectrum in avg_spectrum.items():
        ax1.plot(freqs, spectrum, label=f"Label {label}", alpha=0.7)
    ax1.set_title("Full Frequency Spectrum (Linear Scale)")
    ax1.set_xlabel("Frequency (cycles per token)")
    ax1.set_ylabel("Magnitude")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 2, 2)
    for label, spectrum in avg_spectrum.items():
        ax2.plot(freqs[low_idx], spectrum[low_idx], label=f"Label {label}", alpha=0.7)
    ax2.set_title(f"Low Frequency Detail (≤ {low_freq_cutoff} cycles/token)")
    ax2.set_xlabel("Frequency (cycles per token)")
    ax2.set_ylabel("Magnitude")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(2, 2, 3)
    for label, spectrum in avg_spectrum.items():
        ax3.plot(freqs[high_idx], spectrum[high_idx], label=f"Label {label}", alpha=0.7)
    ax3.set_title(f"High Frequency Detail (> {low_freq_cutoff} cycles/token)")
    ax3.set_xlabel("Frequency (cycles per token)")
    ax3.set_ylabel("Magnitude")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(2, 2, 4)
    for label, spectrum in avg_spectrum.items():
        ax4.semilogy(freqs, spectrum + 1e-10, label=f"Label {label}", alpha=0.7)
    ax4.set_title("Full Frequency Spectrum (Log Scale)")
    ax4.set_xlabel("Frequency (cycles per token)")
    ax4.set_ylabel("Magnitude (log scale)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    save_plot(fig, "perplexity_frequency_bands.png", plots_dir)

    for label, spectrum in avg_spectrum.items():
        low = spectrum[low_idx]
        high = spectrum[high_idx]
        logger.info(
            f"Label {label} — n_seqs={len(label_data[label])} "
            f"low(max={np.max(low):.4f},mean={np.mean(low):.4f}) "
            f"high(max={np.max(high):.4f},mean={np.mean(high):.4f})"
        )

    return median_length, avg_spectrum, freqs


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="Path to raw_data.csv with per-token perplexity columns.",
    )
    p.add_argument(
        "--signal-column",
        type=str,
        default="telescope_perplexity_per_token",
        help="Column holding the per-token signal to analyze.",
    )
    p.add_argument(
        "--low-freq-cutoff",
        type=float,
        default=0.1,
        help="Frequency threshold (cycles/token) splitting low vs high band.",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on the number of rows loaded from the CSV.",
    )
    add_common_args(p, "per_token/fft_spectrum")
    return p


def main() -> None:
    args = build_parser().parse_args()
    out_dir = resolve_common_args(args, "per_token/fft_spectrum")
    logger = get_logger("fft_spectrum", args.log_level)

    max_rows = args.max_rows
    if args.quick:
        max_rows = min(max_rows or 200, 200)

    logger.info(f"Loading {args.input_csv}")
    df = pd.read_csv(args.input_csv, nrows=max_rows)
    valid = df[df[args.signal_column].apply(is_valid_perplexity_data)].copy()
    logger.info(
        f"Total={len(df)}, valid={len(valid)} ({len(valid) / max(len(df),1) * 100:.1f}%)"
    )
    logger.info(f"Label distribution: {valid['y_labels'].value_counts().to_dict()}")

    analyze_signal_frequency(
        valid, out_dir, args.signal_column, args.low_freq_cutoff, logger
    )
    logger.info(f"Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
