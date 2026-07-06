#!/usr/bin/env python3
"""Train a logistic regression classifier on FFT amplitude features of per-token signals."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from ablations._common import (
    add_common_args,
    get_logger,
    process_array_string,
    resolve_common_args,
    save_plot,
)


def extract_fft_features(signal: np.ndarray) -> np.ndarray:
    fft_result = np.fft.fft(signal)
    mag = np.abs(fft_result)[: len(signal) // 2]
    return mag / max(len(signal), 1)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", type=str, required=True)
    p.add_argument("--signal-column", type=str, default="telescope_perplexity_per_token")
    p.add_argument("--label-column", type=str, default="y_labels")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--top-k-frequencies", type=int, default=10)
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on the number of rows loaded from the CSV.",
    )
    add_common_args(p, "per_token/fft_classifier")
    return p


def main() -> None:
    args = build_parser().parse_args()
    out_dir = resolve_common_args(args, "per_token/fft_classifier")
    logger = get_logger("fft_classifier", args.log_level)

    max_rows = args.max_rows
    if args.quick:
        max_rows = min(max_rows or 500, 500)

    logger.info(f"Loading {args.input_csv}")
    df = pd.read_csv(args.input_csv, nrows=max_rows)
    logger.info(f"Loaded {len(df)} rows")

    features, labels, lengths = [], [], []
    for _, row in df.iterrows():
        signal = process_array_string(row[args.signal_column])
        if signal.size == 0:
            continue
        lengths.append(len(signal))
        features.append(extract_fft_features(signal))
        labels.append(row[args.label_column])

    if not features:
        logger.error("No valid signals found; aborting.")
        return

    median_length = int(np.median(lengths))
    logger.info(f"Median signal length: {median_length}")

    half = median_length // 2
    X = np.array(
        [
            np.pad(f, (0, half - len(f))) if len(f) < half else f[:half]
            for f in features
        ]
    )
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=args.max_iter, random_state=args.seed)
    cv_scores = cross_val_score(clf, X_train_s, y_train, cv=args.cv_folds)
    logger.info(f"CV scores: {cv_scores}")
    logger.info(f"Mean CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    clf.fit(X_train_s, y_train)
    logger.info(f"Train acc: {clf.score(X_train_s, y_train):.3f}")
    logger.info(f"Test acc:  {clf.score(X_test_s, y_test):.3f}")
    y_pred = clf.predict(X_test_s)
    logger.info("\n" + classification_report(y_test, y_pred))
    logger.info(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")

    feature_importance = np.abs(clf.coef_[0])
    freqs = np.fft.fftfreq(median_length)[:half]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(freqs, feature_importance)
    ax.set_title("FFT Amplitude Feature Importance")
    ax.set_xlabel("Frequency (cycles per token)")
    ax.set_ylabel("Absolute Coefficient Value")
    fig.tight_layout()
    save_plot(fig, "feature_importance.png", out_dir)

    top = np.argsort(feature_importance)[-args.top_k_frequencies :][::-1]
    logger.info(f"\nTop {args.top_k_frequencies} discriminative frequencies:")
    for idx in top:
        logger.info(
            f"  freq={freqs[idx]:.4f} cycles/token  importance={feature_importance[idx]:.4f}"
        )

    logger.info(f"Outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
