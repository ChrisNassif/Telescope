#!/usr/bin/env python3
"""Assemble per-model / per-dataset sequence-modeling CSVs from raw experiment output.

For each combination of --models × --datasets, reads
``<experiment-folder>/<model>_<dataset>_dataset/raw_data.csv`` and writes
``<output-dir>/<dataset>_<model>_dataset/full.csv`` with the per-token perplexity
columns plus labels.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from ablations._common import add_common_args, ensure_dir, get_logger, resolve_common_args


DEFAULT_MODELS = ["smollm_360M"]
DEFAULT_DATASETS = ["detect_llm_text", "ai_human", "hc3", "hc3_plus", "esl_gpt4o"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--experiment-folder",
        type=str,
        default="experiment_results_new_extra_features",
        help="Root folder containing raw per-model/per-dataset outputs.",
    )
    p.add_argument("--models", type=str, nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--datasets", type=str, nargs="+", default=DEFAULT_DATASETS)
    p.add_argument(
        "--metric-columns",
        type=str,
        nargs="+",
        default=["telescope_perplexity_per_token", "cross_perplexity_per_token", "perplexity_per_token"],
        help="Columns kept in each output CSV (must all exist in raw_data.csv).",
    )
    p.add_argument(
        "--dropna-column",
        type=str,
        default="telescope_perplexity_per_token",
        help="Rows with NaN/inf in this column are dropped.",
    )
    add_common_args(p, "sequence_modeling/build_dataset")
    return p


def main() -> None:
    args = build_parser().parse_args()
    out_dir = resolve_common_args(args, "sequence_modeling/build_dataset")
    logger = get_logger("build_dataset", args.log_level)

    exp_root = Path(args.experiment_folder)
    for model in args.models:
        for dataset in args.datasets:
            src = exp_root / f"{model}_{dataset}_dataset" / "raw_data.csv"
            if not src.exists():
                logger.warning(f"missing input: {src}")
                continue
            df = pd.read_csv(src).replace([np.inf, -np.inf], np.nan).dropna(subset=[args.dropna_column])
            payload = {col: df[col].to_numpy() for col in args.metric_columns if col in df.columns}
            payload["labels"] = df["y_labels"].astype(bool).to_numpy()
            out = ensure_dir(out_dir / f"{dataset}_{model}_dataset")
            out_csv = out / "full.csv"
            pd.DataFrame(payload).to_csv(out_csv, index=False)
            logger.info(f"wrote {out_csv} ({len(df)} rows)")


if __name__ == "__main__":
    main()
