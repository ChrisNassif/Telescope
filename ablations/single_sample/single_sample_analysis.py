#!/usr/bin/env python3
"""Compute Telescope perplexity for a single text sample."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ablations._common import add_common_args, get_logger, load_detector, resolve_common_args


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM-360M-Instruct")
    p.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Path to a UTF-8 text file to score.",
    )
    p.add_argument(
        "--text",
        type=str,
        default=None,
        help="Inline text to score (alternative to --input-file).",
    )
    add_common_args(p, "single_sample")
    return p


def main() -> None:
    args = build_parser().parse_args()
    resolve_common_args(args, "single_sample")
    logger = get_logger("single_sample", args.log_level)

    if args.input_file:
        text = Path(args.input_file).read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    else:
        raise SystemExit("Provide either --input-file or --text.")

    logger.info(f"Loading detector {args.model}")
    detector = load_detector(args.model)
    score = detector.compute_all_metrics(text)["telescope_perplexity"]
    logger.info(f"Detection score (telescope_perplexity): {score}")
    print(score)


if __name__ == "__main__":
    main()
