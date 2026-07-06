#!/usr/bin/env python3
"""Top-level CLI dispatcher for Telescope ablation scripts.

Examples
--------
List every registered ablation:

    python run_ablations.py --list

Run a specific ablation, forwarding any remaining args to its own CLI:

    python run_ablations.py fft_spectrum --input-csv path/to/raw_data.csv --quick
    python run_ablations.py ngram_perplexity --sample-size 500 --ngrams 2 3
    python run_ablations.py train_pythia --model-name EleutherAI/pythia-160m \\
        --prompts prompts.txt --interval 5000 --plot

Print the underlying CLI for a specific ablation:

    python run_ablations.py fft_spectrum --help

Run every ablation with default arguments (skips ones that need required flags):

    python run_ablations.py --all
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))


# ablation-name -> (module-path, one-line-description, requires-args)
ABLATIONS: Dict[str, Dict[str, object]] = {
    "fft_spectrum": {
        "module": "ablations.per_token.fft_spectrum",
        "desc": "FFT frequency-band analysis of per-token telescope perplexity.",
        "requires_args": True,
    },
    "fft_classifier": {
        "module": "ablations.per_token.fft_classifier",
        "desc": "Logistic regression on FFT amplitude features.",
        "requires_args": True,
    },
    "spike_metrics": {
        "module": "ablations.per_token.spike_metrics",
        "desc": "Per-token perplexity spike detection, bidirectional and distribution analysis.",
        "requires_args": True,
    },
    "ngram_perplexity": {
        "module": "ablations.per_token.ngram_perplexity",
        "desc": "N-gram perplexity comparison (HC3 dataset).",
        "requires_args": False,
    },
    "sampling_analysis": {
        "module": "ablations.sampling.sampling_analysis",
        "desc": "Compare Telescope scores across generation sampling strategies.",
        "requires_args": False,
    },
    "single_sample": {
        "module": "ablations.single_sample.single_sample_analysis",
        "desc": "Score a single text sample.",
        "requires_args": True,
    },
    "token_distribution": {
        "module": "ablations.single_token_distribution.token_distribution",
        "desc": "Token probability distribution analyses (entropy, next-token bump, peak exclusion).",
        "requires_args": True,
    },
    "train_pythia": {
        "module": "ablations.training.train_pythia_checkpoints",
        "desc": "Evaluate Pythia model checkpoints across training steps.",
        "requires_args": True,
    },
    "dataset_separability": {
        "module": "ablations.training.dataset_separability",
        "desc": "Compare separability between two HuggingFace datasets.",
        "requires_args": True,
    },
    "build_sequence_dataset": {
        "module": "ablations.sequence_modeling.build_dataset",
        "desc": "Build sequence-modeling CSVs from raw experiment output.",
        "requires_args": False,
    },
    "train_sequence_model": {
        "module": "ablations.sequence_modeling.train_sequence_model",
        "desc": "Train an LSTM classifier on per-token perplexity sequences.",
        "requires_args": True,
    },
}


def print_list() -> None:
    print("Available ablations:\n")
    name_w = max(len(n) for n in ABLATIONS) + 2
    for name, info in ABLATIONS.items():
        marker = " *" if info["requires_args"] else "  "
        print(f"{marker}{name:<{name_w}}{info['desc']}")
    print("\n(* = requires arguments; won't be run by --all with defaults)")


def dispatch(name: str, argv: List[str]) -> int:
    if name not in ABLATIONS:
        print(f"error: unknown ablation '{name}'. Use --list to see options.", file=sys.stderr)
        return 2
    module = importlib.import_module(ABLATIONS[name]["module"])
    # Replace sys.argv so the sub-script's argparse sees only what the user meant for it.
    saved = sys.argv
    sys.argv = [name] + argv
    try:
        module.main()
    finally:
        sys.argv = saved
    return 0


def run_all() -> int:
    failed: List[str] = []
    for name, info in ABLATIONS.items():
        if info["requires_args"]:
            print(f"[skip] {name} (requires arguments)")
            continue
        print(f"\n=== running {name} ===")
        try:
            dispatch(name, [])
        except SystemExit as e:
            if e.code not in (0, None):
                failed.append(name)
        except Exception as e:
            print(f"[fail] {name}: {e}")
            failed.append(name)
    if failed:
        print(f"\nFailures: {failed}")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    parser.add_argument("--list", action="store_true", help="List available ablations and exit.")
    parser.add_argument("--all", action="store_true", help="Run every ablation with default args.")
    parser.add_argument("-h", "--help", action="store_true", help="Show this help and exit.")
    parser.add_argument(
        "ablation", nargs="?",
        help="Name of the ablation to run (see --list).",
    )
    parser.add_argument(
        "rest", nargs=argparse.REMAINDER,
        help="Arguments forwarded to the underlying ablation script.",
    )
    args = parser.parse_args()

    if args.help or (not args.list and not args.all and not args.ablation):
        parser.print_help()
        print()
        print_list()
        return 0
    if args.list:
        print_list()
        return 0
    if args.all:
        return run_all()

    return dispatch(args.ablation, args.rest)


if __name__ == "__main__":
    raise SystemExit(main())
