"""Argparse helpers shared across ablation scripts."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from ablations._common.io import default_output_dir, ensure_dir


def add_common_args(parser: argparse.ArgumentParser, ablation_name: str) -> None:
    """Attach flags every ablation supports so the CLI is uniform."""
    group = parser.add_argument_group("common")
    group.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Directory to write outputs. Default: outputs/{ablation_name}/",
    )
    group.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Torch device (falls back to CPU if CUDA unavailable).",
    )
    group.add_argument("--seed", type=int, default=42, help="Random seed.")
    group.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    group.add_argument(
        "--quick",
        action="store_true",
        help="Smoke-test mode: dramatically reduce sample sizes / iteration counts.",
    )


def resolve_common_args(args: argparse.Namespace, ablation_name: str) -> Path:
    """Apply seeds, resolve output dir, ensure it exists. Returns the output Path."""
    random.seed(args.seed)
    try:
        import numpy as np

        np.random.seed(args.seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except ImportError:
        pass

    out = Path(args.output_dir) if args.output_dir else default_output_dir(ablation_name)
    return ensure_dir(out)
