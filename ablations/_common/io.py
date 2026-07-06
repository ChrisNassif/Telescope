"""IO and array-parsing helpers used across per-token ablations."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUTS_ROOT = REPO_ROOT / "outputs"


def process_array_string(array_string: str) -> np.ndarray:
    """Convert a stringified numeric array (as stored in raw_data.csv) into a numpy array.

    Handles the loose whitespace/newline format produced by earlier experiment dumps.
    Returns an empty array on failure rather than raising.
    """
    try:
        cleaned = array_string.strip("[]").replace("\n", " ")
        values = [x.strip() for x in cleaned.split() if x.strip()]
        return np.array([float(x.rstrip(",")) for x in values])
    except Exception:
        return np.array([])


def is_valid_perplexity_data(data_string: str) -> bool:
    """A per-token perplexity string is only valid if it wasn't truncated with '...'."""
    return isinstance(data_string, str) and "..." not in data_string


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_plot(fig, filename: str, plots_dir: Union[str, Path]) -> Path:
    """Save a matplotlib figure into plots_dir and close it."""
    import matplotlib.pyplot as plt

    plots_dir = ensure_dir(plots_dir)
    filepath = plots_dir / filename
    fig.savefig(filepath)
    plt.close(fig)
    return filepath


def default_output_dir(ablation_name: str) -> Path:
    """Default output location for a given ablation, under repo-root/outputs/."""
    return DEFAULT_OUTPUTS_ROOT / ablation_name
