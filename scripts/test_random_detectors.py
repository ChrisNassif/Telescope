import sys
from typing import Any, Dict, List, Optional, Tuple, Set
argv: List[str] = sys.argv[1:]
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm

from llm_text_detectors.random_functions import generate_unique_detectors, AggregateNode


### START GLOBALS -------------------------------------------------------------------------

# Path to cached .npz file (can also be passed via argv[0])
LOGITS_FILE: str = argv[0] if len(argv) > 0 else "saved_logits/smollm2_135M_esl_gpt4o_dataset.npz"

NUM_RANDOM_DETECTORS_TO_TEST: int = 500
MAX_AST_TREE_DEPTH: int = 3
TOP_K_RESULTS_TO_DISPLAY: int = 20
OUTPUT_SUMMARY_CSV: str = "experiment_results/random_detectors_summary.csv"

### END GLOBALS ---------------------------------------------------------------------------


def load_cached_logits(npz_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Loads y_labels, sample_offsets, sample_lengths, and 16 token metric arrays from NPZ."""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Cached NPZ file not found at: {npz_path}")

    data = np.load(npz_path)
    y_labels = data["y_labels"]
    sample_offsets = data["sample_offsets"]
    sample_lengths = data["sample_lengths"]

    token_data: Dict[str, np.ndarray] = {}
    for key in data.files:
        if key not in ("y_labels", "sample_offsets", "sample_lengths"):
            token_data[key] = data[key].astype(np.float32)

    return y_labels, sample_offsets, sample_lengths, token_data


def compute_best_f1(y_true: np.ndarray, scores: np.ndarray, num_thresholds: int = 100) -> float:
    """Finds maximum F1 score across linear decision thresholds using vectorized broadcasting."""
    clean_mask = ~np.isnan(scores) & ~np.isinf(scores)
    if not np.any(clean_mask):
        return 0.5

    y_clean = y_true[clean_mask]
    s_clean = scores[clean_mask]

    min_s, max_s = np.min(s_clean), np.max(s_clean)
    if min_s == max_s:
        return 0.5

    thresholds = np.linspace(min_s, max_s, num_thresholds)
    
    # Vectorized evaluation across all thresholds at once
    preds_pos = (s_clean[:, None] >= thresholds[None, :])  # Shape: (N, num_thresholds)
    tp_pos = np.sum(preds_pos & (y_clean[:, None] == 1), axis=0)
    fp_pos = np.sum(preds_pos & (y_clean[:, None] == 0), axis=0)
    fn_pos = np.sum((~preds_pos) & (y_clean[:, None] == 1), axis=0)
    f1_pos = 2 * tp_pos / np.maximum(2 * tp_pos + fp_pos + fn_pos, 1e-10)

    preds_neg = (s_clean[:, None] <= thresholds[None, :])
    tp_neg = np.sum(preds_neg & (y_clean[:, None] == 1), axis=0)
    fp_neg = np.sum(preds_neg & (y_clean[:, None] == 0), axis=0)
    fn_neg = np.sum((~preds_neg) & (y_clean[:, None] == 1), axis=0)
    f1_neg = 2 * tp_neg / np.maximum(2 * tp_neg + fp_neg + fn_neg, 1e-10)

    return float(max(np.max(f1_pos), np.max(f1_neg)))


def main():
    print(f"Loading cached logits from: {LOGITS_FILE}")
    y_labels, sample_offsets, sample_lengths, token_data = load_cached_logits(LOGITS_FILE)
    print(f"Loaded {len(y_labels)} samples with {len(token_data)} metric channels.")

    # Validation batch for fast Spearman rank correlation check
    val_size = min(200, len(y_labels))
    val_offsets = sample_offsets[:val_size]
    val_lengths = sample_lengths[:val_size]

    print(f"Generating {NUM_RANDOM_DETECTORS_TO_TEST} unique non-redundant random functions...")
    detectors = generate_unique_detectors(
        num_detectors=NUM_RANDOM_DETECTORS_TO_TEST,
        validation_token_data=token_data,
        sample_offsets=val_offsets,
        sample_lengths=val_lengths,
        max_depth=MAX_AST_TREE_DEPTH,
    )

    results: List[Dict[str, Any]] = []

    print("Evaluating random detectors across all samples...")
    for idx, det in enumerate(tqdm(detectors)):
        formula = det.to_string()
        canon_formula = det.canonicalize().to_string()

        try:
            with np.errstate(all='ignore'), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = det.evaluate_sequence(token_data, sample_offsets, sample_lengths)

                clean_mask = ~np.isnan(scores) & ~np.isinf(scores)
                if np.sum(clean_mask) < len(scores) * 0.9 or np.std(scores[clean_mask]) < 1e-9:
                    auc = 0.5
                    f1 = 0.0
                else:
                    y_clean = y_labels[clean_mask]
                    s_clean = scores[clean_mask]
                    auc = float(roc_auc_score(y_clean, s_clean))
                    # If AUC < 0.5, inverted score thresholding gives 1 - AUC
                    if auc < 0.5:
                        auc = 1.0 - auc

                    f1 = compute_best_f1(y_clean, s_clean)
        except Exception as e:
            formula = f"{formula} (FAILED: {e})"
            auc = 0.5
            f1 = 0.0

        results.append({
            "index": idx + 1,
            "formula": formula,
            "canonical_formula": canon_formula,
            "roc_auc": round(auc, 4),
            "best_f1": round(f1, 4),
        })

    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values(by="roc_auc", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 80)
    print(f"TOP {TOP_K_RESULTS_TO_DISPLAY} BEST PERFORMING RANDOM DETECTORS")
    print("=" * 80)

    for i in range(min(TOP_K_RESULTS_TO_DISPLAY, len(df_res))):
        row = df_res.iloc[i]
        print(f"Rank #{i+1:02d} | ROC-AUC: {row['roc_auc']:.4f} | F1: {row['best_f1']:.4f} | Formula: {row['formula']}")

    os.makedirs(os.path.dirname(OUTPUT_SUMMARY_CSV), exist_ok=True)
    df_res.to_csv(OUTPUT_SUMMARY_CSV, index=False)
    print(f"\nSaved full evaluation results to: {OUTPUT_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
