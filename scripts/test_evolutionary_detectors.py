#!/usr/bin/env python3
"""
Evolutionary Genetic Algorithm for Symbolic AI Text Detectors.

Runs an evolutionary search over AST detector expressions, using top-performing
building blocks from successful detectors to generate offspring through subtree
crossover and mutation.
"""

import sys
from typing import Any, Dict, List, Optional, Tuple, Set
argv: List[str] = sys.argv[1:]
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
import random
import warnings
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm

from llm_text_detectors.random_functions import (
    NonRedundantDetectorPool,
    generate_random_detector,
    AggregateNode,
)
from llm_text_detectors.evolutionary_functions import (
    tournament_selection,
    crossover_detectors,
    mutate_detector,
    clone_ast,
)


### START GLOBALS -------------------------------------------------------------------------

# Path to cached .npz file (can also be passed via argv[0])
LOGITS_FILE: str = argv[0] if len(argv) > 0 else "saved_logits/smollm2_135M_esl_gpt4o_dataset.npz"

POPULATION_SIZE: int = 100
NUM_GENERATIONS: int = 20
TOURNAMENT_SIZE: int = 4
CROSSOVER_PROBABILITY: float = 0.7
MUTATION_PROBABILITY: float = 0.4
ELITE_COUNT: int = 5
MAX_AST_TREE_DEPTH: int = 3
TOP_K_RESULTS_TO_DISPLAY: int = 20
OUTPUT_SUMMARY_CSV: str = "experiment_results/evolutionary_detectors_summary.csv"

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


def compute_fitness(detector: AggregateNode, token_data: Dict[str, np.ndarray], sample_offsets: np.ndarray, sample_lengths: np.ndarray, y_labels: np.ndarray) -> Tuple[float, float, Optional[np.ndarray]]:
    """Evaluates detector scores and returns (ROC-AUC fitness, F1 score, raw scores array)."""
    try:
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = detector.evaluate_sequence(token_data, sample_offsets, sample_lengths)

            clean_mask = ~np.isnan(scores) & ~np.isinf(scores)
            if np.sum(clean_mask) < len(scores) * 0.9 or np.std(scores[clean_mask]) < 1e-9:
                return 0.5, 0.0, None

            y_clean = y_labels[clean_mask]
            s_clean = scores[clean_mask]

            auc = float(roc_auc_score(y_clean, s_clean))
            if auc < 0.5:
                auc = 1.0 - auc

            # Vectorized F1 calculation across thresholds
            min_s, max_s = np.min(s_clean), np.max(s_clean)
            if min_s == max_s:
                f1_val = 0.5
            else:
                thresholds = np.linspace(min_s, max_s, 50)
                preds_pos = (s_clean[:, None] >= thresholds[None, :])
                tp_pos = np.sum(preds_pos & (y_clean[:, None] == 1), axis=0)
                fp_pos = np.sum(preds_pos & (y_clean[:, None] == 0), axis=0)
                fn_pos = np.sum((~preds_pos) & (y_clean[:, None] == 1), axis=0)
                f1_pos = 2 * tp_pos / np.maximum(2 * tp_pos + fp_pos + fn_pos, 1e-10)

                preds_neg = (s_clean[:, None] <= thresholds[None, :])
                tp_neg = np.sum(preds_neg & (y_clean[:, None] == 1), axis=0)
                fp_neg = np.sum(preds_neg & (y_clean[:, None] == 0), axis=0)
                fn_neg = np.sum((~preds_neg) & (y_clean[:, None] == 1), axis=0)
                f1_neg = 2 * tp_neg / np.maximum(2 * tp_neg + fp_neg + fn_neg, 1e-10)

                f1_val = float(max(np.max(f1_pos), np.max(f1_neg)))

            return auc, f1_val, scores
    except Exception:
        return 0.5, 0.0, None


def main():
    print(f"Loading cached logits from: {LOGITS_FILE}")
    y_labels, sample_offsets, sample_lengths, token_data = load_cached_logits(LOGITS_FILE)
    print(f"Loaded {len(y_labels)} samples with {len(token_data)} metric channels.")

    val_size = min(200, len(y_labels))
    val_offsets = sample_offsets[:val_size]
    val_lengths = sample_lengths[:val_size]

    pool = NonRedundantDetectorPool()

    print(f"\nInitializing initial population of {POPULATION_SIZE} detectors...")
    population: List[AggregateNode] = []
    fitness_scores: List[float] = []
    f1_scores: List[float] = []

    attempts = 0
    while len(population) < POPULATION_SIZE and attempts < 10000:
        attempts += 1
        cand = generate_random_detector(max_depth=MAX_AST_TREE_DEPTH)
        if pool.is_redundant(cand, token_data, val_offsets, val_lengths):
            continue

        auc, f1, scores = compute_fitness(cand, token_data, sample_offsets, sample_lengths, y_labels)
        if scores is not None and auc > 0.5:
            pool.add_detector(cand, scores[:val_size])
            population.append(cand)
            fitness_scores.append(auc)
            f1_scores.append(f1)

    print(f"Initial population ready ({len(population)} detectors). Best AUC: {max(fitness_scores):.4f}\n")

    best_all_time_auc = 0.0
    best_all_time_detector = None
    all_evaluated_history: Dict[str, Dict[str, Any]] = {}

    for gen in range(1, NUM_GENERATIONS + 1):
        # Sort population by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]
        f1_scores = [f1_scores[i] for i in sorted_indices]

        gen_best_auc = fitness_scores[0]
        gen_mean_auc = float(np.mean(fitness_scores))

        if gen_best_auc > best_all_time_auc:
            best_all_time_auc = gen_best_auc
            best_all_time_detector = population[0]

        # Record top detectors in history
        for ind, score, f1 in zip(population, fitness_scores, f1_scores):
            formula = ind.to_string()
            if formula not in all_evaluated_history:
                all_evaluated_history[formula] = {
                    "formula": formula,
                    "canonical_formula": ind.canonicalize().to_string(),
                    "roc_auc": round(score, 4),
                    "best_f1": round(f1, 4),
                    "generation": gen,
                }

        print(f"Gen {gen:02d}/{NUM_GENERATIONS:02d} | Best ROC-AUC: {gen_best_auc:.4f} | Mean ROC-AUC: {gen_mean_auc:.4f} | Top: {population[0].to_string()[:60]}")

        # Elitism: retain top ELITE_COUNT intact
        new_population: List[AggregateNode] = [clone_ast(ind) for ind in population[:ELITE_COUNT]]
        new_fitness: List[float] = fitness_scores[:ELITE_COUNT].copy()
        new_f1: List[float] = f1_scores[:ELITE_COUNT].copy()

        gen_attempts = 0
        while len(new_population) < POPULATION_SIZE and gen_attempts < 2000:
            gen_attempts += 1
            p1 = tournament_selection(population, fitness_scores, tournament_size=TOURNAMENT_SIZE)

            if random.random() < CROSSOVER_PROBABILITY:
                p2 = tournament_selection(population, fitness_scores, tournament_size=TOURNAMENT_SIZE)
                child = crossover_detectors(p1, p2)
            else:
                child = clone_ast(p1)

            if random.random() < MUTATION_PROBABILITY:
                child = mutate_detector(child, max_depth=MAX_AST_TREE_DEPTH)

            if pool.is_redundant(child, token_data, val_offsets, val_lengths):
                continue

            auc, f1, scores = compute_fitness(child, token_data, sample_offsets, sample_lengths, y_labels)
            if scores is not None:
                pool.add_detector(child, scores[:val_size])
                new_population.append(child)
                new_fitness.append(auc)
                new_f1.append(f1)

        population = new_population
        fitness_scores = new_fitness
        f1_scores = new_f1

    # Output top results
    df_results = pd.DataFrame(list(all_evaluated_history.values()))
    df_results = df_results.sort_values(by="roc_auc", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 80)
    print(f"TOP {TOP_K_RESULTS_TO_DISPLAY} EVOLVED DETECTORS ACROSS ALL GENERATIONS")
    print("=" * 80)

    for i in range(min(TOP_K_RESULTS_TO_DISPLAY, len(df_results))):
        row = df_results.iloc[i]
        print(f"Rank #{i+1:02d} | ROC-AUC: {row['roc_auc']:.4f} | F1: {row['best_f1']:.4f} | Formula: {row['formula']}")

    os.makedirs(os.path.dirname(OUTPUT_SUMMARY_CSV), exist_ok=True)
    df_results.to_csv(OUTPUT_SUMMARY_CSV, index=False)
    print(f"\nSaved full evolutionary search results to: {OUTPUT_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
