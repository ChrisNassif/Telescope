#!/usr/bin/env python3
"""Analyze the separability of two datasets under perplexity / cross-perplexity / Binoculars scores."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
from datasets import load_dataset
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ablations._common import add_common_args, get_logger, resolve_common_args
from llm_text_detectors import Detectors


def compute_perplexity(enc, logits, median=False, temperature=1.0):
    return Detectors._compute_perplexity(None, enc, logits, median, temperature)


def compute_cross_perplexity(perf_enc, perf_logits, obs_model, device, pad_token_id, median, temperature):
    with torch.no_grad():
        obs_logits = obs_model(**perf_enc).logits
    return Detectors._compute_entropy(
        None, obs_logits.to(device), perf_logits.to(device), perf_enc.to(device),
        pad_token_id, median=median, temperature=temperature,
    )


def compute_binoculars(perp, cross_perp):
    return perp / np.maximum(cross_perp, 1e-10)


def sample_from_dataset(dataset, n: int, text_fields: List[str], min_len: int) -> List[str]:
    samples = []
    for i, item in enumerate(dataset):
        if i >= n:
            break
        text = next((item[f] for f in text_fields if f in item), None)
        if text and len(text.strip()) > min_len:
            samples.append(text)
    return samples


def process_dataset_batch(
    texts, perf_model, obs_model, tokenizer, device, max_length, batch_size, median, temperature
) -> Tuple[List[float], List[float], List[float]]:
    ppl_all, xppl_all, bino_all = [], [], []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        with torch.no_grad():
            out = perf_model(**enc)
        ppl = compute_perplexity(enc, out.logits, median, temperature)
        xppl = compute_cross_perplexity(enc, out.logits, obs_model, device, tokenizer.pad_token_id, median, temperature)
        bino = compute_binoculars(ppl, xppl)
        ppl_all.extend(ppl); xppl_all.extend(xppl); bino_all.extend(bino)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return ppl_all, xppl_all, bino_all


def compute_classification_metrics(y_true, y_scores, name, logger) -> Dict[str, Any]:
    auc_ = roc_auc_score(y_true, y_scores)
    threshold = float(np.median(y_scores))
    y_pred = (y_scores > threshold).astype(int)
    metrics = {
        "auc": auc_,
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    logger.info(f"{name}: AUC={metrics['auc']:.4f} F1={metrics['f1']:.4f} "
                f"P={metrics['precision']:.4f} R={metrics['recall']:.4f}")
    return metrics


def create_visualizations(a_scores, b_scores, output_dir: Path):
    plt.figure(figsize=(18, 6))
    metrics = ["perplexity", "cross_perplexity", "binoculars"]
    titles = ["Perplexity", "Cross-Perplexity", "Binoculars"]
    for i, (m, t) in enumerate(zip(metrics, titles)):
        plt.subplot(1, 3, i + 1)
        plt.hist(a_scores[m], bins=50, alpha=0.5, label="Dataset A", density=True)
        plt.hist(b_scores[m], bins=50, alpha=0.5, label="Dataset B", density=True)
        plt.xlabel(t); plt.ylabel("Density"); plt.legend()
        plt.title(f"Distribution of {t} Scores")
    plt.tight_layout()
    plt.savefig(output_dir / "score_distributions.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.scatter(a_scores["perplexity"], a_scores["cross_perplexity"], alpha=0.5, label="Dataset A", s=30)
    plt.scatter(b_scores["perplexity"], b_scores["cross_perplexity"], alpha=0.5, label="Dataset B", s=30)
    plt.xlabel("Perplexity"); plt.ylabel("Cross-Perplexity"); plt.legend()
    plt.title("Perplexity vs Cross-Perplexity"); plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "perplexity_scatter.png", dpi=300)
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--performer-model", type=str, required=True)
    p.add_argument("--observer-model", type=str, required=True)
    p.add_argument("--dataset-a", type=str, required=True)
    p.add_argument("--dataset-a-config", type=str, default=None)
    p.add_argument("--dataset-b", type=str, required=True)
    p.add_argument("--dataset-b-config", type=str, default=None)
    p.add_argument("--sample-size", type=int, default=1000)
    p.add_argument("--min-text-length", type=int, default=100)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--median", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--text-fields", type=str, nargs="+", default=["text", "content", "document"])
    add_common_args(p, "training/dataset_separability")
    return p


def main() -> None:
    args = build_parser().parse_args()
    out_dir = resolve_common_args(args, "training/dataset_separability")
    logger = get_logger("dataset_separability", args.log_level)

    if args.quick:
        args.sample_size = min(args.sample_size, 50)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA unavailable — using CPU")

    logger.info(f"Performer: {args.performer_model}")
    perf_tokenizer = AutoTokenizer.from_pretrained(args.performer_model)
    perf_model = AutoModelForCausalLM.from_pretrained(args.performer_model).to(device)
    logger.info(f"Observer: {args.observer_model}")
    obs_model = AutoModelForCausalLM.from_pretrained(args.observer_model).to(device)

    if perf_tokenizer.pad_token is None:
        perf_tokenizer.pad_token = perf_tokenizer.eos_token
        perf_model.config.pad_token_id = perf_model.config.eos_token_id

    logger.info(f"Loading dataset A: {args.dataset_a}/{args.dataset_a_config}")
    ds_a = load_dataset(args.dataset_a, args.dataset_a_config, split="train", streaming=True)
    logger.info(f"Loading dataset B: {args.dataset_b}/{args.dataset_b_config}")
    ds_b = load_dataset(args.dataset_b, args.dataset_b_config, split="train", streaming=True)

    a_samples = sample_from_dataset(ds_a, args.sample_size, args.text_fields, args.min_text_length)
    b_samples = sample_from_dataset(ds_b, args.sample_size, args.text_fields, args.min_text_length)
    logger.info(f"Collected {len(a_samples)} A, {len(b_samples)} B")

    a_ppl, a_xppl, a_bino = process_dataset_batch(
        a_samples, perf_model, obs_model, perf_tokenizer, device,
        args.max_length, args.batch_size, args.median, args.temperature,
    )
    b_ppl, b_xppl, b_bino = process_dataset_batch(
        b_samples, perf_model, obs_model, perf_tokenizer, device,
        args.max_length, args.batch_size, args.median, args.temperature,
    )

    a_scores = {"perplexity": a_ppl, "cross_perplexity": a_xppl, "binoculars": a_bino}
    b_scores = {"perplexity": b_ppl, "cross_perplexity": b_xppl, "binoculars": b_bino}
    y_true = np.concatenate([np.zeros(len(a_ppl)), np.ones(len(b_ppl))])

    results: Dict[str, Any] = {
        "dataset_info": {
            "dataset_a": args.dataset_a, "dataset_a_config": args.dataset_a_config,
            "dataset_a_samples": len(a_samples),
            "dataset_b": args.dataset_b, "dataset_b_config": args.dataset_b_config,
            "dataset_b_samples": len(b_samples),
        },
        "model_info": {"performer_model": args.performer_model, "observer_model": args.observer_model},
        "parameters": {
            "max_length": args.max_length, "batch_size": args.batch_size,
            "median": args.median, "temperature": args.temperature,
        },
    }
    for name, key in [("Perplexity", "perplexity"), ("Cross-Perplexity", "cross_perplexity"), ("Binoculars", "binoculars")]:
        y_scores = np.concatenate([a_scores[key], b_scores[key]])
        results[f"{key}_metrics"] = compute_classification_metrics(y_true, y_scores, name, logger)

    results["dataset_a_scores"] = {k: [float(v) for v in vs] for k, vs in a_scores.items()}
    results["dataset_b_scores"] = {k: [float(v) for v in vs] for k, vs in b_scores.items()}

    (out_dir / "separability_results.json").write_text(json.dumps(results, indent=2))
    create_visualizations(a_scores, b_scores, out_dir)
    logger.info(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
