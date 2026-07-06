#!/usr/bin/env python3
"""N-gram perplexity comparison for human vs AI text detection.

Splits texts from the HC3 dataset into N-word chunks and compares Telescope vs
standard perplexity ROC/AUC across a range of N values.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm

from ablations._common import add_common_args, get_logger, load_detector, resolve_common_args


def split_into_ngrams(text: str, n: int) -> List[str]:
    words = text.split()
    if len(words) < n:
        return [text]
    return [" ".join(words[i : i + n]) for i in range(len(words) - (n - 1))]


def perplexity_of_texts(
    detector,
    texts: List[str],
    device: torch.device,
    batch_size: int,
    use_standard: bool,
    max_length: int,
    median: bool,
    temperature: float,
    logger,
) -> List[float]:
    detector.performer_model.eval()
    results: List[float] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="perplexity"):
        for text in texts[i : i + batch_size]:
            if not isinstance(text, str) or not text.strip():
                results.append(float("nan"))
                continue
            try:
                enc = detector.performer_tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=max_length
                ).to(device)
                with torch.no_grad():
                    logits = detector.performer_model(**enc).logits
                if use_standard:
                    ppl = np.exp(
                        detector._compute_perplexity(enc, logits, median=median, temperature=temperature)
                    )
                else:
                    ppl = detector._compute_telescope_perplexity(enc, logits, median, temperature)
                results.append(float(ppl[0]))
            except Exception as e:
                logger.warning(f"perplexity error: {e}")
                results.append(float("nan"))
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    return results


def ngram_perplexity_of_texts(
    detector,
    texts: List[str],
    device: torch.device,
    n: int,
    batch_size: int,
    use_standard: bool,
    max_length: int,
    median: bool,
    temperature: float,
    logger,
) -> List[float]:
    detector.performer_model.eval()
    out: List[float] = []
    for text in tqdm(texts, desc=f"{n}-gram"):
        if not isinstance(text, str) or not text.strip():
            out.append(float("nan"))
            continue
        ngrams = split_into_ngrams(text, n)
        text_results: List[float] = []
        for i in range(0, len(ngrams), batch_size):
            batch = ngrams[i : i + batch_size]
            try:
                enc = detector.performer_tokenizer(
                    batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
                ).to(device)
                with torch.no_grad():
                    logits = detector.performer_model(**enc).logits
                if use_standard:
                    ppl = np.exp(
                        detector._compute_perplexity(enc, logits, median=median, temperature=temperature)
                    )
                else:
                    ppl = detector._compute_telescope_perplexity(enc, logits, median, temperature)
                text_results.extend(ppl)
            except Exception as e:
                logger.warning(f"ngram batch error: {e}")
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        out.append(float(np.mean(text_results)) if text_results else float("nan"))
    return out


def load_hc3(sample_size: int, min_len: int, logger) -> Tuple[List[str], List[str]]:
    logger.info("Loading HC3...")
    ds = load_dataset("Hello-SimpleAI/HC3", "all", split="train")
    if len(ds) > sample_size:
        ds = ds.select(range(sample_size))
    human, ai = [], []
    for item in ds:
        h = item.get("human_answers", item.get("human_answer", item.get("human")))
        a = item.get("chatgpt_answers", item.get("chatgpt_answer", item.get("chatgpt")))
        for tgt, src in [(human, h), (ai, a)]:
            if src is None:
                continue
            if isinstance(src, list):
                tgt.extend(s for s in src if s and len(s.strip()) > min_len)
            elif len(src.strip()) > min_len:
                tgt.append(src)
    logger.info(f"HC3 loaded: {len(human)} human, {len(ai)} AI")
    return human, ai


def roc_and_auc(results: Dict[str, Dict[str, List[float]]], out_file: Path) -> Dict[str, Dict[str, float]]:
    plt.figure(figsize=(12, 10))
    metrics = {}
    colors = ["blue", "red", "green", "purple", "orange", "brown"]
    for i, (method, data) in enumerate(results.items()):
        h = [x for x in data["human"] if not np.isnan(x)]
        a = [x for x in data["ai"] if not np.isnan(x)]
        if not h or not a:
            continue
        y_true = [1] * len(h) + [0] * len(a)
        y_score = h + a
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2, label=f"{method} (AUC={roc_auc:.3f})")
        metrics[method] = {
            "auc": roc_auc,
            "avg_human": float(np.nanmean(data["human"])),
            "avg_ai": float(np.nanmean(data["ai"])),
        }
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    return metrics


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM-135M")
    p.add_argument("--sample-size", type=int, default=5000)
    p.add_argument("--min-text-length", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--ngram-batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--median", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--ngrams", type=int, nargs="+", default=[2, 3])
    add_common_args(p, "per_token/ngram_perplexity")
    return p


def main() -> None:
    args = build_parser().parse_args()
    out_dir = resolve_common_args(args, "per_token/ngram_perplexity")
    logger = get_logger("ngram_perplexity", args.log_level)

    if args.quick:
        args.sample_size = min(args.sample_size, 50)
        args.ngrams = args.ngrams[:1]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA unavailable — using CPU")

    logger.info(f"Loading detector {args.model}")
    detector = load_detector(args.model)
    if detector.performer_tokenizer.pad_token is None:
        detector.performer_tokenizer.pad_token = detector.performer_tokenizer.eos_token

    human, ai = load_hc3(args.sample_size, args.min_text_length, logger)

    all_results: Dict[str, Dict[str, Dict[str, List[float]]]] = {"telescope": {}, "standard": {}}

    for ppl_type, use_standard in [("telescope", False), ("standard", True)]:
        logger.info(f"Normal perplexity ({ppl_type})...")
        human_normal = perplexity_of_texts(
            detector, human, device, args.batch_size, use_standard,
            args.max_length, args.median, args.temperature, logger,
        )
        ai_normal = perplexity_of_texts(
            detector, ai, device, args.batch_size, use_standard,
            args.max_length, args.median, args.temperature, logger,
        )
        all_results[ppl_type]["Normal"] = {"human": human_normal, "ai": ai_normal}

        for n in args.ngrams:
            if n == 1:
                continue
            logger.info(f"{n}-gram perplexity ({ppl_type})...")
            h = ngram_perplexity_of_texts(
                detector, human, device, n, args.ngram_batch_size, use_standard,
                args.max_length, args.median, args.temperature, logger,
            )
            a = ngram_perplexity_of_texts(
                detector, ai, device, n, args.ngram_batch_size, use_standard,
                args.max_length, args.median, args.temperature, logger,
            )
            all_results[ppl_type][f"{n}-gram"] = {"human": h, "ai": a}

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    telescope_metrics = roc_and_auc(all_results["telescope"], out_dir / "telescope_roc_curves.png")
    standard_metrics = roc_and_auc(all_results["standard"], out_dir / "standard_roc_curves.png")

    with open(out_dir / "ngram_perplexity_results.json", "w") as f:
        json.dump(
            {
                "model": args.model,
                "parameters": {
                    "sample_size": args.sample_size,
                    "max_length": args.max_length,
                    "median": args.median,
                    "temperature": args.temperature,
                    "ngrams": args.ngrams,
                },
                "telescope_metrics": telescope_metrics,
                "standard_metrics": standard_metrics,
            },
            f,
            indent=2,
        )

    logger.info(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
