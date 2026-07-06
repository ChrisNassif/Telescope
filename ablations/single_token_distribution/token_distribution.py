#!/usr/bin/env python3
"""Compare token probability distributions between human and AI text.

Runs four analyses:
- distribution uniformity (avg entropy)
- next-token probability bump
- current-token-in-next-position probability ratios (with and without peak)
- peak-exclusion detection score
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import transformers
import matplotlib.pyplot as plt
import seaborn as sns

from ablations._common import (
    add_common_args,
    ensure_dir,
    get_logger,
    resolve_common_args,
    save_plot,
)
from llm_text_detectors import Detectors


def telescope_perplexity(encoding, logits, median=False, temperature=1.0):
    return Detectors._compute_telescope_perplexity(None, encoding, logits, median, temperature)


def get_token_probs(model, tokenizer, text: str):
    enc = tokenizer(text, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(**enc)
    probs = torch.softmax(out.logits, dim=-1)
    return probs, enc, out.logits


def analyze_distribution_uniformity(human, ai, model, tokenizer) -> Tuple[float, float]:
    def avg_entropy(texts):
        e = []
        for t in texts:
            probs, _, _ = get_token_probs(model, tokenizer, t)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            e.append(entropy.mean().item())
        return float(np.mean(e))
    return avg_entropy(human), avg_entropy(ai)


def analyze_next_token_bump(human, ai, model, tokenizer) -> Tuple[float, float]:
    def avg_next(texts):
        vals = []
        for t in texts:
            probs, enc, _ = get_token_probs(model, tokenizer, t)
            next_tokens = enc.input_ids[0, 1:]
            next_prob = probs[0, :-1, next_tokens]
            vals.append(next_prob.mean().item())
        return float(np.mean(vals))
    return avg_next(human), avg_next(ai)


def analyze_probability_ratios(human, ai, model, tokenizer) -> dict:
    def stats(texts):
        r_all, r_no_peak = [], []
        for t in texts:
            probs, enc, _ = get_token_probs(model, tokenizer, t)
            cur = enc.input_ids[0, :-1]
            for pos in range(len(cur)):
                next_probs = probs[0, pos + 1]
                tok = cur[pos]
                p_tok = next_probs[tok].item()
                r_all.append(p_tok / torch.mean(next_probs).item())
                top_prob, _ = torch.max(next_probs, dim=0)
                sum_no_peak = torch.sum(next_probs) - top_prob
                avg_no_peak = (sum_no_peak / (len(next_probs) - 1)).item()
                r_no_peak.append(p_tok / avg_no_peak)
        return {"with_peak": float(np.mean(r_all)), "without_peak": float(np.mean(r_no_peak))}
    return {"human": stats(human), "ai": stats(ai)}


def analyze_peak_exclusion(human, ai, model, tokenizer) -> dict:
    def stats(texts):
        ratios = []
        for t in texts:
            probs, enc, _ = get_token_probs(model, tokenizer, t)
            cur = enc.input_ids[0, :-1]
            for pos in range(len(cur)):
                next_probs = probs[0, pos + 1]
                tok = cur[pos]
                p_tok = next_probs[tok].item()
                top_prob, _ = torch.max(next_probs, dim=0)
                sum_no_peak = torch.sum(next_probs) - top_prob
                avg_no_peak = (sum_no_peak / (len(next_probs) - 1)).item()
                ratios.append(min(p_tok / avg_no_peak, 1e6))
        return {
            "mean_ratio": float(np.mean(ratios)),
            "median_ratio": float(np.median(ratios)),
            "std_ratio": float(np.std(ratios)),
            "max_ratio": float(np.max(ratios)),
            "min_ratio": float(np.min(ratios)),
        }

    def score(s):
        return float(np.log1p(s["mean_ratio"]) * np.log1p(s["max_ratio"]) * (1 + s["std_ratio"]))

    h = stats(human)
    a = stats(ai)
    return {
        "human": {"stats": h, "detection_score": score(h)},
        "ai": {"stats": a, "detection_score": score(a)},
    }


def calculate_perplexities(human, ai, model, tokenizer) -> Tuple[float, float]:
    def avg(texts):
        p = []
        for t in texts:
            _, enc, logits = get_token_probs(model, tokenizer, t)
            p.extend(telescope_perplexity(enc, logits))
        return float(np.mean(p))
    return avg(human), avg(ai)


def visualize_summary(human_entropy, ai_entropy, human_next, ai_next, human_ppl, ai_ppl, peak_results, out_dir: Path):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    sns.barplot(x=["Human", "AI"], y=[human_entropy, ai_entropy], ax=ax1)
    ax1.set_title("Distribution Uniformity (Entropy)")
    sns.barplot(x=["Human", "AI"], y=[human_next, ai_next], ax=ax2)
    ax2.set_title("Next Token Probability")
    sns.barplot(x=["Human", "AI"], y=[human_ppl, ai_ppl], ax=ax3)
    ax3.set_title("Telescope Perplexity")
    sns.barplot(
        x=["Human", "AI"],
        y=[peak_results["human"]["detection_score"], peak_results["ai"]["detection_score"]],
        ax=ax4,
    )
    ax4.set_title("Peak Exclusion Detection Score")
    ax4.set_yscale("log")
    fig.tight_layout()
    save_plot(fig, "all_results.png", out_dir)


def visualize_peak_exclusion(results, out_dir: Path):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    for ax, key, title in [
        (ax1, "mean_ratio", "Mean Peak Exclusion Ratio"),
        (ax2, "std_ratio", "Std Deviation of Ratios"),
        (ax3, "max_ratio", "Max Ratio"),
    ]:
        sns.barplot(
            x=["Human", "AI"],
            y=[results["human"]["stats"][key], results["ai"]["stats"][key]],
            ax=ax,
        )
        ax.set_title(title)
        ax.set_yscale("log")
    sns.barplot(
        x=["Human", "AI"],
        y=[results["human"]["detection_score"], results["ai"]["detection_score"]],
        ax=ax4,
    )
    ax4.set_title("Overall Detection Score")
    ax4.set_yscale("log")
    fig.tight_layout()
    save_plot(fig, "peak_exclusion_analysis.png", out_dir)


def load_texts(path: Path) -> List[str]:
    """Load one-text-per-record JSON (list of strings) or use the provided text as-is."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict) and "texts" in data:
        return [str(x) for x in data["texts"]]
    raise ValueError(f"Unrecognized text-file format for {path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM-360M-Instruct")
    p.add_argument(
        "--human-texts",
        type=str,
        default=None,
        help="Path to JSON with a list of human text strings (or a dict with 'texts' key).",
    )
    p.add_argument(
        "--ai-texts",
        type=str,
        default=None,
        help="Path to JSON with a list of AI-generated text strings.",
    )
    p.add_argument(
        "--use-demo-texts",
        action="store_true",
        help="Use the built-in single-sample human/AI text pair (for smoke testing).",
    )
    add_common_args(p, "single_token_distribution")
    return p


_DEMO_HUMAN = [
    "This is a placeholder human-authored sentence used for smoke-testing the pipeline. "
    "Provide --human-texts / --ai-texts for real analyses."
]
_DEMO_AI = [
    "This is a placeholder AI-generated sentence used for smoke-testing the pipeline. "
    "Provide --human-texts / --ai-texts for real analyses."
]


def main() -> None:
    args = build_parser().parse_args()
    out_dir = resolve_common_args(args, "single_token_distribution")
    logger = get_logger("token_distribution", args.log_level)

    if args.use_demo_texts:
        human, ai = _DEMO_HUMAN, _DEMO_AI
    else:
        if not args.human_texts or not args.ai_texts:
            raise SystemExit("Provide --human-texts and --ai-texts (or --use-demo-texts).")
        human = load_texts(Path(args.human_texts))
        ai = load_texts(Path(args.ai_texts))

    if args.quick:
        human = human[:2]
        ai = ai[:2]

    logger.info(f"Loading model {args.model}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model)

    human_ent, ai_ent = analyze_distribution_uniformity(human, ai, model, tokenizer)
    human_next, ai_next = analyze_next_token_bump(human, ai, model, tokenizer)
    human_ppl, ai_ppl = calculate_perplexities(human, ai, model, tokenizer)
    ratios = analyze_probability_ratios(human, ai, model, tokenizer)
    peak_results = analyze_peak_exclusion(human, ai, model, tokenizer)

    results = {
        "entropy": {"human": human_ent, "ai": ai_ent},
        "next_token_prob": {"human": human_next, "ai": ai_next},
        "telescope_perplexity": {"human": human_ppl, "ai": ai_ppl},
        "probability_ratios": ratios,
        "peak_exclusion": peak_results,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    for section, payload in results.items():
        logger.info(f"{section}: {payload}")

    visualize_summary(human_ent, ai_ent, human_next, ai_next, human_ppl, ai_ppl, peak_results, out_dir)
    visualize_peak_exclusion(peak_results, out_dir)
    logger.info(f"Outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
