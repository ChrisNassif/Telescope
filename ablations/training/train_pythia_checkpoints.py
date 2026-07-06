#!/usr/bin/env python3
"""Evaluate Pythia model checkpoints at fixed step intervals.

For each checkpoint, generates samples for the supplied prompts, computes
Telescope perplexity and attention-entropy statistics, and appends the results
to a resumable CSV.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from ablations._common import add_common_args, get_logger, load_detector, resolve_common_args


PYTHIA_TOTAL_STEPS = 143000


def compute_attention_entropy(w: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    w = (w + eps) / (w + eps).sum(dim=-1, keepdim=True)
    return -torch.sum(w * torch.log2(w), dim=-1)


def compute_attention_stats(model, input_ids, attention_mask=None) -> Dict:
    model.eval()
    with torch.no_grad():
        out = model(input_ids, attention_mask=attention_mask, output_attentions=True)
    per_layer = []
    for i, la in enumerate(out.attentions):
        ent = compute_attention_entropy(la)
        max_a = la.max(dim=-1)[0]
        sparsity = (la < 0.1).float().mean()
        per_layer.append(
            {
                "layer": i,
                "mean_entropy": ent.mean().item(),
                "max_entropy": ent.max().item(),
                "min_entropy": ent.min().item(),
                "mean_max_attention": max_a.mean().item(),
                "attention_sparsity": sparsity.item(),
            }
        )
    return {
        "avg_attention_entropy": float(np.mean([s["mean_entropy"] for s in per_layer])),
        "std_attention_entropy": float(np.std([s["mean_entropy"] for s in per_layer])),
        "avg_attention_sparsity": float(np.mean([s["attention_sparsity"] for s in per_layer])),
        "avg_max_attention": float(np.mean([s["mean_max_attention"] for s in per_layer])),
        "layer_stats": per_layer,
    }


def get_evaluation_steps(interval: int, start: int, end: int) -> List[int]:
    steps = list(range(start, min(end + 1, PYTHIA_TOTAL_STEPS + 1), interval))
    if steps and steps[-1] != PYTHIA_TOTAL_STEPS and PYTHIA_TOTAL_STEPS <= end:
        steps.append(PYTHIA_TOTAL_STEPS)
    return steps


def verify_checkpoint_results(output_dir: Path, step: int) -> bool:
    r = output_dir / f"results_step{step}.json"
    s = output_dir / f"samples_step{step}.json"
    if not (r.exists() and s.exists()):
        return False
    try:
        rd = json.loads(r.read_text())
        sd = json.loads(s.read_text())
        for f in ("sample_size", "attention_entropy", "telescope_perplexity"):
            if f not in rd:
                return False
        for f in ("prompts", "generated_samples", "attention_statistics"):
            if f not in sd:
                return False
        return len(sd["prompts"]) == len(sd["generated_samples"])
    except Exception:
        return False


def generate_samples(
    model_name: str,
    step: int,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
    logger,
) -> Tuple[List[str], List[Dict]]:
    with tempfile.TemporaryDirectory() as tmp:
        model = GPTNeoXForCausalLM.from_pretrained(
            model_name, revision=f"step{step}", cache_dir=tmp, torch_dtype=torch.float16
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=f"step{step}", cache_dir=tmp)
        texts, stats = [], []
        for prompt in prompts:
            inp = tokenizer(prompt, return_tensors="pt").to(device)
            stats.append(compute_attention_stats(model, inp.input_ids, inp.attention_mask))
            outputs = model.generate(
                **inp,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
            )
            texts.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
        del model, tokenizer
        if device == "cuda":
            torch.cuda.empty_cache()
        return texts, stats


def compute_batch_metrics(sample_texts, attention_stats, telescope, logger) -> Optional[Dict]:
    all_ppl, all_attn = [], []
    for text, stat in zip(sample_texts, attention_stats):
        try:
            ppl = float(telescope.compute_telescope_perplexity(text))
            all_ppl.append(ppl)
            all_attn.append(
                {
                    "attention_entropy": stat["avg_attention_entropy"],
                    "attention_entropy_std": stat["std_attention_entropy"],
                    "attention_sparsity": stat["avg_attention_sparsity"],
                    "max_attention": stat["avg_max_attention"],
                }
            )
        except Exception as e:
            logger.error(f"metric error: {e}")
            continue
    if not all_ppl:
        return None
    n = len(all_ppl)
    mean_ppl = float(np.mean(all_ppl))
    std_ppl = float(np.std(all_ppl))
    ci_lo = mean_ppl - 2 * std_ppl / np.sqrt(n)
    ci_hi = mean_ppl + 2 * std_ppl / np.sqrt(n)
    agg = {}
    for k in all_attn[0]:
        vals = [m[k] for m in all_attn]
        agg[k] = float(np.mean(vals))
        agg[f"{k}_std"] = float(np.std(vals))
        agg[f"{k}_ci_95_lower"] = agg[k] - 2 * agg[f"{k}_std"] / np.sqrt(len(vals))
        agg[f"{k}_ci_95_upper"] = agg[k] + 2 * agg[f"{k}_std"] / np.sqrt(len(vals))
    return {
        "sample_size": n,
        "telescope_perplexity": mean_ppl,
        "perplexity_std": std_ppl,
        "perplexity_ci_95_lower": ci_lo,
        "perplexity_ci_95_upper": ci_hi,
        **agg,
    }


def evaluate_checkpoint(model_name, step, prompts, output_dir, telescope, args, logger):
    samples_file = output_dir / f"samples_step{step}.json"
    results_file = output_dir / f"results_step{step}.json"
    csv_file = output_dir / "evaluation_results.csv"

    logger.info(f"Generating samples at step {step}")
    generated, attn_stats = generate_samples(
        model_name, step, prompts, args.max_new_tokens, args.temperature, args.top_p, args.device, logger
    )
    (samples_file.with_suffix(".json.tmp")).write_text(
        json.dumps(
            {"prompts": prompts, "generated_samples": generated, "attention_statistics": attn_stats,
             "evaluation_timestamp": time.time(), "step": step},
            indent=2,
        )
    )

    metrics = compute_batch_metrics(generated, attn_stats, telescope, logger)
    if not metrics:
        logger.warning(f"No metrics for step {step}")
        return
    metrics.update({"evaluation_timestamp": time.time(), "step": step, "model_name": model_name})
    (results_file.with_suffix(".json.tmp")).write_text(json.dumps(metrics, indent=4))
    os.rename(str(samples_file) + ".tmp", str(samples_file))
    os.rename(str(results_file) + ".tmp", str(results_file))

    csv_exists = csv_file.exists()
    with csv_file.open("a") as f:
        if not csv_exists:
            f.write(
                "step,telescope_perplexity,perplexity_ci_lower,perplexity_ci_upper,"
                "attention_entropy,attention_entropy_ci_lower,attention_entropy_ci_upper,"
                "attention_sparsity,sample_size\n"
            )
        f.write(
            f"{step},{metrics['telescope_perplexity']},{metrics['perplexity_ci_95_lower']},"
            f"{metrics['perplexity_ci_95_upper']},{metrics['attention_entropy']},"
            f"{metrics['attention_entropy_ci_95_lower']},{metrics['attention_entropy_ci_95_upper']},"
            f"{metrics['attention_sparsity']},{metrics['sample_size']}\n"
        )
    logger.info(f"Wrote {results_file}")


def plot_results(csv_file: Path, output_dir: Path, logger) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(csv_file).replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < 2:
        logger.warning("Not enough data to plot")
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    ax1.plot(df["step"], df["telescope_perplexity"], "b-", label="Telescope Perplexity")
    ax1.fill_between(df["step"], df["perplexity_ci_lower"], df["perplexity_ci_upper"], alpha=0.2)
    ax1.set_ylabel("Perplexity"); ax1.legend(); ax1.grid(True, alpha=0.7)
    ax2.plot(df["step"], df["attention_entropy"], "g-", label="Attention Entropy")
    ax2.fill_between(df["step"], df["attention_entropy_ci_lower"], df["attention_entropy_ci_upper"], alpha=0.2)
    ax2.plot(df["step"], df["attention_sparsity"], "r-", label="Attention Sparsity")
    ax2.set_xlabel("Training Steps"); ax2.legend(); ax2.grid(True, alpha=0.7)
    fig.tight_layout()
    (output_dir / "evaluation_plots.png").parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_dir / "evaluation_plots.png", dpi=300, bbox_inches="tight")
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name", type=str, required=True, help="e.g. EleutherAI/pythia-160m")
    p.add_argument("--prompts", type=str, required=True, help="File with prompts, one per line.")
    p.add_argument("--interval", type=int, default=1000)
    p.add_argument("--start-step", type=int, default=0)
    p.add_argument("--end-step", type=int, default=PYTHIA_TOTAL_STEPS)
    p.add_argument("--force-restart", action="store_true")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--telescope-model", type=str, default="HuggingFaceTB/SmolLM-360M",
                   help="Model used by the Telescope Detectors instance.")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    add_common_args(p, "training/pythia_checkpoints")
    return p


def main() -> None:
    args = build_parser().parse_args()
    out_dir = resolve_common_args(args, "training/pythia_checkpoints")
    logger = get_logger("train_pythia", args.log_level)

    if args.quick:
        args.interval = max(args.interval, 50000)
        args.max_new_tokens = min(args.max_new_tokens, 50)

    prompts_path = Path(args.prompts)
    if not prompts_path.exists():
        raise SystemExit(f"Prompts file not found: {prompts_path}")
    prompts = [ln.strip() for ln in prompts_path.read_text().splitlines() if ln.strip()]
    logger.info(f"Loaded {len(prompts)} prompts")

    steps = get_evaluation_steps(args.interval, args.start_step, args.end_step)
    logger.info(f"Will evaluate {len(steps)} checkpoints")

    csv_file = out_dir / "evaluation_results.csv"
    progress_file = out_dir / "evaluation_progress.json"
    if args.force_restart:
        for f in (csv_file, progress_file):
            if f.exists():
                f.unlink()
    else:
        completed = {s for s in steps if verify_checkpoint_results(out_dir, s)}
        steps = [s for s in steps if s not in completed]
        if completed:
            logger.info(f"Resuming ({len(completed)} already done)")

    if not steps:
        logger.info("All requested steps completed.")
        if args.plot and csv_file.exists():
            plot_results(csv_file, out_dir, logger)
        return

    logger.info(f"Loading Telescope with {args.telescope_model}")
    telescope = load_detector(args.telescope_model)

    total = len(steps)
    for i, step in enumerate(steps):
        logger.info(f"Checkpoint {i+1}/{total}: step {step}")
        try:
            evaluate_checkpoint(args.model_name, step, prompts, out_dir, telescope, args, logger)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            progress_file.write_text(
                json.dumps(
                    {"last_completed_step": step, "completed_count": i + 1, "total_steps": total,
                     "remaining_steps": steps[i + 1:], "timestamp": time.time()},
                    indent=2,
                )
            )
        except Exception as e:
            logger.error(f"Error at step {step}: {e}")
            continue

    if args.plot and csv_file.exists():
        plot_results(csv_file, out_dir, logger)
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
