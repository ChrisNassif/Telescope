#!/usr/bin/env python3
"""Compare Telescope metrics across sampling strategies for a generator model.

For each sampling config (greedy, beam, temperature, nucleus), generate N samples,
score them with a Telescope Detectors instance, and report mean/std metrics.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ablations._common import add_common_args, get_logger, load_detector, resolve_common_args


SAMPLING_CONFIGS: Dict[str, Dict] = {
    "greedy": {"do_sample": False, "num_beams": 1},
    "beam_search_2": {"do_sample": False, "num_beams": 2},
    "beam_search_5": {"do_sample": False, "num_beams": 5},
    "temperature_0.7": {"do_sample": True, "temperature": 0.7},
    "temperature_1.0": {"do_sample": True, "temperature": 1.0},
    "temperature_1.5": {"do_sample": True, "temperature": 1.5},
    "nucleus_0.9": {"do_sample": True, "temperature": 1.0, "top_p": 0.9},
    "nucleus_0.7": {"do_sample": True, "temperature": 1.0, "top_p": 0.7},
}


def generate_with_sampling(
    model, tokenizer, prompt: str, max_length: int, num_samples: int, params: Dict
) -> List[str]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen_params = {
        "max_length": max_length,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    do_sample = params.get("do_sample", False)
    if do_sample:
        for k in ("temperature", "top_p"):
            if k in params:
                gen_params[k] = params[k]
    gen_params["do_sample"] = do_sample
    if "num_beams" in params:
        gen_params["num_beams"] = params["num_beams"]

    if gen_params.get("num_beams", 1) > 1:
        gen_params["num_return_sequences"] = min(num_samples, gen_params["num_beams"])
    else:
        gen_params["num_return_sequences"] = num_samples if do_sample else 1

    try:
        outputs = model.generate(**inputs, **gen_params)
        texts = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
        while len(texts) < num_samples:
            texts.extend(texts[: num_samples - len(texts)])
        return texts[:num_samples]
    except Exception as e:
        print(f"Generation error: {e}")
        return [prompt] * num_samples


def process_metrics(metrics: Dict) -> Dict:
    out = {}
    for k, v in metrics.items():
        if isinstance(v, (torch.Tensor, np.ndarray)):
            out[k] = float(v.item()) if getattr(v, "ndim", 0) > 0 else float(v)
        else:
            out[k] = v
    return out


def analyze_sampling(
    telescope,
    prompt: str,
    generator_model_name: str,
    auth_token: str,
    configs: Dict[str, Dict],
    num_samples: int,
    max_length: int,
    quantize_4bit: bool,
    logger,
) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(generator_model_name, token=auth_token, trust_remote_code=True)
    kwargs = {"token": auth_token, "trust_remote_code": True, "device_map": "auto", "torch_dtype": torch.float16}
    if quantize_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(generator_model_name, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = []
    for name, params in configs.items():
        logger.info(f"Testing {name}")
        samples = generate_with_sampling(model, tokenizer, prompt, max_length, num_samples, params)
        method_rows = []
        for sample in samples:
            try:
                ppl, xppl, extras = telescope.compute_all_metrics(sample)
                method_rows.append(
                    {
                        "method": name,
                        "telescope_perplexity": float(np.asarray(ppl).item()),
                        "telescope_cross_perplexity": float(np.asarray(xppl).item()),
                        "telescope_perplexity_divided_by_cross_perplexity": float(
                            np.asarray(ppl / xppl).item()
                        ),
                        **process_metrics(extras),
                    }
                )
            except Exception as e:
                logger.warning(f"metric error for {name}: {e}")
                continue

        if method_rows:
            mdf = pd.DataFrame(method_rows).replace([np.inf, -np.inf], np.nan)
            mean = mdf.mean(numeric_only=True, skipna=True)
            std = mdf.std(numeric_only=True, skipna=True)
            rows.append(
                {
                    "method": name,
                    "mean_telescope_perplexity_divided_by_cross_perplexity": float(
                        mean["telescope_perplexity_divided_by_cross_perplexity"]
                    ),
                    "std_telescope_perplexity_divided_by_cross_perplexity": float(
                        std["telescope_perplexity_divided_by_cross_perplexity"]
                    ),
                    "mean_perplexity": float(mean["telescope_perplexity"]),
                    "std_perplexity": float(std["telescope_perplexity"]),
                    "mean_cross_perplexity": float(mean["telescope_cross_perplexity"]),
                    "std_cross_perplexity": float(std["telescope_cross_perplexity"]),
                    "mean_entropy_ratio": float(mean.get("entropy_ratio", 0)),
                    "mean_kl_divergence": float(mean.get("kl_divergence", 0)),
                    "mean_performer_distribution_overlap": float(
                        mean.get("performer_distribution_overlap", 0)
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--observer-model", type=str, default="HuggingFaceTB/SmolLM-135M",
        help="Model used as observer by Telescope."
    )
    p.add_argument(
        "--performer-model", type=str, default="HuggingFaceTB/SmolLM-135M-instruct",
        help="Model used as performer by Telescope."
    )
    p.add_argument(
        "--generator-model", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
        help="Model whose outputs are scored across sampling strategies."
    )
    p.add_argument("--prompt", type=str, default="Write a short story about a robot learning to feel emotions:")
    p.add_argument("--num-samples", type=int, default=500)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--no-quantize", action="store_true", help="Skip 4-bit quantization of generator.")
    p.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=None,
        help=f"Subset of sampling methods to run. Default: all of {list(SAMPLING_CONFIGS.keys())}",
    )
    add_common_args(p, "sampling")
    return p


def main() -> None:
    args = build_parser().parse_args()
    out_dir = resolve_common_args(args, "sampling")
    logger = get_logger("sampling_analysis", args.log_level)

    if args.quick:
        args.num_samples = min(args.num_samples, 10)
        args.max_length = min(args.max_length, 256)

    configs = SAMPLING_CONFIGS
    if args.methods:
        configs = {k: v for k, v in SAMPLING_CONFIGS.items() if k in args.methods}

    from llm_text_detectors.utils import get_hugging_face_auth_token
    token = get_hugging_face_auth_token()

    logger.info(f"Loading Telescope (observer={args.observer_model}, performer={args.performer_model})")
    telescope = load_detector(args.observer_model, args.performer_model, token)

    df = analyze_sampling(
        telescope,
        args.prompt,
        args.generator_model,
        token,
        configs,
        args.num_samples,
        args.max_length,
        quantize_4bit=not args.no_quantize,
        logger=logger,
    )

    if df.empty:
        logger.warning("No results generated.")
        return

    out_csv = out_dir / "sampling_analysis_results.csv"
    df.to_csv(out_csv, index=False)
    with pd.option_context("display.float_format", "{:.6f}".format):
        logger.info("\n" + df.to_string())
    logger.info(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
