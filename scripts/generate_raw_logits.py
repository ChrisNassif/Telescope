import sys
from typing import Any, Dict, List, Optional, Tuple, Set
argv: List[str] = sys.argv[1:]
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from llm_text_detectors import Detectors


### START GLOBALS -------------------------------------------------------------------------

# Model performer/observer pairs to extract logits for
MODEL_PERFORMER_OBSERVER_PAIRS_TO_TEST: Dict[str, Tuple[str, str]] = {
    # "smollm_135M": ("HuggingFaceTB/SmolLM-135M-Instruct", "HuggingFaceTB/SmolLM-135M"),
    "smollm2_135M": ("HuggingFaceTB/SmolLM2-135M-Instruct", "HuggingFaceTB/SmolLM2-135M"),
    # "smollm_360M": ("HuggingFaceTB/SmolLM-360M-Instruct", "HuggingFaceTB/SmolLM-360M"),
    # "smollm2_360M": ("HuggingFaceTB/SmolLM2-360M-Instruct", "HuggingFaceTB/SmolLM2-360M"),
    # "smollm_1_7B": ("HuggingFaceTB/SmolLM-1.7B-Instruct", "HuggingFaceTB/SmolLM-1.7B"),
    # "smollm2_1_7B": ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "HuggingFaceTB/SmolLM2-1.7B"),
    # "falcon_7B": ("tiiuae/falcon-7b-instruct", "tiiuae/falcon-7b"),
    # "gemma2_2B": ("google/gemma-2-2b-it", "google/gemma-2-2b"),
    # "llama3_8B": ("meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B"),
    # "gemma2_9B": ("google/gemma-2-9b-it", "google/gemma-2-9b"),
    # "gpt_neo_2_7B": ("EleutherAI/gpt-neo-2.7B", "EleutherAI/gpt-neo-2.7B"),
    # "gpt_j_6B": ("EleutherAI/gpt-j-6b", "EleutherAI/gpt-j-6b"),
}

# Dataset file in the datasets folder (can also be passed as argv[0])
DATASET_FILE: str = argv[0] if len(argv) > 0 else "ESL_GPT4o_Dataset.csv"
DATASET_FOLDER: str = "datasets"

MAX_NUMBER_OF_SAMPLES: int = 1000
OUTPUT_LOGITS_FOLDER: str = "saved_logits"
DEVICE: str = "cuda:0" if torch.cuda.is_available() else "cpu"

### END GLOBALS ---------------------------------------------------------------------------


def load_dataset(dataset_path: str) -> pd.DataFrame:
    """Loads dataset CSV and returns DataFrame with normalized text and label columns."""
    if not os.path.exists(dataset_path):
        alt_path = os.path.join(DATASET_FOLDER, dataset_path)
        if os.path.exists(alt_path):
            dataset_path = alt_path
        else:
            raise FileNotFoundError(f"Dataset CSV not found at {dataset_path}")

    df = pd.read_csv(dataset_path)

    # Resolve text column
    text_col = None
    for col in ["text", "original_texts", "prompt"]:
        if col in df.columns:
            text_col = col
            break
    if text_col is None:
        text_col = df.columns[0]

    # Resolve label column
    label_col = None
    for col in ["generated", "y_labels", "label"]:
        if col in df.columns:
            label_col = col
            break
    if label_col is None:
        label_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    df = df.dropna(subset=[text_col, label_col]).reset_index(drop=True)
    df["clean_text"] = df[text_col].astype(str)
    df["clean_label"] = df[label_col].astype(int)
    return df


@torch.inference_mode()
def compute_16_metrics_for_sample(
    detectors: Detectors, text: str, device: str = "cpu"
) -> Dict[str, np.ndarray]:
    """Computes 16 per-token metric streams for a single text sample."""
    performer_logits, observer_logits, text_encodings = detectors._compute_logits(
        text, detectors.performer_model, detectors.observer_model, detectors.performer_tokenizer, device=device
    )

    performer_logits = performer_logits.to(torch.float32)
    observer_logits = observer_logits.to(torch.float32)

    input_ids = text_encodings["input_ids"]
    if input_ids.ndim > 1:
        input_ids = input_ids.squeeze(0)

    target_ids = input_ids[1:]
    num_tokens = target_ids.size(0)

    if num_tokens <= 0:
        return {}

    L_shifted = performer_logits[:-1]
    probs = torch.softmax(L_shifted, dim=-1)
    lprobs = torch.log_softmax(L_shifted, dim=-1)

    target_ids_exp = target_ids.unsqueeze(-1)
    token_probs = probs.gather(-1, target_ids_exp).squeeze(-1)
    log_prob = torch.log(token_probs.clamp(min=1e-10))

    ranks = (probs > token_probs.unsqueeze(-1)).sum(dim=-1) + 1
    log_rank = torch.log(ranks.float())

    # Shifted metrics for k = 1, 2, 3
    shifted_results: Dict[str, np.ndarray] = {}
    for k in [1, 2, 3]:
        if k == 1:
            s_ids = input_ids[:-1]
        else:
            prefix = input_ids[: k - 1]
            suffix = input_ids[:-k]
            s_ids = torch.cat([prefix, suffix])

        s_ids_exp = s_ids.unsqueeze(-1)
        s_token_probs = probs.gather(-1, s_ids_exp).squeeze(-1)
        s_log_prob = torch.log(s_token_probs.clamp(min=1e-10))
        s_ranks = (probs > s_token_probs.unsqueeze(-1)).sum(dim=-1) + 1
        s_log_rank = torch.log(s_ranks.float())

        shifted_results[f"shift{k}_log_prob"] = s_log_prob.cpu().numpy().astype(np.float16)
        shifted_results[f"shift{k}_rank"] = s_ranks.cpu().numpy().astype(np.float16)
        shifted_results[f"shift{k}_log_rank"] = s_log_rank.cpu().numpy().astype(np.float16)

    entropy = -torch.sum(probs * torch.log2(probs.clamp(min=1e-10)), dim=-1)

    mean_ref = torch.sum(probs * lprobs, dim=-1)
    var_ref = torch.sum(probs * torch.square(lprobs), dim=-1) - torch.square(mean_ref)

    topk_probs, _ = torch.topk(probs, k=min(10, probs.size(-1)), dim=-1)
    max_prob = topk_probs[:, 0]
    top2_prob = topk_probs[:, 1] if topk_probs.size(-1) > 1 else max_prob
    margin_prob = max_prob - top2_prob

    max_logit = torch.max(L_shifted, dim=-1).values
    target_logit = L_shifted.gather(-1, target_ids_exp).squeeze(-1)
    target_logit_diff = max_logit - target_logit

    top5_mass = torch.sum(topk_probs[:, : min(5, topk_probs.size(-1))], dim=-1)
    top10_mass = torch.sum(topk_probs[:, : min(10, topk_probs.size(-1))], dim=-1)

    actual_probs = torch.zeros_like(probs)
    actual_probs[torch.arange(num_tokens, device=device), target_ids] = 1.0
    tv_dist = 0.5 * torch.sum(torch.abs(probs - actual_probs), dim=-1)

    if detectors.performer_model != detectors.observer_model:
        probs_obs = torch.softmax(observer_logits[:-1], dim=-1)
        cross_loss = -torch.sum(probs_obs * lprobs, dim=-1)
    else:
        cross_loss = -mean_ref

    zero_vec = torch.tensor([0.0], device=device)
    if num_tokens > 1:
        d_log_prob = torch.cat([zero_vec, log_prob[1:] - log_prob[:-1]])
        d_entropy = torch.cat([zero_vec, entropy[1:] - entropy[:-1]])
    else:
        d_log_prob = zero_vec
        d_entropy = zero_vec

    out_dict = {
        "log_prob": log_prob.cpu().numpy().astype(np.float16),
        "rank": ranks.cpu().numpy().astype(np.float16),
        "log_rank": log_rank.cpu().numpy().astype(np.float16),
        "entropy": entropy.cpu().numpy().astype(np.float16),
        "mean_ref": mean_ref.cpu().numpy().astype(np.float16),
        "var_ref": var_ref.cpu().numpy().astype(np.float16),
        "max_prob": max_prob.cpu().numpy().astype(np.float16),
        "top2_prob": top2_prob.cpu().numpy().astype(np.float16),
        "margin_prob": margin_prob.cpu().numpy().astype(np.float16),
        "target_logit_diff": target_logit_diff.cpu().numpy().astype(np.float16),
        "top5_mass": top5_mass.cpu().numpy().astype(np.float16),
        "top10_mass": top10_mass.cpu().numpy().astype(np.float16),
        "tv_dist": tv_dist.cpu().numpy().astype(np.float16),
        "cross_loss": cross_loss.cpu().numpy().astype(np.float16),
        "d_log_prob": d_log_prob.cpu().numpy().astype(np.float16),
        "d_entropy": d_entropy.cpu().numpy().astype(np.float16),
    }
    out_dict.update(shifted_results)
    return out_dict


def main():
    print(f"Loading dataset: {DATASET_FILE}")
    df = load_dataset(DATASET_FILE)
    if MAX_NUMBER_OF_SAMPLES > 0 and len(df) > MAX_NUMBER_OF_SAMPLES:
        df = df.iloc[:MAX_NUMBER_OF_SAMPLES].reset_index(drop=True)
    print(f"Loaded {len(df)} samples.")

    os.makedirs(OUTPUT_LOGITS_FOLDER, exist_ok=True)
    dataset_name = os.path.basename(DATASET_FILE).lower().replace(".csv", "")

    for model_codename, (performer_repo, observer_repo) in MODEL_PERFORMER_OBSERVER_PAIRS_TO_TEST.items():
        print(f"\n=======================================================")
        print(f"Processing model: {model_codename} ({performer_repo})")
        print(f"=======================================================")

        detectors = Detectors(
            performer_model_huggingface_name=performer_repo,
            observer_model_huggingface_name=observer_repo,
            device=DEVICE,
        )

        metric_streams: Dict[str, List[np.ndarray]] = {m: [] for m in [
            "log_prob", "shift1_log_prob", "shift2_log_prob", "shift3_log_prob",
            "rank", "shift1_rank", "shift2_rank", "shift3_rank",
            "log_rank", "shift1_log_rank", "shift2_log_rank", "shift3_log_rank",
            "entropy", "mean_ref", "var_ref", "max_prob", "top2_prob", "margin_prob",
            "target_logit_diff", "top5_mass", "top10_mass", "tv_dist", "cross_loss",
            "d_log_prob", "d_entropy"
        ]}
        sample_offsets: List[int] = []
        sample_lengths: List[int] = []
        y_labels: List[int] = []
        current_offset = 0

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            text = str(row["clean_text"])
            label = int(row["clean_label"])

            res = compute_16_metrics_for_sample(detectors, text, device=DEVICE)
            if not res or len(res["log_prob"]) == 0:
                continue

            seq_len = len(res["log_prob"])
            sample_offsets.append(current_offset)
            sample_lengths.append(seq_len)
            current_offset += seq_len
            y_labels.append(label)

            for m_name in metric_streams.keys():
                metric_streams[m_name].append(res[m_name])

        if len(sample_offsets) == 0:
            print(f"No samples processed for {model_codename}.")
            continue

        output_filename = f"{model_codename}_{dataset_name}.npz"
        output_path = os.path.join(OUTPUT_LOGITS_FOLDER, output_filename)

        save_kwargs = {
            "y_labels": np.array(y_labels, dtype=np.uint8),
            "sample_offsets": np.array(sample_offsets, dtype=np.int32),
            "sample_lengths": np.array(sample_lengths, dtype=np.int32),
        }

        for m_name, arrays in metric_streams.items():
            concatenated = np.concatenate(arrays) if len(arrays) > 0 else np.array([], dtype=np.float16)
            save_kwargs[m_name] = concatenated

        np.savez_compressed(output_path, **save_kwargs)
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Successfully saved compressed logits to {output_path} ({file_size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
