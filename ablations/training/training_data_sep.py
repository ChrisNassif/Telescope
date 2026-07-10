#!/usr/bin/env python3
"""
Dataset Separability Analysis Tool
Analyzes the separability of two datasets using perplexity, cross-perplexity, and Binoculars scores.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import logging
from typing import List, Tuple, Dict, Any
from llm_text_detectors import Detectors

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Loss functions
ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
softmax_fn = torch.nn.Softmax(dim=-1)


def compute_perplexity(
    encoding: transformers.BatchEncoding,
    logits: torch.Tensor,
    median: bool = False,
    temperature: float = 1.0
) -> np.ndarray:
    """Calculate perplexity scores for each sample in a batch."""
    return Detectors._compute_perplexity(None, encoding, logits, median, temperature)


def compute_cross_perplexity(
    performer_encoding: transformers.BatchEncoding,
    performer_logits: torch.Tensor,
    observer_model: transformers.PreTrainedModel,
    device: torch.device,
    pad_token_id: int,
    median: bool = False,
    temperature: float = 1.0
) -> np.ndarray:
    """Calculate cross-perplexity between a performer and observer model."""
    with torch.no_grad():
        observer_outputs = observer_model(**performer_encoding)
        observer_logits = observer_outputs.logits

    return Detectors._compute_entropy(
        None,
        observer_logits.to(device),
        performer_logits.to(device),
        performer_encoding.to(device),
        pad_token_id,
        median=median,
        temperature=temperature
    )


def compute_binoculars_score(perplexity: np.ndarray, cross_perplexity: np.ndarray) -> np.ndarray:
    """Calculate the Binoculars score (perplexity / cross_perplexity)."""
    cross_perplexity = np.maximum(cross_perplexity, 1e-10)
    return perplexity / cross_perplexity


def sample_from_dataset(dataset, n_samples: int, text_fields: List[str]) -> List[str]:
    """Sample texts from a streaming dataset."""
    samples = []
    for i, item in enumerate(dataset):
        if i >= n_samples:
            break
        
        text = None
        for field in text_fields:
            if field in item:
                text = item[field]
                break
        
        if text and len(text.strip()) > 100:
            samples.append(text)
    
    return samples


def process_dataset_batch(
    texts: List[str],
    performer_model: transformers.PreTrainedModel,
    observer_model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    device: torch.device,
    max_length: int,
    batch_size: int,
    median: bool,
    temperature: float
) -> Tuple[List[float], List[float], List[float]]:
    """Process a dataset and compute all metrics."""
    perplexities = []
    cross_perplexities = []
    binoculars_scores = []
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:min(i+batch_size, len(texts))]
        encodings = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True,
            max_length=max_length, 
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = performer_model(**encodings)

        batch_ppl = compute_perplexity(encodings, outputs.logits, median=median, temperature=temperature)
        batch_cross_ppl = compute_cross_perplexity(
            encodings, outputs.logits, observer_model, device, tokenizer.pad_token_id, median=median, temperature=temperature
        )
        batch_bino = compute_binoculars_score(batch_ppl, batch_cross_ppl)

        perplexities.extend(batch_ppl)
        cross_perplexities.extend(batch_cross_ppl)
        binoculars_scores.extend(batch_bino)
        
        torch.cuda.empty_cache()
    
    return perplexities, cross_perplexities, binoculars_scores


def compute_classification_metrics(y_true: np.ndarray, y_scores: np.ndarray, metric_name: str) -> Dict[str, Any]:
    """Compute classification metrics for a given score."""
    auc = roc_auc_score(y_true, y_scores)
    threshold = np.median(y_scores)
    y_pred = (y_scores > threshold).astype(int)
    
    metrics = {
        "auc": auc,
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }
    
    logger.info(f"\n{metric_name} Results:")
    logger.info(f"AUC: {metrics['auc']:.4f}")
    logger.info(f"F1 Score: {metrics['f1']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    
    return metrics


def create_visualizations(
    dataset_a_scores: Dict[str, List[float]],
    dataset_b_scores: Dict[str, List[float]],
    output_dir: str
) -> None:
    """Create and save visualization plots."""
    # Distribution plots
    plt.figure(figsize=(18, 6))
    
    metrics = ["perplexity", "cross_perplexity", "binoculars"]
    titles = ["Perplexity Scores", "Cross-Perplexity Scores", "Binoculars Scores"]
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        plt.subplot(1, 3, i + 1)
        plt.hist(dataset_a_scores[metric], bins=50, alpha=0.5, label='Dataset A', density=True)
        plt.hist(dataset_b_scores[metric], bins=50, alpha=0.5, label='Dataset B', density=True)
        plt.xlabel(title.replace(" Scores", ""))
        plt.ylabel('Density')
        plt.legend()
        plt.title(f'Distribution of {title}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'score_distributions.png'), dpi=300)
    plt.close()
    
    # Scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(dataset_a_scores["perplexity"], dataset_a_scores["cross_perplexity"], 
                alpha=0.5, label='Dataset A', s=30)
    plt.scatter(dataset_b_scores["perplexity"], dataset_b_scores["cross_perplexity"], 
                alpha=0.5, label='Dataset B', s=30)
    plt.xlabel('Perplexity')
    plt.ylabel('Cross-Perplexity')
    plt.legend()
    plt.title('Perplexity vs Cross-Perplexity')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'perplexity_scatter.png'), dpi=300)
    plt.close()
    
    logger.info(f"Visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Analyze separability between two datasets')
    
    # Model arguments
    parser.add_argument('--performer-model', type=str, required=True,
                        help='Name or path of the performer model')
    parser.add_argument('--observer-model', type=str, required=True,
                        help='Name or path of the observer model')
    
    # Dataset arguments
    parser.add_argument('--dataset-a', type=str, required=True,
                        help='Name of the first dataset')
    parser.add_argument('--dataset-a-config', type=str, default=None,
                        help='Configuration of the first dataset')
    parser.add_argument('--dataset-b', type=str, required=True,
                        help='Name of the second dataset')
    parser.add_argument('--dataset-b-config', type=str, default=None,
                        help='Configuration of the second dataset')
    
    # Processing arguments
    parser.add_argument('--sample-size', type=int, default=1000,
                        help='Number of samples from each dataset')
    parser.add_argument('--max-length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for processing')
    parser.add_argument('--median', action='store_true',
                        help='Use median instead of mean for perplexity')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for softmax')
    parser.add_argument('--text-fields', type=str, nargs='+', 
                        default=['text', 'content', 'document'],
                        help='Field names to look for text in datasets')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='./separability_results',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
    
    # Load models
    logger.info(f"Loading performer model: {args.performer_model}")
    performer_tokenizer = AutoTokenizer.from_pretrained(args.performer_model)
    performer_model = AutoModelForCausalLM.from_pretrained(args.performer_model).to(device)
    
    logger.info(f"Loading observer model: {args.observer_model}")
    observer_model = AutoModelForCausalLM.from_pretrained(args.observer_model).to(device)
    
    # Fix padding tokens
    if performer_tokenizer.pad_token is None:
        performer_tokenizer.pad_token = performer_tokenizer.eos_token
        performer_model.config.pad_token_id = performer_model.config.eos_token_id
    
    # Load datasets
    logger.info(f"Loading dataset A: {args.dataset_a}/{args.dataset_a_config}")
    dataset_a = load_dataset(args.dataset_a, args.dataset_a_config, split="train", streaming=True)
    
    logger.info(f"Loading dataset B: {args.dataset_b}/{args.dataset_b_config}")
    dataset_b = load_dataset(args.dataset_b, args.dataset_b_config, split="train", streaming=True)
    
    # Sample from datasets
    logger.info(f"Sampling {args.sample_size} examples from dataset A...")
    dataset_a_samples = sample_from_dataset(dataset_a, args.sample_size, args.text_fields)
    
    logger.info(f"Sampling {args.sample_size} examples from dataset B...")
    dataset_b_samples = sample_from_dataset(dataset_b, args.sample_size, args.text_fields)
    
    logger.info(f"Collected {len(dataset_a_samples)} samples from dataset A "
                f"and {len(dataset_b_samples)} samples from dataset B")
    
    # Process dataset A
    logger.info("Processing dataset A samples...")
    a_ppl, a_cross_ppl, a_bino = process_dataset_batch(
        dataset_a_samples, performer_model, observer_model, performer_tokenizer,
        device, args.max_length, args.batch_size, args.median, args.temperature
    )
    
    # Process dataset B
    logger.info("Processing dataset B samples...")
    b_ppl, b_cross_ppl, b_bino = process_dataset_batch(
        dataset_b_samples, performer_model, observer_model, performer_tokenizer,
        device, args.max_length, args.batch_size, args.median, args.temperature
    )
    
    # Prepare data for metrics
    dataset_a_scores = {
        "perplexity": a_ppl,
        "cross_perplexity": a_cross_ppl,
        "binoculars": a_bino
    }
    
    dataset_b_scores = {
        "perplexity": b_ppl,
        "cross_perplexity": b_cross_ppl,
        "binoculars": b_bino
    }
    
    # Create labels
    y_true = np.concatenate([np.zeros(len(a_ppl)), np.ones(len(b_ppl))])
    
    # Compute metrics for each score type
    results = {
        "dataset_info": {
            "dataset_a": args.dataset_a,
            "dataset_a_config": args.dataset_a_config,
            "dataset_a_samples": len(dataset_a_samples),
            "dataset_b": args.dataset_b,
            "dataset_b_config": args.dataset_b_config,
            "dataset_b_samples": len(dataset_b_samples)
        },
        "model_info": {
            "performer_model": args.performer_model,
            "observer_model": args.observer_model
        },
        "parameters": {
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "median": args.median,
            "temperature": args.temperature
        }
    }
    
    # Compute metrics for each score type
    for metric_name, score_key in [("Perplexity", "perplexity"), 
                                   ("Cross-Perplexity", "cross_perplexity"), 
                                   ("Binoculars", "binoculars")]:
        y_scores = np.concatenate([dataset_a_scores[score_key], dataset_b_scores[score_key]])
        metrics = compute_classification_metrics(y_true, y_scores, metric_name)
        results[f"{score_key}_metrics"] = metrics
    
    # Save raw scores
    results["dataset_a_scores"] = {k: [float(v) for v in vals] for k, vals in dataset_a_scores.items()}
    results["dataset_b_scores"] = {k: [float(v) for v in vals] for k, vals in dataset_b_scores.items()}
    
    # Save results
    results_file = os.path.join(args.output_dir, 'separability_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_file}")
    
    # Create visualizations
    create_visualizations(dataset_a_scores, dataset_b_scores, args.output_dir)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY OF RESULTS")
    logger.info("="*60)
    for metric in ["perplexity", "cross_perplexity", "binoculars"]:
        auc = results[f"{metric}_metrics"]["auc"]
        f1 = results[f"{metric}_metrics"]["f1"]
        logger.info(f"{metric.upper()}: AUC={auc:.4f}, F1={f1:.4f}")
    
    logger.info("\nAnalysis complete!")


if __name__ == "__main__":
    main()