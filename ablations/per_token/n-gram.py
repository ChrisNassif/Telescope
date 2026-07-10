"""
N-gram Perplexity Analysis for Human vs AI Text Detection
Compares telescope and standard perplexity metrics using different n-gram approaches.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import torch
import numpy as np
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
config_path = os.path.join(project_root, "config.yaml")
PLOT_COLORS = yaml.safe_load(open(config_path))["plot_colors"]
import json
import logging
from typing import List, Dict, Tuple, Optional
from llm_text_detectors import Detectors

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def split_into_ngrams(text: str, n: int = 2) -> List[str]:
    """Split text into n-grams (groups of n consecutive words)."""
    words = text.split()
    if len(words) < n:
        return [text]
    ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-(n-1))]
    return ngrams


def calculate_perplexity_batch(
    detector: Detectors,
    texts: List[str],
    device: torch.device,
    batch_size: int = 8,
    use_standard_ppl: bool = False,
    max_length: int = 512,
    median: bool = False,
    temperature: float = 1.0
) -> List[float]:
    """Calculate perplexity for texts in batches."""
    detector.performer_model.eval()
    results = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch_texts = texts[i:i+batch_size]
        batch_results = []

        with torch.no_grad():
            for text in batch_texts:
                if not isinstance(text, str) or not text.strip():
                    batch_results.append(float('nan'))
                    continue

                try:
                    encodings = detector.performer_tokenizer(
                        text, 
                        return_tensors="pt", 
                        truncation=True, 
                        max_length=max_length
                    ).to(device)
                    
                    outputs = detector.performer_model(**encodings)
                    logits = outputs.logits

                    if use_standard_ppl:
                        ppl = detector._compute_perplexity(encodings, logits, median=median, temperature=temperature)
                        ppl = np.exp(ppl)
                    else:
                        ppl = detector._compute_telescope_perplexity(encodings, logits, median, temperature)

                    batch_results.append(ppl[0])

                except Exception as e:
                    logger.warning(f"Error processing text: {str(e)}")
                    batch_results.append(float('nan'))

                finally:
                    if 'encodings' in locals():
                        del encodings
                    if 'outputs' in locals():
                        del outputs
                    if 'logits' in locals():
                        del logits
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        results.extend(batch_results)

    return results


def calculate_ngram_perplexity(
    detector: Detectors,
    texts: List[str],
    device: torch.device,
    n: int = 2,
    batch_size: int = 32,
    use_standard_ppl: bool = False,
    max_length: int = 512,
    median: bool = False,
    temperature: float = 1.0
) -> List[float]:
    """Calculate perplexity by splitting each text into n-grams first."""
    detector.performer_model.eval()
    all_results = []

    with torch.no_grad():
        for text in tqdm(texts, desc=f"{n}-gram Perplexity"):
            if not isinstance(text, str) or not text.strip():
                all_results.append(float('nan'))
                continue

            ngrams = split_into_ngrams(text, n)
            if not ngrams:
                all_results.append(float('nan'))
                continue

            text_results = []
            for i in range(0, len(ngrams), batch_size):
                batch_ngrams = ngrams[i:i+batch_size]

                try:
                    batch_encodings = detector.performer_tokenizer(
                        batch_ngrams, 
                        padding=True, 
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt"
                    ).to(device)
                    
                    batch_outputs = detector.performer_model(**batch_encodings)
                    batch_logits = batch_outputs.logits

                    if use_standard_ppl:
                        batch_ppl = detector._compute_perplexity(batch_encodings, batch_logits, median=median, temperature=temperature)
                        batch_ppl = np.exp(batch_ppl)
                    else:
                        batch_ppl = detector._compute_telescope_perplexity(batch_encodings, batch_logits, median, temperature)

                    text_results.extend(batch_ppl)

                except Exception as e:
                    logger.warning(f"Error processing n-gram batch: {str(e)}")
                    for ngram in batch_ngrams:
                        try:
                            encodings = detector.performer_tokenizer(ngram, return_tensors="pt").to(device)
                            outputs = detector.performer_model(**encodings)
                            logits = outputs.logits

                            if use_standard_ppl:
                                ppl = detector._compute_perplexity(encodings, logits, median=median, temperature=temperature)
                                ppl = np.exp(ppl)
                            else:
                                ppl = detector._compute_telescope_perplexity(encodings, logits, median, temperature)

                            text_results.append(ppl[0])
                        except Exception as e2:
                            logger.warning(f"Error processing individual n-gram: {str(e2)}")

                finally:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            if text_results:
                all_results.append(np.mean(text_results))
            else:
                all_results.append(float('nan'))

    return all_results


def generate_roc_curves(
    perplexity_results: Dict[str, Dict[str, List[float]]],
    output_file: str
) -> Dict[str, Dict[str, float]]:
    """Generate ROC curves and calculate AUC for all methods."""
    plt.figure(figsize=(12, 10))
    results_dict = {}

    colors = PLOT_COLORS

    for i, (method_name, method_data) in enumerate(perplexity_results.items()):
        human_ppl = [x for x in method_data['human'] if not np.isnan(x)]
        ai_ppl = [x for x in method_data['ai'] if not np.isnan(x)]

        y_true = [1] * len(human_ppl) + [0] * len(ai_ppl)
        y_scores = human_ppl + ai_ppl

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                 label=f'{method_name} (AUC = {roc_auc:.3f})')

        results_dict[method_name] = {
            'auc': roc_auc,
            'avg_human': np.nanmean(method_data['human']),
            'avg_ai': np.nanmean(method_data['ai']),
            'std_human': np.nanstd(method_data['human']),
            'std_ai': np.nanstd(method_data['ai'])
        }

    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves: N-gram Perplexity Methods')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

    return results_dict


def load_hc3_dataset(sample_size: int) -> Tuple[List[str], List[str]]:
    """Load HC3 dataset and extract human and AI texts."""
    logger.info("Loading HC3 dataset...")
    dataset = load_dataset("Hello-SimpleAI/HC3", "all", split="train")

    if len(dataset) > sample_size:
        dataset = dataset.select(range(sample_size))
        logger.info(f"Using {sample_size} samples from the dataset")

    human_texts = []
    ai_texts = []

    for item in dataset:
        human_answer = item.get("human_answers", item.get("human_answer", item.get("human", None)))
        ai_answer = item.get("chatgpt_answers", item.get("chatgpt_answer", item.get("chatgpt", None)))

        if human_answer and ai_answer:
            if isinstance(human_answer, list):
                human_texts.extend([h for h in human_answer if h and len(h.strip()) > 50])
            elif len(human_answer.strip()) > 50:
                human_texts.append(human_answer)

            if isinstance(ai_answer, list):
                ai_texts.extend([a for a in ai_answer if a and len(a.strip()) > 50])
            elif len(ai_answer.strip()) > 50:
                ai_texts.append(ai_answer)

    logger.info(f"Collected {len(human_texts)} human texts and {len(ai_texts)} AI texts")
    return human_texts, ai_texts


def main():
    parser = argparse.ArgumentParser(description='N-gram perplexity analysis for human vs AI text detection')
    
    parser.add_argument('--model', type=str, default='HuggingFaceTB/SmolLM-135M',
                        help='Model to use for perplexity calculation')
    parser.add_argument('--sample-size', type=int, default=5000,
                        help='Number of samples to process from dataset')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for normal perplexity')
    parser.add_argument('--ngram-batch-size', type=int, default=32,
                        help='Batch size for n-gram processing')
    parser.add_argument('--max-length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--median', action='store_true',
                        help='Use median instead of mean for telescope perplexity')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for softmax')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--output-dir', type=str, default='./ngram_perplexity_results',
                        help='Directory to save results')
    parser.add_argument('--ngrams', type=int, nargs='+', default=[1, 2, 3],
                        help='N-gram sizes to test')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
    
    # Load model via Detectors
    logger.info(f"Loading model: {args.model}")
    detector = Detectors(args.model)
    
    if detector.performer_tokenizer.pad_token is None:
        detector.performer_tokenizer.pad_token = detector.performer_tokenizer.eos_token
    
    # Load dataset
    human_texts, ai_texts = load_hc3_dataset(args.sample_size)
    
    # Results storage
    all_results = {
        'telescope': {},
        'standard': {}
    }
    
    # Calculate normal perplexity
    logger.info("Calculating normal perplexity...")
    for ppl_type, use_standard in [('telescope', False), ('standard', True)]:
        logger.info(f"Processing {ppl_type} perplexity...")
        
        human_normal = calculate_perplexity_batch(
            detector, human_texts, device,
            batch_size=args.batch_size,
            use_standard_ppl=use_standard,
            max_length=args.max_length,
            median=args.median,
            temperature=args.temperature
        )
        
        ai_normal = calculate_perplexity_batch(
            detector, ai_texts, device,
            batch_size=args.batch_size,
            use_standard_ppl=use_standard,
            max_length=args.max_length,
            median=args.median,
            temperature=args.temperature
        )
        
        all_results[ppl_type]['Normal'] = {
            'human': human_normal,
            'ai': ai_normal
        }
        
        # Calculate n-gram perplexities
        for n in args.ngrams:
            if n == 1:
                continue  # Skip unigram for normal texts
                
            logger.info(f"Calculating {n}-gram {ppl_type} perplexity...")
            
            human_ngram = calculate_ngram_perplexity(
                detector, human_texts, device,
                n=n,
                batch_size=args.ngram_batch_size,
                use_standard_ppl=use_standard,
                max_length=args.max_length,
                median=args.median,
                temperature=args.temperature
            )
            
            ai_ngram = calculate_ngram_perplexity(
                detector, ai_texts, device,
                n=n,
                batch_size=args.ngram_batch_size,
                use_standard_ppl=use_standard,
                max_length=args.max_length,
                median=args.median,
                temperature=args.temperature
            )
            
            all_results[ppl_type][f'{n}-gram'] = {
                'human': human_ngram,
                'ai': ai_ngram
            }
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Generate visualizations and compute metrics
    logger.info("Generating ROC curves...")
    
    telescope_metrics = generate_roc_curves(
        all_results['telescope'],
        os.path.join(args.output_dir, 'telescope_roc_curves.png')
    )
    
    standard_metrics = generate_roc_curves(
        all_results['standard'],
        os.path.join(args.output_dir, 'standard_roc_curves.png')
    )
    
    # Create comparison plot
    plt.figure(figsize=(14, 10))
    
    colors = PLOT_COLORS
    for i, (method_name, _) in enumerate(all_results['telescope'].items()):
        if method_name in telescope_metrics:
            human_ppl = [x for x in all_results['telescope'][method_name]['human'] if not np.isnan(x)]
            ai_ppl = [x for x in all_results['telescope'][method_name]['ai'] if not np.isnan(x)]
            y_true = [1] * len(human_ppl) + [0] * len(ai_ppl)
            y_scores = human_ppl + ai_ppl
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2, linestyle='-',
                     label=f'Telescope {method_name} (AUC = {telescope_metrics[method_name]["auc"]:.3f})')
    
    for i, (method_name, _) in enumerate(all_results['standard'].items()):
        if method_name in standard_metrics:
            human_ppl = [x for x in all_results['standard'][method_name]['human'] if not np.isnan(x)]
            ai_ppl = [x for x in all_results['standard'][method_name]['ai'] if not np.isnan(x)]
            y_true = [1] * len(human_ppl) + [0] * len(ai_ppl)
            y_scores = human_ppl + ai_ppl
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2, linestyle='--',
                     label=f'Standard {method_name} (AUC = {standard_metrics[method_name]["auc"]:.3f})')
    
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves: Telescope vs Standard Perplexity')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'perplexity_comparison.png'), dpi=300)
    plt.close()
    
    # Save results
    results = {
        'model': args.model,
        'parameters': {
            'sample_size': args.sample_size,
            'max_length': args.max_length,
            'median': args.median,
            'temperature': args.temperature,
            'ngrams': args.ngrams
        },
        'telescope_metrics': telescope_metrics,
        'standard_metrics': standard_metrics
    }
    
    results_file = os.path.join(args.output_dir, 'ngram_perplexity_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("RESULTS SUMMARY")
    logger.info("="*60)
    
    logger.info("\nTelescope Perplexity Results:")
    for method, metrics in telescope_metrics.items():
        logger.info(f"{method}: AUC = {metrics['auc']:.4f}, "
                   f"Avg Human = {metrics['avg_human']:.2f}, "
                   f"Avg AI = {metrics['avg_ai']:.2f}")
    
    logger.info("\nStandard Perplexity Results:")
    for method, metrics in standard_metrics.items():
        logger.info(f"{method}: AUC = {metrics['auc']:.4f}, "
                   f"Avg Human = {metrics['avg_human']:.2f}, "
                   f"Avg AI = {metrics['avg_ai']:.2f}")
    
    logger.info(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()