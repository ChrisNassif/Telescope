#!/usr/bin/env python3
"""
Pythia Model Evaluation Script
Evaluates Pythia model checkpoints at regular intervals, computing attention entropy
and perplexity metrics with confidence intervals.
"""

import os
import sys
import argparse
import torch
import numpy as np
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from typing import List, Optional, Dict, Any, Tuple
import logging
import json
from pathlib import Path
import tempfile
import time

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llm_text_detectors import Detectors

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
PYTHIA_TOTAL_STEPS = 143000



def compute_attention_entropy(attention_weights: torch.Tensor) -> torch.Tensor:
    """Compute the entropy of attention distributions."""
    eps = 1e-8
    attention_weights = attention_weights + eps
    attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)
    entropy = -torch.sum(attention_weights * torch.log2(attention_weights), dim=-1)
    return entropy


def compute_attention_stats(model: GPTNeoXForCausalLM,
                          input_ids: torch.Tensor,
                          attention_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """Compute statistics about attention distribution."""
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )
        
        attention_weights = outputs.attentions
        layer_stats = []
        
        for layer_idx, layer_attention in enumerate(attention_weights):
            layer_entropy = compute_attention_entropy(layer_attention)
            max_attention = layer_attention.max(dim=-1)[0]
            attention_sparsity = (layer_attention < 0.1).float().mean()
            
            layer_stats.append({
                'layer': layer_idx,
                'mean_entropy': layer_entropy.mean().item(),
                'max_entropy': layer_entropy.max().item(),
                'min_entropy': layer_entropy.min().item(),
                'mean_max_attention': max_attention.mean().item(),
                'attention_sparsity': attention_sparsity.item()
            })
        
        aggregate_stats = {
            'avg_attention_entropy': np.mean([s['mean_entropy'] for s in layer_stats]),
            'std_attention_entropy': np.std([s['mean_entropy'] for s in layer_stats]),
            'avg_attention_sparsity': np.mean([s['attention_sparsity'] for s in layer_stats]),
            'avg_max_attention': np.mean([s['mean_max_attention'] for s in layer_stats]),
            'layer_stats': layer_stats
        }
        
        return aggregate_stats


def get_evaluation_steps(interval: int, start_step: int, end_step: int) -> List[int]:
    """Generate list of steps to evaluate at regular intervals."""
    steps = list(range(start_step, min(end_step + 1, PYTHIA_TOTAL_STEPS + 1), interval))
    if steps[-1] != PYTHIA_TOTAL_STEPS and PYTHIA_TOTAL_STEPS <= end_step:
        steps.append(PYTHIA_TOTAL_STEPS)
    return steps


def verify_checkpoint_results(output_dir: str, step: int) -> bool:
    """Verify that a checkpoint evaluation is complete and valid."""
    results_file = os.path.join(output_dir, f"results_step{step}.json")
    samples_file = os.path.join(output_dir, f"samples_step{step}.json")
    
    try:
        if not os.path.exists(results_file) or not os.path.exists(samples_file):
            return False
        
        with open(results_file, 'r') as f:
            results = json.load(f)
            required_fields = ['sample_size', 'attention_entropy', 'telescope_perplexity']
            for field in required_fields:
                if field not in results:
                    return False
        
        with open(samples_file, 'r') as f:
            samples = json.load(f)
            required_fields = ['prompts', 'generated_samples', 'attention_statistics']
            for field in required_fields:
                if field not in samples:
                    return False
        
        if len(samples['prompts']) != len(samples['generated_samples']):
            return False
        
        return True
        
    except Exception:
        return False


def generate_samples(model_name: str, step: int, prompts: List[str], 
                    device: str = "cuda") -> Tuple[List[str], List[Dict]]:
    """Generate samples using Pythia checkpoint and return attention statistics."""
    model_name = model_name.strip()
    logger.info(f"Generating samples using {model_name} at step {step}")
    
    with tempfile.TemporaryDirectory() as temp_cache_dir:
        try:
            model = GPTNeoXForCausalLM.from_pretrained(
                model_name,
                revision=f"step{step}",
                cache_dir=temp_cache_dir,
                torch_dtype=torch.float16
            ).to(device)
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                revision=f"step{step}",
                cache_dir=temp_cache_dir
            )
            
            generated_texts = []
            attention_stats = []
            
            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                
                stats = compute_attention_stats(model, inputs.input_ids, inputs.attention_mask)
                attention_stats.append(stats)
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id
                )
                
                text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_texts.append(text)
            
            del model
            del tokenizer
            if device == "cuda":
                torch.cuda.empty_cache()
            
            return generated_texts, attention_stats
            
        except Exception as e:
            logger.error(f"Error generating samples: {str(e)}")
            raise


def compute_batch_metrics(step: int, sample_texts: List[str],
                         attention_stats: List[Dict], telescope: Detectors) -> Optional[Dict[str, float]]:
    """Compute metrics for multiple samples and return averaged results."""
    all_perplexities = []
    all_attention_metrics = []
    
    for text, attn_stat in zip(sample_texts, attention_stats):
        try:
            ppl = telescope.compute_telescope_perplexity(text)
            all_perplexities.append(float(ppl))
            
            attention_metrics = {
                'attention_entropy': attn_stat['avg_attention_entropy'],
                'attention_entropy_std': attn_stat['std_attention_entropy'],
                'attention_sparsity': attn_stat['avg_attention_sparsity'],
                'max_attention': attn_stat['avg_max_attention']
            }
            all_attention_metrics.append(attention_metrics)
            
        except Exception as e:
            logger.error(f"Error computing metrics for sample: {str(e)}")
            continue
    
    if all_perplexities and all_attention_metrics:
        mean_ppl = np.mean(all_perplexities)
        std_ppl = np.std(all_perplexities)
        n = len(all_perplexities)
        
        ci_95_lower = mean_ppl - 2 * (std_ppl / np.sqrt(n))
        ci_95_upper = mean_ppl + 2 * (std_ppl / np.sqrt(n))
        
        avg_attention = {}
        for key in all_attention_metrics[0].keys():
            values = [m[key] for m in all_attention_metrics if key in m and m[key] is not None]
            if values:
                avg_attention[key] = np.mean(values)
                std_key = f"{key}_std"
                avg_attention[std_key] = np.std(values)
                
                n_att = len(values)
                ci_lower = avg_attention[key] - 2 * (np.std(values) / np.sqrt(n_att))
                ci_upper = avg_attention[key] + 2 * (np.std(values) / np.sqrt(n_att))
                avg_attention[f"{key}_ci_95_lower"] = ci_lower
                avg_attention[f"{key}_ci_95_upper"] = ci_upper
        
        return {
            'sample_size': n,
            'telescope_perplexity': mean_ppl,
            'perplexity_std': std_ppl,
            'perplexity_ci_95_lower': ci_95_lower,
            'perplexity_ci_95_upper': ci_95_upper,
            **avg_attention
        }
    
    return None


def evaluate_checkpoint(model_name: str, step: int, prompts: List[str],
                       output_dir: str, telescope: Detectors, device: str = "cuda") -> None:
    """Run evaluations on the model and save results."""
    model_name = model_name.strip()
    logger.info(f"Evaluating {model_name} at step {step}")
    
    samples_file = os.path.join(output_dir, f"samples_step{step}.json")
    results_file = os.path.join(output_dir, f"results_step{step}.json")
    csv_file = os.path.join(output_dir, "evaluation_results.csv")
    
    samples_temp = samples_file + ".tmp"
    results_temp = results_file + ".tmp"
    
    try:
        logger.info(f"Generating samples for step {step}...")
        generated_samples, attention_stats = generate_samples(model_name, step, prompts, device)
        
        with open(samples_temp, 'w') as f:
            json.dump({
                'prompts': prompts,
                'generated_samples': generated_samples,
                'attention_statistics': attention_stats,
                'evaluation_timestamp': time.time(),
                'step': step
            }, f, indent=2)
        
        logger.info(f"Computing metrics for step {step}...")
        metrics = compute_batch_metrics(step, generated_samples, attention_stats, telescope)
        
        if metrics:
            metrics['evaluation_timestamp'] = time.time()
            metrics['step'] = step
            metrics['model_name'] = model_name
            
            with open(results_temp, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            os.rename(samples_temp, samples_file)
            os.rename(results_temp, results_file)
            
            logger.info(f"Results saved to {results_file}")
            
            csv_exists = os.path.exists(csv_file)
            lock_file = csv_file + ".lock"
            wait_time = 0
            max_wait = 30
            
            while os.path.exists(lock_file) and wait_time < max_wait:
                time.sleep(0.1)
                wait_time += 0.1
            
            try:
                with open(lock_file, 'w') as f:
                    f.write(str(os.getpid()))
                
                with open(csv_file, 'a') as f:
                    if not csv_exists:
                        f.write("step,telescope_perplexity,perplexity_ci_lower,perplexity_ci_upper,"
                               "attention_entropy,attention_entropy_ci_lower,attention_entropy_ci_upper,"
                               "attention_sparsity,sample_size\n")
                    
                    f.write(f"{step},{metrics['telescope_perplexity']},"
                           f"{metrics['perplexity_ci_95_lower']},{metrics['perplexity_ci_95_upper']},"
                           f"{metrics['attention_entropy']},{metrics['attention_entropy_ci_95_lower']},"
                           f"{metrics['attention_entropy_ci_95_upper']},{metrics['attention_sparsity']},"
                           f"{metrics['sample_size']}\n")
                
                logger.info(f"Results appended to {csv_file}")
                
            finally:
                if os.path.exists(lock_file):
                    os.remove(lock_file)
        
        else:
            logger.warning(f"No metrics computed for step {step}")
    
    except Exception as e:
        logger.error(f"Evaluation error for step {step}: {str(e)}")
        for temp_file in [samples_temp, results_temp]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
        raise


def plot_results(csv_file: str, output_dir: str) -> None:
    """Create plots of perplexity and attention metrics with confidence intervals."""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        
        df = pd.read_csv(csv_file)
        if len(df) < 2:
            logger.warning("Not enough data points to create a meaningful plot")
            return
        
        cols_to_check = [
            'step', 'telescope_perplexity', 'perplexity_ci_lower', 'perplexity_ci_upper',
            'attention_entropy', 'attention_entropy_ci_lower', 'attention_entropy_ci_upper',
            'attention_sparsity'
        ]
        available_cols = [col for col in cols_to_check if col in df.columns]
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=available_cols)
        if len(df) < 2:
            logger.warning("No valid data points after filtering")
            return
        
        import yaml
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(project_root, "config.yaml")
        PLOT_COLORS = yaml.safe_load(open(config_path))["plot_colors"]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Perplexity plot
        ax1.plot(df['step'], df['telescope_perplexity'], color=PLOT_COLORS[0], label='Telescope Perplexity')
        ax1.fill_between(
            df['step'],
            df['perplexity_ci_lower'],
            df['perplexity_ci_upper'],
            color=PLOT_COLORS[0],
            alpha=0.2,
            label='95% Confidence Interval'
        )
        ax1.set_ylabel('Perplexity')
        ax1.set_title('Model Perplexity and Attention Metrics During Training')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Attention metrics plot
        ax2.plot(df['step'], df['attention_entropy'], color=PLOT_COLORS[2], label='Attention Entropy')
        ax2.fill_between(
            df['step'],
            df['attention_entropy_ci_lower'],
            df['attention_entropy_ci_upper'],
            color=PLOT_COLORS[2],
            alpha=0.2
        )
        ax2.plot(df['step'], df['attention_sparsity'], color=PLOT_COLORS[3], label='Attention Sparsity')
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Attention Metrics')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'evaluation_plots.png')
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        logger.info(f"Plots saved to {plot_path}")
        
    except ImportError:
        logger.warning("Matplotlib or pandas not installed. Skipping plot generation.")
    except Exception as e:
        logger.error(f"Error creating plots: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Pythia model checkpoints')
    parser.add_argument('model_name', type=str, help='HuggingFace model name (e.g., EleutherAI/pythia-160m)')
    parser.add_argument('--prompts', type=str, required=True, help='File containing prompts (one per line)')
    parser.add_argument('--output-dir', type=str, default='./pythia_eval_results', help='Output directory')
    parser.add_argument('--interval', type=int, default=1000, help='Step interval for evaluation')
    parser.add_argument('--start-step', type=int, default=0, help='Starting step')
    parser.add_argument('--end-step', type=int, default=143000, help='Ending step')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--force-restart', action='store_true', help='Force restart from beginning')
    parser.add_argument('--plot', action='store_true', help='Generate plots after evaluation')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.prompts):
        logger.error(f"Prompts file not found: {args.prompts}")
        sys.exit(1)
    
    # Load prompts
    with open(args.prompts, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    logger.info(f"Loaded {len(prompts)} prompts")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get evaluation steps
    steps_to_evaluate = get_evaluation_steps(args.interval, args.start_step, args.end_step)
    logger.info(f"Will evaluate {len(steps_to_evaluate)} checkpoints: {steps_to_evaluate}")
    
    # Handle resumption
    progress_file = os.path.join(args.output_dir, "evaluation_progress.json")
    csv_file = os.path.join(args.output_dir, "evaluation_results.csv")
    
    if args.force_restart:
        if os.path.exists(progress_file):
            os.remove(progress_file)
        if os.path.exists(csv_file):
            os.remove(csv_file)
        logger.info("Force restart - cleared previous progress")
    else:
        completed_steps = set()
        for step in steps_to_evaluate:
            if verify_checkpoint_results(args.output_dir, step):
                completed_steps.add(step)
                logger.info(f"Found complete results for step {step}")
        
        steps_to_evaluate = [s for s in steps_to_evaluate if s not in completed_steps]
        
        if completed_steps:
            logger.info(f"Resuming evaluation ({len(completed_steps)} steps already completed)")
    
    if not steps_to_evaluate:
        logger.info("All requested steps have been completed!")
        if args.plot and os.path.exists(csv_file):
            plot_results(csv_file, args.output_dir)
        return
    
    # Initialize CSV file if needed
    if not os.path.exists(csv_file):
        with open(csv_file, 'w') as f:
            f.write("step,telescope_perplexity,perplexity_ci_lower,perplexity_ci_upper,"
                   "attention_entropy,attention_entropy_ci_lower,attention_entropy_ci_upper,"
                   "attention_sparsity,sample_size\n")
    
    # Initialize Telescope
    model_name = "HuggingFaceTB/SmolLM-360M"
    telescope = Detectors(model_name)
    
    # Evaluate checkpoints
    total_steps = len(steps_to_evaluate)
    for idx, step in enumerate(steps_to_evaluate):
        logger.info(f"Evaluating checkpoint {idx+1}/{total_steps}: step {step}")
        
        try:
            evaluate_checkpoint(
                args.model_name,
                step,
                prompts,
                args.output_dir,
                telescope,
                device=args.device
            )
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            progress_data = {
                "last_completed_step": step,
                "completed_count": idx + 1,
                "total_steps": total_steps,
                "remaining_steps": steps_to_evaluate[idx+1:],
                "progress_percentage": round((idx + 1) / total_steps * 100, 1),
                "timestamp": time.time()
            }
            
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
            
            logger.info(f"Progress: {idx+1}/{total_steps} ({progress_data['progress_percentage']}%)")
            
        except Exception as e:
            logger.error(f"Error processing step {step}: {str(e)}")
            continue
    
    # Clean up Telescope
    del telescope
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Generate plots
    if args.plot and os.path.exists(csv_file):
        plot_results(csv_file, args.output_dir)
    
    logger.info("Evaluation complete!")

if __name__ == "__main__":
    main()