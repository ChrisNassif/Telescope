import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import wandb
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

class LLMTrainingMetricsLogger:
    def __init__(
        self,
        telescope,
        log_dir: str = "training_logs",
        tensorboard: bool = True,
        log_interval: int = 100,
        save_interval: int = 500,
        device: str = "cuda:0",
        use_wandb: bool = False
    ):
        """
        Initialize the metrics logger for LLM training.

        Args:
            telescope: Telescope object for computing metrics
            log_dir: Directory to save logs
            tensorboard: Whether to use TensorBoard logging
            log_interval: Steps between logging metrics
            save_interval: Steps between saving metrics to disk
            device: Device to use for computations
            use_wandb: Whether to use Weights & Biases logging
        """
        self.telescope = telescope
        self.device = device
        self.log_dir = Path(log_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.tensorboard = tensorboard
        self.writer = SummaryWriter(log_dir=str(self.log_dir)) if tensorboard else None
        
        self.use_wandb = use_wandb
        self.log_interval = log_interval
        self.save_interval = save_interval
        
        # Initialize metric storage
        self.metrics_history = {
            'step': [],
            'telescope_perplexity': [],
            'telescope_cross_perplexity': [],
            'causal_perplexity': [],
            'performer_model_entropy': [],
            'observer_model_entropy': [],
            'entropy_ratio': [],
            'kl_divergence': [],
            'performer_total_variation_distance': [],
            'observer_total_variation_distance': [],
            'performer_distribution_overlap': [],
            'observer_distribution_overlap': [],
            'performer_logits_std': [],
            'observer_logits_std': []
        }
        
        # Save initial configuration
        self._save_config()

    def _save_config(self) -> None:
        """Save logger configuration to JSON file."""
        config = {
            'observer_model_name': self.telescope.observer_model.config.name_or_path,
            'performer_model_name': self.telescope.performer_model.config.name_or_path,
            'log_interval': self.log_interval,
            'save_interval': self.save_interval,
            'tensorboard': self.tensorboard,
            'device': self.device,
            'created_at': datetime.now().isoformat()
        }
        with open(self.log_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=4)

    def _serialize_value(self, obj: Any) -> Any:
        """Convert non-serializable types to JSON-serializable formats."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        elif isinstance(obj, (list, tuple)):
            return [self._serialize_value(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._serialize_value(value) for key, value in obj.items()}
        return obj

    def to_number(self, val: Any) -> Union[float, int]:
        """Convert tensor or numpy values to Python numbers."""
        if torch.is_tensor(val):
            return val.cpu().item()
        elif isinstance(val, (np.integer, np.floating)):
            return val.item()
        return val

    def compute_and_log_metrics(self, step: int, text: str) -> None:
        """
        Compute and log metrics using the telescope.
        
        Args:
            step: Current training step
            text: Input text for computing metrics
        """
        # Only compute metrics every log_interval steps
        if step % self.log_interval != 0:
            return
        
        try:
            # Compute metrics using telescope
            telescope_ppl, telescope_xppl, extra_metrics = self.telescope.compute_all_metrics(
                text, device=self.device
            )
            
            self.log_metrics(step, telescope_ppl, telescope_xppl, extra_metrics)
        except Exception as e:
            print(f"Error computing metrics at step {step}: {e}")

    def log_metrics(
        self,
        step: int,
        telescope_perplexity: float,
        telescope_cross_perplexity: float,
        extra_metrics: Dict[str, Any]
    ) -> None:
        """
        Log metrics to various backends and store in history.
        
        Args:
            step: Current training step
            telescope_perplexity: Main perplexity metric
            telescope_cross_perplexity: Cross perplexity metric
            extra_metrics: Dictionary of additional metrics
        """
        # Store metrics
        self.metrics_history['step'].append(step)
        self.metrics_history['telescope_perplexity'].append(
            self.to_number(telescope_perplexity)
        )
        self.metrics_history['telescope_cross_perplexity'].append(
            self.to_number(telescope_cross_perplexity)
        )
        
        for metric_name, value in extra_metrics.items():
            if metric_name in self.metrics_history:
                self.metrics_history[metric_name].append(self.to_number(value))
        
        # Log to TensorBoard
        if self.tensorboard and self.writer is not None:
            self.writer.add_scalar(
                'telescope_perplexity',
                self.to_number(telescope_perplexity),
                step
            )
            self.writer.add_scalar(
                'telescope_cross_perplexity',
                self.to_number(telescope_cross_perplexity),
                step
            )
            for metric_name, value in extra_metrics.items():
                self.writer.add_scalar(metric_name, self.to_number(value), step)
        
        # Log to wandb
        if self.use_wandb:
            log_dict = {
                'telescope_perplexity': self.to_number(telescope_perplexity),
                'telescope_cross_perplexity': self.to_number(telescope_cross_perplexity)
            }
            log_dict.update({
                k: self.to_number(v) for k, v in extra_metrics.items()
            })
            wandb.log(log_dict, step=step)
        
        # Save to disk periodically
        if step % self.save_interval == 0:
            self.save_metrics()

    def save_metrics(self) -> None:
        """Save metrics to CSV and JSON files."""
        try:
            # Sanitize metrics history
            sanitized_history = {
                key: [
                    float(v) if isinstance(v, (int, float, np.integer, np.floating))
                    else self._serialize_value(v)
                    for v in values
                ]
                for key, values in self.metrics_history.items()
            }
            
            # Save to CSV
            df = pd.DataFrame(sanitized_history)
            df.to_csv(self.log_dir / 'metrics.csv', index=False)
            
            # Save latest metrics as JSON
            latest_metrics = {
                key: sanitized_history[key][-1]
                for key in sanitized_history
                if sanitized_history[key]
            }
            with open(self.log_dir / 'latest_metrics.json', 'w') as f:
                json.dump(latest_metrics, f, indent=4)
                
        except Exception as e:
            print(f"Error saving metrics: {e}")

    def plot_metrics(
        self,
        metrics_to_plot: Optional[List[str]] = None,
        save: bool = True
    ) -> None:
        """
        Plot metrics over training steps.
        
        Args:
            metrics_to_plot: List of metric names to plot. If None, plot all metrics
            save: Whether to save plots to disk
        """
        if metrics_to_plot is None:
            metrics_to_plot = [k for k in self.metrics_history.keys() if k != 'step']
            
        for metric in metrics_to_plot:
            if metric == 'step' or not self.metrics_history[metric]:
                continue
                
            plt.figure(figsize=(10, 6))
            plt.plot(
                self.metrics_history['step'],
                self.metrics_history[metric]
            )
            plt.title(f'{metric} over Training Steps')
            plt.xlabel('Training Steps')
            plt.ylabel(metric)
            plt.grid(True)
            
            if save:
                plt.savefig(self.log_dir / f'{metric}_plot.png')
            plt.close()

    def close(self) -> None:
        """Clean up resources."""
        if self.tensorboard and self.writer is not None:
            self.writer.close()