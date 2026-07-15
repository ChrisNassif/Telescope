import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
import yaml

PLOT_COLORS: List[str] = yaml.safe_load(open("config.yaml"))["plot_colors"]

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif'],
    'font.size': 36,
    'axes.titlesize': 42,
    'axes.labelsize': 36,
    'xtick.labelsize': 33,
    'ytick.labelsize': 33,
    'legend.fontsize': 33,
    'figure.titlesize': 48,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 4,
    'lines.markersize': 18,
    'axes.linewidth': 3
})

EXPERIMENT_FOLDER_NAME: str = "experiment_results"
ANALYSIS_OUTPUT_FOLDER_NAME: str = "experiment_analyses"
ANALYSIS_NAME: str = "calibration_plots"





def format_metric_name(metric_name: str) -> str:
    """Convert metric names to properly formatted display names."""
    name_mapping: Dict[str, str] = {
        'binoculars_score': 'Binoculars Score',
        'telescope_perplexity': 'Telescope Perplexity',
        'telescope_perplexity_divided_by_cross_perplexity': 'Telescope Perplexity Divided by Cross Perplexity'
    }
    return name_mapping.get(metric_name, metric_name.replace('_', ' ').title())

def create_output_folders(datasets: List[str]) -> None:
    """Create output directory structure."""
    os.makedirs(f"{ANALYSIS_OUTPUT_FOLDER_NAME}/{ANALYSIS_NAME}", exist_ok=True)
    os.makedirs(f"{ANALYSIS_OUTPUT_FOLDER_NAME}/{ANALYSIS_NAME}/combined", exist_ok=True)
    
    dataset: str
    for dataset in datasets:
        os.makedirs(f"{ANALYSIS_OUTPUT_FOLDER_NAME}/{ANALYSIS_NAME}/{dataset}", exist_ok=True)

def load_and_combine_data(model_name: str, dataset_name: str) -> pd.DataFrame:
    """Load data from a specific model and dataset combination."""
    file_path: str = f"{EXPERIMENT_FOLDER_NAME}/{model_name}_{dataset_name}_dataset/raw_data.csv"
    print(file_path)
    df: pd.DataFrame = pd.read_csv(file_path)
    df['dataset'] = dataset_name
    return df

def create_calibration_plot(
    data: pd.DataFrame,
    metric: str,
    model_name: str,
    output_path: str,
    dataset_name: Optional[str] = None,
    n_bins: int = 10
) -> None:
    """Create and save calibration plots for a specific metric, with both normal and isotonic regression."""
    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots(figsize=(16, 16))
    plt.subplots_adjust(bottom=0.25) 
    plt.style.use('seaborn-v0_8-whitegrid')
    
    y_true: np.ndarray = data['y_labels'].values
    
    metric_values: np.ndarray = data[metric].values
    
    min_val: float = float(np.min(metric_values))
    max_val: float = float(np.max(metric_values))
    
    normalized_values: np.ndarray
    if max_val > min_val:
        normalized_values = (metric_values - min_val) / (max_val - min_val)
    else:
        normalized_values = np.zeros_like(metric_values)
    
    ai_mean: float = float(np.mean(normalized_values[y_true == 1]))
    human_mean: float = float(np.mean(normalized_values[y_true == 0]))
    
    # If AI mean is lower than human mean, flip the scores
    # We want higher scores to indicate AI text for consistency
    y_prob: np.ndarray
    direction: str
    if ai_mean < human_mean:
        y_prob = 1 - normalized_values
        direction = "flipped"
    else:
        y_prob = normalized_values
        direction = "original"
        
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=3.5, alpha=0.7, label='Perfectly calibrated')
    
    prob_true: np.ndarray
    prob_pred: np.ndarray
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    ax.plot(
        prob_pred, prob_true, marker='o', markersize=14, linewidth=5, 
        color=PLOT_COLORS[0], label=f'Original (Brier: {brier_score_loss(y_true, y_prob):.3f})'
    )
    
    isotonic_regression: IsotonicRegression = IsotonicRegression(out_of_bounds='clip')
    isotonic_regression.fit(y_prob, y_true)
    y_prob_isotonic: np.ndarray = isotonic_regression.predict(y_prob)
    
    prob_true_isotonic: np.ndarray
    prob_pred_isotonic: np.ndarray
    prob_true_isotonic, prob_pred_isotonic = calibration_curve(y_true, y_prob_isotonic, n_bins=n_bins)
    
    ax.plot(
        prob_pred_isotonic, prob_true_isotonic, marker='s', markersize=14, linewidth=5,
        color=PLOT_COLORS[1], label=f'Isotonic (Brier: {brier_score_loss(y_true, y_prob_isotonic):.3f})'
    )
    
    formatted_metric: str = format_metric_name(metric)
    title: str
    if dataset_name:
        title = f'{formatted_metric} Calibration - {model_name} - {dataset_name}'
    else:
        title = f'{formatted_metric} Calibration - {model_name}'
        
    ax.set_title(title, fontsize=42, pad=20)
    ax.set_xlabel('Predicted Probability', labelpad=20)
    ax.set_ylabel('True Probability (Fraction of Positives)', labelpad=20)
    

    legend: Any = ax.legend(
        loc='upper center', bbox_to_anchor=(0.5, -0.15),
        frameon=True, framealpha=0.9, borderpad=1.2,
        fontsize=33, ncol=3
    )
    
    legend.get_frame().set_linewidth(3)
    
    ax.set_aspect('equal', adjustable='box')
    
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=2)
    
    ax.tick_params(width=3, length=10, pad=10)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()



def save_calibration_stats(
    data: pd.DataFrame,
    metric: str,
    model_name: str,
    output_path: str,
    dataset_name: Optional[str] = None
) -> None:
    """Save detailed calibration statistics to a text file."""
    y_true: np.ndarray = data['y_labels'].values
    metric_values: np.ndarray = data[metric].values
    
    min_val: float = float(np.min(metric_values))
    max_val: float = float(np.max(metric_values))
    
    normalized_values: np.ndarray
    if max_val > min_val:
        normalized_values = (metric_values - min_val) / (max_val - min_val)
    else:
        normalized_values = np.zeros_like(metric_values)
    
    ai_mean: float = float(np.mean(normalized_values[y_true == 1]))
    human_mean: float = float(np.mean(normalized_values[y_true == 0]))
    
    y_prob: np.ndarray
    direction: str
    if ai_mean < human_mean:
        y_prob = 1 - normalized_values
        direction = "flipped"
    else:
        y_prob = normalized_values
        direction = "original"
    
    isotonic_regression: IsotonicRegression = IsotonicRegression(out_of_bounds='clip')
    isotonic_regression.fit(y_prob, y_true)
    y_prob_isotonic: np.ndarray = isotonic_regression.predict(y_prob)
    
    brier_original: float = float(brier_score_loss(y_true, y_prob))
    brier_isotonic: float = float(brier_score_loss(y_true, y_prob_isotonic))
    
    formatted_metric: str = format_metric_name(metric)
    
    with open(output_path, 'w') as f:
        f.write(f"Calibration statistics for {model_name} - {formatted_metric}")
        if dataset_name:
            f.write(f" - {dataset_name}")
        f.write("\n\n")
        
        f.write(f"Score direction: {direction}\n")
        if direction == "flipped":
            f.write("(Lower original values indicate AI text)\n\n")
        else:
            f.write("(Higher original values indicate AI text)\n\n")
            
        f.write(f"Number of samples: {len(y_true)}\n")
        f.write(f"Number of positives (AI): {np.sum(y_true)}\n")
        f.write(f"Number of negatives (human): {len(y_true) - np.sum(y_true)}\n\n")
        
        f.write(f"Original Brier score: {brier_original:.4f}\n")
        f.write(f"Isotonic calibration Brier score: {brier_isotonic:.4f}\n")
        f.write(f"Improvement: {(brier_original - brier_isotonic) / brier_original * 100:.2f}%\n\n")
        
        f.write(f"AI mean (original metric): {np.mean(metric_values[y_true == 1]):.4f}\n")
        f.write(f"Human mean (original metric): {np.mean(metric_values[y_true == 0]):.4f}\n")
        f.write(f"Difference: {np.mean(metric_values[y_true == 1]) - np.mean(metric_values[y_true == 0]):.4f}\n\n")
        
        f.write("Metric percentiles:\n")
        percentiles: List[int] = [0, 10, 25, 50, 75, 90, 100]
        
        f.write("Overall:\n")
        p: int
        for p in percentiles:
            f.write(f"{p}th percentile: {np.percentile(metric_values, p):.4f}\n")
        
        f.write("\nHuman texts:\n")
        human_values: np.ndarray = metric_values[y_true == 0]
        for p in percentiles:
            f.write(f"{p}th percentile: {np.percentile(human_values, p):.4f}\n")
        
        f.write("\nAI texts:\n")
        ai_values: np.ndarray = metric_values[y_true == 1]
        for p in percentiles:
            f.write(f"{p}th percentile: {np.percentile(ai_values, p):.4f}\n")




def main() -> None:

    model_features: Dict[str, List[str]] = {
        "falcon_7B": ["binoculars_score", "telescope_perplexity"],
        "gemma2_9B": ["binoculars_score", "telescope_perplexity"],
        "smollm_360M": ["binoculars_score", "telescope_perplexity"],
        "smollm_135M": ["binoculars_score", "telescope_perplexity"],
        "smollm_1_7B": ["binoculars_score", "telescope_perplexity"],
        "smollm2_360M": ["binoculars_score", "telescope_perplexity"],
        "smollm2_135M": ["binoculars_score", "telescope_perplexity"],
        "smollm2_1_7B": ["binoculars_score", "telescope_perplexity"],
    }
    
    datasets: List[str] = [
        "detect_llm_text",
        "ai_human",
        "hc3",
        "hc3_plus",
        "esl_gpt4o"
    ]
    
    create_output_folders(datasets)
    
    model_name: str
    features: List[str]
    for model_name, features in model_features.items():
        dataset: str
        for dataset in datasets:
            try:
                df: pd.DataFrame = load_and_combine_data(model_name, dataset)

                feature: str
                for feature in features:
                    output_path: str = f"{ANALYSIS_OUTPUT_FOLDER_NAME}/{ANALYSIS_NAME}/{dataset}/{model_name}_{feature}_calibration.png"
                    create_calibration_plot(df, feature, model_name, output_path, dataset)
                    print(f"Created calibration plot for {model_name} - {feature} - {dataset}")
                    
                    stats_path: str = f"{ANALYSIS_OUTPUT_FOLDER_NAME}/{ANALYSIS_NAME}/{dataset}/{model_name}_{feature}_calibration_stats.txt"
                    save_calibration_stats(df, feature, model_name, stats_path, dataset)
                    
            except FileNotFoundError:
                print(f"Warning: No data found for {model_name} on {dataset}")
                continue
        

        try:
            model_data: List[pd.DataFrame] = []
            for dataset in datasets:
                try:
                    df = load_and_combine_data(model_name, dataset)
                    model_data.append(df)
                except FileNotFoundError:
                    continue
            
            if model_data:
                combined_data: pd.DataFrame = pd.concat(model_data, ignore_index=True)
                
                for feature in features:
                    output_path = f"{ANALYSIS_OUTPUT_FOLDER_NAME}/{ANALYSIS_NAME}/combined/{model_name}_{feature}_calibration.png"
                    create_calibration_plot(combined_data, feature, model_name, output_path)
                    print(f"Created combined calibration plot for {model_name} - {feature}")
                    
                    stats_path = f"{ANALYSIS_OUTPUT_FOLDER_NAME}/{ANALYSIS_NAME}/combined/{model_name}_{feature}_calibration_stats.txt"
                    save_calibration_stats(combined_data, feature, model_name, stats_path)
        
        except Exception as e:
            print(f"Error processing combined data for {model_name}: {str(e)}")



if __name__ == "__main__":
    main()