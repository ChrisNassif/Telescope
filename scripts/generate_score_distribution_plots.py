import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

from llm_text_detectors.utils import calculate_optimal_bin_count

PLOT_COLORS = yaml.safe_load(open("config.yaml"))["plot_colors"]



### START GLOBALS -------------------------------------------------------------------------

EXPERIMENT_FOLDER_NAME = "experiment_results"
ANALYSIS_OUTPUT_FOLDER_NAME = "experiment_analyses"
ANALYSIS_NAME = "metric_distributions"

METRIC_CODENAMES_TO_TEST = {
    "falcon_7B": ["binoculars_score", "telescope_perplexity", "telescope_perplexity_divided_by_cross_perplexity"],
    "gemma2_9B": ["binoculars_score", "telescope_perplexity", "telescope_perplexity_divided_by_cross_perplexity"],
    "smollm_360M": ["binoculars_score", "telescope_perplexity", "telescope_perplexity_divided_by_cross_perplexity"],
    "smollm_135M": ["binoculars_score", "telescope_perplexity", "telescope_perplexity_divided_by_cross_perplexity"],
    "smollm_1_7B": ["binoculars_score", "telescope_perplexity", "telescope_perplexity_divided_by_cross_perplexity"],
    "smollm2_360M": ["binoculars_score", "telescope_perplexity", "telescope_perplexity_divided_by_cross_perplexity"],
    "smollm2_135M": ["binoculars_score", "telescope_perplexity", "telescope_perplexity_divided_by_cross_perplexity"],
    "smollm2_1_7B": ["binoculars_score", "telescope_perplexity", "telescope_perplexity_divided_by_cross_perplexity"],
}

DATASET_CODENAMES_TO_TEST = [
    "detect_llm_text",
    "ai_human",
    "hc3",
    "hc3_plus",
    "esl_gpt4o",
    
    "ghostbusters_essay_gpt",
    "ghostbusters_news_gpt",
    "ghostbusters_creative_gpt",
    "ghostbusters_essay_gpt4o",
    "ghostbusters_creative_gpt4o",
    "ghostbusters_news_claude",
    "ghostbusters_creative_claude",
    "ghostbusters_essay_claude",
    "ghostbusters_essay_deepseek",
    "ghostbusters_creative_deepseek",
]

### END GLOBALS -------------------------------------------------------------------------



    
def create_output_folders(datasets):
    """Create output directory structure."""
    os.makedirs(f"{ANALYSIS_OUTPUT_FOLDER_NAME}/{ANALYSIS_NAME}", exist_ok=True)
    os.makedirs(f"{ANALYSIS_OUTPUT_FOLDER_NAME}/{ANALYSIS_NAME}/combined", exist_ok=True)

    for dataset in datasets:
        os.makedirs(f"{ANALYSIS_OUTPUT_FOLDER_NAME}/{ANALYSIS_NAME}/{dataset}", exist_ok=True)


def load_and_combine_data(model_name, dataset_name):
    """Load data from a specific model and dataset combination."""
    file_path = f"{EXPERIMENT_FOLDER_NAME}/{model_name}_{dataset_name}_dataset/raw_data.csv"
    print(file_path)
    df = pd.read_csv(file_path)
    df['dataset'] = dataset_name
    return df



def create_distribution_plot(data, metric_codename, model_codename, output_path, dataset_codename=None):
    """Create and save a histogram plot for a specific metric."""
    plt.figure(figsize=(12, 6))
    
    human_data = data[data['y_labels'] == 0][metric_codename]
    ai_data = data[data['y_labels'] == 1][metric_codename]
    
    # Calculate optimal bins using combined data based on the Freedman-Diaconis rule
    all_data = data[metric_codename]
    n_bins = calculate_optimal_bin_count(all_data)
    
    plt.hist(human_data, bins=n_bins, alpha=0.5, label=f'Human (n={len(human_data)})', color=PLOT_COLORS[0], density=True)
    plt.hist(ai_data, bins=n_bins, alpha=0.5, label=f'AI (n={len(ai_data)})', color=PLOT_COLORS[3], density=True)
    
    title = f'Distribution of {metric_codename} for {model_codename}'
    if dataset_codename:
        title += f'\nDataset: {dataset_codename}'
        
    plt.title(title)
    plt.xlabel(metric_codename)
    plt.ylabel('Density')
    plt.legend()
    
    # Add median lines
    plt.axvline(human_data.median(), color=PLOT_COLORS[0], linestyle='--', alpha=0.5, label=f'Human median: {human_data.median():.3f}')
    plt.axvline(ai_data.median(), color=PLOT_COLORS[3], linestyle='--', alpha=0.5, label=f'AI median: {ai_data.median():.3f}')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    create_output_folders(DATASET_CODENAMES_TO_TEST)
    
    for model_codename, metric_codenames in METRIC_CODENAMES_TO_TEST.items():
        for dataset_codename in DATASET_CODENAMES_TO_TEST:
            try:
                df = load_and_combine_data(model_codename, dataset_codename)
            except FileNotFoundError:
                print(f"Warning: No data found for {model_codename} on {dataset_codename}")
                continue

            for metric_codename in metric_codenames:
                output_path = f"{ANALYSIS_OUTPUT_FOLDER_NAME}/{ANALYSIS_NAME}/{dataset_codename}/{model_codename}_{metric_codename}_distribution.png"
                create_distribution_plot(df, metric_codename, model_codename, output_path, dataset_codename)
                print(f"Created distribution plot for {model_codename} - {metric_codename} - {dataset_codename}")
                
                # Save summary statistics to text file
                stats_path = f"{ANALYSIS_OUTPUT_FOLDER_NAME}/{ANALYSIS_NAME}/{dataset_codename}/{model_codename}_{metric_codename}_stats.txt"
                with open(stats_path, 'w') as f:
                    f.write(f"Summary statistics for {model_codename} - {metric_codename} - {dataset_codename}:\n")
                    f.write("\nHuman texts:\n")
                    f.write(str(df[df['y_labels'] == 0][metric_codename].describe()))
                    f.write("\n\nAI texts:\n")
                    f.write(str(df[df['y_labels'] == 1][metric_codename].describe()))
        
        
        # Combine data from all datasets for this model
        model_data = []
        for dataset_codename in DATASET_CODENAMES_TO_TEST:
            try:
                df = load_and_combine_data(model_codename, dataset_codename)
                model_data.append(df)
            except FileNotFoundError:
                print(f"WARNING: Model and dataset combination not found: {model_codename, dataset_codename}")
                continue
        
        if not model_data:
            print(f"WARNING: Could not process any of the datasets for model: {model_codename}")
            return

        combined_data = pd.concat(model_data, ignore_index=True)
        
        for metric_codename in metric_codenames:
            output_path = f"{ANALYSIS_OUTPUT_FOLDER_NAME}/{ANALYSIS_NAME}/combined/{model_codename}_{metric_codename}_distribution.png"
            create_distribution_plot(combined_data, metric_codename, model_codename, output_path)
            print(f"Created combined distribution plot for {model_codename} - {metric_codename}")
            
            # Save combined summary statistics to text file
            stats_path = f"{ANALYSIS_OUTPUT_FOLDER_NAME}/{ANALYSIS_NAME}/combined/{model_codename}_{metric_codename}_stats.txt"
            with open(stats_path, 'w') as f:
                f.write(f"Combined summary statistics for {model_codename} - {metric_codename}:\n")
                f.write("\nHuman texts:\n")
                f.write(str(combined_data[combined_data['y_labels'] == 0][metric_codename].describe()))
                f.write("\n\nAI texts:\n")
                f.write(str(combined_data[combined_data['y_labels'] == 1][metric_codename].describe()))
    


if __name__ == "__main__":
    main()