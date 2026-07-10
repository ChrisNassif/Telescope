# TODO: WE NEED TO GET DETERMINISTIC COLORS INSTEAD OF TRYING TO JUST RANDOMLY PICK COLORS

import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.insert(0, os.getcwd())

import numpy as np
from sklearn.metrics import cohen_kappa_score, mutual_info_score, normalized_mutual_info_score
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import yaml

from llm_text_detectors.utils import create_logistic_regression_classifier




### START GLOBALS -------------------------------------------------------------------------

EXPERIMENT_FOLDER_NAME = "experiment_results"
ANALYSIS_OUTPUT_FOLDER_NAME = "experiment_analyses"
ANALYSIS_NAME = "error_independence_analysis"


METRIC_CODENAMES_TO_TEST = {
    "gemma2_2B": ["telescope_perplexity", "binoculars_score", "perplexity", "log_rank_ratio", "fast_detectgpt"],
    "gemma2_9B": ["telescope_perplexity", "binoculars_score", "perplexity", "log_rank_ratio", "fast_detectgpt"],
    "llama3_8B": ["telescope_perplexity", "binoculars_score", "perplexity", "log_rank_ratio", "fast_detectgpt"],
    "falcon_7B":  ["telescope_perplexity", "binoculars_score", "perplexity", "log_rank_ratio", "fast_detectgpt"],
    
    "smollm_135M": ["telescope_perplexity", "binoculars_score", "perplexity", "log_rank_ratio", "fast_detectgpt"],
    "smollm_360M": ["telescope_perplexity", "binoculars_score", "perplexity", "log_rank_ratio", "fast_detectgpt"],
    "smollm_1_7B": ["telescope_perplexity", "binoculars_score", "perplexity", "log_rank_ratio", "fast_detectgpt"],
    "smollm2_135M": ["telescope_perplexity", "binoculars_score", "perplexity", "log_rank_ratio", "fast_detectgpt"],
    "smollm2_360M": ["telescope_perplexity", "binoculars_score", "perplexity", "log_rank_ratio", "fast_detectgpt"],
    "smollm2_1_7B": ["telescope_perplexity", "binoculars_score", "perplexity", "log_rank_ratio", "fast_detectgpt"],
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


os.makedirs(f"{ANALYSIS_OUTPUT_FOLDER_NAME}/{ANALYSIS_NAME}", exist_ok=True)

# a dictionary that maps a model's codename (for instance smollm2_360M) 
# to a presentable, paper-ready name (for instance SmolLM2 360M)
MODEL_CODENAME_TO_MODEL_DISPLAYNAME = yaml.safe_load(open("config.yaml"))["model_codenames_to_model_displaynames"]





def calculate_agreement_statistics(predictions1, predictions2, actual_labels):
    """
    Calculate agreement statistics between two classifiers
    
    Args:
        predictions1: Probability predictions from first classifier
        predictions2: Probability predictions from second classifier
        actual_labels: Ground truth labels
        
    Returns:
        kappa: Cohen's Kappa statistic
        q_statistic: Yule's Q statistic
        mutual_info: Normalized mutual information between errors
    """
    # Verify all arrays have the same length
    assert len(predictions1) == len(predictions2) == len(actual_labels), \
        f"Arrays must have same length: {len(predictions1)}, {len(predictions2)}, {len(actual_labels)}"
    
    # Convert predictions to binary if they aren't already
    pred1_binary = (predictions1 > 0.5).astype(int)
    pred2_binary = (predictions2 > 0.5).astype(int)
    
    # Calculate Kappa statistic (agreement beyond chance)
    kappa = cohen_kappa_score(pred1_binary, pred2_binary)
    
    # Calculate contingency table for Q-statistic
    n11 = np.sum((pred1_binary == 1) & (pred2_binary == 1))
    n10 = np.sum((pred1_binary == 1) & (pred2_binary == 0))
    n01 = np.sum((pred1_binary == 0) & (pred2_binary == 1))
    n00 = np.sum((pred1_binary == 0) & (pred2_binary == 0))
    
    # Calculate Yule's Q statistic (ranges from -1 to 1)
    # Q = (n11*n00 - n10*n01) / (n11*n00 + n10*n01)
    numerator = n11 * n00 - n10 * n01
    denominator = n11 * n00 + n10 * n01
    q_statistic = numerator / denominator if denominator != 0 else 0
    
    # Calculate mutual information between errors
    errors1 = (pred1_binary != actual_labels).astype(int)
    errors2 = (pred2_binary != actual_labels).astype(int)
    
    # Calculate mutual information between errors
    # Handle the case where one classifier gets everything right or wrong
    # (which would cause NMI to be undefined)
    if len(np.unique(errors1)) < 2 or len(np.unique(errors2)) < 2:
        mutual_info = 0.0  # No mutual information if one has no variation
    else:
        mutual_info = normalized_mutual_info_score(errors1, errors2)
    
    return kappa, q_statistic, mutual_info


def plot_agreement_heatmap(matrix, labels, stat_name, dataset_codename, vmin=0, vmax=1):
    """
    Create a heatmap visualization for an agreement statistic
    
    Args:
        matrix: 2D numpy array with statistic values
        labels: List of labels for axes
        stat_name: Name of the statistic being plotted
        dataset_codename: Codename of the dataset
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
    """
    plt.figure(figsize=(14, 12))
    
    # Choose appropriate colormap
    if stat_name == 'mutual_info':
        # Higher values indicate more dependency between errors
        cmap = 'viridis'
        title = f"Error Mutual Information - {dataset_codename}"
    elif stat_name == 'q_statistic':
        # Range is -1 to 1
        # Negative values: classifiers tend to make errors on different examples
        # Positive values: classifiers tend to make errors on the same examples
        cmap = 'coolwarm'
        title = f"Yule's Q-statistic - {dataset_codename}"
    else:  # kappa
        # Higher values indicate more agreement
        cmap = 'YlGnBu'
        title = f"Cohen's Kappa - {dataset_codename}"
    
    # Create heatmap with smaller font size for annotations
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': stat_name.replace('_', ' ').title()},
        annot_kws={"size": 6}  # Smaller annotation font size
    )
    
    plt.title(title, fontsize=16, pad=20)
    plt.xticks(rotation=90, fontsize=6)  # Smaller x-tick labels
    plt.yticks(rotation=0, fontsize=6)   # Smaller y-tick labels
    plt.tight_layout()
    
    # Save the plot
    output_path = f"{ANALYSIS_OUTPUT_FOLDER_NAME}/{ANALYSIS_NAME}/{stat_name}_{dataset_codename}_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {stat_name} heatmap for {dataset_codename} to {output_path}")


def process_dataset(dataset_codename):
    """
    Process a single dataset to calculate agreement statistics between classifiers
    
    Args:
        dataset_codename: Name of the dataset to process
        
    Returns:
        Dictionary containing results for this dataset
    """
    print(f"\nProcessing dataset: {dataset_codename}")
    
    # For storing classifier info
    classifiers = {}  # Store trained classifiers
    predictions = {}  # Store classifier predictions
    feature_values = {}  # Store feature values
    actual_labels = None  # Will store ground truth labels
    classifier_keys = []  # Keys to identify each classifier
    data_lengths = {}  # Store data lengths for each classifier
    
    # Process each model and feature
    for model_codename, features_from_experiment in METRIC_CODENAMES_TO_TEST.items():
        
        model_displayname = MODEL_CODENAME_TO_MODEL_DISPLAYNAME[model_codename]
        df_path = f"{EXPERIMENT_FOLDER_NAME}/{model_codename}_{dataset_codename}_dataset/raw_data.csv"
        
        # Load the CSV once per model
        try:
            df = pd.read_csv(df_path)
            df = df.replace([np.inf, -np.inf], np.nan)
        except Exception as e:
            print(f"  Error loading dataset for {model_codename} on {dataset_codename}: {e}")
            continue
            
        for feature in features_from_experiment:
            # Create a key for this classifier
            classifier_key = f"{model_displayname}_{feature}"
            
            try:
                # Drop NaN only for the specific feature and y_labels
                feature_df = df.dropna(subset=[feature, "y_labels"])
                
                if len(feature_df) < 50:  # Skip if too few samples
                    print(f"  Skipping {classifier_key} - insufficient data")
                    continue
                
                # Store feature values and labels
                feature_values[classifier_key] = feature_df[feature].values
                data_lengths[classifier_key] = len(feature_df)
                
                if actual_labels is None:
                    actual_labels = feature_df["y_labels"].values
                
                # Create and train the classifier
                X = feature_df[[feature]].values
                y = feature_df["y_labels"].values
                classifier = create_logistic_regression_classifier(X, y)
                
                # Store the classifier and its predictions
                classifiers[classifier_key] = classifier
                predictions[classifier_key] = classifier.predict_proba(X)[:, 1]
                classifier_keys.append(classifier_key)
                
                print(f"  Processed {classifier_key}")
                
            except Exception as e:
                print(f"  Error processing {classifier_key}: {e}")
                continue
            
    
    # Calculate agreement statistics for all pairs
    n_classifiers = len(classifier_keys)
    
    # Initialize matrices for each statistic
    kappa_matrix = np.zeros((n_classifiers, n_classifiers))
    q_statistic_matrix = np.zeros((n_classifiers, n_classifiers))
    mutual_info_matrix = np.zeros((n_classifiers, n_classifiers))
    
    # Calculate statistics for all pairs
    for i in range(n_classifiers):
        for j in range(n_classifiers):
            key_i = classifier_keys[i]
            key_j = classifier_keys[j]
            
            if i == j:
                # Perfect agreement with self
                kappa_matrix[i, j] = 1.0
                q_statistic_matrix[i, j] = 1.0
                mutual_info_matrix[i, j] = 1.0
            else:
                # Calculate agreement statistics
                pred_i = predictions[key_i]
                pred_j = predictions[key_j]
                
                # Ensure arrays have the same length
                min_length = min(len(pred_i), len(pred_j), len(actual_labels))
                pred_i_trimmed = pred_i[:min_length]
                pred_j_trimmed = pred_j[:min_length]
                actual_labels_trimmed = actual_labels[:min_length]
                
                kappa, q, mi = calculate_agreement_statistics(pred_i_trimmed, pred_j_trimmed, actual_labels_trimmed)
                
                kappa_matrix[i, j] = kappa
                q_statistic_matrix[i, j] = q
                mutual_info_matrix[i, j] = mi
    
    # Create heatmaps
    plot_agreement_heatmap(kappa_matrix, classifier_keys, 'kappa', dataset_codename)
    plot_agreement_heatmap(q_statistic_matrix, classifier_keys, 'q_statistic', dataset_codename, vmin=-1, vmax=1)
    plot_agreement_heatmap(mutual_info_matrix, classifier_keys, 'mutual_info', dataset_codename)
    
    # Calculate average statistics for each classifier (how independent it is from others)
    average_kappa = np.zeros(n_classifiers)
    average_q_statistic = np.zeros(n_classifiers)
    average_mutual_information = np.zeros(n_classifiers)
    
    for i in range(n_classifiers):
        # Calculate average agreement with all other classifiers (excluding self)
        others = list(range(n_classifiers))
        others.remove(i)
        
        average_kappa[i] = np.mean(kappa_matrix[i, others])
        average_q_statistic[i] = np.mean(q_statistic_matrix[i, others])
        average_mutual_information[i] = np.mean(mutual_info_matrix[i, others])
    
    summary_df = pd.DataFrame({
        'Classifier': classifier_keys,
        'Average_Kappa': average_kappa,
        'Average_Q_Statistic': average_q_statistic,
        'Average_Mutual_Info': average_mutual_information,
    })
    
    summary_df = summary_df.sort_values('Average_Q_Statistic')
    
    summary_path = f"{ANALYSIS_OUTPUT_FOLDER_NAME}/{ANALYSIS_NAME}/{dataset_codename}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")
    
    # Free memory explicitly
    plt.close('all')
    gc.collect()
    
    return {
        'kappa': kappa_matrix,
        'q_statistic': q_statistic_matrix,
        'mutual_info': mutual_info_matrix,
        'classifier_keys': classifier_keys,
        'summary': summary_df
    }


def aggregate_results(all_results):
    """
    Aggregate results across all datasets
    
    Args:
        all_results: Dictionary of results for each dataset
    """
    all_keys = set()
    for dataset_results in all_results.values():
        all_keys.update(dataset_results['classifier_keys'])
    
    all_keys = sorted(list(all_keys))
    n_keys = len(all_keys)
    
    agg_kappa = np.zeros((n_keys, n_keys))
    agg_q = np.zeros((n_keys, n_keys))
    agg_mi = np.zeros((n_keys, n_keys))
    
    count_matrix = np.zeros((n_keys, n_keys))
    
    for dataset_codename, results in all_results.items():
        keys = results['classifier_keys']
        
        for i, key_i in enumerate(keys):
            i_global = all_keys.index(key_i)
            
            for j, key_j in enumerate(keys):
                j_global = all_keys.index(key_j)
                
                agg_kappa[i_global, j_global] += results['kappa'][i, j]
                agg_q[i_global, j_global] += results['q_statistic'][i, j]
                agg_mi[i_global, j_global] += results['mutual_info'][i, j]
                count_matrix[i_global, j_global] += 1
    
    mask = count_matrix > 0
    agg_kappa[mask] /= count_matrix[mask]
    agg_q[mask] /= count_matrix[mask]
    agg_mi[mask] /= count_matrix[mask]
    
    np.fill_diagonal(agg_kappa, 1.0)
    np.fill_diagonal(agg_q, 1.0)
    np.fill_diagonal(agg_mi, 1.0)
    
    plot_agreement_heatmap(agg_kappa, all_keys, 'kappa', 'aggregated')
    plot_agreement_heatmap(agg_q, all_keys, 'q_statistic', 'aggregated', vmin=-1, vmax=1)
    plot_agreement_heatmap(agg_mi, all_keys, 'mutual_info', 'aggregated')
    
    average_kappa = np.zeros(n_keys)
    average_q_statistic = np.zeros(n_keys)
    average_mutual_information = np.zeros(n_keys)
    
    for i in range(n_keys):
        others = list(range(n_keys))
        others.remove(i)
        
        kappa_values = [agg_kappa[i, j] for j in others if count_matrix[i, j] > 0]
        q_values = [agg_q[i, j] for j in others if count_matrix[i, j] > 0]
        mi_values = [agg_mi[i, j] for j in others if count_matrix[i, j] > 0]
        
        average_kappa[i] = np.mean(kappa_values) if kappa_values else np.nan
        average_q_statistic[i] = np.mean(q_values) if q_values else np.nan
        average_mutual_information[i] = np.mean(mi_values) if mi_values else np.nan
    

    agg_summary_df = pd.DataFrame({
        'Classifier': all_keys,
        'Average_Kappa': average_kappa,
        'Average_Q_Statistic': average_q_statistic,
        'Average_Mutual_Info': average_mutual_information,
    })
    

    agg_summary_df = agg_summary_df.sort_values('Average_Q_Statistic')
    
    agg_summary_path = f"{ANALYSIS_OUTPUT_FOLDER_NAME}/{ANALYSIS_NAME}/aggregated_summary.csv"
    agg_summary_df.to_csv(agg_summary_path, index=False)
    print(f"\nSaved aggregated summary to {agg_summary_path}")
    
    print("\nTop 5 most independent classifiers (lowest Q-statistic):")
    print(agg_summary_df.head(5)[['Classifier', 'Average_Q_Statistic']])
    
    print("\nTop 5 most dependent classifiers (highest Q-statistic):")
    print(agg_summary_df.tail(5)[['Classifier', 'Average_Q_Statistic']])




def main():
    print(f"Starting error independence analysis for {len(DATASET_CODENAMES_TO_TEST)} datasets")
    
    all_results = {}
    for dataset_codename in DATASET_CODENAMES_TO_TEST:
        all_results[dataset_codename] = process_dataset(dataset_codename)
    
    aggregate_results(all_results)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()