import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from llm_text_detectors.utils import calc_bins


### START GLOBALS -------------------------------------------------------------------------

EXPERIMENT_FOLDER_NAME = "experiment_results"
ANALYSIS_OUTPUT_FOLDER_NAME = "experiment_analyses"
ANALYSIS_NAME = "threshold_transfer"


SOURCE_DATASET_CODENAME = "ghostbusters_essay_gpt4o"
TARGET_DATASET_CODENAME = "ghostbusters_essay_claude"

MODEL_CODENAME_TO_TEST = "falcon_7B"
METRIC_CODENAMES_TO_TEST = ["telescope_perplexity", "binoculars_score"]

### END GLOBALS -------------------------------------------------------------------------




def create_output_folders():
    """Create output directory structure."""
    # Create main output directory and subfolders
    os.makedirs(f"{ANALYSIS_OUTPUT_FOLDER_NAME}/{ANALYSIS_NAME}", exist_ok=True)

def load_data(model_codename, dataset_codename):
    """Load data from a specific model and dataset combination."""
    file_path = f"{EXPERIMENT_FOLDER_NAME}/{model_codename}_{dataset_codename}_dataset/raw_data.csv"
    df = pd.read_csv(file_path)
    df['dataset'] = dataset_codename  # Add dataset name to the dataframe
    return df


def find_optimal_threshold(data, metric_codename):
    """Find optimal threshold that maximizes accuracy on the source dataset."""
    # Get values and true labels
    values = data[metric_codename].values
    labels = data['y_labels'].values
    
    # Try different thresholds and find the one with best accuracy
    thresholds = np.sort(values)
    best_acc = 0
    best_threshold = 0
    
    for threshold in thresholds:
        # Predict AI if metric value is below threshold (assuming lower values indicate AI)
        # This may need to be reversed depending on your specific metrics
        predictions = (values < threshold).astype(int)
        
        # Calculate accuracy
        accuracy = (predictions == labels).mean()
        
        # If this approach gives consistently poor results, we might be using the wrong direction
        if accuracy < 0.5:
            predictions = (values >= threshold).astype(int)
            accuracy = (predictions == labels).mean()
        
        if accuracy > best_acc:
            best_acc = accuracy
            best_threshold = threshold
    
    return best_threshold, best_acc




def create_threshold_transfer_plot(
        source_data, target_data, metric_codename, model_codename, 
        source_dataset_codename, target_dataset_codename, output_path
    ):
    
    """Create and save a histogram plot with transferred threshold."""
    plt.figure(figsize=(14, 8))
    
    # Get target dataset data for each class
    target_human_data = target_data[target_data['y_labels'] == 0][metric_codename]
    target_ai_data = target_data[target_data['y_labels'] == 1][metric_codename]
    
    # Calculate optimal bins for target data
    target_all_data = target_data[metric_codename]
    n_bins = calc_bins(target_all_data)
    
    # Create histograms for target dataset
    plt.hist(target_human_data, bins=n_bins, alpha=0.5, label=f'Target Human (n={len(target_human_data)})', color='lightblue', density=True)
    plt.hist(target_ai_data, bins=n_bins, alpha=0.5, label=f'Target AI (n={len(target_ai_data)})', color='lightcoral', density=True)
    
    # Find optimal threshold on source dataset
    source_threshold, source_accuracy = find_optimal_threshold(source_data, metric_codename)
    
    # Calculate performance on target dataset using source threshold
    target_values = target_data[metric_codename].values
    target_labels = target_data['y_labels'].values
    
    # Predict using the source threshold
    # Adjust the comparison based on your metric (< or > threshold)
    predictions = (target_values < source_threshold).astype(int)
    
    # If accuracy is less than 0.5, we might need to flip the prediction direction
    target_accuracy = (predictions == target_labels).mean()
    if target_accuracy < 0.5:
        predictions = (target_values >= source_threshold).astype(int)
        target_accuracy = (predictions == target_labels).mean()
    
    # Calculate confusion matrix for additional metrics
    tn, fp, fn, tp = confusion_matrix(target_labels, predictions).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
    
    # Draw the threshold line
    plt.axvline(source_threshold, color='purple', linestyle='-', linewidth=2, label=f'Source Threshold: {source_threshold:.3f}')
    
    # Add title and labels
    title = (
        f'Distribution of {metric_codename} for {model_codename}\n' 
        f'Source: {source_dataset_codename} (Acc: {source_accuracy:.3f}) → ' 
        f'Target: {target_dataset_codename} (Acc: {target_accuracy:.3f})'
    )
    
    plt.title(title)
    plt.xlabel(metric_codename)
    plt.ylabel('Density')
    
    # Add performance metrics annotation
    plt.figtext(
        0.15, 0.02, 
        f"Target Performance with Source Threshold:\n"
        f"Accuracy: {target_accuracy:.3f}, Sensitivity: {sensitivity:.3f}, "
        f"Specificity: {specificity:.3f}",
        bbox=dict(facecolor='white', alpha=0.8)
    )
    
    # Add median lines for reference
    plt.axvline(target_human_data.median(), color='blue', linestyle='--', alpha=0.5, label=f'Human median: {target_human_data.median():.3f}')
    plt.axvline(target_ai_data.median(), color='red', linestyle='--', alpha=0.5, label=f'AI median: {target_ai_data.median():.3f}')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Return performance metrics for logging
    return {
        'source_threshold': source_threshold,
        'source_accuracy': source_accuracy,
        'target_accuracy': target_accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity
    }




def main():
    
    # Create folder structure
    create_output_folders()
    
    try:
        # Load the datasets
        source_data = load_data(MODEL_CODENAME_TO_TEST, SOURCE_DATASET_CODENAME)
        target_data = load_data(MODEL_CODENAME_TO_TEST, TARGET_DATASET_CODENAME)
        
        # Process each metric
        for metric_codename in METRIC_CODENAMES_TO_TEST:
            # Create the output path
            output_path = (
                f"{ANALYSIS_OUTPUT_FOLDER_NAME}/{ANALYSIS_NAME}/{MODEL_CODENAME_TO_TEST}_"
                f"{metric_codename}_{SOURCE_DATASET_CODENAME}_to_{TARGET_DATASET_CODENAME}.png"
            )
            
            # Create the threshold transfer plot
            results = create_threshold_transfer_plot(
                source_data, target_data, metric_codename, MODEL_CODENAME_TO_TEST,
                SOURCE_DATASET_CODENAME, TARGET_DATASET_CODENAME, output_path
            )
            
            # Save performance metrics to text file
            stats_path = (
                f"{ANALYSIS_OUTPUT_FOLDER_NAME}/{ANALYSIS_NAME}/"
                f"{MODEL_CODENAME_TO_TEST}_{metric_codename}_{SOURCE_DATASET_CODENAME}_to_{TARGET_DATASET_CODENAME}_stats.txt"
            )
            
            with open(stats_path, 'w') as f:
                f.write(f"Threshold Transfer Results - {MODEL_CODENAME_TO_TEST} - {metric_codename}\n")
                f.write(f"Source Dataset: {SOURCE_DATASET_CODENAME}\n")
                f.write(f"Target Dataset: {TARGET_DATASET_CODENAME}\n\n")
                f.write(f"Source Optimal Threshold: {results['source_threshold']:.6f}\n")
                f.write(f"Source Accuracy: {results['source_accuracy']:.4f}\n\n")
                f.write(f"Target Performance with Source Threshold:\n")
                f.write(f"Accuracy: {results['target_accuracy']:.4f}\n")
                f.write(f"Sensitivity (TPR): {results['sensitivity']:.4f}\n")
                f.write(f"Specificity (TNR): {results['specificity']:.4f}\n")
                
            print(f"Created threshold transfer plot for {metric_codename}: {SOURCE_DATASET_CODENAME} → {TARGET_DATASET_CODENAME}")
            
            
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        print("Please check that the specified model and datasets exist in the experiment results folder.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")




if __name__ == "__main__":
    main()