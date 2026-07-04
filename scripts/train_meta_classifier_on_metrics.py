import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import pandas as pd
import pickle
from typing import Dict

EXPERIMENT_FOLDER_NAME = "experiment_results"
ANALYSIS_OUTPUT_FOLDER_NAME = "experiment_analyses"
ANALYSIS_NAME = "telescope_variants_performance"

# Separate features by model
model_features = {
    "telescope_smollm_360M_telescope_gemma2_9B": {
        "smollm": [
            "score",
            "performer_model_entropy",
            "kl_divergence",
            "perplexity"
        ],
    }
}

datasets_to_test = [
    "essay",
    "ai_human",
    "hc3",
    "hc3_plus",
    "custom4o"
]

def train_multi_feature_classifier(features_df: pd.DataFrame, labels: np.ndarray) -> Pipeline:
    """
    Trains a logistic regression classifier using multiple features
    
    Args:
        features_df: DataFrame containing all features to use
        labels: Array of binary labels
    
    Returns:
        Trained classifier pipeline
    """
    clf: Pipeline = make_pipeline(StandardScaler(), LogisticRegression())
    clf.fit(features_df, labels)
    return clf

def evaluate_classifier(clf: Pipeline, features_df: pd.DataFrame, labels: np.ndarray) -> Dict[str, float]:
    """
    Evaluates a classifier using ROC AUC and F1 score
    
    Args:
        clf: Trained classifier
        features_df: DataFrame containing features
        labels: Array of true labels
    
    Returns:
        Dictionary containing evaluation metrics
    """
    predictions = clf.predict_proba(features_df)[:, 1]  # Get probability of positive class
    
    # Calculate ROC AUC
    fpr, tpr, _ = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)
    
    # Calculate F1 score
    precision, recall, thresholds = precision_recall_curve(labels, predictions)
    f1_scores = 2 * recall * precision / (recall + precision)
    best_f1score = np.max(f1_scores[~np.isnan(f1_scores)])
    
    return {
        "roc_auc": roc_auc,
        "f1_score": best_f1score
    }

def main():
    results = {}
    
    for experiment_prefix, model_feature_groups in model_features.items():
        results[experiment_prefix] = {}
        
        for model_name, features_list in model_feature_groups.items():
            print(f"\nProcessing {model_name} features...")
            results[experiment_prefix][model_name] = {}
            
            for train_dataset in datasets_to_test:
                print(f"Training on {train_dataset}...")
                
                train_df = pd.read_csv(f"{EXPERIMENT_FOLDER_NAME}/{experiment_prefix}_{train_dataset}_dataset/raw_data.csv")
                
                X_train = train_df[features_list]
                y_train = train_df["y_labels"]
                classifier = train_multi_feature_classifier(X_train, y_train)
                
                for eval_dataset in datasets_to_test:
                    print(f"Evaluating on {eval_dataset}...")
                    
                    eval_df = pd.read_csv(f"{EXPERIMENT_FOLDER_NAME}/{experiment_prefix}_{eval_dataset}_dataset/raw_data.csv")
                    X_eval = eval_df[features_list]
                    y_eval = eval_df["y_labels"]
                    
                    metrics = evaluate_classifier(classifier, X_eval, y_eval)
                    
                    if train_dataset not in results[experiment_prefix][model_name]:
                        results[experiment_prefix][model_name][train_dataset] = {}
                    results[experiment_prefix][model_name][train_dataset][eval_dataset] = metrics
                    
                    print(f"Model: {model_name}, ROC AUC: {metrics['roc_auc']:.3f}, F1 Score: {metrics['f1_score']:.3f}")
    

    with open(f"{ANALYSIS_OUTPUT_FOLDER_NAME}/{ANALYSIS_NAME}_separate_models_zeroshot.pkl", "wb") as f:
        pickle.dump(results, f)
    
    summary_data = []
    for exp_prefix in results:
        for model_name in results[exp_prefix]:
            for train_dataset in results[exp_prefix][model_name]:
                for eval_dataset in results[exp_prefix][model_name][train_dataset]:
                    metrics = results[exp_prefix][model_name][train_dataset][eval_dataset]
                    summary_data.append({
                        "Experiment": exp_prefix,
                        "Model": model_name,
                        "Train Dataset": train_dataset,
                        "Eval Dataset": eval_dataset,
                        "ROC AUC": metrics["roc_auc"],
                        "F1 Score": metrics["f1_score"]
                    })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f"{ANALYSIS_OUTPUT_FOLDER_NAME}/{ANALYSIS_NAME}_separate_models_zeroshot_summary.csv", index=False)



if __name__ == "__main__":
    main()