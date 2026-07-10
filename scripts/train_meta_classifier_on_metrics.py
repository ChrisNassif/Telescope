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



model_features = {
    "smollm_360M": [
        "telescope_perplexity",
        # "performer_model_entropy",
        # "kl_divergence",
        "perplexity"
    ],
    "gemma2_9B": [
        "telescope_perplexity",
        "binoculars_score",
        "perplexity",
        # "cross_perplexity",
        # "kl_divergence",
        # "performer_model_entropy",
        "log_rank_ratio",
        "fast_detectgpt"
    ],
    "falcon_7B": [
        "telescope_perplexity",
        "binoculars_score",
        "perplexity",
        # "cross_perplexity",
        # "kl_divergence",
        # "performer_model_entropy",
        "log_rank_ratio"
    ],
    "llama3_8B": [
        "telescope_perplexity",
        "binoculars_score",
        "perplexity",
        # "cross_perplexity",
        # "kl_divergence",
        # "performer_model_entropy",
        "log_rank_ratio"
    ]
}

datasets_to_test = [
    "detect_llm_text",
    "ai_human",
    "hc3",
    "hc3_plus",
    "esl_gpt4o"
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
    classifier: Pipeline = make_pipeline(StandardScaler(), LogisticRegression())
    classifier.fit(features_df, labels)
    return classifier

def evaluate_classifier(classifier: Pipeline, features_df: pd.DataFrame, labels: np.ndarray) -> Dict[str, float]:
    """
    Evaluates a classifier using ROC AUC and F1 score
    
    Args:
        classifier: Trained classifier
        features_df: DataFrame containing features
        labels: Array of true labels
    
    Returns:
        Dictionary containing evaluation metrics
    """
    predictions = classifier.predict_proba(features_df)[:, 1]  # Get probability of positive class
    
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
    
    # Ensure output directory exists
    os.makedirs(ANALYSIS_OUTPUT_FOLDER_NAME, exist_ok=True)
    
    for model_name, features_list in model_features.items():
        results[model_name] = {}
        print(f"\nProcessing model: {model_name}")
        
        for train_dataset in datasets_to_test:
            train_path = f"{EXPERIMENT_FOLDER_NAME}/{model_name}_{train_dataset}_dataset/raw_data.csv"
            if not os.path.exists(train_path):
                print(f"Skipping training dataset: {train_dataset} (path not found)")
                continue
                
            print(f"Training on {train_dataset}...")
            train_df = pd.read_csv(train_path)
            # Replace infinity values with NaN so they get dropped
            train_df = train_df.replace([np.inf, -np.inf], np.nan)
            
            # Select only features that actually exist in this raw data
            available_features = [f for f in features_list if f in train_df.columns]
            
            # Drop rows with NaN and infinite values in the selected features
            initial_train_len = len(train_df)

            train_df = train_df.dropna(subset=available_features)
            if len(train_df) == 0:
                print(f"  Warning: Skipping training on {train_dataset} because it contains no valid samples after dropping NaNs/Infs for features {available_features}.")
                continue
            if len(train_df) < initial_train_len:
                print(f"  Warning: Dropping {initial_train_len - len(train_df)} rows containing NaNs/Infs from training data.")
                
            X_train = train_df[available_features]
            y_train = train_df["y_labels"]
            classifier = train_multi_feature_classifier(X_train, y_train)
            
            results[model_name][train_dataset] = {}
            
            for eval_dataset in datasets_to_test:
                eval_path = f"{EXPERIMENT_FOLDER_NAME}/{model_name}_{eval_dataset}_dataset/raw_data.csv"
                if not os.path.exists(eval_path):
                    continue
                
                eval_df = pd.read_csv(eval_path)
                # Replace infinity values with NaN so they get dropped
                eval_df = eval_df.replace([np.inf, -np.inf], np.nan)
                
                # Check if all training features exist in evaluation dataset
                missing_features = [f for f in available_features if f not in eval_df.columns]
                if missing_features:
                    print(f"  Warning: Skipping evaluation on {eval_dataset} because it is missing features: {missing_features}.")
                    continue
                
                # Drop rows with NaN and infinite values in the selected features
                initial_eval_len = len(eval_df)
                eval_df = eval_df.dropna(subset=available_features)
                if len(eval_df) == 0:
                    print(f"  Warning: Skipping evaluation on {eval_dataset} because it contains no valid samples after dropping NaNs/Infs for features {available_features}.")
                    continue
                if len(eval_df) < initial_eval_len:
                    print(f"  Warning: Dropping {initial_eval_len - len(eval_df)} rows containing NaNs/Infs from evaluation data.")
                    
                X_eval = eval_df[available_features]
                y_eval = eval_df["y_labels"]
                
                metrics = evaluate_classifier(classifier, X_eval, y_eval)
                results[model_name][train_dataset][eval_dataset] = metrics
                print(f"  Eval on {eval_dataset} -> ROC AUC: {metrics['roc_auc']:.3f}, F1: {metrics['f1_score']:.3f}")
    
    with open(f"{ANALYSIS_OUTPUT_FOLDER_NAME}/{ANALYSIS_NAME}_separate_models_zeroshot.pkl", "wb") as f:
        pickle.dump(results, f)
    
    summary_data = []
    for model_name in results:
        for train_dataset in results[model_name]:
            for eval_dataset in results[model_name][train_dataset]:
                metrics = results[model_name][train_dataset][eval_dataset]
                summary_data.append({
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