import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.insert(0, os.getcwd())

import pandas as pd
import numpy as np

from confidenceinterval import f1_score, roc_auc_score

import yaml, json
from copy import deepcopy

from llm_text_detectors.utils import create_logistic_regression_classifier



### START GLOBALS -------------------------------------------------------------------------

SHOULD_PERFORM_TRANSFERABILITY_TEST = False
SHOULD_AVERAGE_RESULTS_ACROSS_REFERENCE_MODEL = False

EXPERIMENT_FOLDER_NAME = "experiment_results"
ANALYSIS_OUTPUT_FOLDER_NAME = "experiment_analyses"
RAW_RESULTS_FILE_NAME = "experiment_analyses/raw_results"

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

    "gpt_neo_2_7B": ["telescope_perplexity", "binoculars_score", "perplexity", "log_rank_ratio", "fast_detectgpt"],
    "gpt_j_6B": ["telescope_perplexity", "binoculars_score", "perplexity", "log_rank_ratio", "fast_detectgpt"]
}

DATASET_CODENAMES_TO_TEST = [
    "detect_llm_text",
    "ai_human",
    "hc3",
    "hc3_plus",
    "esl_gpt4o",
    # # "m4_multilingual",
    # # "m4_monolingual",
    
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

    "ghostbusters_perturb_char_basic_50",
    "ghostbusters_perturb_char_cap_50",
    "ghostbusters_perturb_char_space_50",
    "ghostbusters_perturb_para_adj_5",
    "ghostbusters_perturb_para_paraph_5",
    "ghostbusters_perturb_sent_adj_5",
    "ghostbusters_perturb_sent_paraph_5",
    "ghostbusters_perturb_word_adj_50",
    "ghostbusters_perturb_word_syn_50",
    
    
    # "ghostbusters_creative_gpt4o_adversarial_prompt",
    # "ghostbusters_creative_gpt4o_low_temperature",
    # "ghostbusters_creative_gpt4o_high_temperature",
    # "ghostbusters_essay_gpt4o_adversarial_prompt",
    # "ghostbusters_essay_gpt4o_adversarial_prompt2",
    # "ghostbusters_essay_gpt4o_low_temperature",
    # "ghostbusters_essay_gpt4o_high_temperature",
    # "m4_arabic_chatgpt",
    # "m4_chinese_chatgpt",
    # "m4_english_wikipedia_chatgpt",
    # "m4_russian_chatgpt",
    # "m4_urdu_chatgpt",
    

    # "ghostbusters_perturb_character_basic_50",
    # "ghostbusters_perturb_character_capitalization_50",
    # "ghostbusters_perturb_character_space_50",
    # "ghostbusters_perturb_paragraph_adjacent_5",
    # "ghostbusters_perturb_paragraph_paraphrase_5",
    # "ghostbusters_perturb_sentence_adjacent_5",
    # "ghostbusters_perturb_sentence_paraphrase_5",
    # "ghostbusters_perturb_word_adjacent_50",
    # "ghostbusters_perturb_word_synonym_50",


]

### END GLOBALS -------------------------------------------------------------------------



# a dictionary that maps a dataset's codename (for instance ghostbusters_essay_gpt) to a presentable, paper-ready name (for instance GB Essay ChatGPT)
DATASET_CODENAME_TO_DATASET_DISPLAYNAME = yaml.safe_load(open("config.yaml"))["dataset_codenames_to_dataset_displaynames"]

# a dictionary that maps a model's codename (for instance smollm2_360M) to a presentable, paper-ready name (for instance SmolLM2 360M)
MODEL_CODENAME_TO_MODEL_DISPLAYNAME = yaml.safe_load(open("config.yaml"))["model_codenames_to_model_displaynames"]




def generate_latex_table_from_data(result_dict, dataset_codenames_to_show, metric_codenames_to_show, score_name, score_type):
    
    for dataset_codename in dataset_codenames_to_show:
        print("\midrule\n\multirow{12}{*}" + r"{" + f"{DATASET_CODENAME_TO_DATASET_DISPLAYNAME[dataset_codename]}" + r"}")
        
        for model_codename, metric_codenames_from_experiment in metric_codenames_to_show.items():
            
            model_displayname = MODEL_CODENAME_TO_MODEL_DISPLAYNAME[model_codename]
            
            # figure out which metric to bold to highlight best performance if this is a raw score
            if score_type == float:
                best_metric_codenames: list[str] = []
                best_metric_score = -np.inf
                for metric_codename in metric_codenames_from_experiment:
                    score = result_dict[(dataset_codename, model_displayname, metric_codename)][score_name]
                    
                    if score > best_metric_score:
                        best_metric_codenames = [metric_codename,]
                        best_metric_score = score 
                        
                    if score == best_metric_score:
                        best_metric_codenames.append(metric_codename)
                
                
                
            stuff_to_print = ""
            for metric_codename in metric_codenames_from_experiment:
                score = result_dict[(dataset_codename, model_displayname, metric_codename)][score_name]
                
                if score_type == tuple:
                    if isinstance(score, (tuple, list, np.ndarray)) and len(score) >= 2 and not pd.isna(score[0]) and not pd.isna(score[1]):
                        stuff_to_print += f"& ({score[0]:.5f}, {score[1]:.5f}) "
                    else:
                        stuff_to_print += f"& (nan, nan) "
                
                if score_type == float:
                    if pd.isna(score):
                        stuff_to_print += f"& nan "
                    elif metric_codename in best_metric_codenames:
                        stuff_to_print += f"& \\textbf{{{score:.5f}}} "
                    else:
                        stuff_to_print += f"& {score:.5f} "
                        
                    
            print(f"& {model_displayname} {stuff_to_print} \\\\")
            
        print()
        



def generate_latex_table_from_data_averaged_across_reference_models(result_dict, dataset_codenames_to_show, metric_codenames_to_show, score_name, score_type): 
   
    # generate latex code for results averaged across reference models (AUROC)
    for dataset_codename in dataset_codenames_to_show:
        
        if score_type == float: # for normal score values
            total_scores = {metric_codename: 0 for metric_codenames_from_experiment in metric_codenames_to_show.values() for metric_codename in metric_codenames_from_experiment}
        
        elif score_type == tuple:  # for confidence intervals
            total_scores = {metric_codename: (0, 0) for metric_codenames_from_experiment in metric_codenames_to_show.values() for metric_codename in metric_codenames_from_experiment} 
        
        else:
            raise Exception("score_type not correct")
        
        
        number_of_each_metric = {metric_codename: 0 for metric_codenames_from_experiment in metric_codenames_to_show.values() for metric_codename in metric_codenames_from_experiment}

        # compute the total scores of each model-metric combination
        for model_codename, metric_codenames_from_experiment in metric_codenames_to_show.items():
                        
            model_displayname = MODEL_CODENAME_TO_MODEL_DISPLAYNAME[model_codename]

            for metric_codename in metric_codenames_from_experiment:
                
                score = result_dict[(dataset_codename, model_displayname, metric_codename)][score_name]
                
                if score_type == float:
                    if not pd.isna(score):
                        total_scores[metric_codename] += score
                        number_of_each_metric[metric_codename] += 1
                
                if score_type == tuple:
                    if isinstance(score, (tuple, list, np.ndarray)) and len(score) >= 2 and not pd.isna(score[0]) and not pd.isna(score[1]):
                        total_scores[metric_codename] = (total_scores[metric_codename][0] + score[0], total_scores[metric_codename][1] + score[1])
                        number_of_each_metric[metric_codename] += 1
        
        
        # figure out which metric to bold to highlight best performance if this is a raw score
        if score_type == float:
            best_metric_codenames: list[str] = []
            best_metric_score = -np.inf
            for metric_codename in metric_codenames_from_experiment:
                score = total_scores[metric_codename]
                
                if score > best_metric_score:
                    best_metric_codenames = [metric_codename,]
                    best_metric_score = score 
                    
                if score == best_metric_score:
                    best_metric_codenames.append(metric_codename)
                        
                        
        stuff_to_print = f"{DATASET_CODENAME_TO_DATASET_DISPLAYNAME[dataset_codename]}"
        for metric_codename, total_score in total_scores.items():
            
            if score_type == float:
                num = number_of_each_metric[metric_codename]
                if num > 0:
                    average_score = total_score / num
                    if metric_codename in best_metric_codenames:
                        stuff_to_print += f"& \\textbf{{{average_score:.5f}}}"
                    else:
                        stuff_to_print += f"& {average_score:.5f}"
                else:
                    stuff_to_print += f"& nan"
            
            if score_type == tuple:
                num = number_of_each_metric[metric_codename]
                if num > 0:
                    average_score = (total_score[0] / num, total_score[1] / num)
                    stuff_to_print += f"& ({average_score[0]:.5f}, {average_score[1]:.5f}) "
                else:
                    stuff_to_print += f"& (nan, nan) "
             
        print(stuff_to_print + "& \\\\")








def _calculate_interpolated_tpr(y_true, y_scores, target_fpr):
    """
    Helper function to calculate true positive rate at a fixed false positive rate
    using ROC interpolation. Used by both the point estimate and the bootstrap to ensure consistency.
    """
    desc_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[desc_indices]

    tps = np.cumsum(y_true_sorted, dtype=float)
    fps = np.cumsum(1 - y_true_sorted, dtype=float)
    
    n_pos = tps[-1]
    n_neg = fps[-1]
    
    if n_neg == 0 or n_pos == 0:
        return np.nan

    fprs = fps / n_neg
    tprs = tps / n_pos
    
    fprs = np.concatenate([[0.0], fprs])
    tprs = np.concatenate([[0.0], tprs])
    
    # Linearly interpolate the TPR value at the exact target_fpr
    return np.interp(target_fpr, fprs, tprs)


def bootstrap_tpr_at_fixed_fpr_confidence_interval(y_true, y_scores, target_fpr=0.05, n_bootstraps=1000, alpha=0.95, random_state=None):
    """
    Calculates confidence intervals for TPR at fixed FPR using bootstrapping.
    """
    rng = np.random.default_rng(random_state)
    
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    negatives_mask = (y_true == 0)
    positives_mask = (y_true == 1)
    
    neg_scores = y_scores[negatives_mask]
    pos_scores = y_scores[positives_mask]
    
    n_neg = len(neg_scores)
    n_pos = len(pos_scores)

    if n_neg == 0 or n_pos == 0: 
        return (np.nan, np.nan)

    tpr_bootstraps = np.zeros(n_bootstraps)
    
    # Pre-calculate the percentile target
    percentile_target = (1 - target_fpr) * 100

    for i in range(n_bootstraps):
        # Stratified resampling (preserve class balance of original sample)
        neg_resampled = rng.choice(neg_scores, size=n_neg, replace=True)
        pos_resampled = rng.choice(pos_scores, size=n_pos, replace=True)
        
        # Calculate threshold using linear interpolation
        threshold = np.percentile(neg_resampled, percentile_target, method='linear')
        
        # Calculate TPR using linear interpolation logic: 
        # (count of scores > threshold + fractional count for scores == threshold)
        # However, the standard way to interpolate TPR at a fixed FPR is via the ROC curve.
        pos_sorted = np.sort(pos_resampled)
        # Find where the threshold would sit in the positive distribution
        tpr = 1.0 - (np.searchsorted(pos_sorted, threshold, side='left') / n_pos)
        
        tpr_bootstraps[i] = tpr
    
    lower_p = (1.0 - alpha) / 2.0 * 100
    upper_p = (alpha + (1.0 - alpha) / 2.0) * 100
    
    return np.percentile(tpr_bootstraps, lower_p), np.percentile(tpr_bootstraps, upper_p)


# def tpr_at_fixed_fpr(y_true, y_scores, target_fpr=0.05):
#     """
#     Calculates true positive rate at a specific fixed false positive rate 
#     using linear interpolation of the ROC curve.
#     """
#     y_true = np.array(y_true)
#     y_scores = np.array(y_scores)
    
#     return _calculate_interpolated_tpr(y_true, y_scores, target_fpr)


def tpr_at_fixed_fpr(y_true, y_scores, target_fpr=0.05):
    """
    Calculates true positive rate at a specific fixed false positive rate.
    
    This function avoids the 'optimism bias' of linear interpolation by 
    calculating the exact threshold from the negative class distribution.

    Parameters:
    -----------
    y_true : array-like
        Ground truth labels (0 for negative, 1 for positive).
    y_scores : array-like
        Probability estimates or decision scores.
    target_fpr : float, optional (default=0.05)
        The maximum allowable False Positive Rate (e.g., 0.05 for 5%).

    Returns:
    --------
    float
        The True Positive Rate (Sensitivity) at the given FPR.
    """
    # Ensure inputs are numpy float arrays
    y_true = np.array(y_true).astype(float)
    y_scores = np.array(y_scores).astype(float)
    
    # Split scores by class
    neg_scores = y_scores[y_true == 0]
    pos_scores = y_scores[y_true == 1]
    
    if len(neg_scores) == 0 or len(pos_scores) == 0:
        return np.nan

    # Calculate the threshold using linear interpolation on the negative scores.
    threshold = np.percentile(neg_scores, 100 - (target_fpr * 100), method='linear')
    
    # Calculate the TPR. Using searchsorted on sorted positives provides 
    # the rank-based interpolation.
    pos_sorted = np.sort(pos_scores)
    tpr = 1.0 - (np.searchsorted(pos_sorted, threshold, side='left') / len(pos_scores))
    
    return tpr

def generate_roc_curve_from_metric():
    os.makedirs(ANALYSIS_OUTPUT_FOLDER_NAME, exist_ok=True)
    result_dict = {}
    for model_codename, metric_codenames_from_experiment in METRIC_CODENAMES_TO_TEST.items():
        
        model_displayname = MODEL_CODENAME_TO_MODEL_DISPLAYNAME[model_codename]
        
        for dataset_codename in DATASET_CODENAMES_TO_TEST:
            for metric_codename in metric_codenames_from_experiment:
                result_dict[(dataset_codename, model_displayname, metric_codename)] = dict()
                
                result_dict[(dataset_codename, model_displayname, metric_codename)]["AUROC Confidence Interval"] = np.nan
                result_dict[(dataset_codename, model_displayname, metric_codename)]["F1 Score Confidence Interval"] = np.nan
                result_dict[(dataset_codename, model_displayname, metric_codename)]["Transfered F1 Score Confidence Interval"] = np.nan
                result_dict[(dataset_codename, model_displayname, metric_codename)]["AUROC"] = np.nan
                result_dict[(dataset_codename, model_displayname, metric_codename)]["F1 Score"] = np.nan
                result_dict[(dataset_codename, model_displayname, metric_codename)]["Transfered F1 Score"] = np.nan
                result_dict[(dataset_codename, model_displayname, metric_codename)][r"TPR at FPR 5%"] = np.nan
                result_dict[(dataset_codename, model_displayname, metric_codename)][r"TPR at FPR 5% Confidence Interval"] = np.nan
                    
    
    for model_codename, metric_codenames_from_experiment in METRIC_CODENAMES_TO_TEST.items():
        
        model_displayname = MODEL_CODENAME_TO_MODEL_DISPLAYNAME[model_codename]
        
        for test_dataset_codename in DATASET_CODENAMES_TO_TEST:
            try:
                df = pd.read_csv(f"{EXPERIMENT_FOLDER_NAME}/{model_codename}_{test_dataset_codename}_dataset/raw_data.csv")
            except:
                print(f"{EXPERIMENT_FOLDER_NAME}/{model_codename}_{test_dataset_codename}_dataset/raw_data.csv failed")
                continue
            
            # print(len(df))

            for col in metric_codenames_from_experiment:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna(subset=metric_codenames_from_experiment)

            # Filter out any rows where y_labels is not a valid class identifier due to parsing/quote mismatches
            df = df[df["y_labels"].astype(str).isin(["0", "1", "0.0", "1.0"])]
            y_labels = df["y_labels"].astype(float).astype(int).astype(bool)
            
            if len(np.unique(y_labels)) < 2:
                print(f"  Warning: Skipping {test_dataset_codename} for {model_codename} because y_labels contains only one class: {np.unique(y_labels)}.")
                continue
            
            # print(f"MODEL: {model_displayname}, DATASET: {test_dataset_codename}")
            if SHOULD_PERFORM_TRANSFERABILITY_TEST:
                train_df_list = []
                for train_dataset_index, train_dataset_codename in enumerate(DATASET_CODENAMES_TO_TEST):
                    if (train_dataset_codename == test_dataset_codename): # don't test on the same dataset you train on
                        continue
                    try:
                        train_df = pd.read_csv(f"{EXPERIMENT_FOLDER_NAME}/{model_codename}_{train_dataset_codename}_dataset/raw_data.csv")
                        for col in metric_codenames_from_experiment:
                            if col in train_df.columns:
                                train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
                        train_df = train_df.replace([np.inf, -np.inf], np.nan)
                        train_df = train_df.dropna(subset=metric_codenames_from_experiment)
                        train_df = train_df[train_df["y_labels"].astype(str).isin(["0", "1", "0.0", "1.0"])]
                        train_df = train_df.head(2000)
                        train_df_list.append(train_df)
                    except:
                        pass
                combined_train_df = pd.concat(train_df_list) if train_df_list else None
            else:
                combined_train_df = None

            for metric_codename in metric_codenames_from_experiment:
                                
                if SHOULD_PERFORM_TRANSFERABILITY_TEST and combined_train_df is not None:
                    transfered_classifier = create_logistic_regression_classifier(combined_train_df[[metric_codename,]], combined_train_df["y_labels"])
                    predicted_labels_transfered_classifier = transfered_classifier.predict(df[[metric_codename,]])
                    best_f1score_transfered_classifier, f1_confidence_interval_transfered_classifier = f1_score(y_labels, predicted_labels_transfered_classifier)
                else:
                    best_f1score_transfered_classifier, f1_confidence_interval_transfered_classifier = 0, (0, 0)
                
                
                
                metric_scores = df[[metric_codename,]]
                fixed_scores_for_rocauc = deepcopy(df[metric_codename])
                if metric_codename == "binoculars_score" or metric_codename == "perplexity": 
                    fixed_scores_for_rocauc = -fixed_scores_for_rocauc
                
                
                roc_auc, roc_auc_confidence_interval = roc_auc_score(y_labels, fixed_scores_for_rocauc)
                
                classifier = create_logistic_regression_classifier(metric_scores, y_labels)
                predicted_labels = classifier.predict(df[[metric_codename,]])
                best_f1score, f1_confidence_interval = f1_score(y_labels, predicted_labels)
                tpr_at_fpr_5 = tpr_at_fixed_fpr(y_labels, fixed_scores_for_rocauc, target_fpr=0.05)
                
                
                
                tpr_at_fpr_5_confidence_interval = bootstrap_tpr_at_fixed_fpr_confidence_interval(y_true=y_labels, y_scores=fixed_scores_for_rocauc, target_fpr=0.05)
                roc_auc_confidence_interval = (float(roc_auc_confidence_interval[0]), float(roc_auc_confidence_interval[1]))
                f1_confidence_interval =  (float(f1_confidence_interval[0]), float(f1_confidence_interval[1]))
                f1_confidence_interval_transfered_classifier =  (float(f1_confidence_interval_transfered_classifier[0]), float(f1_confidence_interval_transfered_classifier[1]))

                # print(f"predicted labels length: {len(predicted_labels)}")

                result_dict[(test_dataset_codename, model_displayname, metric_codename)]["F1 Score Confidence Interval"] = f1_confidence_interval
                result_dict[(test_dataset_codename, model_displayname, metric_codename)]["F1 Score"] = float(best_f1score)
                result_dict[(test_dataset_codename, model_displayname, metric_codename)]["Transfered F1 Score Confidence Interval"] = f1_confidence_interval_transfered_classifier
                result_dict[(test_dataset_codename, model_displayname, metric_codename)]["Transfered F1 Score"] = float(best_f1score_transfered_classifier)
                result_dict[(test_dataset_codename, model_displayname, metric_codename)]["AUROC Confidence Interval"] = roc_auc_confidence_interval
                result_dict[(test_dataset_codename, model_displayname, metric_codename)]["AUROC"] = float(roc_auc)
                
                result_dict[(test_dataset_codename, model_displayname, metric_codename)]["TPR at FPR 5%"] = tpr_at_fpr_5
                result_dict[(test_dataset_codename, model_displayname, metric_codename)]["TPR at FPR 5% Confidence Interval"] = tpr_at_fpr_5_confidence_interval


                # print(f"MODEL: {model_displayname}, DATASET: {test_dataset_codename}, METRIC: {metric_codename}, AUROC: {roc_auc}, {roc_auc_confidence_interval}, {best_f1score}, {f1_confidence_interval}")
                # print(f"MODEL: {model_displayname}, DATASET: {test_dataset_codename}, METRIC: {metric_codename}, AUROC: {roc_auc}")
               

    # generate mostly paper ready latex code from the data
    print("\n\n")
   

    generate_latex_table_from_data_averaged_across_reference_models(result_dict, DATASET_CODENAMES_TO_TEST, METRIC_CODENAMES_TO_TEST, "F1 Score", float)
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

    # generate_latex_table_from_data(result_dict, DATASET_CODENAMES_TO_TEST, METRIC_CODENAMES_TO_TEST, "TPR at FPR 5%", float)
    
    if not SHOULD_PERFORM_TRANSFERABILITY_TEST and not SHOULD_AVERAGE_RESULTS_ACROSS_REFERENCE_MODEL:
        print("AUROC:")
        generate_latex_table_from_data(result_dict, DATASET_CODENAMES_TO_TEST, METRIC_CODENAMES_TO_TEST, "AUROC", float)
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
        print("AUROC Confidence Interval:")
        generate_latex_table_from_data(result_dict, DATASET_CODENAMES_TO_TEST, METRIC_CODENAMES_TO_TEST, "AUROC Confidence Interval", tuple)
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
        print("TPR at FPR 5%:")
        generate_latex_table_from_data(result_dict, DATASET_CODENAMES_TO_TEST, METRIC_CODENAMES_TO_TEST, "TPR at FPR 5%", float)
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
        print("TPR at FPR 5% Confidence Interval")
        generate_latex_table_from_data(result_dict, DATASET_CODENAMES_TO_TEST, METRIC_CODENAMES_TO_TEST, "TPR at FPR 5% Confidence Interval", tuple)
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
    
    if not SHOULD_PERFORM_TRANSFERABILITY_TEST and SHOULD_AVERAGE_RESULTS_ACROSS_REFERENCE_MODEL: 
        print("AUROC:")
        generate_latex_table_from_data_averaged_across_reference_models(result_dict, DATASET_CODENAMES_TO_TEST, METRIC_CODENAMES_TO_TEST, "AUROC", float)
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
        print("AUROC Confidence Interval:")
        generate_latex_table_from_data_averaged_across_reference_models(result_dict, DATASET_CODENAMES_TO_TEST, METRIC_CODENAMES_TO_TEST, "AUROC Confidence Interval", tuple)
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
        print("TPR at FPR 5%:")
        generate_latex_table_from_data_averaged_across_reference_models(result_dict, DATASET_CODENAMES_TO_TEST, METRIC_CODENAMES_TO_TEST, "TPR at FPR 5%", float)
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
        print("TPR at FPR 5% Confidence Interval")
        generate_latex_table_from_data_averaged_across_reference_models(result_dict, DATASET_CODENAMES_TO_TEST, METRIC_CODENAMES_TO_TEST, "TPR at FPR 5% Confidence Interval", tuple)
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
    
    if SHOULD_PERFORM_TRANSFERABILITY_TEST and not SHOULD_AVERAGE_RESULTS_ACROSS_REFERENCE_MODEL:
        print("F1 Score:")
        generate_latex_table_from_data(result_dict, DATASET_CODENAMES_TO_TEST, METRIC_CODENAMES_TO_TEST, "Transfered F1 Score", float)
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
        print("F1 Score Confidence Interval:")
        generate_latex_table_from_data(result_dict, DATASET_CODENAMES_TO_TEST, METRIC_CODENAMES_TO_TEST, "Transfered F1 Score Confidence Interval", tuple)
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
    
    if SHOULD_PERFORM_TRANSFERABILITY_TEST and SHOULD_AVERAGE_RESULTS_ACROSS_REFERENCE_MODEL:
        print("F1 Score:")
        generate_latex_table_from_data_averaged_across_reference_models(result_dict, DATASET_CODENAMES_TO_TEST, METRIC_CODENAMES_TO_TEST, "Transfered F1 Score", float)
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
        print("F1 Score Confidence Interval:")
        generate_latex_table_from_data_averaged_across_reference_models(result_dict, DATASET_CODENAMES_TO_TEST, METRIC_CODENAMES_TO_TEST, "Transfered F1 Score Confidence Interval", tuple)
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
    
    
    
    # Save to a json so that we can use these results easily in other things
    # Make sure all of the data gets into the right format since without this, we can't convert the data to a json
    corrected_result_dict = {}
    for model_codename, metric_codenames_from_experiment in METRIC_CODENAMES_TO_TEST.items():
        
        model_displayname = MODEL_CODENAME_TO_MODEL_DISPLAYNAME[model_codename]
        
        corrected_result_dict[model_displayname] = {}
        for metric_codename in metric_codenames_from_experiment:
            corrected_result_dict[model_displayname][metric_codename] = {}
            for dataset_codename in DATASET_CODENAMES_TO_TEST:
                corrected_result_dict[model_displayname][metric_codename][dataset_codename] = result_dict[(dataset_codename, model_displayname, metric_codename)]
    
    with open("results_data.json", "w") as file:
        json.dump(corrected_result_dict, file)
    
    
    
    

if __name__ == "__main__":
    generate_roc_curve_from_metric()
