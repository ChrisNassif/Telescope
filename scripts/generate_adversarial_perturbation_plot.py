# Deterministic colors are fetched from config.yaml

import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import yaml



### START GLOBALS -------------------------------------------------------------------------

EXPERIMENT_FOLDER_NAME = "experiment_results"
ANALYSIS_OUTPUT_FOLDER_NAME = "experiment_analyses"


METRIC_CODENAMES_TO_TEST = {
    # "gemma2_2B": ["binoculars_score"],
    "gemma2_9B": ["telescope_perplexity", "log_rank_ratio"],
    # "llama3_8B": ["telescope_perplexity", "binoculars_score", "perplexity", "log_rank_ratio"],
    "falcon_7B":  ["binoculars_score", "perplexity"],
    
    # "smollm_135M": ["telescope_perplexity", "binoculars_score", "perplexity", "log_rank_ratio"],
    # "smollm_360M": ["telescope_perplexity", "log_rank_ratio"],
    # "smollm_1_7B": ["telescope_perplexity", "binoculars_score", "perplexity", "log_rank_ratio"],
    # "smollm2_135M": ["binoculars_score"],
    # "smollm2_360M": ["telescope_perplexity", "binoculars_score", "perplexity", "log_rank_ratio"],
    # "smollm2_1_7B": ["telescope_perplexity", "binoculars_score", "perplexity", "log_rank_ratio"],
}


# NOTE: Here are the following datasets that you can choose from
# "ghostbusters_perturb_character_basic",
# "ghostbusters_perturb_character_capitalization",
# "ghostbusters_perturb_character_space",
# "ghostbusters_perturb_paragraph_adjacent",
# "ghostbusters_perturb_paragraph_paraphrase",
# "ghostbusters_perturb_sentence_adjacent",
# "ghostbusters_perturb_sentence_paraphrase",
# "ghostbusters_perturb_word_adjacent",
# "ghostbusters_perturb_word_synonym",
PERTURBATION_DATASET_CODENAME_TO_TEST = "ghostbusters_perturb_sentence_paraphrase"
PLOT_TITLE = "Number of Sentence Paraphrase Adversarial Text Perturbations vs the Score of Each Classifier"

NUMBER_OF_PERTURBATION_OCCURANCES_TO_TEST = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    20,
    50,
    100,
    200
]

### END GLOBALS -------------------------------------------------------------------------




# a list of all of the colors that can be used to make plots
PLOT_COLORS = yaml.safe_load(open("config.yaml"))["plot_colors"]

# a dictionary that maps a metric's codename (for instance telescope_perplexity) to a presentable, paper-ready name (for instance Telescope Perplexity)
METRIC_CODENAME_TO_METRIC_DISPLAYNAME = yaml.safe_load(open("config.yaml"))["metric_codenames_to_metric_displaynames"]

# a dictionary that maps a model's codename (for instance smollm2_360M) to a presentable, paper-ready name (for instance SmolLM2 360M)
MODEL_CODENAME_TO_MODEL_DISPLAYNAME = yaml.safe_load(open("config.yaml"))["model_codenames_to_model_displaynames"]

# a dictionary that maps a dataset's codename (for instance ghostbusters_essay_gpt) to a presentable, paper-ready name (for instance GB Essay ChatGPT)
DATASET_CODENAME_TO_DATASET_DISPLAYNAME = yaml.safe_load(open("config.yaml"))["dataset_codenames_to_dataset_displaynames"]







def main():
    
    result_dict = {}   # {(perturbation_dataset_codename, model_codename, metric_codename): (perturbation_level, score)}
    
    for model_codename, metric_codenames_from_experiment in METRIC_CODENAMES_TO_TEST.items():
        
        model_displayname = MODEL_CODENAME_TO_MODEL_DISPLAYNAME[model_codename]
        
        for NUMBER_OF_PERTURBATION_OCCURANCES in NUMBER_OF_PERTURBATION_OCCURANCES_TO_TEST:
            try:
                df = pd.read_csv(f"{EXPERIMENT_FOLDER_NAME}/{model_codename}_{PERTURBATION_DATASET_CODENAME_TO_TEST}_{NUMBER_OF_PERTURBATION_OCCURANCES}/raw_data.csv")
            except:
                print(f"failed to find: {model_codename}, {PERTURBATION_DATASET_CODENAME_TO_TEST}, {NUMBER_OF_PERTURBATION_OCCURANCES}")
                continue
            
            
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna(subset=metric_codenames_from_experiment)
            
            for metric_codename in metric_codenames_from_experiment:                
                
                if (PERTURBATION_DATASET_CODENAME_TO_TEST, model_codename, metric_codename) not in result_dict.keys():
                    result_dict[(PERTURBATION_DATASET_CODENAME_TO_TEST, model_codename, metric_codename)] = []
                
                metric_scores = df[f"{metric_codename}"]
                if metric_codename == "binoculars_score" or metric_codename == "perplexity": 
                    metric_scores = -metric_scores
                    
                y_labels = df["y_labels"]
                
                # this one specifically has a lot of nan values this is a patch fix
                if (metric_codename == "log_rank_ratio" and model_codename == "gemma2_9B_detectllm_lrr" and PERTURBATION_DATASET_CODENAME_TO_TEST == "ghostbusters_perturb_word_synonym" and NUMBER_OF_PERTURBATION_OCCURANCES == 50):
                    continue
                    
                fpr, tpr, thresholds = roc_curve(y_labels, metric_scores)
                roc_auc = auc(fpr, tpr)
                
                result_dict[(PERTURBATION_DATASET_CODENAME_TO_TEST, model_codename, metric_codename)].append((NUMBER_OF_PERTURBATION_OCCURANCES, roc_auc))
                
                print(f"{model_displayname}, {PERTURBATION_DATASET_CODENAME_TO_TEST}, {NUMBER_OF_PERTURBATION_OCCURANCES}, {metric_codename}, {roc_auc}")



 
    fig, ax = plt.subplots()
    color_idx = 0
    for model_codename, metric_codenames_from_experiment in METRIC_CODENAMES_TO_TEST.items():
        
        model_displayname = MODEL_CODENAME_TO_MODEL_DISPLAYNAME[model_codename]
        
        
        for metric_codename in metric_codenames_from_experiment:
            
            metric_displayname = METRIC_CODENAME_TO_METRIC_DISPLAYNAME[metric_codename]
            
            color = PLOT_COLORS[color_idx % len(PLOT_COLORS)]
            color_idx += 1
            
            ax.plot(*zip(*result_dict[(PERTURBATION_DATASET_CODENAME_TO_TEST, model_codename, metric_codename)]), label=f"{metric_displayname} ({model_displayname})", linewidth=7, color=color)     
            ax.scatter(*zip(*result_dict[(PERTURBATION_DATASET_CODENAME_TO_TEST, model_codename, metric_codename)]), color=color)


    plt.xlabel('Number of Perturbations', fontsize=32)
    plt.ylabel('AUROC', fontsize=32)
    plt.title(PLOT_TITLE, fontsize=26)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)


    ax.grid(True, linestyle=':', linewidth=2, alpha=0.7)

    ax.legend(loc='lower left', fontsize=22)
    plt.show()  
            
    
    
if __name__ == "__main__":
    main()
