import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import pandas as pd
import matplotlib.pyplot as plt
import random
import yaml

from confidenceinterval import f1_score, roc_auc_score

from src.utils import create_logistic_regression_classifier



### START GLOBALS -------------------------------------------------------------------------

COLORS = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#bcbd22",  # Olive
    "#17becf",  # Cyan
    "#f5a623",  # Amber
    "#a6cee3",  # Light Blue
    "#b15928",  # Dark Brown
    "#6a3d9a",  # Dark Purple
    "#ffcc00",  # Bright Yellow
]


EXPERIMENT_FOLDER_NAME = "experiment_results"
EXPERIMENT_ANALYSES_FOLDER_NAME = "experiment_analyses"

ANALYSIS_NAME = "performance_over_lengths"


METRIC_CODENAMES_TO_TEST = {
    "gemma2_2B": ["telescope_perplexity", "binoculars_score", "perplexity", "lrr"],
    "gemma2_9B": ["telescope_perplexity", "binoculars_score", "perplexity", "lrr"],
    # "llama3_8B": ["telescope_perplexity", "binoculars_score", "perplexity", "lrr"],
    # "falcon_7B":  ["telescope_perplexity", "binoculars_score", "perplexity", "lrr"],
    
    # "smollm_135M": ["telescope_perplexity", "binoculars_score", "perplexity", "lrr"],
    # "smollm_360M": ["telescope_perplexity", "binoculars_score", "perplexity", "lrr"],
    # "smollm_1_7B": ["telescope_perplexity", "binoculars_score", "perplexity", "lrr"],
    # "smollm2_135M": ["telescope_perplexity", "binoculars_score", "perplexity", "lrr"],
    # "smollm2_360M": ["telescope_perplexity", "binoculars_score", "perplexity", "lrr"],
    # "smollm2_1_7B": ["telescope_perplexity", "binoculars_score", "perplexity", "lrr"],
}


DATASET_CODENAME_TO_TEST = "ghostbusters_news_gpt"


### END GLOBALS -------------------------------------------------------------------------





# a list of all of the colors that can be used to make plots
PLOT_COLORS = yaml.safe_load(open("config.yaml"))["plot_colors"]

# a dictionary that maps a metric's codename (for instance telescope_perplexity) to a presentable, paper-ready name (for instance Telescope Perplexity)
METRIC_CODENAME_TO_METRIC_DISPLAYNAME = yaml.safe_load(open("config.yaml"))["metric_codenames_to_metric_displaynames"]

# a dictionary that maps a model's codename (for instance smollm2_360M) to a presentable, paper-ready name (for instance SmolLM2 360M)
MODEL_CODENAME_TO_PROPER_MODEL_NAME = yaml.safe_load(open("config.yaml"))["model_codenames_to_model_displaynames"]

# a dictionary that maps a dataset's codename (for instance ghostbusters_essay_gpt) to a presentable, paper-ready name (for instance GB Essay ChatGPT)
DATASET_CODENAME_TO_DATASET_DISPLAYNAME = yaml.safe_load(open("config.yaml"))["dataset_codenames_to_dataset_displaynames"]
DATASET_DISPLAYNAME = DATASET_CODENAME_TO_DATASET_DISPLAYNAME[DATASET_CODENAME_TO_TEST]





def test_length_cutoffs(
    df, 
    predicted_labels,
    actual_labels,  
    number_of_points_to_test = 20, 
    range_to_test = (0, 1000)
    ):

    
    length_cutoffs_to_test = np.linspace(range_to_test[0], range_to_test[1], number_of_points_to_test)
    results = []
        
    for length_cutoff in length_cutoffs_to_test:
        number_correct = 0
        number_incorrect = 0

        for index, text in enumerate(df["original_texts"]):
            if (len(text.split(" ")) < length_cutoff):
                continue

            if ((predicted_labels[index] > 0.5 and actual_labels[index] > 0.5) \
                or (predicted_labels[index] < 0.5 and actual_labels[index] < 0.5)):
                number_correct += 1
            else:
                number_incorrect += 1
        
        if (number_correct + number_incorrect == 0): 
            results.append(1)

        else:
            accuracy = number_correct / (number_correct + number_incorrect)
            results.append(accuracy)
            
    return length_cutoffs_to_test, results
            
            
            
def main():
    plt.figure()
    
    for model_codename, metric_codenames_from_experiment in METRIC_CODENAMES_TO_TEST.items():
        
        model_displayname = MODEL_CODENAME_TO_PROPER_MODEL_NAME[model_codename]
        
        df = pd.read_csv(f"{EXPERIMENT_FOLDER_NAME}/{model_codename}_{DATASET_CODENAME_TO_TEST}_dataset/raw_data.csv")
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=metric_codenames_from_experiment)
        df = df.reset_index()
 
 
        for metric_codename in metric_codenames_from_experiment:
            
            metric_displayname = METRIC_CODENAME_TO_METRIC_DISPLAYNAME[metric_codename]
                    
            classifier = create_logistic_regression_classifier(df[[metric_codename,]], df["y_labels"])
            
            predicted_labels = classifier.predict(df[[metric_codename,]])

            length_cutoffs_to_test, results = test_length_cutoffs(df, predicted_labels, df["y_labels"])
            plt.plot(length_cutoffs_to_test, results, color=random.choice(COLORS), lw=7, label=f"{metric_displayname} {model_displayname}")
    
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.title(f"Minimum Number of Words in Text vs Detector Accuracy in {DATASET_DISPLAYNAME}", fontsize=32)
    plt.xlabel("Minimum Number of Words In Subsample", fontsize=32)
    plt.ylabel("Accuracy", fontsize=32)
    
    plt.legend(loc="lower right", fontsize=26)
    
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.grid(True, linestyle=':', linewidth=2, alpha=0.7)
    # plt.savefig(f"{EXPERIMENT_ANALYSES_FOLDER_NAME}/{ANALYSIS_NAME}/fig.png")

    plt.show()
    plt.close()
        
            
    
    
    
if __name__ == "__main__":
    main()