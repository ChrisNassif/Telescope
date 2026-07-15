import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.insert(0, os.getcwd())

from typing import Dict, List, Tuple, Any
import numpy as np
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import yaml

from confidenceinterval import f1_score, roc_auc_score

from llm_text_detectors.utils import create_logistic_regression_classifier



### START GLOBALS -------------------------------------------------------------------------





EXPERIMENT_FOLDER_NAME: str = "experiment_results"
EXPERIMENT_ANALYSES_FOLDER_NAME: str = "experiment_analyses"

ANALYSIS_NAME: str = "performance_over_lengths"


METRIC_CODENAMES_TO_TEST: Dict[str, List[str]] = {
    "gemma2_2B": ["telescope_perplexity", "binoculars_score", "perplexity", "log_rank_ratio"],
    "gemma2_9B": ["telescope_perplexity", "binoculars_score", "perplexity", "log_rank_ratio"],
    # "llama3_8B": ["telescope_perplexity", "binoculars_score", "perplexity", "log_rank_ratio"],
    # "falcon_7B":  ["telescope_perplexity", "binoculars_score", "perplexity", "log_rank_ratio"],
    
    # "smollm_135M": ["telescope_perplexity", "binoculars_score", "perplexity", "log_rank_ratio"],
    # "smollm_360M": ["telescope_perplexity", "binoculars_score", "perplexity", "log_rank_ratio"],
    # "smollm_1_7B": ["telescope_perplexity", "binoculars_score", "perplexity", "log_rank_ratio"],
    # "smollm2_135M": ["telescope_perplexity", "binoculars_score", "perplexity", "log_rank_ratio"],
    # "smollm2_360M": ["telescope_perplexity", "binoculars_score", "perplexity", "log_rank_ratio"],
    # "smollm2_1_7B": ["telescope_perplexity", "binoculars_score", "perplexity", "log_rank_ratio"],
}


DATASET_CODENAME_TO_TEST: str = "ghostbusters_news_gpt"


### END GLOBALS -------------------------------------------------------------------------





# a list of all of the colors that can be used to make plots
PLOT_COLORS: List[str] = yaml.safe_load(open("config.yaml"))["plot_colors"]

# a dictionary that maps a metric's codename (for instance telescope_perplexity) to a presentable, paper-ready name (for instance Telescope Perplexity)
METRIC_CODENAME_TO_METRIC_DISPLAYNAME: Dict[str, str] = yaml.safe_load(open("config.yaml"))["metric_codenames_to_metric_displaynames"]

# a dictionary that maps a model's codename (for instance smollm2_360M) to a presentable, paper-ready name (for instance SmolLM2 360M)
MODEL_CODENAME_TO_PROPER_MODEL_NAME: Dict[str, str] = yaml.safe_load(open("config.yaml"))["model_codenames_to_model_displaynames"]

# a dictionary that maps a dataset's codename (for instance ghostbusters_essay_gpt) to a presentable, paper-ready name (for instance GB Essay ChatGPT)
DATASET_CODENAME_TO_DATASET_DISPLAYNAME: Dict[str, str] = yaml.safe_load(open("config.yaml"))["dataset_codenames_to_dataset_displaynames"]
DATASET_DISPLAYNAME: str = DATASET_CODENAME_TO_DATASET_DISPLAYNAME[DATASET_CODENAME_TO_TEST]





def test_length_cutoffs(
    df: pd.DataFrame, 
    predicted_labels: np.ndarray,
    actual_labels: np.ndarray,  
    number_of_points_to_test: int = 20, 
    range_to_test: Tuple[int, int] = (0, 1000)
) -> Tuple[np.ndarray, List[float]]:

    
    length_cutoffs_to_test: np.ndarray = np.linspace(range_to_test[0], range_to_test[1], number_of_points_to_test)
    results: List[float] = []
        
    length_cutoff: float
    for length_cutoff in length_cutoffs_to_test:
        number_correct: int = 0
        number_incorrect: int = 0

        index: int
        text: str
        for index, text in enumerate(df["original_texts"]):
            if (len(text.split(" ")) < length_cutoff):
                continue

            if ((predicted_labels[index] > 0.5 and actual_labels[index] > 0.5) \
                or (predicted_labels[index] < 0.5 and actual_labels[index] < 0.5)):
                number_correct += 1
            else:
                number_incorrect += 1
        
        if (number_correct + number_incorrect == 0): 
            results.append(1.0)

        else:
            accuracy: float = number_correct / (number_correct + number_incorrect)
            results.append(accuracy)
            
    return length_cutoffs_to_test, results
            
            
            
def main() -> None:
    plt.figure()
    
    color_idx: int = 0
    model_codename: str
    metric_codenames_from_experiment: List[str]
    for model_codename, metric_codenames_from_experiment in METRIC_CODENAMES_TO_TEST.items():
        
        model_displayname: str = MODEL_CODENAME_TO_PROPER_MODEL_NAME[model_codename]
        
        df: pd.DataFrame = pd.read_csv(f"{EXPERIMENT_FOLDER_NAME}/{model_codename}_{DATASET_CODENAME_TO_TEST}_dataset/raw_data.csv")
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=metric_codenames_from_experiment)
        df = df.reset_index()
 
 
        metric_codename: str
        for metric_codename in metric_codenames_from_experiment:
            
            metric_displayname: str = METRIC_CODENAME_TO_METRIC_DISPLAYNAME[metric_codename]
                    
            classifier: Any = create_logistic_regression_classifier(df[[metric_codename,]], df["y_labels"])
            
            predicted_labels: np.ndarray = classifier.predict(df[[metric_codename,]])

            length_cutoffs_to_test: np.ndarray
            results: List[float]
            length_cutoffs_to_test, results = test_length_cutoffs(df, predicted_labels, df["y_labels"])
            color: str = PLOT_COLORS[color_idx % len(PLOT_COLORS)]
            color_idx += 1
            plt.plot(length_cutoffs_to_test, results, color=color, lw=7, label=f"{metric_displayname} {model_displayname}")
    
    
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