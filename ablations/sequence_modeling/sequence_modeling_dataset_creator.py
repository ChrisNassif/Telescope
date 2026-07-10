import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import ast 


SEED = 42

EXPERIMENT_FOLDER_NAME = "experiment_results_new_extra_features"


METRIC_CODENAMES_TO_TEST = {
    # "gemma2_2B": ["telescope_perplexity_per_token",],
    # "gemma2_9B": ["telescope_perplexity_per_token",],
    # "llama3_8B": ["telescope_perplexity_per_token",],
    # "falcon_7B":  ["telescope_perplexity_per_token",],
    
    # "smollm_135M": ["telescope_perplexity_per_token",],
    "smollm_360M": ["telescope_perplexity_per_token",],
    # "smollm_1_7B": ["telescope_perplexity_per_token",],
    # "smollm2_135M": ["telescope_perplexity_per_token",],
    # "smollm2_360M": ["telescope_perplexity_per_token",],
    # "smollm2_1_7B": ["telescope_perplexity_per_token",],
}

DATASET_CODENAMES_TO_TEST = [
    
    "detect_llm_text",
    "ai_human",
    "hc3",
    "hc3_plus",
    "esl_gpt4o",
    
    # "ghostbusters_essay_gpt",
    # "ghostbusters_news_gpt",
    # "ghostbusters_creative_gpt",
    # "ghostbusters_essay_gpt4o",
    # "ghostbusters_creative_gpt4o",
    # "ghostbusters_news_claude",
    # "ghostbusters_creative_claude",
    # "ghostbusters_essay_claude",
    # "ghostbusters_essay_deepseek",
    # "ghostbusters_creative_deepseek",
]



for model_codename, metric_codenames_from_experiment in METRIC_CODENAMES_TO_TEST.items():        
    for dataset_codename in DATASET_CODENAMES_TO_TEST:

        df = pd.read_csv(f"{EXPERIMENT_FOLDER_NAME}/{model_codename}_{dataset_codename}_dataset/raw_data.csv")
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=metric_codenames_from_experiment)
        
        telescope_perplexity_per_token = df["telescope_perplexity_per_token"].to_numpy()
        cross_perplexity_per_token = df["cross_perplexity_per_token"].to_numpy()
        perplexity_per_token = df["perplexity_per_token"].to_numpy()   


        labels = df["y_labels"].astype(bool).to_numpy()
                
        full_dataframe = pd.DataFrame({
            "telescope_perplexity_per_token": telescope_perplexity_per_token, 
            "cross_perplexity_per_token": cross_perplexity_per_token,
            "perplexity_per_token": perplexity_per_token,
            "labels": labels
        })
        
        os.mkdir(f"sequence_modeling/sequence_modeling_datasets/{dataset_codename}_smollm_360M_dataset")
        full_dataframe.to_csv(f"sequence_modeling/sequence_modeling_datasets/{dataset_codename}_smollm_360M_dataset/full.csv", index=False)
        
