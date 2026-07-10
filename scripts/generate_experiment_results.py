import sys; argv = sys.argv[1:]
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
import pandas as pd
from tqdm import tqdm
import numpy as np

from typing import Mapping
from transformers import BitsAndBytesConfig
from llm_text_detectors import Detectors
import hashlib
import time


### START GLOBALS -------------------------------------------------------------------------

BITS_AND_BYTES_QUANTIZATION_CONFIG = None
# BITS_AND_BYTES_QUANTIZATION_CONFIG = BitsAndBytesConfig(load_in_8bit=True)

MAX_NUMBER_OF_SAMPLES = 10_000
MINIMUM_NUMBER_OF_WORDS_IN_SAMPLE = 100
MAXIMUM_NUMBER_OF_WORDS_IN_SAMPLE = 5000
# EXPERIMENT_FOLDER = "experiment_results_temp_execution_time2_lrr"
EXPERIMENT_FOLDER = "experiment_results"

# model_codename: (PERFORMER_MODEL_HUGGINGFACE_REPOSITORY, OBSERVER_MODEL_HUGGINGFACE_REPOSITORY)
MODEL_PERFORMER_OBSERVER_PAIRS_TO_TEST = {
    # "smollm_135M": ("HuggingFaceTB/SmolLM-135M-Instruct", "HuggingFaceTB/SmolLM-135M"),
    # "smollm2_135M": ("HuggingFaceTB/SmolLM2-135M-Instruct", "HuggingFaceTB/SmolLM2-135M"),
    # "smollm_360M": ("HuggingFaceTB/SmolLM-360M-Instruct", "HuggingFaceTB/SmolLM-360M"),
    # "smollm2_360M": ("HuggingFaceTB/SmolLM2-360M-Instruct", "HuggingFaceTB/SmolLM2-360M"),
    # "smollm_1_7B": ("HuggingFaceTB/SmolLM-1.7B-Instruct", "HuggingFaceTB/SmolLM-1.7B"),
    # "smollm2_1_7B": ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "HuggingFaceTB/SmolLM2-1.7B"),
    # "falcon_7B": ("tiiuae/falcon-7b-instruct", "tiiuae/falcon-7b"),
    # "gemma2_2B": ("google/gemma-2-2b-it", "google/gemma-2-2b"),
    # "llama3_8B": ("meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B"),
    # "gemma2_9B": ("google/gemma-2-9b-it", "google/gemma-2-9b"),

    # FOR FAST-DETECTGPT
    # "gpt_neo_2_7B": ("EleutherAI/gpt-neo-2.7B", "EleutherAI/gpt-neo-2.7B"),
    "gpt_j_6B": ("EleutherAI/gpt-j-6b", "EleutherAI/gpt-j-6b"),   
}


# "Detect_LLM_Text_Dataset.csv" 
# "AI_Human_Dataset.csv"
# "ESL_GPT4o_Dataset.csv"
# "Ghostbusters_Dataset.csv"
# "HC3_Dataset.csv"
# "HC3_Plus_Dataset.csv"
# "M4_English_Wikipedia_ChatGPT_Dataset.csv"
# "M4_Russian_ChatGPT_Dataset.csv"
# if a dataset is in the "datasets" folder, then you can input it here to test on that dataset
# DATASET_FILE = "Ghostbusters_Creative_GPT_Dataset.csv"
DATASET_FILE = argv[0]
DATASET_FOLDER = "datasets"

### END GLOBALS ---------------------------------------------------------------------------



start_time = time.time()

# INFO: Some Example Experiment Names:
# EXPERIMENT_NAME = "smollm_360M_ai_human_dataset"
# Generate Experiment Names to Know What Folders To Save the Experiments To
experiment_name_list = []
for model_codename in MODEL_PERFORMER_OBSERVER_PAIRS_TO_TEST.keys(): 
    experiment_name = model_codename + "_"
    experiment_name += DATASET_FILE.lower()[:-4]    # remove .csv
    experiment_name_list.append(experiment_name)

print(f"experiment name list: {experiment_name_list}")
print()




def compute_accuracy_based_on_threshold(y_labels: np.ndarray, y_scores: np.ndarray, detection_threshold: float):
    number_correct = 0
    total = 0
        
    for i in range(len(y_scores)):
        total += 1      
        if y_labels[i] and y_scores[i] > detection_threshold: 
            number_correct+=1
            
        if not y_labels[i] and y_scores[i] < detection_threshold: 
            number_correct+=1
        
    return number_correct/total
    
    
def hash_text(text):
    """Hash text content for fast duplicate detection."""
    return hashlib.md5(str(text).encode()).hexdigest()



def load_existing_results(experiment_folder, experiment_name_list):
    """Load existing experiment CSVs, safely deduplicate them, and return hash sets."""
    existing_data = {}
    processed_hashes = {}
    
    for index, model_name in enumerate(MODEL_PERFORMER_OBSERVER_PAIRS_TO_TEST.keys()):
        csv_path = f'{experiment_folder}/{experiment_name_list[index]}/raw_data.csv'
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if 'original_texts' in df.columns and 'y_labels' in df.columns and len(df) > 0:
                    initial_len = len(df)
                    df = df.drop_duplicates(subset=['original_texts']).reset_index(drop=True)
                    if len(df) != initial_len:
                        print(f"Removed {initial_len - len(df)} duplicate rows from historical CSV for {model_name}")

                    existing_data[model_name] = df
                    processed_hashes[model_name] = set(hash_text(t) for t in df['original_texts'])
                    print(f"Loaded {len(df)} distinct existing results for {model_name}")
                    continue
            except Exception as e:
                print(f"Failed to load existing results for {model_name}: {e}")
        
        existing_data[model_name] = None
        processed_hashes[model_name] = set()
    
    return existing_data, processed_hashes


def save_experiment(
        per_model_labels: dict[str, list], 
        metrics_for_each_model: Mapping[str, Mapping[str, list]], 
        per_model_texts: dict[str, list],
        existing_data: dict,
        filepath: str, 
    ):
    """Saves new records to disk, updates base data frames, and flushes memory batches."""
    for index, model_name in enumerate(MODEL_PERFORMER_OBSERVER_PAIRS_TO_TEST.keys()):
        save_directory = f'{filepath}/{experiment_name_list[index]}'
        print(f"Attempting to save to directory: {save_directory}")  
        if not os.path.exists(save_directory):
            os.makedirs(save_directory, exist_ok=True)
        
        if len(per_model_labels[model_name]) > 0:
            new_df = pd.DataFrame({
                "y_labels": per_model_labels[model_name],
                "original_texts": per_model_texts[model_name],
                **(metrics_for_each_model[model_name])
            })
        else:
            new_df = pd.DataFrame()
        
        if existing_data.get(model_name) is not None and len(existing_data[model_name]) > 0:
            df = pd.concat([existing_data[model_name], new_df], ignore_index=True) if len(new_df) > 0 else existing_data[model_name]
        else:
            df = new_df
        
        if len(df) > 0:
            df = df.drop_duplicates(subset=['original_texts']).reset_index(drop=True)
            df.to_csv(f'{save_directory}/raw_data.csv', index=False)
            
            existing_data[model_name] = df

        per_model_labels[model_name].clear()
        per_model_texts[model_name].clear()
        for metric_name in metrics_for_each_model[model_name].keys():
            metrics_for_each_model[model_name][metric_name].clear()


def main():
    dataset = pd.read_csv(f"{DATASET_FOLDER}/{DATASET_FILE}").sample(frac=1, random_state=42).reset_index(drop=True)
        
    text_dataset = dataset["text"]
    is_ai_generated_dataset = dataset["generated"]
    
    existing_data, processed_hashes = load_existing_results(EXPERIMENT_FOLDER, experiment_name_list)
    
    # Initialize detectors based on configuration
    text_detectors: dict[str, Detectors] = {}
    metrics_for_each_model = {}
    per_model_labels = {}
    per_model_texts = {}

    for text_detector_name, (performer_model, observer_model) in MODEL_PERFORMER_OBSERVER_PAIRS_TO_TEST.items():
        text_detector = Detectors(
            performer_model, observer_model,
            BITS_AND_BYTES_QUANTIZATION_CONFIG
        )
        
        text_detectors[text_detector_name] = text_detector
        metrics_for_each_model[text_detector_name] = {}
        per_model_labels[text_detector_name] = []
        per_model_texts[text_detector_name] = []
    
    number_of_samples_examined = 0
    if processed_hashes:
        number_of_samples_examined = max(len(hashes) for hashes in processed_hashes.values())
    print(f"Starting experiment. Already collected in existing data: {number_of_samples_examined} samples.")

    for index, (text_data, is_ai_generated) in enumerate(zip(text_dataset, is_ai_generated_dataset)):
        print(index)
        if (type(text_data) != str): continue
        if (len(text_data.split(" ")) < MINIMUM_NUMBER_OF_WORDS_IN_SAMPLE): continue
        if (len(text_data.split(" ")) > MAXIMUM_NUMBER_OF_WORDS_IN_SAMPLE): continue
        
        text_h = hash_text(text_data)
        
        # Skip if ALL models already have this text recorded
        if all(text_h in processed_hashes[name] for name in text_detectors):
            continue
        
        if number_of_samples_examined >= MAX_NUMBER_OF_SAMPLES: 
            print(f"Reached Target Collection Size ({MAX_NUMBER_OF_SAMPLES}). Stopping loop.")
            break
            
        number_of_samples_examined += 1

        # Run Telescope metrics for each model pair
        for text_detector_name, text_detector in text_detectors.items():

            if text_h in processed_hashes[text_detector_name]:
                continue  # This model already processed this text
            
            metrics_dict = text_detector.compute_all_metrics(text_data)

            for metric_name, metric_value in metrics_dict.items():
                if metric_name not in metrics_for_each_model[text_detector_name].keys():
                    metrics_for_each_model[text_detector_name][metric_name] = []
                
                if isinstance(metric_value, (np.ndarray, list)):
                    metrics_for_each_model[text_detector_name][metric_name].append(metric_value)
                else:
                    metrics_for_each_model[text_detector_name][metric_name].append(float(metric_value))

            per_model_labels[text_detector_name].append(is_ai_generated)
            per_model_texts[text_detector_name].append(text_data)
            
            processed_hashes[text_detector_name].add(text_h)
        
        if index > 1 and index % 50 == 0: 
            save_experiment(per_model_labels, metrics_for_each_model, per_model_texts, existing_data, EXPERIMENT_FOLDER)
        
    save_experiment(per_model_labels, metrics_for_each_model, per_model_texts, existing_data, EXPERIMENT_FOLDER)

if __name__ == "__main__":
    main()

    print(f"execution time: {time.time() - start_time}")