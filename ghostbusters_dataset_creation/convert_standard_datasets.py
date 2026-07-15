import os
import pandas as pd
from typing import Dict, List, Any



### START GLOBALS -------------------------------------------------------------------------

BASE_DIRECTORY = "original_ghostbusters_datasets"
OUTPUT_DIRECTORY = "datasets"

# Specific category to convert.
# Possible categories: "wp", "essay", "reuter"
# Set to None to convert all categories.
CATEGORY = None

# Specific model directory to convert.
# Possible model directories: 
#   "claude" 
#   "gpt"
#   "deepseek"
#   "gpt4o"
#   "gpt4o_adversarial_prompt" 
#   "gpt4o_adversarial_prompt2" 
#   "gpt4o_high_temperature" 
#   "gpt4o_low_temperature"
#   "gpt_prompt1"
#   "gpt_prompt2"
#   "gpt_semantic"
#   "gpt_writing"
# Set to None to convert all models.
MODEL = None

### END GLOBALS ---------------------------------------------------------------------------



CATEGORY_MAPPINGS: Dict[str, str] = {
    "wp": "Creative",
    "essay": "Essay",
    "reuter": "News"
}

MODEL_MAPPINGS: Dict[str, str] = {
    "claude": "Claude",
    "gpt": "GPT",
    "deepseek": "Deepseek",
    "gpt4o": "GPT4o",
    "gpt4o_adversarial_prompt": "GPT4o_Adversarial_Prompt",
    "gpt4o_adversarial_prompt2": "GPT4o_Adversarial_Prompt2",
    "gpt4o_high_temperature": "GPT4o_High_Temperature",
    "gpt4o_low_temperature": "GPT4o_Low_Temperature",
    "gpt_prompt1": "GPT_Prompt1",
    "gpt_prompt2": "GPT_Prompt2",
    "gpt_semantic": "GPT_Semantic",
    "gpt_writing": "GPT_Writing"
}



def clean_model_name(name: str) -> str:
    """
    Cleans model directory name to get the proper camelcase display name.
    """
    if name in MODEL_MAPPINGS:
        return MODEL_MAPPINGS[name]
    return name.replace("_", " ").title().replace(" ", "_")

def process_flat_dataset(human_directory: str, ai_directory: str) -> List[Dict[str, Any]]:
    """
    Processes flat datasets (like essay and wp) where files are stored directly under category/model/.
    """
    data: List[Dict[str, Any]] = []
    for filename in os.listdir(human_directory):
        human_file = os.path.join(human_directory, filename)
        ai_file = os.path.join(ai_directory, filename)
        
        if not os.path.isfile(human_file) or not os.path.isfile(ai_file):
            continue
            
        with open(human_file, "r", encoding="utf-8", errors="ignore") as f:
            human_text = f.read()
        with open(ai_file, "r", encoding="utf-8", errors="ignore") as f:
            ai_text = f.read()
            
        data.append({"text": human_text, "generated": 0})
        data.append({"text": ai_text, "generated": 1})
        
    return data

def process_nested_dataset(human_directory: str, ai_directory: str) -> List[Dict[str, Any]]:
    """
    Processes nested author-directory datasets (like reuter).
    """
    data: List[Dict[str, Any]] = []
    for author_directory_name in os.listdir(human_directory):
        human_author_directory = os.path.join(human_directory, author_directory_name)
        ai_author_directory = os.path.join(ai_directory, author_directory_name)
        
        if not os.path.isdir(human_author_directory) or not os.path.isdir(ai_author_directory):
            continue
            
        for filename in os.listdir(human_author_directory):
            human_file = os.path.join(human_author_directory, filename)
            ai_file = os.path.join(ai_author_directory, filename)
            
            if not os.path.isfile(human_file) or not os.path.isfile(ai_file):
                continue
                
            with open(human_file, "r", encoding="utf-8", errors="ignore") as f:
                human_text = f.read()
            with open(ai_file, "r", encoding="utf-8", errors="ignore") as f:
                ai_text = f.read()
                
            data.append({"text": human_text, "generated": 0})
            data.append({"text": ai_text, "generated": 1})
            
    return data




def main() -> None:
    base_directory = BASE_DIRECTORY
    output_directory = OUTPUT_DIRECTORY
    
    if not os.path.exists(base_directory):
        alt_path = os.path.join("ghostbusters_dataset_creation", base_directory)
        if os.path.exists(alt_path):
            base_directory = alt_path
            
    if not os.path.exists(output_directory):
        alt_output_dir = os.path.join("datasets")
        if os.path.exists(alt_output_dir):
            output_directory = alt_output_dir
        else:
            os.makedirs(output_directory, exist_ok=True)
            
    if CATEGORY:
        categories = [CATEGORY]
    else:
        categories = ["wp", "essay", "reuter"]
    
    for category in categories:
        category_directory = os.path.join(base_directory, category)
        if not os.path.isdir(category_directory):
            continue
            
        human_directory = os.path.join(category_directory, "human")
        if not os.path.isdir(human_directory):
            print(f"Skipping category {category}: 'human' directory not found.")
            continue
            

        if MODEL:
            model_directories = [MODEL]
        else:
            model_directories = []
            for d in os.listdir(category_directory):
                if os.path.isdir(os.path.join(category_directory, d)) and d not in ("human", "prompts"):
                    model_directories.append(d)
        
        for model in model_directories:
            ai_directory = os.path.join(category_directory, model)
            if not os.path.isdir(ai_directory):
                print(f"Skipping model {model} in category {category}: Directory does not exist.")
                continue
                
            print(f"Converting {category}/{model}...")
            
            if category == "reuter":
                data = process_nested_dataset(human_directory, ai_directory)
            else:
                data = process_flat_dataset(human_directory, ai_directory)
                
            if not data:
                print(f"No aligned data found for {category}/{model}.")
                continue
                
            cat_display = CATEGORY_MAPPINGS.get(category, category.title())
            model_display = clean_model_name(model)
            
            output_filename = f"Ghostbusters_{cat_display}_{model_display}_Dataset.csv"
            output_path = os.path.join(output_directory, output_filename)
            
            df = pd.DataFrame(data, columns=["text", "generated"])
            df.to_csv(output_path, index=False)
            print(f"Generated {output_path} with {len(df)} records.")





if __name__ == "__main__":
    main()
