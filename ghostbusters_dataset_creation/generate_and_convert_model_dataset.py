import os
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from typing import Dict, List, Any, Optional



### START GLOBALS -------------------------------------------------------------------------

MODEL_NAME = "gpt-4o-mini"
MODEL_FOLDER_NAME = "gpt4o_adversarial_prompt3"

# Specific category to generate/convert.
# Possible categories: "wp", "essay"
CATEGORY = "essay"

TEMPERATURE = 1.0


# System prompts from original experiments:
#
# Creative writing adversarial system prompt:
# SYSTEM_PROMPT = "You are a helpful assistant. The following is a creative writing prompt. Please follow it, but make sure that you try to repeat key words one after another for emphasis while still following the prompt."
#
# Creative writing system prompt:
# SYSTEM_PROMPT = "You are a helpful assistant. The following is a creative writing prompt. Please follow it."
#
# Adversarial essay system prompt:
# SYSTEM_PROMPT = "You are a helpful assistant. Try to repeat key words one after another for emphasis while still following the prompt."
#
# Adversarial essay 2 system prompt:
# SYSTEM_PROMPT = "You are a helpful assistant. During the generation process, selectively use the same words for synonyms."
#
# Default system prompt:

SYSTEM_PROMPT = "You are a helpful assistant."

BASE_DIRECTORY = "original_ghostbusters_datasets"
OUTPUT_DIRECTORY = "datasets_temp"
OUTPUT_FILENAME = "Ghostbusters_Essay_GPT4o_Adversarial_Prompt3_Dataset.csv"

# Optional: API Key and API Base URL. If None, environment defaults will be used.
API_KEY = None
BASE_URL = None

### END GLOBALS ---------------------------------------------------------------------------




# The news dataset unfortunately does not contain prompts to allow us to 
# directly recreate the dataset with different models
CATEGORY_MAPPINGS: Dict[str, str] = {
    "wp": "Creative",
    "essay": "Essay",
}

def clean_model_name(name: str) -> str:
    """
    Cleans model folder name for display format in output files.
    """
    return name.replace("_", " ").title().replace(" ", "_")


def get_client(model: str, api_key: Optional[str], base_url: Optional[str]) -> OpenAI:
    """
    Instantiates an OpenAI client with proper environment credentials.
    """
    if "deepseek" in model.lower():
        key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        url = base_url or "https://api.deepseek.com"
        if not key:
            raise ValueError("DeepSeek API key must be provided via the script's global API_KEY variable or the DEEPSEEK_API_KEY environment variable.")
        return OpenAI(api_key=key, base_url=url)
    else:
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OpenAI API key must be provided via the script's global API_KEY variable or the OPENAI_API_KEY environment variable.")
        return OpenAI(api_key=key)



def generate_texts(client: OpenAI, model: str, prompt_directory: str, output_model_directory: str, system_prompt: str, temperature: float) -> None:
    """
    Generates AI text from original prompts using the selected LLM and saves them locally.
    """
    os.makedirs(output_model_directory, exist_ok=True)
    prompt_files = [f for f in os.listdir(prompt_directory) if os.path.isfile(os.path.join(prompt_directory, f))]
    
    print(f"Checking prompts in {prompt_directory}...")
    for index, filename in enumerate(tqdm(prompt_files)):

        if index > 5:
            continue
        
        prompt_path = os.path.join(prompt_directory, filename)
        target_path = os.path.join(output_model_directory, filename)
        
        print(filename)
        print(target_path)
        if os.path.isfile(target_path):
            continue
            
        with open(prompt_path, "r", encoding="utf-8", errors="ignore") as f:
            prompt = f.read()
            

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        ai_generated_text = response.choices[0].message.content
        if ai_generated_text:
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(ai_generated_text)



def create_dataset(human_directory: str, ai_directory: str, output_path: str) -> None:
    """
    Pairs human baseline texts and LLM-generated texts and outputs a formatted CSV.
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
        
    if not data:
        print(f"Warning: No aligned data found to convert between {human_directory} and {ai_directory}.")
        return
        
    df = pd.DataFrame(data, columns=["text", "generated"])
    df.to_csv(output_path, index=False)
    print(f"Generated unified CSV dataset at {output_path} with {len(df)} records.")



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
            
    category_directory = os.path.join(base_directory, CATEGORY)
    prompt_directory = os.path.join(category_directory, "prompts")
    human_directory = os.path.join(category_directory, "human")
    ai_directory = os.path.join(category_directory, MODEL_FOLDER_NAME)
    
    if not os.path.isdir(prompt_directory):
        raise FileNotFoundError(f"Prompts folder not found at: {prompt_directory}")
    if not os.path.isdir(human_directory):
        raise FileNotFoundError(f"Human baseline folder not found at: {human_directory}")
        
    client = get_client(MODEL_NAME, API_KEY, BASE_URL)
    
    print(f"Generating LLM texts using model {MODEL_NAME}...")
    generate_texts(client, MODEL_NAME, prompt_directory, ai_directory, SYSTEM_PROMPT, TEMPERATURE)
    
    print("Creating final CSV dataset...")
    cat_display = CATEGORY_MAPPINGS.get(CATEGORY, CATEGORY.title())
    model_display = clean_model_name(MODEL_FOLDER_NAME)
    output_filename = OUTPUT_FILENAME
    output_path = os.path.join(output_directory, output_filename)
    
    create_dataset(human_directory, ai_directory, output_path)



if __name__ == "__main__":
    main()
