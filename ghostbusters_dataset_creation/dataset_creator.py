import os
from openai import OpenAI
from openai.types.chat import ChatCompletion
import time

# WARNING, THIS SCRIPT DOES NOT WORK ON THE REUTERS (NEWS) PART OF THE GHOSTBUSTERS DATASET


### START GLOBALS -------------------------------------------------------------------------


# Target model is either "gpt4o" or "deepseek"
TARGET_MODEL = "gpt4o"

# Supported dataset folders are wp and essay. "wp" is creative writing and "essay" is essay writing
DATASET_FOLDER = "essay" # wp, essay
MODEL_NAME = "gpt4o_adversarial_prompt2"
TEMPERATURE = 1

# Creative writing adversarial system prompt
# SYSTEM_PROMPT = "You are a helpful assistant. The following is a creative writing prompt. Please follow it, but make sure that you try to repeat key words one after another for emphasis while still following the prompt."

# Creative writing system prompt
# SYSTEM_PROMPT = "You are a helpful assistant. The following is a creative writing prompt. Please follow it."

# Adversarial essay system prompt
# SYSTEM_PROMPT = "You are a helpful assistant. Try to repeat key words one after another for emphasis while still following the prompt."

# Adversarial essay 2 system prompt
#SYSTEM_PROMPT = "You are a helpful assistant. During the generation process, selectively use the same words for synonyms."

# System prompt for everything else
SYSTEM_PROMPT = "You are a helpful assistant."

### END GLOBALS ---------------------------------------------------------------------------



if TARGET_MODEL == "gpt4o":
    client = OpenAI()
elif TARGET_MODEL == "deepseek":
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
else:
    raise Exception("Invalid target model used.")



def prompt_model_gpt4o(prompt) -> ChatCompletion:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=TEMPERATURE
    )
    return response.choices[0].message.content


def prompt_model_deepseek(prompt) -> ChatCompletion:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content



ai_written_text_directory = f'original_ghostbusters_datasets/{DATASET_FOLDER}/{MODEL_NAME}'
prompt_text_directory = f'original_ghostbusters_datasets/{DATASET_FOLDER}/prompts'

if not os.path.isdir(ai_written_text_directory):
    os.mkdir(ai_written_text_directory)

for index, filename in enumerate(os.listdir(prompt_text_directory)):
    print(f"filename: {filename}")
    print(index)
    
    prompt_text_file = os.path.join(prompt_text_directory, filename)
    
    if not os.path.isfile(prompt_text_file): continue
    
    prompt = "\n".join(open(prompt_text_file).readlines())
    
    if TARGET_MODEL == "gpt4o":
        ai_generated_text = prompt_model_gpt4o(prompt)
    elif TARGET_MODEL == "deepseek":
        ai_generated_text = prompt_model_deepseek(prompt)
    else:
        raise Exception("Invalid target model used.")


    ai_generated_text_file = os.path.join(ai_written_text_directory, filename)
    
    with open(ai_generated_text_file, "w") as text_file:
        text_file.write(ai_generated_text)