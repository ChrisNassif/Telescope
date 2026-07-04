import os
from openai import OpenAI
from openai.types.chat import ChatCompletion
import time

# WARNING, THIS SCRIPT DOES NOT WORK ON THE REUTERS (NEWS) PART OF THE GHOSTBUSTERS DATASET

client = OpenAI()
# client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")


DATASET_FOLDER = "essay" # wp, essay
MODEL_NAME = "gpt4o_adversarial_prompt2"
# SYSTEM_PROMPT = "You are a helpful assistant. The following is a creative writing prompt. Please follow it, but make sure that you try to repeat key words one after another for emphasis while still following the prompt."  # Creative writing adversarial prompt
# SYSTEM_PROMPT = "You are a helpful assistant. The following is a creative writing prompt. Please follow it."  # Creative writing prompt
# SYSTEM_PROMPT = "You are a helpful assistant. Try to repeat key words one after another for emphasis while still following the prompt." # system prompt for adversarial essay
SYSTEM_PROMPT = "You are a helpful assistant. During the generation process, selectively use the same words for synonyms." # system prompt for adversarial essay 2
# SYSTEM_PROMPT = "You are a helpful assistant." # system prompt for everything else
TEMPERATURE = 1

def prompt_model(prompt) -> ChatCompletion:
    start_time = time.time()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=TEMPERATURE
    )
    print(time.time() - start_time)
    return response.choices[0].message.content

# def prompt_model(prompt) -> ChatCompletion:
#     response = client.chat.completions.create(
#         model="deepseek-chat",
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             {
#                 "role": "user",
#                 "content": prompt
#             }
#         ]
#     )
#     return response.choices[0].message.content



ai_written_text_directory = f'ghostbuster-data/{DATASET_FOLDER}/{MODEL_NAME}'
prompt_text_directory = f'ghostbuster-data/{DATASET_FOLDER}/prompts'

if not os.path.isdir(ai_written_text_directory):
    os.mkdir(ai_written_text_directory)

for index, filename in enumerate(os.listdir(prompt_text_directory)):
    print(f"filename: {filename}")
    print(index)
    
    prompt_text_file = os.path.join(prompt_text_directory, filename)
    
    if not os.path.isfile(prompt_text_file): continue
    
    prompt = "\n".join(open(prompt_text_file).readlines())
    
    ai_generated_text = prompt_model(prompt)
    
    ai_generated_text_file = os.path.join(ai_written_text_directory, filename)
    
    with open(ai_generated_text_file, "w") as text_file:
        text_file.write(ai_generated_text)