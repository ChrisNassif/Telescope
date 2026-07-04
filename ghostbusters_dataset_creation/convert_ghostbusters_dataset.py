import os
import pandas as pd
import csv

# WARNING, THIS SCRIPT DOES NOT WORK ON THE REUTERS (NEWS) PART OF THE GHOSTBUSTERS DATASET
# PLEASE LOOK AT THE SPECIFIC CONVERSION FILE FOR THAT PART OF THE DATASET

MODEL_CODENAME = "gpt4o_adversarial_prompt2"
MODEL_DISPLAYNAME = "GPT4o_Adversarial_Prompt2"

DATASET_TYPE = "essay" # wp, reuter, essay
DATASET_TYPE_DISPLAYNAME = "Essay" # Creative, News, Essay

human_written_text_directory = f'original_ghostbusters_datasets/{DATASET_TYPE}/human'
ai_written_text_directory = f'original_ghostbusters_datasets/{DATASET_TYPE}/{MODEL_CODENAME}'
prompt_text_directory = f'original_ghostbusters_datasets/{DATASET_TYPE}/prompts'

csv_writer = csv.DictWriter(open(f"Ghostbusters_{DATASET_TYPE_DISPLAYNAME}_{MODEL_DISPLAYNAME}_Dataset.csv", "w+"), fieldnames=["text", "generated"])
csv_writer.writeheader()

data = []
for filename in os.listdir(human_written_text_directory):
    print(f"filename: {filename}")
    
    human_written_text_file = os.path.join(human_written_text_directory, filename)
    ai_written_text_file = os.path.join(ai_written_text_directory, filename)
    
    if not os.path.isfile(human_written_text_file): continue
    if not os.path.isfile(ai_written_text_file): continue
    
    human_written_text = "\n".join(open(human_written_text_file).readlines())
    ai_written_text = "\n".join(open(ai_written_text_file).readlines())
    
    data.append({"text": human_written_text, "generated": 0})
    data.append({"text": ai_written_text, "generated": 1})

csv_writer.writerows(data)
