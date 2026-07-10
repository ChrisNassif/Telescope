import os
import pandas as pd
import csv

# TODO MAKE A SPECIAL CASE FOR THE REUTERS DATASET IN THIS SCRIPT
# WARNING, THIS SCRIPT DOES NOT WORK ON THE REUTERS (NEWS) PART OF THE GHOSTBUSTERS DATASET
# PLEASE LOOK AT THE SPECIFIC CONVERSION FILE FOR THAT PART OF THE DATASET



### START GLOBALS -------------------------------------------------------------------------

DATASET_CODENAME = "gpt4o_adversarial_prompt2"
DATASET_DISPLAYNAME = "GPT4o_Adversarial_Prompt2"

# wp, reuter, essay
DATASET_TYPE = "essay"

# Creative, News, Essay
DATASET_TYPE_DISPLAYNAME = "Essay"

### END GLOBALS ---------------------------------------------------------------------------



human_written_text_directory = f'original_ghostbusters_datasets/{DATASET_TYPE}/human'
ai_written_text_directory = f'original_ghostbusters_datasets/{DATASET_TYPE}/{DATASET_CODENAME}'

csv_writer = csv.DictWriter(open(f"Ghostbusters_{DATASET_TYPE_DISPLAYNAME}_{DATASET_DISPLAYNAME}_Dataset.csv", "w+"), fieldnames=["text", "generated"])
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
