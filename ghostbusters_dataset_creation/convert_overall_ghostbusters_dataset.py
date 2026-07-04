import os
import pandas as pd
import csv

# WARNING, THIS SCRIPT DOES NOT WORK ON THE REUTERS (NEWS) PART OF THE GHOSTBUSTERS DATASET
# PLEASE LOOK AT THE SPECIFIC CONVERSION FILE FOR THAT PART OF THE DATASET

# (human written, ai written)
text_directories = [
    {
        'original_ghostbusters_datasets/essay/human': [
            'original_ghostbusters_datasets/essay/gpt', 
            'original_ghostbusters_datasets/essay/gpt_prompt1',
            'original_ghostbusters_datasets/essay/gpt_prompt2',
            'original_ghostbusters_datasets/essay/gpt_semantic',
            'original_ghostbusters_datasets/essay/gpt_writing',
            'original_ghostbusters_datasets/essay/claude'
        ],
        'original_ghostbusters_datasets/wp/human': [
            'original_ghostbusters_datasets/wp/gpt',
            'original_ghostbusters_datasets/wp/gpt_prompt1',
            'original_ghostbusters_datasets/wp/gpt_prompt2',
            'original_ghostbusters_datasets/wp/gpt_semantic',
            'original_ghostbusters_datasets/wp/gpt_writing',
            'original_ghostbusters_datasets/wp/claude'
        ]
    }
]

# news is formatted differently, so we need to treat it differently
news_text_directories = [
    {
        'original_ghostbusters_datasets/essay/human': [
            'original_ghostbusters_datasets/reuter/gpt',
            'original_ghostbusters_datasets/reuter/gpt_prompt1',
            'original_ghostbusters_datasets/reuter/gpt_prompt2',
            'original_ghostbusters_datasets/reuter/gpt_semantic',
            'original_ghostbusters_datasets/reuter/gpt_writing',
            'original_ghostbusters_datasets/reuter/claude'
        ]
    }
]



csv_writer = csv.DictWriter(open("Ghostbusters_Perturbed_Dataset.csv", "w+"), fieldnames=["text", "generated"])
csv_writer.writeheader()

data = []

# TODO FIX THIS SO THAT WE ONLY GO THROUGH THE HUMAN WRITTEN TEXT ONCE AND THEN GO THROUGH ALL OF THE GPT GENERATED THINGS
# for human_written_text_directory, ai_written_text_directories in text_directories:
#     for filename in os.listdir(human_written_text_directory):
#         print(f"filename: {filename}")
        
#         human_written_text_file = os.path.join(human_written_text_directory, filename)
        
#         if not os.path.isfile(human_written_text_file): continue
        
#         human_written_text = "\n".join(open(human_written_text_file).readlines())
        
#         data.append({"text": human_written_text, "generated": 0})

        
#     for ai_written_text_directory in os.listdir(ai_written_text_directories):
#         absolute_ai_written_text_directory = os.path.join(ai_written_text_directory, filename)
        
        
#         for filename in os.listdir(absolute_ai_written_text_directory):
#             ai_written_text_file = os.path.join(absolute_ai_written_text_directory, filename)
            
#             if not os.path.isfile(ai_written_text_file): continue
#             ai_written_text = "\n".join(open(ai_written_text_file).readlines())
            
#             data.append({"text": ai_written_text, "generated": 1})





# for human_written_text_directory, ai_written_text_directory in news_text_directories:
#     for foldername in os.listdir(human_written_text_directory):
#         print(f"filename: {foldername}")

#         human_written_subfolder = os.path.join(human_written_text_directory, foldername)
#         ai_written_subfolder = os.path.join(ai_written_text_directory, foldername)
        
#         for filename in os.listdir(human_written_subfolder):
            
#             human_written_text_file = os.path.join(human_written_subfolder, filename)
#             ai_written_text_file = os.path.join(ai_written_subfolder, filename)
            
#             print(human_written_text_file)
#             if not os.path.isfile(human_written_text_file): continue
#             if not os.path.isfile(ai_written_text_file): continue
            
#             human_written_text = "\n".join(open(human_written_text_file).readlines())
#             ai_written_text = "\n".join(open(ai_written_text_file).readlines())
            
#             data.append({"text": human_written_text, "generated": 0})
#             data.append({"text": ai_written_text, "generated": 1})




# HANDLE PERTURBATIONS FOLDER
labels = open("original_ghostbusters_datasets/perturb/labels.txt").readlines()

for perturbation_type_folder_path in os.listdir("original_ghostbusters_datasets/perturb"):
    
    perturbation_type_folder_path = os.path.join("original_ghostbusters_datasets/perturb", perturbation_type_folder_path)
    if os.path.isfile(perturbation_type_folder_path): continue
    
    for level_of_perturbation_folder_name in os.listdir(perturbation_type_folder_path):
        
        level_of_perturbation_folder_path = os.path.join(perturbation_type_folder_path, level_of_perturbation_folder_name)
        if os.path.isfile(level_of_perturbation_folder_path): continue
        
        for text_file_name in os.listdir(level_of_perturbation_folder_path):
            
            text_file_path = os.path.join(level_of_perturbation_folder_path, text_file_name)
            if not os.path.isfile(text_file_path): continue
            
            text = "\n".join(open(text_file_path).readlines())
            
            index = int(text_file_name[:-4])
            data.append({"text": text, "generated": int(labels[index])})


csv_writer.writerows(data)
