import os
import pandas as pd
import csv

# WARNING, THIS SCRIPT DOES NOT WORK ON THE REUTERS (NEWS) PART OF THE GHOSTBUSTERS DATASET
# PLEASE LOOK AT THE SPECIFIC CONVERSION FILE FOR THAT PART OF THE DATASET
attack_datasets_to_create = [
    ("Paragraph_Adjacent", "para_adj", 0), # name, attack type, level
    ("Paragraph_Adjacent", "para_adj", 1), # name, attack type, level
    ("Paragraph_Adjacent", "para_adj", 2), # name, attack type, level
    ("Paragraph_Adjacent", "para_adj", 3), # name, attack type, level
    ("Paragraph_Adjacent", "para_adj", 4), # name, attack type, level
    ("Paragraph_Adjacent", "para_adj", 5), # name, attack type, level
    ("Paragraph_Adjacent", "para_adj", 6), # name, attack type, level
    ("Paragraph_Adjacent", "para_adj", 7), # name, attack type, level
    ("Paragraph_Adjacent", "para_adj", 8), # name, attack type, level
    ("Paragraph_Adjacent", "para_adj", 9), # name, attack type, level
    ("Paragraph_Adjacent", "para_adj", 10), # name, attack type, level
    
    ("Paragraph_Paraphrase", "para_paraph", 0), # name, attack type, level
    ("Paragraph_Paraphrase", "para_paraph", 1), # name, attack type, level
    ("Paragraph_Paraphrase", "para_paraph", 2), # name, attack type, level
    ("Paragraph_Paraphrase", "para_paraph", 3), # name, attack type, level
    ("Paragraph_Paraphrase", "para_paraph", 4), # name, attack type, level
    ("Paragraph_Paraphrase", "para_paraph", 5), # name, attack type, level
    ("Paragraph_Paraphrase", "para_paraph", 6), # name, attack type, level
    ("Paragraph_Paraphrase", "para_paraph", 7), # name, attack type, level
    ("Paragraph_Paraphrase", "para_paraph", 8), # name, attack type, level
    ("Paragraph_Paraphrase", "para_paraph", 9), # name, attack type, level
    ("Paragraph_Paraphrase", "para_paraph", 10), # name, attack type, level
    
    ("Sentence_Adjacent", "sent_adj", 0), # name, attack type, level
    ("Sentence_Adjacent", "sent_adj", 1), # name, attack type, level
    ("Sentence_Adjacent", "sent_adj", 2), # name, attack type, level
    ("Sentence_Adjacent", "sent_adj", 3), # name, attack type, level
    ("Sentence_Adjacent", "sent_adj", 4), # name, attack type, level
    ("Sentence_Adjacent", "sent_adj", 5), # name, attack type, level
    ("Sentence_Adjacent", "sent_adj", 6), # name, attack type, level
    ("Sentence_Adjacent", "sent_adj", 7), # name, attack type, level
    ("Sentence_Adjacent", "sent_adj", 8), # name, attack type, level
    ("Sentence_Adjacent", "sent_adj", 9), # name, attack type, level
    ("Sentence_Adjacent", "sent_adj", 10), # name, attack type, level
    
        
    ("Sentence_Paraphrase", "sent_paraph", 0), # name, attack type, level
    ("Sentence_Paraphrase", "sent_paraph", 1), # name, attack type, level
    ("Sentence_Paraphrase", "sent_paraph", 2), # name, attack type, level
    ("Sentence_Paraphrase", "sent_paraph", 3), # name, attack type, level
    ("Sentence_Paraphrase", "sent_paraph", 4), # name, attack type, level
    ("Sentence_Paraphrase", "sent_paraph", 5), # name, attack type, level
    ("Sentence_Paraphrase", "sent_paraph", 6), # name, attack type, level
    ("Sentence_Paraphrase", "sent_paraph", 7), # name, attack type, level
    ("Sentence_Paraphrase", "sent_paraph", 8), # name, attack type, level
    ("Sentence_Paraphrase", "sent_paraph", 9), # name, attack type, level
    ("Sentence_Paraphrase", "sent_paraph", 10), # name, attack type, level

]

for name, attack_type, level in attack_datasets_to_create:
    
    text_directory = f"ghostbuster-data/perturb/{attack_type}/{level}/"
    labels_file = "ghostbuster-data/labels.txt"

    csv_writer = csv.DictWriter(open(f"Ghostbusters_Perturb_{name}_{level}.csv", "w+"), fieldnames=["text", "generated"])
    csv_writer.writeheader()

    labels = open("ghostbuster-data/perturb/labels.txt").readlines()
    print(labels)
    data = []
    for text_filename in os.listdir(text_directory):
        
        # print(text_filename)
        text_file = os.path.join(text_directory, text_filename)
        
        if not os.path.isfile(text_file): continue
        
        text = "\n".join(open(text_file).readlines())
        
        index = int(text_filename[:-4])
        # print(labels[index].strip())
        if int(labels[index].strip()) == 0:
            data.append({"text": text, "generated": 0})
        
        else:
            data.append({"text": text, "generated": 1})



    csv_writer.writerows(data)
