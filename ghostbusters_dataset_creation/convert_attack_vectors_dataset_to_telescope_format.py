import os
import pandas as pd
import csv


### START GLOBALS -------------------------------------------------------------------------

# The formatting of the tuples in this list is as follows:
# tuple of (name, attack type, perturbation level) 
# The "perturbation level" the number of this specific type of perturbation)
ATTACK_DATASETS_TO_CREATE = [
    ("Paragraph_Adjacent", "para_adj", 0),
    ("Paragraph_Adjacent", "para_adj", 1),
    ("Paragraph_Adjacent", "para_adj", 2),
    ("Paragraph_Adjacent", "para_adj", 3),
    ("Paragraph_Adjacent", "para_adj", 4),
    ("Paragraph_Adjacent", "para_adj", 5),
    ("Paragraph_Adjacent", "para_adj", 6),
    ("Paragraph_Adjacent", "para_adj", 7),
    ("Paragraph_Adjacent", "para_adj", 8),
    ("Paragraph_Adjacent", "para_adj", 9),
    ("Paragraph_Adjacent", "para_adj", 10),
    
    ("Paragraph_Paraphrase", "para_paraph", 0),
    ("Paragraph_Paraphrase", "para_paraph", 1),
    ("Paragraph_Paraphrase", "para_paraph", 2),
    ("Paragraph_Paraphrase", "para_paraph", 3),
    ("Paragraph_Paraphrase", "para_paraph", 4),
    ("Paragraph_Paraphrase", "para_paraph", 5),
    ("Paragraph_Paraphrase", "para_paraph", 6),
    ("Paragraph_Paraphrase", "para_paraph", 7),
    ("Paragraph_Paraphrase", "para_paraph", 8),
    ("Paragraph_Paraphrase", "para_paraph", 9),
    ("Paragraph_Paraphrase", "para_paraph", 10),
    
    ("Sentence_Adjacent", "sent_adj", 0),
    ("Sentence_Adjacent", "sent_adj", 1),
    ("Sentence_Adjacent", "sent_adj", 2),
    ("Sentence_Adjacent", "sent_adj", 3),
    ("Sentence_Adjacent", "sent_adj", 4),
    ("Sentence_Adjacent", "sent_adj", 5),
    ("Sentence_Adjacent", "sent_adj", 6),
    ("Sentence_Adjacent", "sent_adj", 7),
    ("Sentence_Adjacent", "sent_adj", 8),
    ("Sentence_Adjacent", "sent_adj", 9),
    ("Sentence_Adjacent", "sent_adj", 10), 
    
    ("Sentence_Paraphrase", "sent_paraph", 0),
    ("Sentence_Paraphrase", "sent_paraph", 1),
    ("Sentence_Paraphrase", "sent_paraph", 2),
    ("Sentence_Paraphrase", "sent_paraph", 3),
    ("Sentence_Paraphrase", "sent_paraph", 4),
    ("Sentence_Paraphrase", "sent_paraph", 5),
    ("Sentence_Paraphrase", "sent_paraph", 6),
    ("Sentence_Paraphrase", "sent_paraph", 7),
    ("Sentence_Paraphrase", "sent_paraph", 8),
    ("Sentence_Paraphrase", "sent_paraph", 9),
    ("Sentence_Paraphrase", "sent_paraph", 10),
]


### END GLOBALS ---------------------------------------------------------------------------





for name, attack_type, level in ATTACK_DATASETS_TO_CREATE:
    
    text_directory = f"original_ghostbusters_datasets/perturb/{attack_type}/{level}/"
    labels_file = "original_ghostbusters_datasets/labels.txt"

    csv_writer = csv.DictWriter(open(f"Ghostbusters_Perturb_{name}_{level}.csv", "w+"), fieldnames=["text", "generated"])
    csv_writer.writeheader()

    labels = open("original_ghostbusters_datasets/perturb/labels.txt").readlines()
    data = []
    for text_filename in os.listdir(text_directory):
    
        text_file = os.path.join(text_directory, text_filename)    
        if not os.path.isfile(text_file): continue
        
        text = "\n".join(open(text_file).readlines())
        index = int(text_filename[:-4])

        if int(labels[index].strip()) == 0:
            data.append({"text": text, "generated": 0})
        else:
            data.append({"text": text, "generated": 1})

    csv_writer.writerows(data)
