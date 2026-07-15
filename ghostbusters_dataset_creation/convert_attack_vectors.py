import os
import pandas as pd
from typing import Dict, List, Any

### START GLOBALS -------------------------------------------------------------------------

PERTURBATION_DIRECTORY = "original_ghostbusters_datasets/perturb"
OUTPUT_DIRECTORY = "datasets"

# Specific perturbation type to convert.
# Possible perturbation types: "char_basic", "char_cap", "char_space", "para_adj", "para_paraph", "sent_adj", "sent_paraph", "word_adj", "word_syn"
# Set to None to convert all perturbation types.
PERTURBATION_TYPE = None

# Specific perturbation level to convert.
# Possible perturbation levels depend on the chosen perturbation type:
# - For "char_basic", "char_cap", "char_space", "word_adj", "word_syn": 
#   "0", "1", "2", "3", "4", "5", "10", "20", "50", "100", "200"
# - For "para_adj", "para_paraph", "sent_adj", "sent_paraph": 
#   "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"
# Set to None to convert all perturbation levels.
PERTURBATION_LEVEL = None

### END GLOBALS ---------------------------------------------------------------------------

PERTURBATION_MAPPINGS: Dict[str, str] = {
    "char_basic": "Character_Basic",
    "char_cap": "Character_Capitalization",
    "char_space": "Character_Space",
    "para_adj": "Paragraph_Adjacent",
    "para_paraph": "Paragraph_Paraphrase",
    "sent_adj": "Sentence_Adjacent",
    "sent_paraph": "Sentence_Paraphrase",
    "word_adj": "Word_Adjacent",
    "word_syn": "Word_Synonym",
}

def convert_perturbation_to_telescope_format(
    perturbation_directory: str,
    output_directory: str,
    labels_file: str,
    perturbation_type: str,
    perturbation_level: str
) -> None:
    perturbation_type_directory = os.path.join(perturbation_directory, perturbation_type)
    perturbation_level_directory = os.path.join(perturbation_type_directory, perturbation_level)
    
    if not os.path.isdir(perturbation_level_directory):
        print(f"Skipping: Directory {perturbation_level_directory} does not exist.")
        return
        
    # Load labels
    with open(labels_file, "r") as f:
        labels = [line.strip() for line in f.readlines()]
        
    mapped_name = PERTURBATION_MAPPINGS.get(perturbation_type, perturbation_type.title())
    output_filename = f"Ghostbusters_Perturb_{mapped_name}_{perturbation_level}.csv"
    output_path = os.path.join(output_directory, output_filename)
    
    data: List[Dict[str, Any]] = []
    
    for filename in os.listdir(perturbation_level_directory):
        file_path = os.path.join(perturbation_level_directory, filename)
        if not os.path.isfile(file_path) or not filename.endswith(".txt"):
            continue
            
        try:
            index = int(filename[:-4])
        except ValueError:
            continue
            
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
            
        if index < len(labels):
            label = int(labels[index])
            data.append({"text": text, "generated": label})
            
    if not data:
        print(f"No valid text files found in {perturbation_level_directory}.")
        return
        
    df = pd.DataFrame(data, columns=["text", "generated"])
    df.to_csv(output_path, index=False)
    print(f"Successfully generated {output_path} with {len(df)} records.")



def main() -> None:
    perturbation_directory = PERTURBATION_DIRECTORY
    output_directory = OUTPUT_DIRECTORY
    
    if not os.path.exists(perturbation_directory):
        alt_path = os.path.join("ghostbusters_dataset_creation", perturbation_directory)
        if os.path.exists(alt_path):
            perturbation_directory = alt_path
            
    labels_file = os.path.join(perturbation_directory, "labels.txt")
    if not os.path.isfile(labels_file):
        raise FileNotFoundError(f"Labels file not found at: {labels_file}")
        
    if not os.path.exists(output_directory):
        alt_output_dir = os.path.join("datasets")
        if os.path.exists(alt_output_dir):
            output_directory = alt_output_dir
        else:
            os.makedirs(output_directory, exist_ok=True)
            

    if PERTURBATION_TYPE:
        perturbation_types_to_convert = [PERTURBATION_TYPE]
    else:
        perturbation_types_to_convert = []
        for d in os.listdir(perturbation_directory):
            if os.path.isdir(os.path.join(perturbation_directory, d)):
                perturbation_types_to_convert.append(d)
    
    for perturbation_type in perturbation_types_to_convert:
        if perturbation_type not in PERTURBATION_MAPPINGS:
            continue
        perturbation_type_path = os.path.join(perturbation_directory, perturbation_type)
        
        if PERTURBATION_LEVEL:
            perturbation_levels_to_convert = [PERTURBATION_LEVEL]
        else:
            perturbation_levels_to_convert = []
            for d in os.listdir(perturbation_type_path):
                if os.path.isdir(os.path.join(perturbation_type_path, d)):
                    perturbation_levels_to_convert.append(d)
        
        for perturbation_level in perturbation_levels_to_convert:
            convert_perturbation_to_telescope_format(perturbation_directory, output_directory, labels_file, perturbation_type, perturbation_level)




if __name__ == "__main__":
    main()
