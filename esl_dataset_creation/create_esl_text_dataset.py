import pandas as pd
import openai
import time
from tqdm import tqdm
import os
import json
from typing import Optional, Set



### START GLOBALS -------------------------------------------------------------------------

INPUT_FILE: str = "train.csv"
OUTPUT_FILE: str = "ESL_GPT4o_Dataset.csv"
CHECKPOINT_FILE: str = "checkpoint.json"

API_KEY: Optional[str] = os.getenv('OPENAI_API_KEY')

BATCH_SIZE: int = 100

### END GLOBALS ---------------------------------------------------------------------------




def rewrite_text(text: str, client: openai.OpenAI) -> str:
    """
    Sends text to GPT-4 for rewriting with strict output requirements.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are tasked with rewriting text. Provide ONLY the rewritten version, with no additional comments, explanations, or formatting. Maintain the core meaning while improving clarity and structure."},
            {"role": "user", "content": f"Rewrite this text. Provide only the rewritten version:\n\n{text}"}
        ],
        temperature=0.7
    )
    
    rewritten = response.choices[0].message.content.strip()
    
    if rewritten.startswith('```') and rewritten.endswith('```'):
        rewritten = rewritten[3:-3].strip()
    
    return rewritten



def save_checkpoint(checkpoint_file: str, processed_ids: Set[str]) -> None:
    """Save the IDs of processed texts"""
    with open(checkpoint_file, 'w') as f:
        json.dump(list(processed_ids), f)

def load_checkpoint(checkpoint_file: str) -> Set[str]:
    """Load the IDs of previously processed texts"""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return set(json.load(f))
    return set()


def main(input_file: str, output_file: str, api_key: str, batch_size: int = 100, checkpoint_file: str = 'checkpoint.json') -> None:
    """
    Processes the input CSV in batches and creates a new CSV with original and rewritten text.
    Supports resuming from last checkpoint.
    """
    client = openai.OpenAI(api_key=api_key)
    
    print(f"Reading input file: {input_file}")
    df = pd.read_csv(input_file)
    
    processed_ids = load_checkpoint(checkpoint_file)
    print(f"Found {len(processed_ids)} previously processed texts")
    
    df_to_process = df[~df['text_id'].astype(str).isin(processed_ids)].copy()
    
    if len(df_to_process) == 0:
        print("All texts have been processed!")
        return
    
    print(f"Total texts to process: {len(df_to_process)}")
    
    for start_index in range(0, len(df_to_process), batch_size):
        batch_df = df_to_process.iloc[start_index:start_index + batch_size]
        print(f"\nProcessing batch {start_index//batch_size + 1} ({len(batch_df)} texts)")
        
        batch_rows = []
        
        for _, row in tqdm(batch_df.iterrows(), total=len(batch_df)):
            time.sleep(0.5) 
            rewritten = rewrite_text(row['full_text'], client)
            rewritten_val = rewritten if rewritten else "Error in processing"
            
            # Original text
            batch_rows.append({'text': row['full_text'], 'generated': 0})
            
            # Rewritten text
            batch_rows.append({'text': rewritten_val, 'generated': 1})
            
            processed_ids.add(str(row['text_id']))
            save_checkpoint(checkpoint_file, processed_ids)
        
        batch_output = pd.DataFrame(batch_rows, columns=['text', 'generated'])
        
        if os.path.exists(output_file):
            batch_output.to_csv(output_file, mode='a', header=False, index=False)
        else:
            batch_output.to_csv(output_file, index=False)
        
        print(f"\nBatch {start_index//batch_size + 1} complete. Saved to {output_file}")
        
        if start_index + batch_size < len(df_to_process):
            response = input("\nDo you want to process the next batch? (y/n): ").lower()
            if response != 'y':
                print("Processing paused. Run the script again to continue from where you left off.")
                break



if __name__ == "__main__":
    api_key = API_KEY
    if not api_key:
        api_key = input("Please enter your OpenAI API key: ").strip()
    main(INPUT_FILE, OUTPUT_FILE, api_key, BATCH_SIZE, CHECKPOINT_FILE)