import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llm_text_detectors import Detectors
import os

# Rest of your code
reference_model = "HuggingFaceTB/SmolLM-360M-Instruct"
detector = Detectors(reference_model)

# Load text from file
file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Single_sample", "sample_text.txt")

with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

# Compute score
score = detector.compute_all_metrics(text)['telescope_perplexity']
print(f"Detection score: {score}")