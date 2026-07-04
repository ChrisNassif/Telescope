import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_text_detectors import Detectors
from llm_text_detectors.utils import get_hugging_face_auth_token


TEXT_SAMPLE = "Distinguishing Large Language Model (LLM) generated text from human writing is a critical challenge. While LLMs master complex linguistic features, we hypothesize their training process leaves indelible marks: \textbf{1)} LLMs develop strong biases regarding fundamental local sequence probabilities, particularly an aversion to token repetition, very early in training, and \textbf{2)} these biases persist as ``frozen habits'' or developmental artifacts, differing systematically from human text patterns. To probe this phenomenon, we introduce \texttt{Telescope}, a metric evaluating the model's likelihood of the token it just processed, $P(s_i | s_{1:i})$. Although frustratingly simple, we argue \texttt{Telescope} effectively probes this deep-seated repetition bias. Our empirical investigation reveals that the \texttt{Telescope} signature indeed emerges early in pre-training and remains stable, supporting the ``frozen habit'' hypothesis related to local sequence likelihoods. Furthermore, we demonstrate that \texttt{Telescope} enables highly effective zero-shot LLM detection, achieving state-of-the-art or competitive performance across diverse datasets (including new evaluation sets we introduce), reference models, and perturbation schemes, often with greater efficiency. This work identifies persistent, early-learned biases in local repetition likelihood as a key LLM differentiator and validates a simple yet conceptually grounded probe, Telescope, for detecting these developmental artifacts."



PERFORMER_MODEL_NAME = "HuggingFaceTB/SmolLM-360M-Instruct"
OBSERVER_MODEL_NAME = "HuggingFaceTB/SmolLM-360M"

def main():
    hugging_face_auth_token = get_hugging_face_auth_token()
    telescope_detector = Detectors(OBSERVER_MODEL_NAME, PERFORMER_MODEL_NAME, hugging_face_auth_token)
    telescope_perplexity = telescope_detector.compute_telescope_perplexity(TEXT_SAMPLE)
    
    print(telescope_perplexity)


if __name__ == "__main__":
    main()