from telescope_llm_text_detection import Detectors


TEXT_SAMPLE = "This is a text sample."



PERFORMER_MODEL_NAME = "HuggingFaceTB/SmolLM-360M-Instruct"
OBSERVER_MODEL_NAME = "HuggingFaceTB/SmolLM-360M"

def main():
    telescope_detector = Detectors(PERFORMER_MODEL_NAME, OBSERVER_MODEL_NAME)
    print("Telescope Perplexity:", telescope_detector.compute_telescope_perplexity(TEXT_SAMPLE))
    print("Perplexity:", telescope_detector.compute_perplexity(TEXT_SAMPLE))
    print("Binoculars Score:", telescope_detector.compute_binoculars_score(TEXT_SAMPLE))
    print("DetectLLM LRR:", telescope_detector.compute_detectllm_log_rank_ratio(TEXT_SAMPLE))
    print("Fast-DetectGPT:", telescope_detector.compute_fast_detectgpt(TEXT_SAMPLE))


if __name__ == "__main__":
    main()