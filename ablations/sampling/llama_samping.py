import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llm_text_detectors import Detectors
import torch
import pandas as pd
import numpy as np
from typing import Dict, List
import time


def generate_with_sampling(
    model,
    tokenizer,
    prompt: str,
    max_length: int = 1024,
    num_samples: int = 500,
    sampling_params: Dict = None
) -> List[str]:
    """Generate text using different sampling parameters"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Base generation parameters
    gen_params = {
        "max_length": max_length,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    if sampling_params:
        # Handle temperature and top_p based on do_sample
        do_sample = sampling_params.get("do_sample", False)
        
        # Only add temperature and top_p if sampling is enabled
        if do_sample:
            if "temperature" in sampling_params:
                gen_params["temperature"] = sampling_params["temperature"]
            if "top_p" in sampling_params:
                gen_params["top_p"] = sampling_params["top_p"]
        else:
            # Explicitly remove to avoid warnings
            gen_params.pop("temperature", None)
            gen_params.pop("top_p", None)
        
        # Add other parameters
        gen_params["do_sample"] = do_sample
        if "num_beams" in sampling_params:
            gen_params["num_beams"] = sampling_params["num_beams"]
    
    # Adjust num_return_sequences based on beam size
    if gen_params.get("num_beams", 1) > 1:
        gen_params["num_return_sequences"] = min(num_samples, gen_params["num_beams"])
    else:
        gen_params["num_return_sequences"] = num_samples if gen_params.get("do_sample", False) else 1
    
    # Generate sequences
    try:
        outputs = model.generate(**inputs, **gen_params)
        
        # Decode outputs
        generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        
        # If we need more sequences, duplicate them
        while len(generated_texts) < num_samples:
            generated_texts.extend(generated_texts[:num_samples - len(generated_texts)])
            
        return generated_texts[:num_samples]
        
    except Exception as e:
        print(f"Generation error: {str(e)}")
        return [prompt] * num_samples  # Return original prompt as fallback   
def process_metrics(metrics: Dict) -> Dict:
    """Convert all metric values to Python floats"""
    processed = {}
    for k, v in metrics.items():
        if isinstance(v, (torch.Tensor, np.ndarray)):
            # Handle multi-dimensional arrays
            if hasattr(v, 'ndim') and v.ndim > 0:
                processed[k] = float(v.item())  # Use item() to safely extract scalar value
            else:
                processed[k] = float(v)
        else:
            processed[k] = v
    return processed


# TODO THIS NEEDS TO GET FIXED ASAP NONE OF THESE ARGUMENTS MAKE SENSE
# TODO AUTH TOKEN SHOULD NOT BE PASSED IN??? WHAT IS THE TELESCOPE OBJECT BEING PASSED IN
# TODO WHY ISN'T THIS BEING LOADED IN USING THE UTILS
# TODO ONCE THIS IS FIXED REMOVE THE get_hugging_face_token() function from imports and usage
def analyze_sampling_methods(telescope, prompt: str, model_name: str, auth_token: str):
    results = []
    
    # Define different sampling configurations to test
    sampling_configs = {
        "greedy": {
            "do_sample": False,
            "num_beams": 1
        },
        "beam_search_2": {
            "do_sample": False,
            "num_beams": 2
        },
        "beam_search_5": {
            "do_sample": False,
            "num_beams": 5
        },
        "temperature_0.7": {
            "do_sample": True,
            "temperature": 0.7
        },
        "temperature_1.0": {
            "do_sample": True,
            "temperature": 1.0
        },
        "temperature_1.5": {
            "do_sample": True,
            "temperature": 1.5
        },
        "nucleus_0.9": {
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 0.9
        },
        "nucleus_0.7": {
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 0.7
        },
    }
    
    try:
        # Configure quantization for Llama model
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        # Load model with proper authentication
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=auth_token,
            trust_remote_code=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=auth_token,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto"
        )
        
        # Set pad token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return pd.DataFrame()
    
    for method_name, params in sampling_configs.items():
        print(f"Testing {method_name}...")
        
        try:
            # Generate samples
            samples = generate_with_sampling(model, tokenizer, prompt, sampling_params=params)
            
            # Analyze each sample
            method_results = []
            for sample in samples:
                try:
                    perplexity, cross_perplexity, metrics = telescope.compute_all_metrics(sample)
                    
                    # Process metrics and ensure scalar values
                    result = {
                        "method": method_name,
                        "telescope_perplexity": float(np.asarray(perplexity).item()),
                        "telescope_cross_perplexity": float(np.asarray(cross_perplexity).item()),
                        "telescope_perplexity_divided_by_cross_perplexity": float(np.asarray(perplexity/cross_perplexity).item()),
                        **process_metrics(metrics)
                    }
                    method_results.append(result)
                except Exception as e:
                    print(f"Error analyzing sample for {method_name}: {str(e)}")
                    continue
            
            if method_results:
                # Calculate mean and std for this method
                method_df = pd.DataFrame(method_results)
                
                # Replace infinity values with NaN
                method_df = method_df.replace([np.inf, -np.inf], np.nan)
                
                # Calculate statistics, handling NaN values
                mean_results = method_df.mean(numeric_only=True, skipna=True)
                std_results = method_df.std(numeric_only=True, skipna=True)
                
                results.append({
                    "method": method_name,
                    "mean_telescope_perplexity_divided_by_cross_perplexity": float(mean_results["telescope_perplexity_divided_by_cross_perplexity"]),
                    "std_telescope_perplexity_divided_by_cross_perplexity": float(std_results["telescope_perplexity_divided_by_cross_perplexity"]),
                    "mean_perplexity": float(mean_results["telescope_perplexity"]),
                    "std_perplexity": float(std_results["telescope_perplexity"]),
                    "mean_cross_perplexity": float(mean_results["telescope_cross_perplexity"]),
                    "std_cross_perplexity": float(std_results["telescope_cross_perplexity"]),
                    "mean_entropy_ratio": float(mean_results.get("entropy_ratio", 0)),
                    "mean_kl_divergence": float(mean_results.get("kl_divergence", 0)),
                    "mean_performer_distribution_overlap": float(mean_results.get("performer_distribution_overlap", 0))
                })
            
        except Exception as e:
            print(f"Error with {method_name}: {str(e)}")
            continue
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Initialize Telescope
    observer_model = "HuggingFaceTB/SmolLM-135M"
    performer_model = "HuggingFaceTB/SmolLM-135M-instruct"
    
    try:
        from llm_text_detectors.utils import get_hugging_face_auth_token
        token = get_hugging_face_auth_token()
            
        telescope = Detectors(performer_model, observer_model)
        
        # Llama model configuration
        llama_model = "meta-llama/Llama-3.2-1B-Instruct"
        
        # Test prompt
        prompt = "Write a short story about a robot learning to feel emotions:"
        
        # Run analysis with Llama model
        results_df = analyze_sampling_methods(telescope, prompt, llama_model, token)
        
        if not results_df.empty:
            print("\nResults:")
            with pd.option_context('display.float_format', '{:.6f}'.format):
                print(results_df.to_string())
            
            # Save results
            os.makedirs("sampling", exist_ok=True)
            results_df.to_csv("sampling/sampling_analysis_results.csv", index=False)
        else:
            print("No results were generated. Please check the errors above.")
            
    except Exception as e:
        print(f"Error running analysis: {str(e)}")