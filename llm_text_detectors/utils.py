from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer
import torch
import os
import numpy as np
import huggingface_hub
from typing import Tuple

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression



def load_model_and_tokenizer(model_path: str, hugging_face_auth_token: str = None, quantization_config = None) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    print(f"Loading tokenizer from {model_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, token = hugging_face_auth_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Pad token set to EOS token: ", tokenizer.pad_token)
    print("Tokenizer loaded successfully")

    # Check CUDA availability
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
    else:
        print("CUDA is not available. Using CPU.")

    if model_path in {"EleutherAI/gpt-neo-2.7B", "EleutherAI/gpt-j-6b"}:
        dtype = torch.float32
    else:
        dtype = torch.float16

    print(f"Using dtype: {dtype} for {model_path}")

    # Load the base model
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        token=hugging_face_auth_token,
        quantization_config=quantization_config,
        device_map="auto",
#        attn_implementation="flash_attention_2",
        torch_dtype=dtype
    )

    print("Base model loaded successfully")

    return model, tokenizer


def get_hugging_face_auth_token():
    token = os.environ.get("HF_TOKEN")
    if not token:
        huggingface_hub.login()
    else:
        raise EnvironmentError("HF_TOKEN environment variable is not set and login failed. Please set token with your Hugging Face token or login.")
    return token


def create_logistic_regression_classifier(metric, labels) -> Pipeline:
    """
    Uses a logistic regression classifier to determine the classification threshold for a single metric.

    This should be equivalent to finding the decision threshold that maximizes accuracy, and a bonus is that
    the logistic regression creates a probability distribution to directly quantify how sure the classifier is.
    """
    if hasattr(metric, 'shape') and len(metric.shape) == 1:
        metric = metric.reshape(-1, 1)
    clf: Pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    clf.fit(metric, labels)
    return clf


def calc_bins(x):
    """Calculate optimal number of bins using the Freedman-Diaconis rule."""
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    bin_width = 2 * iqr / (len(x) ** (1/3))
    if bin_width == 0 or bin_width != bin_width:  # handles 0 and NaN
        return 30
    bins = int(np.ceil((x.max() - x.min()) / bin_width))
    return min(bins, 50)


def print_vram(step_name: str, device: str = "cuda:0"):
    # Convert bytes to Gigabytes
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    
    print(f"[{step_name}] VRAM Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")

