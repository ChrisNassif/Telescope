from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer
import torch
import os
import numpy as np

from typing import Tuple, Optional, Any, Union

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression



def load_model_and_tokenizer(
    model_path: str,
    hugging_face_auth_token: Optional[str] = None,
    quantization_config: Optional[Any] = None,
    device: Union[str, torch.device] = "cuda:0"
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    print(f"Loading tokenizer from {model_path}")

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_path, token = hugging_face_auth_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Pad token set to EOS token: ", tokenizer.pad_token)
    print("Tokenizer loaded successfully")


    # TODO FIX THIS!!!! WHY IS THIS HERE???
    dtype: torch.dtype
    if model_path in {"EleutherAI/gpt-neo-2.7B", "EleutherAI/gpt-j-6b"}:
        dtype = torch.float32
    else:
        dtype = torch.float16

    print(f"Using dtype: {dtype} for {model_path}")

    # Load the base model
    print("Loading base model...")
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_path,
        token=hugging_face_auth_token,
        quantization_config=quantization_config,
        device_map=device,
#        attn_implementation="flash_attention_2",
        torch_dtype=dtype
    )

    print("Base model loaded successfully")

    return model, tokenizer


def get_hugging_face_auth_token() -> str:
    token: Optional[str] = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN environment variable is not set. Please set it with your Hugging Face token.")
    return token


def create_logistic_regression_classifier(metric: Any, labels: Any) -> Pipeline:
    """
    Uses a logistic regression classifier to determine the classification threshold for a single metric.

    This should be equivalent to finding the decision threshold that maximizes accuracy, and a bonus is that
    the logistic regression creates a probability distribution to directly quantify how sure the classifier is.
    """
    if hasattr(metric, 'shape') and len(metric.shape) == 1:
        metric = metric.reshape(-1, 1)
    classifier: Pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    classifier.fit(metric, labels)
    return classifier


def calculate_optimal_bin_count(x: Any) -> int:
    """
    Calculate the optimal number of bins for a histogram using the Freedman-Diaconis rule.

    The Freedman-Diaconis rule is designed to minimize the difference between the empirical
    probability distribution of the sample and the theoretical probability distribution. 
    It is less sensitive to outliers than Scott's rule and works well for large datasets.

    The formula is:
        bin_width = 2 * IQR(x) / (n ** (1/3))
        number_of_bins = ceil((max(x) - min(x)) / bin_width)
    where IQR is the Interquartile Range (75th percentile - 25th percentile) and n is the
    number of data points.

    Edge cases handled:
    - If the dataset is empty, it returns a default of 30 bins.
    - If the Interquartile Range is 0 (e.g., heavily concentrated data) or the calculated
      width is NaN, a default value of 30 bins is returned.
    - The returned bin count is capped at 50 to prevent excessively large numbers of bins.

    Parameters
    ----------
    x : array-like
        The input data (e.g., list, numpy array, or pandas Series) for which the bin count
        needs to be calculated.

    Returns
    -------
    int
        The optimal number of bins (between 1 and 50).
    """
    x_arr: np.ndarray = np.asarray(x)
    if len(x_arr) == 0:
        return 30
    quartile_75: float
    quartile_25: float
    quartile_75, quartile_25 = np.percentile(x_arr, [75, 25])
    inter_quartile_range: float = quartile_75 - quartile_25
    bin_width: float = 2 * inter_quartile_range / (len(x_arr) ** (1/3))
    if bin_width == 0 or bin_width != bin_width:  # handles 0 and NaN
        return 30
    bins: int = int(np.ceil((x_arr.max() - x_arr.min()) / bin_width))
    return min(bins, 50)



def print_vram(step_name: str, device: Union[str, torch.device] = "cuda:0") -> None:
    """
    Prints the amount of VRAM that is currently being used. Primarily for debugging.
    """
    # Convert bytes to gigabytes
    allocated: float = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved: float = torch.cuda.memory_reserved(device) / (1024 ** 3)
    
    print(f"[{step_name}] VRAM Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")

