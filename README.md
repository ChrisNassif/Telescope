# Telescope: Improving Zero-Shot Detection of LLM-Generated Content With Token Repetition

<p align="center">
    <b>A novel approach to detecting AI-generated text through token repetition analysis</b>
</p>

<p align="center">
<a href="https://www.python.org/downloads/release/python-399/"><img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+"></a>
</p>

<p align="center">
    Paper (Coming Soon on Arxiv) •
    <a href="https://huggingface.co/datasets/Aanimated/telescope_datasets">Datasets</a> •
    <a href="https://huggingface.co/datasets/Aanimated/telescope_experiment_results">Raw Results</a>
</p>

## Overview

Telescope introduces a new metric for detecting LLM-generated content in zero-shot settings by analyzing token repetition patterns. This repository contains the complete implementation and links to the datasets and experimental results from our research.

## Installation

### Prerequisites
- Python 3.10 or higher
- Hugging Face account (for accessing certain models)

### Setup Instructions

**1. Install Miniconda**  
Follow the [official Miniconda installation guide](https://docs.conda.io/en/latest/miniconda.html) for your operating system.

**2. Install Git and Git LFS**  
Follow the official guides to install [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) and [Git LFS](https://git-lfs.com/) for your operating system.

**3. Download the repository**
```bash
git clone https://github.com/ChrisNassif/Telescope && cd Telescope
```

**4. Create and activate the custom python environment using conda**
```bash
conda env create -f telescope_env.yml
conda init
conda activate telescope
```

**5. Install the package**
The recommended way to install the telescope package is directly from PyPI:
```bash
pip install telescope_llm_text_detection
```

Alternatively, to install it locally in development/editable mode if you would like to edit the code:
```bash
pip install -e .
```

All required packages should now be installed. If you encounter any missing dependencies, issues, or other hiccups during installation or usage, please [open an issue](../../issues).

### Hugging Face Authentication

Some models used in this work require authentication. To set up your Hugging Face token:

**1. Generate a token** at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

**2. Set the `HF_TOKEN` environment variable** (replace `XXXXXXXX` with your actual token):

Unix/Linux/macOS:
```bash
export HF_TOKEN=XXXXXXXX
```

Windows (PowerShell):
```powershell
$env:HF_TOKEN="XXXXXXXX"
```

To make this permanent, add the export line to your `~/.bashrc`, `~/.zshrc`, or equivalent shell profile like so:
```bash
echo "export HF_TOKEN=XXXXXXXX" >> ~/.bashrc
```



### Datasets and Experiment Results

Download the datasets and experiment results using the following commands. Please note that they are fairly large and will consume approximately 38 GB of storage.

```bash
# Download experiment results
git lfs clone https://huggingface.co/datasets/Aanimated/telescope_experiment_results experiment_results

# Download datasets
git lfs clone https://huggingface.co/datasets/Aanimated/telescope_datasets datasets

# Download original ghostbusters datasets (only needed if you want to generate new ghostbusters datasets)
git lfs clone https://github.com/ChrisNassif/telescope_original_ghostbusters_datasets ghostbusters_dataset_creation/original_ghostbusters_datasets
```

## Project Structure

```
telescope/                      # Repository root
├── llm_text_detectors/          # Core package folder (packaged as telescope_llm_text_detection)
│   ├── __init__.py             # Exports Telescope, utils
│   ├── llm_text_detectors.py   # Telescope detector class
│   └── utils.py                # Utility functions (model loading, auth, shared helpers)
├── ablations/                  # Ablation studies
│   ├── per_token/              # Per-token analysis metrics
│   ├── sampling/               # LLM sampling utilities
│   ├── sequence_modeling/      # Sequence modeling dataset tools
│   ├── single_sample/          # Single sample analysis
│   ├── single_token_distribution/  # Token distribution analysis
│   └── training/               # Training utilities and logging
├── scripts/                    # Analysis and experiment scripts
│   ├── generate_experiment_results.py
│   ├── compute_roc_and_f1score_from_metrics.py
│   ├── generate_*.py           # Various plotting/analysis scripts
│   └── ...
├── ghostbusters_dataset_creation/ # Dataset conversion and creation tools
├── datasets/                   # Dataset files
├── experiment_results/         # Pre-computed experiment results
├── config.yaml                 # Global configuration (model/dataset/metric names)
├── pyproject.toml              # Package configuration
└── telescope_env.yml           # Conda environment specification
```

## Key Concepts and Definitions

### Metrics
A **metric** is a numerical value computed from a reference model's outputs. Examples include:
- Telescope Perplexity
- Binoculars Score
- Perplexity
- DetectLLM LRR

Additional experimental metrics are implemented in `llm_text_detectors/llm_text_detectors.py`. Effective metrics show correlation with whether text was LLM-generated.

### Experiment Results
**Experiment results** are CSV files containing data from running detection algorithms on specific datasets with specific reference models. Each result includes:
- Original text samples
- Ground truth labels (human vs. LLM-generated)
- Computed metric values

Browse the `experiment_results` directory to examine the data format.

### Codenames vs. Display Names
To maintain file naming conventions while preserving publication-ready formatting:
- **Codenames**: lowercase with underscores (e.g., `telescope_perplexity`)
- **Display Names**: formatted for publication (e.g., "Telescope Perplexity")

### Performer and Reference Models
This is a concept from the Binoculars paper:
- **Performer Model**: Computes both perplexity and cross-perplexity
- **Observer Model**: Only needed for cross-perplexity computation

For single-model techniques, the reference model defaults to "performer model" by convention.

See the [Binoculars paper](https://arxiv.org/abs/2401.12070) for detailed explanations.

### Configuration
The `config.yaml` file stores global variables including:
- Codenames -> display name mappings for models, datasets, and metrics
- Plot colors



## Usage

### Python API Usage

If you installed the package via PyPI, you can import and use the telescope detector in your own Python code:

```python
from telescope_llm_text_detection import Detectors
from telescope_llm_text_detection.utils import get_hugging_face_auth_token

# Initialize the detector
performer_model = "HuggingFaceTB/SmolLM-360M-Instruct"
observer_model = "HuggingFaceTB/SmolLM-360M"
token = get_hugging_face_auth_token()

detector = Detectors(observer_model, performer_model, token)

# Compute telescope perplexity
text = "Your text sample goes here."
score = detector.compute_telescope_perplexity(text)
print("Telescope Perplexity:", score)
```

### Running Experiments
In lieu of having command line arguments for every script, this codebase instead uses global variables at the top of each runnable script where you can set which arguments you want for things like which datasets or metrics to use. The reason for this is because specifying all of the metrics, datasets, models, etc takes up a lot of space and is annoying to keep track of in the runtime arguments of a script, so we just have all of in an easy place to see and edit. 

If you would like to **Generate new experiment results** by running detection algorithms on datasets:

```bash
python scripts/generate_experiment_results.py
```

This script:
- Runs reference models on text samples
- Calculates metrics (Telescope Perplexity, Binoculars Score, etc.)
- Saves raw results to CSV files

> [!IMPORTANT]
> Running experiments requires significant computational resources and time. Pre-computed results are provided to facilitate analysis without rerunning experiments.

### Analyzing Experiments
**Analyze existing experiment results** to generate:
- ROC curves
- F1-scores
- Threshold transfer characteristics
- Data visualizations

Available analysis scripts (all located in `scripts/`):
- `scripts/compute_roc_and_f1score_from_metrics.py`
- `scripts/generate_adversarial_perturbation_plot.py`
- `scripts/generate_calibration_charts.py`
- `scripts/generate_error_independence_table.py`
- `scripts/generate_length_vs_score_plot.py`
- `scripts/generate_misclassification_plots.py`
- `scripts/generate_score_distribution_plots.py`
- `scripts/generate_threshold_transfer_plot.py`

Experiment results contain only raw metric values. Analysis scripts compute performance metrics like AUROC, precision, and recall.



## Additional Metrics

Various additional metrics are implemented in `llm_text_detectors/llm_text_detectors.py` from our initial large-scale testing phase. While none proved as promising as Telescope Perplexity in our experiments, they remain available for further research and analysis.


## Future Work



## Citation

If you use Telescope in your research, please cite our paper:



```bibtex
@inproceedings{telescope2026,
  title={Telescope: Improving Zero-Shot Detection of {LLM} Generated Content By Measuring Token Repetition Probability},
  author={Christopher Nassif and Josh Cooper},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year={2026}
}
```

## License

CC BY-NC-SA 4.0

## Contact

For questions or collaboration opportunities, please [open an issue](../../issues) or contact [contact information].