# Telescope: Improving Zero-Shot Detection of LLM-Generated Content With Token Repetition

<p align="center">
    <b>A novel approach to detecting AI-generated text through token repetition analysis [Accepted ICML 2026]</b>
</p>

<p align="center">
    <a href="https://arxiv.org/pdf/2607.04061v1"><img src="https://img.shields.io/badge/arXiv-2607.04061-B31B1B.svg" alt="Paper"></a>
    <a href="https://huggingface.co/datasets/Aanimated/telescope_datasets"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-FFD21E" alt="Datasets"></a>
    <a href="https://huggingface.co/datasets/Aanimated/telescope_experiment_results"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Raw%20Results-blue" alt="Raw Results"></a>
    <a href="https://www.python.org/downloads/release/python-399/"><img src="https://img.shields.io/badge/Python-3.9+-3776AB.svg?logo=python&logoColor=white" alt="Python Version"></a>
</p>

<br>

## Overview

Telescope introduces a new metric for detecting LLM-generated content in zero-shot settings by analyzing token repetition patterns. This repository contains the complete implementation and links to the datasets and experimental results from our research.


<br>

### Performance Comparison (Average AUROC)

> [!NOTE]
> Below is the average AUROC performance across 12 reference models on diverse datasets. Bolded values inside badges indicate the best performance per dataset.
> 
> *See our paper's Appendix for the full results.*

| Dataset | Telescope (ours) | Binoculars | Perplexity | DetectLLM | Fast-DetectGPT |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Detect LLM Text** | **`0.99219`** | 0.76588 | 0.89307 | 0.92981 | 0.70085 |
| **AI vs Human** | **`0.95143`** | 0.86297 | 0.90743 | 0.90316 | 0.75608 |
| **HC3** | 0.99155 | 0.99441 | **`0.99471`** | 0.98436 | 0.95584 |
| **HC3 Plus** | **`0.98451`** | 0.88510 | 0.90999 | 0.87758 | 0.83575 |
| **ESL GPT4o Mini** | **`0.99983`** | 0.79637 | 0.82523 | 0.69051 | 0.60603 |
| **GB Essay ChatGPT** | 0.98628 | 0.88434 | **`0.99810`** | 0.99730 | 0.55624 |
| **GB News ChatGPT** | 0.90480 | 0.98773 | 0.98817 | **`0.99050`** | 0.91940 |
| **GB Creative ChatGPT** | **`0.99397`** | 0.91846 | 0.94990 | 0.91852 | 0.52336 |
| **GB Essay GPT4o** | 0.98136 | 0.85505 | **`0.99365`** | 0.99163 | 0.51477 |
| **GB Creative GPT4o** | **`0.99271`** | 0.91276 | 0.92303 | 0.87374 | 0.63813 |
| **GB News Claude** | 0.88038 | **`0.89263`** | 0.87211 | 0.86317 | 0.77787 |
| **GB Creative Claude** | **`0.96604`** | 0.82929 | 0.89304 | 0.87449 | 0.60276 |
| **GB Essay Claude** | 0.94223 | 0.77288 | 0.94310 | **`0.95988`** | 0.61633 |
| **GB Essay Deepseek V3** | 0.98484 | 0.99225 | **`0.99881`** | 0.99680 | 0.82763 |
| **GB Creative Deepseek V3** | 0.98199 | **`0.99569`** | 0.98852 | 0.96391 | 0.90439 |


<br>

## Installation

### Prerequisites
- Python 3.10 or higher
- Hugging Face account (for accessing certain models)

### Setup Instructions For Python Packages

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

### Setup Hugging Face Authentication

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



### Setup and Download Datasets and Experiment Results

Download the datasets and experiment results using the following commands. Please note that they are fairly large and will consume approximately 40 GB of storage.

```bash
# Download experiment results
git lfs clone https://huggingface.co/datasets/Aanimated/telescope_experiment_results experiment_results

# Download datasets
git lfs clone https://huggingface.co/datasets/Aanimated/telescope_datasets datasets

# Download original ghostbusters datasets (only needed if you want to generate new ghostbusters datasets)
git lfs clone https://github.com/ChrisNassif/telescope_original_ghostbusters_datasets ghostbusters_dataset_creation/original_ghostbusters_datasets
```

<br>

## Project Structure

```
telescope/                      # Repository root
├── llm_text_detectors/         # Core package folder (packaged as telescope_llm_text_detection)
│   ├── __init__.py             # Exports Telescope, utils
│   ├── llm_text_detectors.py   # Telescope detector class
│   └── utils.py                # Utility functions (model loading, auth, shared helpers)
├── scripts/                    # Analysis and experiment scripts
│   ├── generate_experiment_results.py  # New experiment result generation
│   ├── compute_roc_and_f1score_from_metrics.py  # evaluation and latex table generation script
│   ├── generate_*.py           # Various plotting/analysis scripts
│   └── ...
├── ablations/                  # Ablation studies
│   ├── per_token/              # Per-token analysis metrics
│   ├── sampling/               # LLM sampling utilities
│   ├── sequence_modeling/      # Sequence modeling dataset tools
│   ├── single_sample/          # Single sample analysis
│   ├── single_token_distribution/  # Token distribution analysis
│   └── training/               # Training utilities and logging
├── ghostbusters_dataset_creation/ # Dataset conversion and creation tools
├── datasets/                   # Dataset files
├── experiment_results/         # Pre-computed experiment results
├── config.yaml                 # Global configuration (model/dataset/metric names)
├── pyproject.toml              # Package configuration
└── telescope_env.yml           # Conda environment specification
```

<br>

## Key Concepts and Definitions

There are some definitions and concepts that will be nice to know before jumping in the code.


### Metrics
A **metric** is a numerical value computed from a reference model's outputs. Examples include:
- Telescope Perplexity
- Binoculars Score
- Perplexity
- DetectLLM Log-Rank Ratio

Additional experimental metrics are implemented in `llm_text_detectors/llm_text_detectors.py`. Effective metrics show correlation with whether text was LLM-generated.

### Experiment Results
**Experiment results** are CSV files containing data from running detection algorithms on specific datasets with specific reference models. Each result includes:
- Original text samples
- Ground truth labels (human: 0 vs. LLM-generated: 1)
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



<br>

## Usage

### Python API Usage

If you installed the package via PyPI, you can import and use the telescope detector in your own Python code:

```python
from telescope_llm_text_detection import Detectors

performer_model = "HuggingFaceTB/SmolLM-360M-Instruct"
observer_model = "HuggingFaceTB/SmolLM-360M"

detector = Detectors(performer_model, observer_model)

text = "Your text sample goes here."
score = detector.compute_telescope_perplexity(text)
print("Telescope Perplexity:", score)
```

The observer model is optional and if you run `detector = Detectors(performer_model)` then the package will not load a separate model and wherever an observer model is needed, it will just use the performer model instead.

### Running Experiments
In lieu of having command line arguments for every script, this codebase instead uses global variables at the top of each runnable script where you can set which arguments you want for things like which datasets or metrics to use. The reason for this is because specifying all of the metrics, datasets, models, etc takes up a lot of space and is annoying to keep track of in the runtime arguments of a script, so we just have all of in an easy place to see and edit. 

If you would like to **generate new experiment results** by running detection algorithms on datasets:

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



<br>

## Additional Metrics

Various additional metrics are implemented in `llm_text_detectors/llm_text_detectors.py` from our initial large-scale testing phase. While none proved as promising as Telescope Perplexity in our experiments, they remain available for further research and analysis.


<br>

## Future Work

Future work may focus on finding novel ways to combine different zero shot LLM text detection metrics by training various types of meta classifiers that are trained on a combination of the original text and various metrics such as telescope_perplexity, binoculars_score, fast_detectgpt, etc etc. 

Additionally, future work may further explore the impact of skipping the metric computation on the first `n` tokens when a metric averages across tokens. For example, telescope perplexity can be computed at a token level and then averaged across all of the tokens in a text; earlier tokens in a sequence may have a bit less information and thus may be less useful to detection performance, so skipping the computation on the first `n` tokens and averaging across the rest of the tokens may reduce variance slightly, especially for shorter texts.

Finally, as new models continue to release at a rapid pace, it is imperative that techniques continue to be benchmarked against the frontier models to ensure that detector performance is still high for popular language models. Likewise, the search for effective reference models also never ends, and it is also not entirely clear why exactly certain reference models perform well with certain techniques and how reference model quantization plays into this, so this is also another area to be explored.


<br>



## License

CC BY-NC-SA 4.0

<br>


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

<br>

## Contact

For questions or collaboration opportunities, please [open an issue](../../issues) or contact [chrisjnassif@gmail.com].