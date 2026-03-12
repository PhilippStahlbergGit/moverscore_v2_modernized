# MoverScore-Modern

**MoverScore-Modern** is a refactored and modernized implementation of the MoverScore metric. This repository addresses significant some of the outdated aspects in the original 2019 codebase, ensuring compatibility with modern Python environments, diverse hardware architectures, and current deep learning libraries.

This version was developed to support the **HyTE (Hybrid Triage Evaluation)** framework for the evaluation of meteorological narratives at the Western Norway University of Applied Sciences (HVL).

## Motivation
The original MoverScore implementation remains a highly effective semantic metric. However, the legacy codebase presents several barriers for modern researchers:
* Hardcoded CUDA requirements that prevent execution on CPU-only environments or Apple Silicon (MPS).
* Incompatibility with Python 3.10+ due to deprecated NumPy and Collections types.
* Outdated `transformers` API calls that trigger errors in recent versions.

## Technical Modernizations
The following changes have been implemented in this version:

### 1. Hardware Autodetect
Removed hardcoded `cuda:0` device selection. The script now utilizes `torch.device` to automatically detect the best available backend (CUDA, MPS, or CPU).

### 2. NumPy 2.0 and Python 3.10+ Compatibility
Fixed crashes caused by deprecated aliases. All instances of `np.float`, `np.int`, and `np.bool` have been replaced with standard dtypes (e.g., `np.float64`).

### 3. Transformers API Updates
* Updated `get_bert_embedding` to interface directly with `last_hidden_state`.
* Replaced deprecated `tokenizer.max_len` with `tokenizer.model_max_length` including safe fallback logic.
* Implemented a lazy-loading singleton pattern for the model and tokenizer to optimize memory usage and avoid import-time initialization failures.

### 4. Robustness and Logging
Added granular logging controls to suppress telemetry and non-critical warnings from the Hugging Face hub and HTTPX libraries.

## Usage

### Prerequisites
* Python 3.9+
* PyTorch 2.0+
* Transformers
* PyEMD

### Basic Example
```python
from moverscore_v2 import word_mover_score, get_idf_dict

# Example data
references = ["High pressure building over the North Sea."]
hypotheses = ["A high pressure system is developing in the North Sea."]

# Pre-calculate IDF dictionaries as required by the metric
idf_dict_ref = get_idf_dict(references)
idf_dict_hyp = get_idf_dict(hypotheses)

# Compute scores
scores = word_mover_score(references, hypotheses, idf_dict_ref, idf_dict_hyp)
print(f"MoverScore: {scores[0]}")
```

### Academic Attribution

This implementation is based on the research presented at EMNLP 2019. If you use this metric in your research, please cite the original authors below.
Original Paper:

```bibtex
@inproceedings{zhao2019moverscore,
  title = {MoverScore: Text Generation Evaluating with Contextualized Embeddings and Earth Mover Distance},
  month = {August},
  year = {2019},
  author = {Wei Zhao, Maxime Peyrard, Fei Liu, Yang Gao, Christian M. Meyer, Steffen Eger},
  address = {Hong Kong, China},
  publisher = {Association for Computational Linguistics},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing},
}

```

Original Repository: [AIPHES/emnlp19-moverscore](https://github.com/AIPHES/emnlp19-moverscore/tree/master?tab=readme-ov-file)

### Maintenance:
This refactored version is maintained by Philipp Stahlberg as part of a Master's Thesis project at the Western Norway University of Applied Sciences. It is provided "as-is" to assist the community in running MoverScore on modern stacks.
