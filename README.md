# Measuring Statutory Semantic Inertia

Official code release for:
**Measuring Statutory Semantic Inertia: A Noise-Calibrated Measurement Design for Separating Legal Semantic Stability from Ordinary Semantic Drift**
*Muhammad Adeel Sajjad & Muhammad Waleed Saleem (2026)*
Zenodo DOI: [10.5281/zenodo.23613023](https://doi.org/10.5281/zenodo.23613023)

## Overview
This repository provides a calibrated measurement design for separating genuine semantic drift from frequency bias, representation noise, and textual inheritance (copied/re-enacted law):
- **Paired domain design**: Evaluates shared vocabulary under an equal token budget.
- **Noise calibration ($E_D, E_J$)**: Uses a split-half baseline to correct frequency confounds.
- **Textual inheritance control**: Deduplicates shared/copied legal provisions via MinHash.
- **Validation**: Validated on synthetic corpora with ground-truth graded drift and on the SemEval-2020 Task 1 English benchmark.

## Installation
```bash
pip install -r requirements.txt
```

## Reproducing Results

### 1. Synthetic Experiments
```bash
# Experiment 1: Sensitivity, reliability, and decile calibration
python run_exp1.py

# Experiment 2: Paired domain testing across 100 simulated worlds
python run_exp2.py

# Generate paper figures
python make_figures.py
```

### 2. SemEval-2020 English Benchmark
```bash
python real_pipeline.py semeval --root data/semeval --runs 3 --budget 0 --out semeval_repeated.json
```

## Citation
```bibtex
@article{sajjad2026measuring,
  title={Measuring Statutory Semantic Inertia: A Noise-Calibrated Measurement Design for Separating Legal Semantic Stability from Ordinary Semantic Drift},
  author={Sajjad, Muhammad Adeel and Saleem, Muhammad Waleed},
  journal={Zenodo Preprint},
  year={2026},
  doi={10.5281/zenodo.23613023}
}
```
