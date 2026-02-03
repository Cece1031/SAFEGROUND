# SafeGround: Know When to Trust GUI Grounding Models via Uncertainty Calibration

A general-purpose framework for uncertainty-aware spatial grounding predictions with statistically guaranteed false discovery rate (FDR) control.

## Overview

![Overview](fig22.pdf)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SafeGround Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Generate Multiple Predictions                               │
│     Model + Stochastic Sampling → N coordinates                │
│                                                                 │
│  2. Compute Uncertainty                                         │
│     Coordinates → Heatmap → Regions → Uncertainty Score         │
│                                                                 │
│  3. Calibrate Threshold                                         │
│     Split Data → Binary Search → Clopper-Pearson Upper Bound    │
│                                                                 │
│  4. Risk-Aware Prediction                                       │
│     Accept if uncertainty ≤ threshold                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
codes/
├── heatmap.py          # Heatmap generation from sampled coordinates
├── regions.py          # Connected region extraction using BFS
├── margin.py           # Margin-based uncertainty measure
├── entropy.py          # Entropy-based uncertainty measure
├── concentration.py    # Concentration (HHI) based uncertainty
├── combined.py         # Weighted combination of uncertainty methods
├── uncertainty.py      # Unified API for uncertainty computation
├── fdr_control.py      # Clopper-Pearson FDR control
└── README.md           # This file
```

## Uncertainty Quantification

### Heatmap Generation (`heatmap.py`)

Converts sampled coordinates into a spatial probability distribution.

```python
from heatmap import create_heatmap_from_samples

heatmap, prob, h, w = create_heatmap_from_samples(
    sampled_coords=[(100, 200), (105, 195), ...],
    resized_width=560,
    resized_height=840,
    patch_size=28
)
```

### Region Extraction (`regions.py`)

Extracts connected regions from heatmap using BFS.

```python
from regions import extract_regions_from_heatmap

region_scores, region_centers = extract_regions_from_heatmap(
    heatmap_prob,
    activation_threshold=0.3
)
```

### Uncertainty Methods

| Method | File | Description | Formula |
|--------|------|-------------|---------|
| `margin` | `margin.py` | Gap between top-2 regions | 1 - (μ₁ - μ₂)/(μ₁ + ε) |
| `entropy` | `entropy.py` | Distribution entropy | -Σp·log(p) / log(n) |
| `concentration` | `concentration.py` | HHI complement | 1 - Σp² |
| `combined` | `combined.py` | Weighted combination | 0.2·margin + 0.2·entropy + 0.6·concentration |

### Unified API (`uncertainty.py`)

```python
from uncertainty import compute_all_uncertainties

uncertainties = compute_all_uncertainties(
    sampled_coords=[(100, 200), (105, 195), ...],
    resized_width=560,
    resized_height=840
)
# Returns: {'margin': 0.3, 'entropy': 0.5, 'concentration': 0.4, 'combined': 0.41}
```

## FDR Control (`fdr_control.py`)

### Clopper-Pearson Method

For observing w errors in m trials, the upper bound is:

```
r_upper = Beta.ppf(1 - α, w + 1, m - w)
```

### Usage

```python
from fdr_control import (
    calibrate_threshold_binary_search,
    run_cross_validation
)

# Calibrate threshold
cal_result = calibrate_threshold_binary_search(
    uncertainties=uncertainties,
    hits=hits,
    alpha=0.05,           # 95% confidence
    target_error_rate=0.3 # Control FDR at 30%
)

# Run cross-validation
results = run_cross_validation(
    all_uncertainties=uncertainties,
    all_hits=hits,
    n_splits=100,
    test_ratio=0.6,
    target_error_rates=[0.3, 0.35, 0.4, 0.45, 0.5]
)
```

### Output Metrics

| Metric | Description |
|--------|-------------|
| `threshold` | Calibrated decision threshold |
| `power` | Fraction of correct predictions retained |
| `abstention_rate` | Fraction of predictions rejected |
| `upper_bound` | Clopper-Pearson confidence bound |
| `empirical_error_rate` | Actual error rate on test set |

## Installation

```bash
pip install numpy scipy
```

## Dependencies

- numpy
- scipy

## Citation

```bibtex
@misc{wang2026safegroundknowtrustgui,
      title={SafeGround: Know When to Trust GUI Grounding Models via Uncertainty Calibration}, 
      author={Qingni Wang and Yue Fan and Xin Eric Wang},
      year={2026},
      eprint={2602.02419},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.02419}, 
}
```
