# SafeGround: Know When to Trust GUI Grounding Models via Uncertainty Calibration

A general-purpose framework for uncertainty-aware spatial grounding predictions with statistically guaranteed false discovery rate (FDR) control.

[Paper](https://arxiv.org/pdf/2602.02419)

## Directory Structure

```
SAFEGROUND/
├── heatmap.py          # Heatmap generation from sampled coordinates
├── regions.py          # Connected region extraction using BFS
├── margin.py           # Margin-based uncertainty measure
├── entropy.py          # Entropy-based uncertainty measure
├── concentration.py    # Concentration (HHI) based uncertainty
├── combined.py         # Weighted combination of uncertainty methods
├── uncertainty.py      # Unified API for uncertainty computation
├── fdr_control.py      # Clopper-Pearson FDR control
├── safeground.png      # Overview figure
└── README.md           # This file
```

## Uncertainty Quantification

Pipeline: Coordinates → Heatmap → Regions → Uncertainty Score

### Heatmap Generation (`heatmap.py`)

Converts sampled coordinates into a spatial probability distribution.

### Region Extraction (`regions.py`)

Extracts connected regions from heatmap using BFS.

### Uncertainty Methods

| Method | Description | Formula |
|--------|-------------|---------|
| `margin` | Gap between top-2 regions | 1 - (μ₁ - μ₂)/(μ₁ + ε) |
| `entropy` | Distribution entropy | -Σp·log(p) / log(n) |
| `concentration` | HHI complement | 1 - Σp² |
| `combined` | Weighted combination | 0.2·margin + 0.2·entropy + 0.6·concentration |

## FDR Control (`fdr_control.py`)

### Clopper-Pearson Method

For observing w errors in m trials, the upper bound is:

```
r_upper = Beta.ppf(1 - α, w + 1, m - w)
```

### Usage

```python
from fdr_control import calibrate_threshold_binary_search

cal_result = calibrate_threshold_binary_search(
    uncertainties=uncertainties,
    hits=hits,
    alpha=0.05,
    target_error_rate=0.3
)
```

### Output Metrics

| Metric | Description |
|--------|-------------|
| `threshold` | Calibrated decision threshold |
| `power` | Fraction of correct predictions retained |
| `abstention_rate` | Fraction of predictions rejected |
| `upper_bound` | Clopper-Pearson confidence bound |

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
