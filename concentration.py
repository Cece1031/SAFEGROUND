"""
Formula:
    HHI = Σ p_i²
    uncertainty = 1 - HHI

Where:
- p_i = normalized region score (sums to 1)
- HHI ranges from 1/n (uniform) to 1 (single region)

Range: [0, 1]
"""

import numpy as np
from typing import List


def compute_region_concentration(sorted_scores: List[float]) -> float:
    """
    Compute concentration-based uncertainty (HHI complement).

    Args:
        sorted_scores: Region scores sorted in descending order

    Returns:
        Concentration uncertainty in [0, 1]
            - 0 = highest concentration (single region) → lowest uncertainty
            - 1 = lowest concentration (uniform) → highest uncertainty

    Examples:
        >>> # Single region (maximum concentration)
        >>> compute_region_concentration([1.0])
        0.1

        >>> # Two equal regions
        >>> compute_region_concentration([0.5, 0.5])
        0.5

        >>> # Three equal regions
        >>> compute_region_concentration([0.33, 0.33, 0.34])
        0.667

        >>> # No regions
        >>> compute_region_concentration([])
        1.0
    """
    if len(sorted_scores) == 0:
        return 1.0

    if len(sorted_scores) == 1:
        return 0.1

    scores_array = np.array(sorted_scores)
    total_score = np.sum(scores_array) + 1e-8
    probs = scores_array / total_score

    hhi = np.sum(probs ** 2)
    uncertainty = 1.0 - hhi

    return max(0.0, min(1.0, uncertainty))


def compute_hhi(sorted_scores: List[float]) -> float:
    """
    Compute raw HHI (Herfindahl-Hirschman Index).

    Args:
        sorted_scores: Region scores

    Returns:
        Raw HHI value (not complemented)
    """
    if len(sorted_scores) == 0:
        return 0.0

    scores_array = np.array(sorted_scores)
    total_score = np.sum(scores_array) + 1e-8
    probs = scores_array / total_score

    return np.sum(probs ** 2)


def compute_concentration_with_details(sorted_scores: List[float]) -> dict:
    """
    Compute concentration with detailed intermediate values.

    Args:
        sorted_scores: Region scores sorted in descending order

    Returns:
        Dictionary with concentration calculation details
    """
    if len(sorted_scores) == 0:
        return {
            'uncertainty': 1.0,
            'hhi': 0.0,
            'n_regions': 0,
            'probs': []
        }

    if len(sorted_scores) == 1:
        return {
            'uncertainty': 0.1,
            'hhi': 1.0,
            'n_regions': 1,
            'probs': [1.0]
        }

    scores_array = np.array(sorted_scores)
    total_score = np.sum(scores_array) + 1e-8
    probs = scores_array / total_score

    hhi = np.sum(probs ** 2)
    uncertainty = max(0.0, min(1.0, 1.0 - hhi))

    return {
        'uncertainty': uncertainty,
        'hhi': hhi,
        'n_regions': len(sorted_scores),
        'probs': probs.tolist()
    }
