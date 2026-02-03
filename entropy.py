"""
Entropy-Based Uncertainty

Measures uncertainty using Shannon entropy of the region score distribution.
Higher entropy means more spread out distribution → higher uncertainty.

Formula:
    H = -Σ p_i * log(p_i)
    normalized_H = H / log(n)


Range: [0, 1] after normalization

"""

import numpy as np
from typing import List


def compute_region_entropy(sorted_scores: List[float]) -> float:
    """
    Compute entropy-based uncertainty from region scores.

    Args:
        sorted_scores: Region scores sorted in descending order

    Returns:
        Normalized entropy in [0, 1]
            - 0 = minimum entropy (concentrated) → lowest uncertainty
            - 1 = maximum entropy (uniform) → highest uncertainty

    Examples:
        >>> # Single region (minimum entropy)
        >>> compute_region_entropy([1.0])
        0.5

        >>> # Two equal regions
        >>> compute_region_entropy([0.5, 0.5])
        1.0

        >>> # Three regions with decreasing scores
        >>> compute_region_entropy([0.6, 0.3, 0.1])
        0.918
    """
    if len(sorted_scores) <= 1:
        return 0.5 if len(sorted_scores) == 1 else 1.0

    scores_array = np.array(sorted_scores)
    total_score = np.sum(scores_array) + 1e-8
    probs = scores_array / total_score

    entropy = -np.sum(probs * np.log(probs + 1e-10))

    max_entropy = np.log(len(probs))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    return min(1.0, normalized_entropy)


def compute_entropy_raw(sorted_scores: List[float]) -> float:
    """
    Compute raw (unnormalized) entropy.

    Args:
        sorted_scores: Region scores

    Returns:
        Raw Shannon entropy (not normalized)
    """
    if len(sorted_scores) == 0:
        return 0.0

    scores_array = np.array(sorted_scores)
    total_score = np.sum(scores_array) + 1e-8
    probs = scores_array / total_score

    entropy = -np.sum(probs * np.log(probs + 1e-10))
    return entropy


def compute_entropy_with_details(sorted_scores: List[float]) -> dict:
    """
    Compute entropy with detailed intermediate values.

    Args:
        sorted_scores: Region scores sorted in descending order

    Returns:
        Dictionary with entropy calculation details
    """
    if len(sorted_scores) <= 1:
        return {
            'uncertainty': 0.5 if len(sorted_scores) == 1 else 1.0,
            'raw_entropy': 0.0,
            'normalized_entropy': 0.5 if len(sorted_scores) == 1 else 1.0,
            'max_entropy': 0.0 if len(sorted_scores) <= 1 else np.log(len(sorted_scores)),
            'n_regions': len(sorted_scores),
            'probs': []
        }

    scores_array = np.array(sorted_scores)
    total_score = np.sum(scores_array) + 1e-8
    probs = scores_array / total_score

    raw_entropy = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = np.log(len(probs))
    normalized_entropy = raw_entropy / max_entropy if max_entropy > 0 else 0.0

    return {
        'uncertainty': min(1.0, normalized_entropy),
        'raw_entropy': raw_entropy,
        'normalized_entropy': normalized_entropy,
        'max_entropy': max_entropy,
        'n_regions': len(sorted_scores),
        'probs': probs.tolist()
    }
