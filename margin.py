"""
Margin-Based Uncertainty

Formula:
    margin = (μ₁ - μ₂) / (μ₁ + ε)
    uncertainty = 1 - margin = 1 - (μ₁ - μ₂) / (μ₁ + ε)

Where:
- μ₁ = highest region score
- μ₂ = second highest region score
- ε = small constant to avoid division by zero

Range: [0, 1]

"""

from typing import List


def compute_region_margin(sorted_scores: List[float]) -> float:
    """
    Compute margin-based uncertainty from region scores.

    Args:
        sorted_scores: Region scores sorted in descending order by score

    Returns:
        Uncertainty value in [0, 1]
            - 0 = highest confidence (large margin between top regions)
            - 1 = highest uncertainty (similar top region scores)

    Examples:
        >>> # High confidence case
        >>> compute_region_margin([0.9, 0.1, 0.0])
        0.122

        >>> # Low confidence case (similar top regions)
        >>> compute_region_margin([0.4, 0.35, 0.25])
        0.930

        >>> # Single region
        >>> compute_region_margin([0.8])
        0.2

        >>> # No regions
        >>> compute_region_margin([])
        1.0
    """
    if len(sorted_scores) < 2:
        if len(sorted_scores) == 0:
            return 1.0
        score = sorted_scores[0]
        return max(0.1, 1.0 - score)

    mu_1, mu_2 = sorted_scores[0], sorted_scores[1]
    if mu_1 == 0:
        return 1.0

    margin = (mu_1 - mu_2) / (mu_1 + 1e-8)
    uncertainty = 1 - margin

    return max(0.0, min(1.0, uncertainty))


def compute_margin_with_details(sorted_scores: List[float]) -> dict:
    """
    Compute margin with detailed intermediate values.

    Args:
        sorted_scores: Region scores sorted in descending order

    Returns:
        Dictionary with margin calculation details
    """
    if len(sorted_scores) < 2:
        if len(sorted_scores) == 0:
            return {
                'uncertainty': 1.0,
                'top_score': 0.0,
                'second_score': 0.0,
                'margin': 0.0,
                'n_regions': 0
            }
        return {
            'uncertainty': max(0.1, 1.0 - sorted_scores[0]),
            'top_score': sorted_scores[0],
            'second_score': 0.0,
            'margin': sorted_scores[0],
            'n_regions': 1
        }

    mu_1, mu_2 = sorted_scores[0], sorted_scores[1]
    if mu_1 == 0:
        margin = 0.0
    else:
        margin = (mu_1 - mu_2) / (mu_1 + 1e-8)

    uncertainty = max(0.0, min(1.0, 1 - margin))

    return {
        'uncertainty': uncertainty,
        'top_score': mu_1,
        'second_score': mu_2,
        'margin': margin,
        'n_regions': len(sorted_scores)
    }
