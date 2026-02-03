"""
Heatmap Generation Module

Converts multiple sampled coordinates into a spatial probability distribution
over a patch grid. This is the foundation for spatial distribution analysis
in uncertainty quantification.

Key Steps:
1. Divide image into non-overlapping patches
2. Count samples falling into each patch
3. Normalize to obtain probability distribution
"""

import numpy as np
from typing import List, Tuple

DEFAULT_PATCH_SIZE = 28


def create_heatmap_from_samples(
    sampled_coords: List[Tuple[float, float]],
    resized_width: int,
    resized_height: int,
    patch_size: int = DEFAULT_PATCH_SIZE
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Convert sampled coordinates to a heatmap over patches.

    This function maps each sampled (x, y) coordinate to its corresponding
    patch in a grid, counting occurrences to build a spatial distribution.

    Args:
        sampled_coords: List of (x, y) coordinates in resized image space
        resized_width: Width of the resized image in pixels
        resized_height: Height of the resized image in pixels
        patch_size: Size of each patch in pixels (default: 28)

    Returns:
        heatmap: Raw count heatmap (n_height, n_width)
        heatmap_prob: Normalized probability distribution
        n_height: Number of patches along height
        n_width: Number of patches along width

    Example:
        >>> coords = [(100, 200), (105, 195), (98, 205)]
        >>> heatmap, prob, h, w = create_heatmap_from_samples(
        ...     coords, resized_width=560, resized_height=840
        ... )
        >>> heatmap.shape  # (30, 20) for patch_size=28
    """
    n_width = resized_width // patch_size
    n_height = resized_height // patch_size

    heatmap = np.zeros((n_height, n_width))

    for (x, y) in sampled_coords:
        if x is None or y is None:
            continue

        patch_x = int((x / resized_width) * n_width)
        patch_y = int((y / resized_height) * n_height)

        patch_x = min(max(patch_x, 0), n_width - 1)
        patch_y = min(max(patch_y, 0), n_height - 1)

        heatmap[patch_y, patch_x] += 1

    total = np.sum(heatmap)
    if total > 0:
        heatmap_prob = heatmap / total
    else:
        heatmap_prob = np.zeros_like(heatmap)

    return heatmap, heatmap_prob, n_height, n_width


def get_patch_coordinates(
    x: float,
    y: float,
    resized_width: int,
    resized_height: int,
    n_width: int,
    n_height: int
) -> Tuple[int, int]:
    """
    Convert absolute coordinates to patch indices.

    Args:
        x: X coordinate in pixel space
        y: Y coordinate in pixel space
        resized_width: Total width in pixels
        resized_height: Total height in pixels
        n_width: Number of patches horizontally
        n_height: Number of patches vertically

    Returns:
        (patch_x, patch_y) indices
    """
    patch_x = int((x / resized_width) * n_width)
    patch_y = int((y / resized_height) * n_height)

    patch_x = min(max(patch_x, 0), n_width - 1)
    patch_y = min(max(patch_y, 0), n_height - 1)

    return patch_x, patch_y


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """
    Normalize heatmap to probability distribution.

    Args:
        heatmap: Raw count heatmap

    Returns:
        Normalized probability distribution
    """
    total = np.sum(heatmap)
    if total > 0:
        return heatmap / total
    return np.zeros_like(heatmap)


def compute_spatial_statistics(heatmap: np.ndarray) -> dict:
    """
    Compute spatial statistics from heatmap.

    Args:
        heatmap: Raw or normalized heatmap

    Returns:
        Dictionary with spatial statistics
    """
    total = np.sum(heatmap)
    if total == 0:
        return {
            'mean_x': 0,
            'mean_y': 0,
            'std_x': 0,
            'std_y': 0,
            'spread': 0
        }

    prob = heatmap / total if np.sum(heatmap) > 0 else heatmap

    h, w = heatmap.shape
    y_coords, x_coords = np.meshgrid(
        np.arange(h) / h,
        np.arange(w) / w,
        indexing='ij'
    )

    mean_x = np.sum(x_coords * prob)
    mean_y = np.sum(y_coords * prob)
    std_x = np.sqrt(np.sum(((x_coords - mean_x) ** 2) * prob))
    std_y = np.sqrt(np.sum(((y_coords - mean_y) ** 2) * prob))

    spread = std_x + std_y

    return {
        'mean_x': mean_x,
        'mean_y': mean_y,
        'std_x': std_x,
        'std_y': std_y,
        'spread': spread
    }
