"""
FDR Control with Clopper-Pearson Confidence Intervals

Clopper-Pearson Method:
-----------------------
For observing w errors in m trials, the Clopper-Pearson upper bound
satisfies:

    P(Binomial(m, p) >= w) >= alpha/2

This is computed using the beta distribution:
    r_upper = Beta.ppf(1 - alpha, w, m - w + 1)

In scipy.stats: stats.beta.ppf(1 - alpha, w_cal + 1, m_cal - w_cal)


"""

import numpy as np
from scipy import stats
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CalibrationResult:
    """Result of threshold calibration."""
    threshold: float
    accepted_samples: int
    errors: int
    empirical_risk: float
    upper_bound: float
    abstention_rate: float


@dataclass
class EvaluationResult:
    """Result of evaluating on test set."""
    threshold: float
    abstention_rate: float
    power: float
    empirical_hit_rate: float
    empirical_error_rate: float
    empirical_risk: float
    upper_bound: float
    feasible: bool
    n_cal: int
    n_test: int


@dataclass
class CrossValidationResult:
    """Result of cross-validation across multiple splits."""
    n_cal: int
    n_test: int
    alpha: float
    n_splits: int
    summary: Dict[float, Dict]


def calibrate_threshold_binary_search(
    uncertainties: List[float],
    hits: List[bool],
    alpha: float = 0.05,
    target_error_rate: float = 0.3
) -> CalibrationResult:
    """
    Calibrate decision threshold using Clopper-Pearson FDR control.

    Uses binary search over candidate thresholds to find the largest
    threshold where the Clopper-Pearson upper bound is below the target
    error rate.

    Args:
        uncertainties: List of uncertainty values for calibration samples
        hits: List of hit/miss indicators (True=correct, False=error)
        alpha: Confidence level for Clopper-Pearson interval (default: 0.05)
        target_error_rate: Target false discovery rate (default: 0.3)

    Returns:
        CalibrationResult with optimal threshold and statistics

    Example:
        >>> uncertainties = [0.1, 0.2, 0.3, 0.5, 0.8]
        >>> hits = [True, True, False, True, False]
        >>> result = calibrate_threshold_binary_search(uncertainties, hits)
        >>> result.threshold
        0.3
    """
    losses = [0 if hit else 1 for hit in hits]
    unique_uncertainties = sorted(set(uncertainties))

    low = 0
    high = len(unique_uncertainties) - 1
    best_threshold = None
    best_result = None

    while low <= high:
        mid = (low + high) // 2
        t_candidate = unique_uncertainties[mid]

        m_cal = 0
        w_cal = 0
        for i, u in enumerate(uncertainties):
            if u <= t_candidate:
                m_cal += 1
                w_cal += losses[i]

        if m_cal > 0:
            if w_cal == m_cal:
                r_upper = 1.0
            elif w_cal == 0:
                r_upper = stats.beta.ppf(1 - alpha, 1, m_cal)
            else:
                r_upper = stats.beta.ppf(1 - alpha, w_cal + 1, m_cal - w_cal)

            if r_upper <= target_error_rate:
                best_threshold = t_candidate
                best_result = {
                    'threshold': float(t_candidate),
                    'accepted_samples': m_cal,
                    'errors': w_cal,
                    'empirical_risk': w_cal / m_cal if m_cal > 0 else 0.0,
                    'upper_bound': r_upper,
                    'abstention_rate': (len(uncertainties) - m_cal) / len(uncertainties)
                }
                low = mid + 1
            else:
                high = mid - 1
        else:
            high = mid - 1

    if best_result is None:
        best_threshold = -float("inf")
        best_result = {
            'threshold': float(best_threshold),
            'accepted_samples': 0,
            'errors': 0,
            'empirical_risk': 0.0,
            'upper_bound': 1.0,
            'abstention_rate': 1.0
        }

    return CalibrationResult(**best_result)


def evaluate_with_threshold(
    cal_result: CalibrationResult,
    test_uncertainties: List[float],
    test_hits: List[bool],
    target_error_rate: float = 0.3
) -> EvaluationResult:
    """
    Evaluate model performance using calibrated threshold.

    Applies the calibrated threshold to test data and computes
    power (recall among correct predictions) and empirical metrics.

    Args:
        cal_result: Calibrated threshold result
        test_uncertainties: Uncertainty values for test samples
        test_hits: Hit/miss indicators for test samples
        target_error_rate: Target error rate for feasibility check

    Returns:
        EvaluationResult with all metrics
    """
    tau = cal_result.threshold

    abstentions = 0
    total_correct = 0
    correct_kept = 0
    predictions = []

    for uncertainty, hit in zip(test_uncertainties, test_hits):
        if hit:
            total_correct += 1
            if uncertainty <= tau:
                correct_kept += 1

        if uncertainty > tau:
            abstentions += 1
        else:
            predictions.append(hit)

    n_total = len(test_uncertainties)
    hit_rate = np.mean(predictions) if predictions else 0.0
    error_rate = 1.0 - hit_rate

    power = correct_kept / total_correct if total_correct > 0 else 0.0

    return EvaluationResult(
        threshold=tau,
        abstention_rate=abstentions / n_total if n_total > 0 else 0,
        power=power,
        empirical_hit_rate=hit_rate,
        empirical_error_rate=error_rate,
        empirical_risk=cal_result.empirical_risk,
        upper_bound=cal_result.upper_bound,
        feasible=cal_result.upper_bound <= target_error_rate,
        n_cal=len(test_uncertainties),
        n_test=len(test_uncertainties)
    )


def run_single_split_evaluation(
    cal_uncertainties: List[float],
    cal_hits: List[bool],
    test_uncertainties: List[float],
    test_hits: List[bool],
    alpha: float = 0.05,
    target_error_rate: float = 0.3
) -> Tuple[CalibrationResult, EvaluationResult]:
    """
    Run calibration and evaluation on a single data split.

    Args:
        cal_uncertainties: Calibration set uncertainties
        cal_hits: Calibration set hits
        test_uncertainties: Test set uncertainties
        test_hits: Test set hits
        alpha: Confidence level
        target_error_rate: Target error rate

    Returns:
        Tuple of (calibration_result, evaluation_result)
    """
    cal_result = calibrate_threshold_binary_search(
        cal_uncertainties, cal_hits, alpha, target_error_rate
    )

    eval_result = evaluate_with_threshold(cal_result, test_uncertainties, test_hits, target_error_rate)

    return cal_result, eval_result


def run_cross_validation(
    all_uncertainties: List[float],
    all_hits: List[bool],
    n_splits: int = 100,
    test_ratio: float = 0.6,
    seed: int = 42,
    alpha: float = 0.05,
    target_error_rates: List[float] = None
) -> CrossValidationResult:
    """
    Run multiple random splits for robust evaluation.

    Args:
        all_uncertainties: All uncertainty values
        all_hits: All hit indicators
        n_splits: Number of random splits
        test_ratio: Ratio of data for testing
        seed: Random seed for reproducibility
        alpha: Confidence level for Clopper-Pearson
        target_error_rates: List of target error rates to evaluate

    Returns:
        CrossValidationResult with aggregated statistics

    Example:
        >>> results = run_cross_validation(
        ...     uncertainties, hits,
        ...     n_splits=100,
        ...     test_ratio=0.6,
        ...     target_error_rates=[0.3, 0.35, 0.4]
        ... )
        >>> for rate, metrics in results.summary.items():
        ...     print(f"Target: {rate}, Power: {metrics['power']['mean']:.3f}")
    """
    if target_error_rates is None:
        target_error_rates = [0.3, 0.35, 0.4, 0.45, 0.5]

    np.random.seed(seed)
    n_total = len(all_uncertainties)
    n_test = int(n_total * test_ratio)
    n_cal = n_total - n_test

    all_indices = np.arange(n_total)

    results_by_rate = {rate: {
        'thresholds': [],
        'abstention_rates': [],
        'power_rates': [],
        'empirical_hit_rates': [],
        'empirical_error_rates': [],
        'upper_bounds': [],
        'feasible_counts': []
    } for rate in target_error_rates}

    for split in range(n_splits):
        np.random.shuffle(all_indices)
        test_indices = all_indices[:n_test]
        cal_indices = all_indices[n_test:]

        test_unc = [all_uncertainties[i] for i in test_indices]
        test_hits_list = [all_hits[i] for i in test_indices]
        cal_unc = [all_uncertainties[i] for i in cal_indices]
        cal_hits_list = [all_hits[i] for i in cal_indices]

        for target_error_rate in target_error_rates:
            cal_result, eval_result = run_single_split_evaluation(
                cal_unc, cal_hits_list, test_unc, test_hits_list,
                alpha, target_error_rate
            )
            r = results_by_rate[target_error_rate]
            r['thresholds'].append(eval_result.threshold)
            r['abstention_rates'].append(eval_result.abstention_rate)
            r['power_rates'].append(eval_result.power)
            r['empirical_hit_rates'].append(eval_result.empirical_hit_rate)
            r['empirical_error_rates'].append(eval_result.empirical_error_rate)
            r['upper_bounds'].append(eval_result.upper_bound)
            r['feasible_counts'].append(1 if eval_result.feasible else 0)

    summary = {}
    for target_error_rate in target_error_rates:
        r = results_by_rate[target_error_rate]
        summary[target_error_rate] = {
            'threshold': {'mean': np.mean(r['thresholds']), 'std': np.std(r['thresholds'])},
            'abstention_rate': {'mean': np.mean(r['abstention_rates']), 'std': np.std(r['abstention_rates'])},
            'power': {'mean': np.mean(r['power_rates']), 'std': np.std(r['power_rates'])},
            'empirical_hit_rate': {'mean': np.mean(r['empirical_hit_rates']), 'std': np.std(r['empirical_hit_rates'])},
            'empirical_error_rate': {'mean': np.mean(r['empirical_error_rates']), 'std': np.std(r['empirical_error_rates'])},
            'upper_bound': {'mean': np.mean(r['upper_bounds']), 'std': np.std(r['upper_bounds'])},
            'feasible_rate': np.mean(r['feasible_counts'])
        }

    return CrossValidationResult(
        n_cal=n_cal,
        n_test=n_test,
        alpha=alpha,
        n_splits=n_splits,
        summary=summary
    )


def compute_clopper_pearson_upper_bound(
    w: int,
    m: int,
    alpha: float = 0.05
) -> float:
    """
    Compute Clopper-Pearson upper confidence bound for binomial proportion.

    The Clopper-Pearson interval is an exact interval that satisfies:
        P(Binomial(m, p) >= w) >= alpha/2

    This is implemented using the beta distribution:
        upper = Beta.ppf(1 - alpha, w, m - w + 1)

    Args:
        w: Number of successes (errors in our case)
        m: Number of trials (total accepted samples)
        alpha: Significance level

    Returns:
        Upper bound of the confidence interval

    Example:
        >>> # 2 errors out of 10 samples, 95% confidence
        >>> compute_clopper_pearson_upper_bound(2, 10, alpha=0.05)
        0.417
    """
    if m == 0:
        return 1.0
    if w == 0:
        return stats.beta.ppf(1 - alpha, 1, m)
    if w == m:
        return 1.0
    return stats.beta.ppf(1 - alpha, w + 1, m - w)


def compute_empirical_error_rate(
    uncertainties: List[float],
    hits: List[bool],
    threshold: float
) -> Tuple[float, int, int]:
    """
    Compute empirical error rate below a given threshold.

    Args:
        uncertainties: List of uncertainty values
        hits: List of hit/miss indicators
        threshold: Decision threshold

    Returns:
        Tuple of (empirical_error_rate, accepted_count, error_count)
    """
    accepted = 0
    errors = 0

    for u, hit in zip(uncertainties, hits):
        if u <= threshold:
            accepted += 1
            if not hit:
                errors += 1

    error_rate = errors / accepted if accepted > 0 else 0.0
    return error_rate, accepted, errors


def find_optimal_threshold(
    uncertainties: List[float],
    hits: List[bool],
    alpha: float = 0.05,
    target_error_rate: float = 0.3
) -> CalibrationResult:
    """
    Find optimal threshold using Clopper-Pearson control.

    This is an alias for calibrate_threshold_binary_search.

    Args:
        uncertainties: List of uncertainty values
        hits: List of hit/miss indicators
        alpha: Confidence level
        target_error_rate: Target error rate

    Returns:
        CalibrationResult with optimal threshold
    """
    return calibrate_threshold_binary_search(uncertainties, hits, alpha, target_error_rate)


def get_uncertainties_by_method(results: List[Dict], method: str) -> List[float]:
    """
    Extract uncertainty values for a specific method from results.

    Args:
        results: List of result dictionaries
        method: Uncertainty method name

    Returns:
        List of uncertainty values
    """
    uncertainties = []
    for r in results:
        unc = r.get('uncertainties', r.get('uncertainty', {}))
        if isinstance(unc, dict):
            uncertainties.append(unc.get(method, 0.0))
        else:
            uncertainties.append(unc if isinstance(unc, (int, float)) else 0.0)
    return uncertainties


def get_hits(results: List[Dict]) -> List[bool]:
    """
    Extract hit/miss indicators from results.

    Args:
        results: List of result dictionaries

    Returns:
        List of boolean hit indicators
    """
    return [r.get('correct', r.get('hit', False)) for r in results]


def evaluate_split(
    cal_results: List[Dict],
    test_results: List[Dict],
    method: str,
    alpha: float = 0.05,
    target_error_rate: float = 0.3
) -> Dict:
    """
    Evaluate a single split with given uncertainty method.

    This is a convenience function that combines data extraction,
    threshold calibration, and evaluation.

    Args:
        cal_results: Calibration set results
        test_results: Test set results
        method: Uncertainty method name
        alpha: Confidence level
        target_error_rate: Target error rate

    Returns:
        Dictionary with evaluation results
    """
    cal_uncertainties = get_uncertainties_by_method(cal_results, method)
    cal_hits = get_hits(cal_results)

    test_uncertainties = get_uncertainties_by_method(test_results, method)
    test_hits = get_hits(test_results)

    cal_result = calibrate_threshold_binary_search(cal_uncertainties, cal_hits, alpha, target_error_rate)

    tau = cal_result.threshold
    predictions = []
    abstentions = 0
    total_correct = 0
    correct_kept = 0

    for uncertainty, hit in zip(test_uncertainties, test_hits):
        if hit:
            total_correct += 1
            if uncertainty <= tau:
                correct_kept += 1

        if uncertainty > tau:
            abstentions += 1
        else:
            predictions.append(hit)

    n_total = len(test_uncertainties)
    hit_rate = np.mean(predictions) if predictions else 0.0
    error_rate = 1.0 - hit_rate

    feasible = cal_result.upper_bound <= target_error_rate

    power = correct_kept / total_correct if total_correct > 0 else 0.0

    return {
        'threshold': tau,
        'abstention_rate': abstentions / n_total if n_total > 0 else 0,
        'power': power,
        'empirical_hit_rate': hit_rate,
        'empirical_error_rate': error_rate,
        'empirical_risk': cal_result.empirical_risk,
        'upper_bound': cal_result.upper_bound,
        'n_cal': len(cal_results),
        'n_test': len(test_results),
        'feasible': feasible
    }
