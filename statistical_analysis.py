import numpy as np
from scipy.stats import f_oneway
from typing import List, Tuple

def perform_anova(group1: List[float], group2: List[float]) -> Tuple[float, float]:    
    """
    Perform ANOVA between two groups and return the p-value and F-statistic.

    Parameters:
    group1 (List[float]): First group of data.
    group2 (List[float]): Second group of data.

    Returns:
    Tuple[float, float]: A tuple containing the p-value and F-statistic.
    """
    if len(group1) == 0 or len(group2) == 0:
        raise ValueError("Data groups must not be empty.")
    result = f_oneway(group1, group2)
    return result.pvalue, result.statistic

def anova_multiple_segments(baseline_data: np.ndarray, meditation_segments: List[float]) -> List[Tuple[float, float]]:
    """
    Perform ANOVA for multiple single-value meditation segments compared to a baseline.

    Parameters:
    baseline_data (np.ndarray): Baseline data.
    meditation_segments (List[float]): List of single-value meditation segment data.

    Returns:
    List[Tuple[float, float]]: A list of tuples containing p-values and F-statistics for each comparison.
    """
    return [perform_anova(baseline_data, [segment]) for segment in meditation_segments]

def print_anova_results(baseline_data: List[float], meditation_data: List[float], meditation_segment_powers: List[List[float]]) -> None:
    """
    Calculate and print ANOVA results for baseline vs. entire meditation period and each meditation segment.

    Parameters:
    baseline_data (List[float]): Baseline data.
    meditation_data (List[float]): Meditation data.
    meditation_segment_powers (List[List[float]]): Powers of each meditation segment.
    """
    entire_period_p_value, entire_period_f_stat = perform_anova(baseline_data, meditation_data)
    print(f"ANOVA p-value between baseline and entire meditation period: {entire_period_p_value:.16f} (F-statistic: {entire_period_f_stat:.4f})")
    
    segment_results = anova_multiple_segments(baseline_data, meditation_segment_powers)
    for i, (p_value, f_stat) in enumerate(segment_results, 1):
        print(f"ANOVA p-value for baseline vs. meditation segment {i}: {p_value:.16f} (F-statistic: {f_stat:.4f})")