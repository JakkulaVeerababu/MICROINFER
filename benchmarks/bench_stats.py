"""
Shared statistical functions for the MicroInfer benchmark suite.
"""

import statistics

def _percentile(data: list, p: float) -> float:
    """Return p-th percentile (0-100) of a sorted list."""
    if not data:
        return 0.0
    data_sorted = sorted(data)
    idx = (p / 100.0) * (len(data_sorted) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(data_sorted) - 1)
    frac = idx - lo
    return data_sorted[lo] * (1 - frac) + data_sorted[hi] * frac

def compute_stats(data: list) -> dict:
    """
    Computes mean, standard deviation, p50, and p99 for a list of floats.
    Standard deviation is 0.0 if there is only 1 data point.
    """
    if not data:
        return {"mean": 0.0, "std": 0.0, "p50": 0.0, "p99": 0.0}
    
    mean = statistics.mean(data)
    std = statistics.stdev(data) if len(data) > 1 else 0.0
    
    return {
        "mean": round(mean, 2),
        "std": round(std, 2),
        "p50": round(_percentile(data, 50), 2),
        "p99": round(_percentile(data, 99), 2),
    }

def flag_outliers(data: list, metric_name: str = "metric") -> list:
    """
    Identifies statistical outliers using the Interquartile Range (IQR) method.
    Returns a list of dicts with the outlier's index and value.
    Logs warnings to stdout.
    """
    if not data or len(data) < 4:
        return []

    q1 = _percentile(data, 25)
    q3 = _percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = []
    for idx, val in enumerate(data):
        if val < lower_bound or val > upper_bound:
            outliers.append({"run_idx": idx + 1, "value": val})
            print(f"  [!] Outlier detected in {metric_name} (run {idx + 1}): {val:.2f} "
                  f"(bounds: {lower_bound:.2f} - {upper_bound:.2f})")
    return outliers
