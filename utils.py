import math
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def format_small_number(number: float) -> str:
    if pd.isna(number):
        return "nan"
    if number == 0:
        return "0.0000"
    if 0 < abs(number) < 1e-4:
        return f"{number:.1e}"
    return f"{number:.4f}"


def calculate_option_proportion(
    df: pd.DataFrame, target_options: Iterable[str], target_value: str, selected_col: str = "selectedoption"
) -> float:
    if df.empty or selected_col not in df.columns:
        return np.nan
    filtered = df[df[selected_col].isin(target_options)]
    if filtered.empty:
        return np.nan
    counts = filtered[selected_col].value_counts()
    return counts.get(target_value, 0) / counts.sum()


def filter_study_problem_trial(
    df: pd.DataFrame,
    study: int,
    problem: int,
    trial: int | None = None,
    trial_range: Tuple[int, int] | None = None,
    attention_only: bool | None = None,
) -> pd.DataFrame:
    mask = (df["study"] == study) & (df["problem"] == problem)
    if attention_only is True and "attention" in df.columns:
        mask &= df["attention"] == True
    if attention_only is False and "attention" in df.columns:
        mask &= df["attention"].isna() | (df["attention"] != True)
    if trial is not None:
        mask &= df["trial"] == trial
    elif trial_range is not None:
        lo, hi = trial_range
        mask &= df["trial"].between(lo, hi)
    return df.loc[mask].copy()


def calculate_cohen_d_independent(m1: float, m2: float, s1: float, s2: float, n1: int, n2: int) -> float:
    if n1 < 2 or n2 < 2 or any(map(np.isnan, [m1, m2, s1, s2])):
        return np.nan
    pooled_var = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)
    if pooled_var <= 0:
        return np.nan if m1 != m2 else 0.0
    return (m1 - m2) / math.sqrt(pooled_var)


def calculate_cohen_d_paired(t_statistic: float, n_pairs: int) -> float:
    if n_pairs < 1 or np.isnan(t_statistic):
        return np.nan
    return 0.0 if t_statistic == 0 else t_statistic / math.sqrt(n_pairs)


def calculate_t_test_stats_independent(
    data1: Iterable[float], data2: Iterable[float], equal_var: bool = True
) -> tuple[float, float, float, float, float, float, float, float]:
    arr1 = np.asarray(list(data1), dtype=float)
    arr2 = np.asarray(list(data2), dtype=float)
    n1, n2 = len(arr1), len(arr2)
    if n1 < 1 or n2 < 1:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    mean1, mean2 = float(arr1.mean()), float(arr2.mean())
    std1 = float(arr1.std(ddof=1)) if n1 > 1 else np.nan
    std2 = float(arr2.std(ddof=1)) if n2 > 1 else np.nan
    t_result = stats.ttest_ind(arr1, arr2, equal_var=equal_var)
    # Explicitly cast to float to satisfy type checker, assuming ttest_ind returns numeric values
    t_stat = float(t_result[0]) # type: ignore
    p_value = float(t_result[1]) # type: ignore
    df = float(n1 + n2 - 2) if n1 + n2 > 2 else np.nan
    cohen_d = calculate_cohen_d_independent(mean1, mean2, std1, std2, n1, n2)
    return t_stat, p_value, df, cohen_d, mean1, mean2, std1, std2


def calculate_t_test_stats_paired(data1: Iterable[float], data2: Iterable[float]) -> tuple[float, float, float, float]:
    arr1 = np.asarray(list(data1), dtype=float)
    arr2 = np.asarray(list(data2), dtype=float)
    if len(arr1) != len(arr2) or len(arr1) < 2:
        return np.nan, np.nan, np.nan, np.nan
    diff = arr1 - arr2
    if np.std(diff, ddof=1) < 1e-9:
        return 0.0, 1.0, len(arr1) - 1, 0.0
    t_stat, p_value = stats.ttest_rel(arr1, arr2)
    df = len(arr1) - 1
    return t_stat, p_value, df, calculate_cohen_d_paired(t_stat, len(arr1))


def binomial_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Approximate 95% CI for a proportion (Wald, clipped to [0,1])."""
    if n <= 0 or np.isnan(p):
        return np.nan, np.nan
    se = math.sqrt(p * (1 - p) / n)
    lo, hi = p - z * se, p + z * se
    return max(0.0, lo), min(1.0, hi)


def binomial_ci_wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval (better behaved than Wald), clipped to [0,1]."""
    if n <= 0 or np.isnan(p):
        return np.nan, np.nan
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    margin = z * math.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2)))
    lo = (center - margin) / denom
    hi = (center + margin) / denom
    return max(0.0, lo), min(1.0, hi)
