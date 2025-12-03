import logging
import os
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_loader import DataLoader
from utils import binomial_ci, calculate_option_proportion, filter_study_problem_trial

logger = logging.getLogger(__name__)


def analyze_trial_by_trial_probabilities(
    df_all: pd.DataFrame, study_number: int, output_dir: str, save_formats: List[str]
) -> None:
    """Plot P(B|B+C) for every trial (1..max) for each ABC/BCD problem in a study, with 95% CI ribbons."""
    df_study = df_all[df_all["study"] == study_number]
    if df_study.empty:
        logger.info("No data for study %s; skipping trial-by-trial plot", study_number)
        return

    for (s_num, problem_base), problems in DataLoader.PROBLEM_PAIRS.items():
        if s_num != study_number:
            continue
        series = []
        markers = ["o", "^", "s", "D", "v", "p", "X"]
        max_trial = int(df_study["trial"].max()) if "trial" in df_study.columns else 100
        trials = list(range(1, max_trial + 1))

        for i, problem_id in enumerate(sorted(problems)):
            df_problem = filter_study_problem_trial(df_study, study_number, problem_id)
            if df_problem.empty:
                continue
            probs: List[float] = []
            lowers: List[float] = []
            uppers: List[float] = []
            for t in trials:
                df_t = df_problem[df_problem["trial"] == t]
                if df_t.empty:
                    probs.append(np.nan)
                    lowers.append(np.nan)
                    uppers.append(np.nan)
                else:
                    p_b = calculate_option_proportion(df_t, ["B", "C"], "B")
                    bc_n = len(df_t[df_t["selectedoption"].isin(["B", "C"])])
                    lo, hi = binomial_ci(p_b, bc_n) if not np.isnan(p_b) else (np.nan, np.nan)
                    probs.append(p_b)
                    lowers.append(lo)
                    uppers.append(hi)
            series.append(
                {
                    "name": DataLoader.get_problem_name(study_number, problem_id),
                    "probs": probs,
                    "low": lowers,
                    "high": uppers,
                    "marker": markers[i % len(markers)],
                }
            )

        if not series:
            continue
        plot_trial_probabilities(series, problem_base, study_number, trials, output_dir, save_formats)


def plot_trial_probabilities(
    series: List[dict], problem_base: str, study_number: int, trials: List[int], output_dir: str, save_formats: List[str]
) -> None:
    plt.figure(figsize=(12, 6))
    for s in series:
        plt.plot(
            trials,
            s["probs"],
            marker=s["marker"],
            linestyle="-",
            label=s["name"],
            markersize=4,
            alpha=0.8,
        )
        plt.fill_between(trials, s["low"], s["high"], alpha=0.15)
    plt.xlabel("Trial")
    plt.ylabel("P(B|B+C)")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title(f"Study {study_number} · {problem_base} · P(B|B+C) by trial")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    for fmt in save_formats:
        fname = os.path.join(output_dir, f"study_{study_number}_{problem_base}_trial_probabilities.{fmt}")
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        logger.info("Saved plot %s", fname)
    plt.close()


def plot_all_studies(df_all: pd.DataFrame, output_dir: str, save_formats: List[str]) -> None:
    studies_with_trials = df_all[df_all["trial"] > 1]["study"].unique()
    for study_num in sorted(studies_with_trials):
        analyze_trial_by_trial_probabilities(df_all, study_num, output_dir, save_formats)
