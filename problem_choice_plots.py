import logging
import os
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from data_loader import DataLoader
from utils import binomial_ci_wilson, filter_study_problem_trial

logger = logging.getLogger(__name__)


def _trial_bucket(df_all: pd.DataFrame, study_number: int) -> str:
    """Return '100-trial' if max trial >=100 else 'one-shot'."""
    df = df_all[df_all["study"] == study_number]
    max_trial = df["trial"].max() if not df.empty else 0
    return "100-trial" if pd.notna(max_trial) and max_trial >= 100 else "one-shot"

def _collect_choice_counts(df_all: pd.DataFrame, study_number: int, problem_base: str) -> pd.DataFrame:
    """Trial-1 choice counts (not rates) for ABC vs BCD within a problem base."""
    key = (study_number, problem_base)
    if key not in DataLoader.PROBLEM_PAIRS:
        return pd.DataFrame()
    problems = DataLoader.PROBLEM_PAIRS[key]
    if len(problems) != 2:
        return pd.DataFrame()

    abc_id = bcd_id = None
    for pid in problems:
        name = DataLoader.get_problem_name(study_number, pid)
        if " ABC" in name:
            abc_id = pid
        elif " BCD" in name:
            bcd_id = pid
    if abc_id is None or bcd_id is None:
        return pd.DataFrame()

    df_abc = filter_study_problem_trial(df_all, study_number, abc_id, trial=1)
    df_bcd = filter_study_problem_trial(df_all, study_number, bcd_id, trial=1)
    if df_abc.empty or df_bcd.empty:
        return pd.DataFrame()

    rows = []
    for opt in ["A", "B", "C"]:
        rows.append(
            {
                "study": study_number,
                "bucket": _trial_bucket(df_all, study_number),
                "problem": problem_base,
                "variant": "ABC",
                "option": opt,
                "count": df_abc["selectedoption"].value_counts().get(opt, 0),
                "total": len(df_abc),
            }
        )
    for opt in ["B", "C", "D"]:
        rows.append(
            {
                "study": study_number,
                "bucket": _trial_bucket(df_all, study_number),
                "problem": problem_base,
                "variant": "BCD",
                "option": opt,
                "count": df_bcd["selectedoption"].value_counts().get(opt, 0),
                "total": len(df_bcd),
            }
        )
    return pd.DataFrame(rows)


def _plot_bar(df_rates: pd.DataFrame, title: str, fname_base: str, save_formats: List[str]) -> None:
    """Draw bars + per-bar Wilson CIs with explicit positioning (no seaborn errorbar logic)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    order = ["A", "B", "C", "D"]
    hue_order = ["ABC", "BCD"]
    x = np.arange(len(order))
    width = 0.35
    palette = sns.color_palette(n_colors=len(hue_order))

    handles: List = []
    labels: List[str] = []

    for h_idx, variant in enumerate(hue_order):
        df_var = df_rates[df_rates["variant"] == variant].set_index("option")
        if df_var.empty:
            continue
        offset = (h_idx - (len(hue_order) - 1) / 2) * width
        color = palette[h_idx]

        for i, opt in enumerate(order):
            if opt not in df_var.index:
                continue
            row = df_var.loc[opt]
            # Ensure scalar values
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            
            rate = float(row["rate"])
            ci_low = float(row["ci_low"]) if pd.notna(row["ci_low"]) else np.nan
            ci_high = float(row["ci_high"]) if pd.notna(row["ci_high"]) else np.nan
            xpos = x[i] + offset

            bar = ax.bar(xpos, rate, width, color=color, edgecolor="black")
            if variant not in labels:
                handles.append(bar[0])
                labels.append(variant)

            ax.text(
                xpos,
                rate + 0.01,
                f"{rate:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

            if not (np.isnan(ci_low) or np.isnan(ci_high)):
                yerr = [[rate - ci_low], [ci_high - rate]]
                ax.errorbar(
                    xpos,
                    rate,
                    yerr=yerr,
                    fmt="none",
                    ecolor="black",
                    capsize=4,
                    linewidth=1,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_title(title)
    ax.set_ylabel("Rate (Trial 1)")
    ax.set_ylim(0, 0.8)
    ax.set_xlabel("Option")
    if handles:
        ax.legend(handles, labels, title="Variant", frameon=False)
    plt.tight_layout()
    os.makedirs(os.path.dirname(fname_base), exist_ok=True)
    for fmt in save_formats:
        fname = f"{fname_base}.{fmt}"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        logger.info("Saved plot %s", fname)
    plt.close()


def plot_problem_choice_distribution(df_all: pd.DataFrame, problem_base: str, studies: Iterable[int], output_dir: str, save_formats: List[str], save_tables: bool) -> None:
    """Per-study plots: each study+superset gets its own bar plot."""
    for study in studies:
        df_counts = _collect_choice_counts(df_all, study, problem_base)
        if df_counts.empty:
            continue
        df_rates = df_counts.assign(rate=lambda d: d["count"] / d["total"])
        df_rates[["ci_low", "ci_high"]] = df_rates.apply(
            lambda r: pd.Series(binomial_ci_wilson(r["rate"], r["total"])), axis=1
        )
        fname_base = os.path.join(output_dir, f"{problem_base}_study_{study}_choice_distribution")
        title = f"Study {study} · {problem_base} · Trial 1 choices"
        _plot_bar(df_rates, title, fname_base, save_formats)
        if save_tables:
            df_rates.to_csv(f"{fname_base}.csv", index=False)


def plot_problem_choice_distribution_aggregated(df_all: pd.DataFrame, problem_base: str, studies: Iterable[int], output_dir: str, save_formats: List[str], save_tables: bool) -> None:
    """Aggregated plots: bucket studies by trial-count category and plot two barsets if data exists."""
    parts: List[pd.DataFrame] = []
    for study in studies:
        df_counts = _collect_choice_counts(df_all, study, problem_base)
        if not df_counts.empty:
            parts.append(df_counts)
    if not parts:
        return
    df_all_counts = pd.concat(parts, ignore_index=True)
    for bucket in ["100-trial", "one-shot"]:
        df_bucket = df_all_counts[df_all_counts["bucket"] == bucket]
        if df_bucket.empty:
            continue
        agg = (
            df_bucket.groupby(["problem", "variant", "option"], as_index=False)[["count", "total"]]
            .sum()
            .assign(rate=lambda d: d["count"] / d["total"])
        )
        agg[["ci_low", "ci_high"]] = agg.apply(lambda r: pd.Series(binomial_ci_wilson(r["rate"], r["total"])), axis=1)
        fname_base = os.path.join(output_dir, f"{problem_base}_{bucket}_agg_choice_distribution")
        title = f"{problem_base} · Trial 1 choices · {bucket}"
        _plot_bar(agg, title, fname_base, save_formats)
        if save_tables:
            agg.to_csv(f"{fname_base}.csv", index=False)


def plot_all_problem_bases(
    df_all: pd.DataFrame, mode: str = "per-study", output_dir: str = "plots", save_formats: Optional[List[str]] = None, save_tables: bool = False
) -> None:
    """
    mode:
      - 'per-study': one plot per study per superset
      - 'aggregate': two plots per superset (100-trial bucket vs one-shot)
      - 'both': do both of the above
    """
    if save_formats is None:
        save_formats = ["png"]
    bases = set(base for (_, base) in DataLoader.PROBLEM_PAIRS.keys())
    studies = sorted(df_all["study"].unique())
    for base in sorted(bases):
        if mode in ("per-study", "both"):
            plot_problem_choice_distribution(df_all, base, studies, output_dir, save_formats, save_tables)
        if mode in ("aggregate", "both"):
            plot_problem_choice_distribution_aggregated(df_all, base, studies, output_dir, save_formats, save_tables)
