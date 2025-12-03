import logging
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from data_loader import DataLoader
from utils import (
    calculate_option_proportion,
    calculate_t_test_stats_independent,
    filter_study_problem_trial,
    format_small_number,
)

logger = logging.getLogger(__name__)


def _identify_pair(problems: List[int], study_number: int, problem_base: str) -> Optional[tuple[int, int]]:
    """Return (abc_id, bcd_id) with a robust fallback if labels are imperfect."""
    if len(problems) < 2:
        return None
    # Prefer explicit ABC/BCD suffixes
    abc_id = bcd_id = None
    for pid in sorted(problems):
        name = DataLoader.get_problem_name(study_number, pid)
        if " ABC" in name:
            abc_id = pid
        elif " BCD" in name:
            bcd_id = pid
    if abc_id is None or bcd_id is None:
        # Fallback: sorted order, warn once
        logger.warning("Falling back to sorted pairing for study %s base %s", study_number, problem_base)
        sorted_ids = sorted(problems)
        abc_id, bcd_id = sorted_ids[0], sorted_ids[1]
    return abc_id, bcd_id


def analyze_pb_bc(
    df_all: pd.DataFrame,
    attention_filters: Iterable[str],
    studies_to_analyze: Iterable[int],
    trial_bins: tuple[int, int] = (2, 100),
) -> pd.DataFrame:
    """Compute P(B|B+C) and independent-sample t-tests for trial 1 and pooled later trials."""
    rows = []
    for (study_number, problem_base), problems in DataLoader.PROBLEM_PAIRS.items():
        if study_number not in studies_to_analyze:
            continue
        pair = _identify_pair(problems, study_number, problem_base)
        if not pair:
            continue
        abc_id, bcd_id = pair

        for attention_filter in attention_filters:
            attention_flag = {"Attention": True, "All": None}.get(attention_filter, None)

            # Trial 1: raw proportions from all B/C choices
            df_abc_t1 = filter_study_problem_trial(df_all, study_number, abc_id, trial=1, attention_only=attention_flag)
            df_bcd_t1 = filter_study_problem_trial(df_all, study_number, bcd_id, trial=1, attention_only=attention_flag)

            p_b_abc = calculate_option_proportion(df_abc_t1, ["B", "C"], "B")
            p_b_bcd = calculate_option_proportion(df_bcd_t1, ["B", "C"], "B")
            effect_t1 = p_b_abc - p_b_bcd if pd.notna(p_b_abc) and pd.notna(p_b_bcd) else np.nan

            # t-test on B/C choosers only
            df_abc_test = df_abc_t1[df_abc_t1["selectedoption"].isin(["B", "C"])]
            df_bcd_test = df_bcd_t1[df_bcd_t1["selectedoption"].isin(["B", "C"])]
            n_abc = df_abc_test["connectid"].nunique()
            n_bcd = df_bcd_test["connectid"].nunique()

            t_stat, p_val, df_tt, cohen_d, *_ = calculate_t_test_stats_independent(
                df_abc_test["selectedoption"].apply(lambda x: 1 if x == "B" else 0),
                df_bcd_test["selectedoption"].apply(lambda x: 1 if x == "B" else 0),
            )

            rows.append(
                [
                    study_number,
                    problem_base,
                    "1",
                    attention_filter,
                    p_b_abc,
                    p_b_bcd,
                    effect_t1,
                    n_abc,
                    n_bcd,
                    df_tt,
                    t_stat,
                    format_small_number(p_val),
                    format_small_number(cohen_d),
                ]
            )

            # Later trials (multi-trial studies only)
            df_abc_multi = filter_study_problem_trial(
                df_all, study_number, abc_id, trial_range=trial_bins, attention_only=attention_flag
            )
            df_bcd_multi = filter_study_problem_trial(
                df_all, study_number, bcd_id, trial_range=trial_bins, attention_only=attention_flag
            )
            if df_abc_multi.empty and df_bcd_multi.empty:
                continue

            participant_probs_abc = df_abc_multi.groupby("connectid")[["selectedoption"]].apply(
                lambda x: calculate_option_proportion(x, ["B", "C"], "B")
            )
            participant_probs_bcd = df_bcd_multi.groupby("connectid")[["selectedoption"]].apply(
                lambda x: calculate_option_proportion(x, ["B", "C"], "B")
            )

            p_b_abc_multi = participant_probs_abc.mean() if not participant_probs_abc.empty else np.nan
            p_b_bcd_multi = participant_probs_bcd.mean() if not participant_probs_bcd.empty else np.nan
            effect_multi = (
                p_b_abc_multi - p_b_bcd_multi if pd.notna(p_b_abc_multi) and pd.notna(p_b_bcd_multi) else np.nan
            )

            valid_abc = participant_probs_abc.dropna()
            valid_bcd = participant_probs_bcd.dropna()
            t_stat_m, p_val_m, df_tt_m, cohen_d_m, *_ = calculate_t_test_stats_independent(valid_abc, valid_bcd)

            rows.append(
                [
                    study_number,
                    problem_base,
                    f"{trial_bins[0]}-{trial_bins[1]}",
                    attention_filter,
                    p_b_abc_multi,
                    p_b_bcd_multi,
                    effect_multi,
                    len(valid_abc),
                    len(valid_bcd),
                    df_tt_m,
                    t_stat_m,
                    format_small_number(p_val_m),
                    format_small_number(cohen_d_m),
                ]
            )

    df = pd.DataFrame(
        rows,
        columns=[
            "Study",
            "Pair",
            "Trial",
            "Filter",
            "ABC P(B|B+C)",
            "BCD P(B|B+C)",
            "Effect",
            "N_ABC",
            "N_BCD",
            "df",
            "t-stat",
            "p-value",
            "Cohen's d",
        ],
    )
    # Stable ordering for reproducibility
    return df.sort_values(["Study", "Pair", "Trial", "Filter"]).reset_index(drop=True)
