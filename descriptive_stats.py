from typing import Iterable, List

import numpy as np
import pandas as pd


def _summarize(df: pd.DataFrame, label: str) -> dict:
    return {
        "filter": label,
        "num_rows": len(df),
        "num_participants": df["connectid"].nunique() if "connectid" in df.columns else np.nan,
        "percent_male": (df["gender"] == "male").mean() * 100 if "gender" in df.columns and not df.empty else np.nan,
        "average_age": df["age"].mean() if "age" in df.columns and not df.empty else np.nan,
        "sd_age": df["age"].std() if "age" in df.columns and len(df["age"].dropna()) > 1 else np.nan,
    }


def count_study_data(df_all: pd.DataFrame, studies_to_analyze: Iterable[int], include_attention: bool = True) -> pd.DataFrame:
    results: List[dict] = []
    for study_id in sorted(studies_to_analyze):
        df_study = df_all[df_all["study"] == study_id]
        if df_study.empty:
            continue
        results.append({"study": study_id, **_summarize(df_study, "All")})
        if include_attention and "attention" in df_study.columns:
            df_attention = df_study[df_study["attention"] == True]
            if not df_attention.empty:
                results.append({"study": study_id, **_summarize(df_attention, "Attention")})
    return pd.DataFrame(results).sort_values(["study", "filter"]).reset_index(drop=True)
