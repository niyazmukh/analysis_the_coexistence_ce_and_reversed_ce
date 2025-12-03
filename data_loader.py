import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd

from config import problem_names

logger = logging.getLogger(__name__)


class DataLoader:
    """Shared data-loading utilities with consistent typing and pairing logic."""

    PROBLEM_NAMES: Dict[Tuple[int, int], str] = problem_names
    PROBLEM_PAIRS: Dict[Tuple[int, str], List[int]] = {}

    DTYPE_MAPPING = {
        "study": "int",
        "age": "float",
        "gender": "string",
        "connectid": "string",
        "trialorder": "float",
        "problem": "int",
        "trial": "int",
        "trialtime": "float",
        "selectedoption": "string",
        "optionorder1": "string",
        "optionorder2": "string",
        "optionorder3": "string",
        "attention": "boolean",
    }

    REQUIRED_COLUMNS = ["study", "problem", "trial", "selectedoption", "connectid"]

    @classmethod
    def load_csv_files(cls, file_paths: List[str]) -> List[pd.DataFrame]:
        dataframes: List[pd.DataFrame] = []
        for path in file_paths:
            df = pd.read_csv(path)
            dataframes.append(df)
            logger.info("Loaded %s rows from %s", len(df), path)
        return dataframes

    @classmethod
    def clean_data(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        missing_required = [c for c in cls.REQUIRED_COLUMNS if c not in df.columns]
        if missing_required:
            raise ValueError(f"Missing required columns: {missing_required}")

        df = df.dropna(subset=cls.REQUIRED_COLUMNS)

        # Harmonize attention flag early so filters are reliable
        if "attention" in df.columns:
            df["attention"] = df["attention"].astype("boolean")
        else:
            df["attention"] = pd.Series([pd.NA] * len(df), dtype="boolean")

        # Cast known columns where present
        for col, dtype in cls.DTYPE_MAPPING.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)  # type: ignore

        # Standardize choice labels to uppercase single letters where applicable
        if "selectedoption" in df.columns:
            df["selectedoption"] = df["selectedoption"].str.strip().str.upper()

        logger.info("Cleaned dataframe -> %s rows, %s columns", len(df), len(df.columns))
        return df

    @classmethod
    def combine_studies(cls, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        combined_df = pd.concat(dataframes, ignore_index=True)
        logger.info("Combined %s dataframes into %s rows", len(dataframes), len(combined_df))
        return combined_df

    @classmethod
    def get_problem_name(cls, study: int, problem: int) -> str:
        return cls.PROBLEM_NAMES.get((study, problem), f"Study{study}_Problem{problem} ABC/BCD")

    @classmethod
    def generate_problem_pairs(cls, limit_to_df: Optional[pd.DataFrame] = None) -> None:
        """Pair ABC/BCD variants; optionally restrict to problems present in limit_to_df."""
        problem_pairs: Dict[Tuple[int, str], List[int]] = {}

        def maybe_add_pair(study: int, prob_id: int, name: str) -> None:
            if limit_to_df is not None:
                if not ((limit_to_df["study"] == study) & (limit_to_df["problem"] == prob_id)).any():
                    return
            base_name = name.replace(" ABC", "").replace(" BCD", "").strip()
            problem_pairs.setdefault((study, base_name), []).append(prob_id)

        for (study, prob_id), name in cls.PROBLEM_NAMES.items():
            maybe_add_pair(study, prob_id, name)

        cls.PROBLEM_PAIRS = problem_pairs
        logger.info("Generated %s problem pairs (pre-filtered)", len(problem_pairs))


# Build pairs at import so downstream modules can rely on them.
DataLoader.generate_problem_pairs()
