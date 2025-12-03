import argparse
import logging
import os
import json
from typing import List, Set

import pandas as pd

from data_loader import DataLoader
from descriptive_stats import count_study_data
from pb_analysis import analyze_pb_bc
from plot_trial_probs import plot_all_studies
from problem_choice_plots import plot_all_problem_bases

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified analysis for studies 12/13 lineage.")
    parser.add_argument("--csv", default="data.csv", help="Input CSV file (default: data.csv)")
    parser.add_argument(
        "--studies",
        default="",
        help="Comma-separated study ids or ranges (e.g., 1,3,5-7). Empty -> prompt picker.",
    )
    parser.add_argument(
        "--attention",
        choices=["all", "attention", "both"],
        default=None,
        help="Attention filter: all participants, attention==True, or both outputs. Empty -> prompt picker.",
    )
    parser.add_argument(
        "--choice-plots",
        choices=["per-study", "aggregate", "both"],
        default=None,
        help="Plot mode for ABC/BCD choice distributions. Empty -> prompt picker.",
    )
    parser.add_argument(
        "--trial-bin",
        default=None,
        help="Range for multi-trial aggregation, inclusive (e.g., 2-50). Empty -> prompt picker.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save tables, figures, and metadata. Empty -> prompt picker.",
    )
    parser.add_argument(
        "--save-figs",
        default=None,
        help="Comma-separated list of figure formats to save (e.g., png,pdf,svg). Empty -> prompt picker.",
    )
    parser.add_argument(
        "--save-tables",
        action="store_true",
        help="Save CSV tables for all analyses (otherwise prompted).",
    )
    parser.add_argument(
        "--save-latex",
        action="store_true",
        help="Save LaTeX tables for key results (otherwise prompted).",
    )
    return parser.parse_args()


def parse_study_input(input_str: str) -> List[int]:
    """Parses comma-separated numbers and ranges into a list of ints (study12 picker logic)."""
    selected: Set[int] = set()
    if not input_str.strip():
        return []
    for part in input_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = map(int, part.split("-"))
            if start > end:
                raise ValueError("Start of range cannot be greater than end.")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    return sorted(selected)


def get_user_attention_preference() -> List[str]:
    while True:
        choice = input("Analyze for attention? (Y/N/Both) [default: Both]: ").strip().upper()
        if not choice or choice == "BOTH":
            return ["All", "Attention"]
        if choice == "Y":
            return ["Attention"]
        if choice == "N":
            return ["All"]
        print("Invalid input. Please enter Y, N, or Both.")


def get_user_choice_plot_mode() -> str:
    while True:
        choice = input("Choice plots mode? (P=per-study/A=aggregate/B=both) [default: P]: ").strip().lower()
        if not choice or choice == "p":
            return "per-study"
        if choice == "a":
            return "aggregate"
        if choice == "b":
            return "both"
        if choice in ("per-study", "aggregate", "both"):
            return choice
        print("Invalid input. Enter P, A, or B.")


def get_user_output_dir(default: str = "outputs") -> str:
    val = input(f"Output directory [default: {default}]: ").strip()
    return val or default


def get_user_save_figs(default: str = "png,pdf") -> List[str]:
    val = input(f"Figure formats (F=png,pdf | P=png | D=pdf | custom list) [default: F]: ").strip().lower()
    if not val or val == "f":
        txt = default
    elif val == "p":
        txt = "png"
    elif val == "d":
        txt = "pdf"
    else:
        txt = val
    return [fmt.strip() for fmt in txt.split(",") if fmt.strip()]


def get_user_trial_bin(default: str = "2-100") -> tuple[int, int]:
    while True:
        val = input(f"Trial bin range (lo-hi) [default: {default}] (T=2-100): ").strip().lower()
        rng = val or default
        if rng == "t":
            rng = default
        if "-" in rng:
            try:
                lo, hi = map(int, rng.split("-"))
                if lo <= hi:
                    return (lo, hi)
            except ValueError:
                pass
        print("Invalid range. Use format lo-hi, e.g., 2-100.")


def get_user_yes_no(prompt: str, default: bool) -> bool:
    while True:
        val = input(f"{prompt} (Y/N) [default: {'Y' if default else 'N'}]: ").strip().lower()
        if not val:
            return default
        if val in ("y", "yes"):
            return True
        if val in ("n", "no"):
            return False
        print("Please enter Y or N.")


def get_user_study_preference(default_studies: List[int]) -> List[int]:
    default_display = ",".join(map(str, default_studies)) if default_studies else "None"
    prompt = f"Enter study numbers (e.g., 1,3,5-7) [default: {default_display}]: "
    while True:
        user_input = input(prompt).strip()
        if not user_input:
            return default_studies
        try:
            parsed = parse_study_input(user_input)
            if parsed:
                return parsed
            print("No valid study numbers entered. Please try again.")
        except ValueError as e:
            print(f"Invalid input: {e}. Please use comma-separated numbers or ranges.")


def parse_studies_arg(study_arg: str, available: List[int]) -> List[int]:
    if not study_arg:
        return []
    try:
        parsed = parse_study_input(study_arg)
        return parsed or []
    except ValueError as e:
        logger.warning("Ignoring invalid --studies argument (%s); will prompt. Error: %s", study_arg, e)
        return []


def main() -> None:
    args = parse_args()

    df_list = DataLoader.load_csv_files([args.csv])
    cleaned = [DataLoader.clean_data(df) for df in df_list]
    df_all = DataLoader.combine_studies(cleaned)

    # Limit problem pairs to problems present in the loaded data
    DataLoader.generate_problem_pairs(limit_to_df=df_all)

    available_studies = sorted(df_all["study"].unique())

    # Study picker: use CLI if provided, else prompt (study12 behavior)
    studies_to_analyze = parse_studies_arg(args.studies, available_studies)
    if not studies_to_analyze:
        studies_to_analyze = get_user_study_preference(available_studies)
    if not studies_to_analyze:
        logger.warning("No studies selected; exiting.")
        return
    logger.info("Analyzing studies: %s", studies_to_analyze)

    # Attention picker: use CLI if provided, else prompt (study12 behavior)
    if args.attention:
        attention_filters = {
            "all": ["All"],
            "attention": ["Attention"],
            "both": ["All", "Attention"],
        }[args.attention]
    else:
        attention_filters = get_user_attention_preference()

    # Choice plot mode picker (consistent prompt behavior)
    choice_plot_mode = args.choice_plots if args.choice_plots else get_user_choice_plot_mode()

    if args.trial_bin:
        lo, hi = map(int, args.trial_bin.split("-"))
        trial_bin = (lo, hi)
    else:
        trial_bin = get_user_trial_bin()

    output_dir = args.output_dir if args.output_dir else get_user_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    fig_formats = [fmt.strip() for fmt in args.save_figs.split(",") if fmt.strip()] if args.save_figs else get_user_save_figs()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1200)
    pd.options.display.float_format = "{:.4f}".format

    print("\nDescriptive statistics")
    desc_df = count_study_data(df_all, studies_to_analyze, include_attention="Attention" in attention_filters)
    print(desc_df)
    save_tables = args.save_tables if args.save_tables else get_user_yes_no("Save CSV tables?", default=False)
    save_latex = args.save_latex if args.save_latex else get_user_yes_no("Save LaTeX tables?", default=False)

    if save_tables:
        desc_df.to_csv(os.path.join(output_dir, "descriptive_stats.csv"), index=False)
    if save_latex:
        desc_df.to_latex(os.path.join(output_dir, "descriptive_stats.tex"), index=False)

    print("\nP(B|B+C) analysis (trial 1 and aggregated later trials)")
    pb_df = analyze_pb_bc(df_all, attention_filters, studies_to_analyze, trial_bins=trial_bin)
    print(pb_df)
    if save_tables:
        pb_df.to_csv(os.path.join(output_dir, "pb_results.csv"), index=False)
    if save_latex:
        pb_df.to_latex(os.path.join(output_dir, "pb_results.tex"), index=False)

    print("\nPlotting trial-by-trial P(B|B+C) for multi-trial studies...")
    plot_all_studies(df_all, output_dir=os.path.join(output_dir, "plots"), save_formats=fig_formats)

    print("\nPlotting ABC/BCD choice distributions (Trial 1) for each superset base...")
    plot_all_problem_bases(
        df_all,
        mode=choice_plot_mode,
        output_dir=os.path.join(output_dir, "plots"),
        save_formats=fig_formats,
        save_tables=save_tables,
    )

    # Run metadata
    run_meta = {
        "data_file": os.path.abspath(args.csv),
        "output_dir": os.path.abspath(output_dir),
        "studies": [int(s) for s in studies_to_analyze],
        "attention_filters": list(attention_filters),
        "choice_plot_mode": choice_plot_mode,
        "trial_bin": [int(trial_bin[0]), int(trial_bin[1])],
        "save_tables": bool(save_tables),
        "save_latex": bool(save_latex),
        "save_figs": list(fig_formats),
    }
    try:
        stat = os.stat(args.csv)
        run_meta["data_size_bytes"] = stat.st_size
        run_meta["data_mtime"] = stat.st_mtime
    except OSError:
        pass
    # Try to capture git commit if available
    try:
        import subprocess

        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__), text=True)
            .strip()
        )
        run_meta["git_commit"] = commit
    except Exception:
        run_meta["git_commit"] = "unknown"

    with open(os.path.join(output_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        import json

        json.dump(run_meta, f, indent=2)


if __name__ == "__main__":
    main()
