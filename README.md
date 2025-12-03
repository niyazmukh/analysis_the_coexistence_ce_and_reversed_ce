# Analysis of the Coexistence of Compromise Effect and Reversed Compromise Effect

This repository contains Python scripts for analyzing behavioral data from studies investigating the Compromise Effect (CE) and Reversed Compromise Effect.

## Project Structure

- `main.py`: The main entry point for running analyses. Handles argument parsing and orchestration of different analysis modules.
- `data_loader.py`: Utilities for loading and cleaning the dataset.
- `descriptive_stats.py`: Generates descriptive statistics of the study data.
- `pb_analysis.py`: Analyzes the probability of choosing option B given B or C are chosen (P(B|B+C)).
- `plot_trial_probs.py`: Generates trial-by-trial probability plots.
- `problem_choice_plots.py`: Generates bar plots for choice distributions (ABC vs BCD variants).
- `utils.py`: Helper functions for statistical calculations (t-tests, Cohen's d, confidence intervals).
- `config.py`: Configuration settings.

## Usage

To run the analysis, use `main.py`. You can specify arguments or use the interactive prompts.

```bash
python main.py
```

### Common Arguments

- `--csv`: Path to the input CSV file (default: `data.csv`).
- `--studies`: Comma-separated list of study IDs to analyze (e.g., `1,3,5-7`).
- `--output-dir`: Directory to save results (default: `outputs`).
- `--save-figs`: Formats to save figures in (e.g., `png,pdf`).

Example:
```bash
python main.py --studies 1,2 --output-dir results --save-figs png
```

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scipy

## Output

The scripts generate:
- CSV and LaTeX tables for descriptive statistics and analysis results.
- Plots showing trial-by-trial probabilities and choice distributions.
