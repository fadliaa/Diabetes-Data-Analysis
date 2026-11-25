"""FDA Diabetes Adverse-Event Cleaning & Exploratory Data Analysis Pipeline.

Fixes & upgrades over the original notebook export:
- Modular, reusable data-cleaning and visualization functions
- Headless chart export (--output-dir) as well as optional interactive display (--show)
- Safe Sex normalization prior to plotting (prevents pie-chart category mismatch crashes)
- Warning-free Seaborn 0.13+ plotting calls
- Automatic fallback to bundled synthetic sample dataset when no external CSV is passed
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")  # Safe default for headless execution unless --show is requested

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


@dataclass
class CleaningReport:
    raw_rows: int
    duplicates_removed: int
    unreported_sex_removed: int
    age_outliers_removed: int
    final_rows: int
    age_lower_bound: float
    age_upper_bound: float


def normalize_sex(series: pd.Series) -> pd.Series:
    """Standardize raw FDA numeric/string sex codes into Male, Female, or Not Reported."""
    cleaned = series.astype(str).str.strip().str.lower()
    mapping = {
        "1": "Male",
        "1.0": "Male",
        "male": "Male",
        "m": "Male",
        "2": "Female",
        "2.0": "Female",
        "female": "Female",
        "f": "Female",
        "0": "Not Reported",
        "unknown": "Not Reported",
        "nan": "Not Reported",
        "": "Not Reported",
    }
    return cleaned.map(mapping).fillna("Not Reported")


def clean_dataset(df: pd.DataFrame, top_n_reactions: int = 10) -> tuple[pd.DataFrame, CleaningReport]:
    """Run full deduplication, demographic standardization, IQR age filtering, and date parsing."""
    raw_rows = len(df)
    cleaned = df.drop_duplicates().reset_index(drop=True)
    duplicates_removed = raw_rows - len(cleaned)

    cleaned["Patient_Age"] = pd.to_numeric(cleaned["Patient_Age"], errors="coerce")
    cleaned["Sex"] = normalize_sex(cleaned["Sex"])

    before_sex_filter = len(cleaned)
    cleaned = cleaned[cleaned["Sex"].isin(["Male", "Female"])].reset_index(drop=True)
    unreported_sex_removed = before_sex_filter - len(cleaned)

    cleaned = cleaned.dropna(subset=["Patient_Age"]).reset_index(drop=True)
    if cleaned.empty:
        return cleaned, CleaningReport(
            raw_rows=raw_rows,
            duplicates_removed=duplicates_removed,
            unreported_sex_removed=unreported_sex_removed,
            age_outliers_removed=0,
            final_rows=0,
            age_lower_bound=0.0,
            age_upper_bound=0.0,
        )

    q1 = float(cleaned["Patient_Age"].quantile(0.25))
    q3 = float(cleaned["Patient_Age"].quantile(0.75))
    iqr = q3 - q1
    lower_bound = max(0.0, q1 - 1.5 * iqr)
    upper_bound = q3 + 1.5 * iqr

    before_outlier = len(cleaned)
    cleaned = cleaned[
        (cleaned["Patient_Age"] >= lower_bound) & (cleaned["Patient_Age"] <= upper_bound)
    ].reset_index(drop=True)
    age_outliers_removed = before_outlier - len(cleaned)

    cleaned["Date"] = pd.to_datetime(cleaned["Date"].astype(str), format="%Y%m%d", errors="coerce")

    top_reactions = cleaned["Reaction"].value_counts().nlargest(top_n_reactions).index
    cleaned["Reaction"] = cleaned["Reaction"].where(cleaned["Reaction"].isin(top_reactions), "Other")

    report = CleaningReport(
        raw_rows=raw_rows,
        duplicates_removed=duplicates_removed,
        unreported_sex_removed=unreported_sex_removed,
        age_outliers_removed=age_outliers_removed,
        final_rows=len(cleaned),
        age_lower_bound=round(lower_bound, 2),
        age_upper_bound=round(upper_bound, 2),
    )
    return cleaned, report


def generate_visualizations(cleaned_df: pd.DataFrame, output_dir: Path, show: bool = False) -> Dict[str, Path]:
    """Generate and save core EDA plots into `output_dir`."""
    if cleaned_df.empty:
        raise ValueError("Cannot generate visualizations from an empty cleaned dataset.")
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")
    saved: Dict[str, Path] = {}

    def _finalize(fig_name: str) -> None:
        out_path = output_dir / fig_name
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()
        saved[fig_name] = out_path

    # 1. Top Adverse Reactions
    plt.figure(figsize=(11, 5.5))
    reaction_counts = cleaned_df["Reaction"].value_counts()
    sns.barplot(
        x=reaction_counts.values,
        y=reaction_counts.index,
        hue=reaction_counts.index,
        palette="viridis",
        legend=False,
    )
    plt.title("Top Adverse Reactions in Diabetes Drug Reports", fontsize=14, fontweight="bold")
    plt.xlabel("Number of Reports")
    plt.ylabel("Reaction Type")
    _finalize("01_top_adverse_reactions.png")

    # 2. Patient Age Distribution with Mean & Median
    plt.figure(figsize=(10, 5))
    ages = cleaned_df["Patient_Age"].dropna()
    plt.hist(ages, bins=25, color="#3b82f6", edgecolor="white", alpha=0.85)
    plt.axvline(ages.mean(), color="#dc2626", linestyle="--", linewidth=2, label=f"Mean: {ages.mean():.1f}")
    plt.axvline(ages.median(), color="#16a34a", linestyle="--", linewidth=2, label=f"Median: {ages.median():.1f}")
    plt.title("Patient Age Distribution (After IQR Outlier Filtering)", fontsize=14, fontweight="bold")
    plt.xlabel("Patient Age (Years)")
    plt.ylabel("Frequency")
    plt.legend()
    _finalize("02_patient_age_distribution.png")

    # 3. Patient Age by Sex Boxplot
    plt.figure(figsize=(8, 5))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PendingDeprecationWarning, module="seaborn")
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="seaborn")
        sns.boxplot(x="Sex", y="Patient_Age", hue="Sex", data=cleaned_df, palette="Set2", legend=False)
    plt.title("Patient Age Distribution by Sex", fontsize=14, fontweight="bold")
    plt.xlabel("Sex")
    plt.ylabel("Patient Age (Years)")
    _finalize("03_age_by_sex_boxplot.png")

    # 4. Adverse Event Reports Over Time (Monthly)
    valid_dates = cleaned_df.dropna(subset=["Date"])
    if not valid_dates.empty:
        plt.figure(figsize=(11, 5))
        events_over_time = valid_dates.groupby(valid_dates["Date"].dt.to_period("M")).size()
        events_over_time.index = events_over_time.index.to_timestamp()
        plt.plot(events_over_time.index, events_over_time.values, marker="o", linewidth=2, color="#0f3d6e")
        plt.title("Monthly Adverse Event Reports Over Time", fontsize=14, fontweight="bold")
        plt.xlabel("Month")
        plt.ylabel("Number of Reports")
        plt.xticks(rotation=45, ha="right")
        _finalize("04_reports_over_time.png")

    # 5. Top 12 Drugs x Reaction Association Heatmap
    plt.figure(figsize=(13, 7))
    top_drugs = cleaned_df["Drug"].value_counts().head(12).index
    filtered = cleaned_df[cleaned_df["Drug"].isin(top_drugs)]
    matrix = pd.crosstab(filtered["Drug"], filtered["Reaction"])
    sns.heatmap(matrix, cmap="Blues", annot=True, fmt="d", linewidths=0.5, cbar_kws={"label": "Reports"})
    plt.title("Drug–Reaction Association Heatmap (Top 12 Drugs)", fontsize=14, fontweight="bold")
    plt.xlabel("Reaction Type")
    plt.ylabel("Drug Name")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    _finalize("05_drug_reaction_heatmap.png")

    # 6. Top Side Effects by Sex (% Normalized)
    plt.figure(figsize=(8.5, 5.5))
    crosstab = pd.crosstab(cleaned_df["Reaction"], cleaned_df["Sex"], normalize="columns") * 100
    sns.heatmap(crosstab, annot=True, fmt=".1f", cmap="Reds", linewidths=0.5)
    plt.title("Adverse Reaction Share by Sex (%)", fontsize=14, fontweight="bold")
    plt.xlabel("Sex")
    plt.ylabel("Reaction Type")
    _finalize("06_reactions_by_sex_heatmap.png")

    return saved


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="FDA Diabetes Adverse-Event EDA Pipeline")
    parser.add_argument(
        "--data",
        type=Path,
        default=repo_root / "data" / "sample_fda_diabetes_data.csv",
        help="Path to FDA adverse-event CSV (defaults to bundled synthetic sample dataset).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "outputs",
        help="Directory where generated PNG charts will be saved.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively in addition to saving PNGs.",
    )
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{args.data}'. Pass --data <path> or run src/generate_sample_data.py."
        )

    raw_df = pd.read_csv(args.data)
    cleaned_df, report = clean_dataset(raw_df)
    saved = generate_visualizations(cleaned_df, args.output_dir, show=args.show)

    print("=== FDA Diabetes Adverse-Event EDA Summary ===")
    print(f"Input file             : {args.data}")
    print(f"Raw rows               : {report.raw_rows}")
    print(f"Duplicates removed     : {report.duplicates_removed}")
    print(f"Unreported sex removed : {report.unreported_sex_removed}")
    print(f"Age outliers removed   : {report.age_outliers_removed} (valid IQR range: {report.age_lower_bound}–{report.age_upper_bound})")
    print(f"Final clean rows       : {report.final_rows}")
    print(f"Charts generated ({len(saved)})  : {args.output_dir}")


if __name__ == "__main__":
    main()
