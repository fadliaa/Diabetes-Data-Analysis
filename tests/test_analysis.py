"""Automated tests for the FDA Diabetes Adverse-Event EDA pipeline."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analysis import clean_dataset, generate_visualizations, normalize_sex
from generate_sample_data import generate_dataset


class TestDiabetesEDA(unittest.TestCase):
    def test_sex_normalization(self) -> None:
        raw = pd.Series(["1", 2, "0", "unknown", "Unknown", "Female", None])
        result = normalize_sex(raw).tolist()
        self.assertEqual(
            result,
            ["Male", "Female", "Not Reported", "Not Reported", "Not Reported", "Female", "Not Reported"],
        )

    def test_end_to_end_cleaning_and_chart_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = generate_dataset(tmp_path / "sample.csv", n_rows=200, seed=7)
            raw_df = pd.read_csv(csv_path)
            cleaned_df, report = clean_dataset(raw_df)

            self.assertEqual(report.raw_rows, 225)
            self.assertGreaterEqual(report.duplicates_removed, 25)
            self.assertGreater(report.unreported_sex_removed, 0)
            self.assertGreater(report.age_outliers_removed, 0)
            self.assertTrue(set(cleaned_df["Sex"].unique()).issubset({"Male", "Female"}))
            self.assertTrue(cleaned_df["Patient_Age"].between(0, 120).all())

            charts = generate_visualizations(cleaned_df, tmp_path / "charts", show=False)
            self.assertEqual(len(charts), 6)
            for chart_path in charts.values():
                self.assertTrue(chart_path.exists() and chart_path.stat().st_size > 1000)


if __name__ == "__main__":
    unittest.main()
