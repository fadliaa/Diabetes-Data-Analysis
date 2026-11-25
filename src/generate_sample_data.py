"""Generate a deterministic synthetic FDA-schema adverse-event CSV for reproducible EDA.

Note: This script produces synthetic demonstration data matching the exact schema of the
original FDA diabetes adverse-event dataset (`Date`, `Drug`, `Reaction`, `Patient_Age`, `Sex`)
so that the cleaning and visualization pipeline can be executed and verified out of the box
without redistributing large third-party FDA exports.
"""

from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path

DRUGS = [
    "Metformin",
    "Insulin Glargine",
    "Semaglutide",
    "Empagliflozin",
    "Sitagliptin",
    "Dulaglutide",
    "Glimepiride",
    "Liraglutide",
    "Dapagliflozin",
    "Pioglitazone",
    "Insulin Lispro",
    "Canagliflozin",
    "Acarbose",
    "Linagliptin",
    "Glipizide",
    "Exenatide",
    "Tirzepatide",
]

REACTIONS = [
    "Hypoglycaemia",
    "Nausea",
    "Diarrhoea",
    "Fatigue",
    "Dizziness",
    "Headache",
    "Vomiting",
    "Blood Glucose Increased",
    "Abdominal Pain",
    "Decreased Appetite",
    "Injection Site Reaction",
    "Dehydration",
    "Ketoacidosis",
    "Weight Decreased",
]


def generate_dataset(output_path: Path, n_rows: int = 650, seed: int = 42) -> Path:
    rng = random.Random(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start_date = date(2023, 1, 1)

    rows = []
    for i in range(n_rows):
        report_date = start_date + timedelta(days=rng.randint(0, 720))
        drug = rng.choices(
            DRUGS,
            weights=[18, 15, 14, 12, 10, 9, 8, 7, 7, 6, 6, 5, 4, 4, 4, 3, 5],
            k=1,
        )[0]
        reaction = rng.choices(
            REACTIONS,
            weights=[20, 17, 14, 11, 9, 8, 8, 7, 6, 6, 4, 3, 3, 2],
            k=1,
        )[0]

        # Include realistic messy entries (age outliers & unstandardized sex codes)
        if i % 37 == 0:
            age: str | int = rng.choice([142, 185, -5, "unknown"])
        else:
            age = max(18, min(92, int(rng.gauss(58, 13))))

        if i % 19 == 0:
            sex = rng.choice(["0", "unknown", "Unknown"])
        else:
            sex = rng.choice(["1", "2"])

        rows.append(
            {
                "Date": report_date.strftime("%Y%m%d"),
                "Drug": drug,
                "Reaction": reaction,
                "Patient_Age": age,
                "Sex": sex,
            }
        )

    # Inject 25 intentional exact duplicates to validate deduplication logic
    rows.extend(rows[:25])

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["Date", "Drug", "Reaction", "Patient_Age", "Sex"]
        )
        writer.writeheader()
        writer.writerows(rows)

    return output_path


if __name__ == "__main__":
    default_out = Path(__file__).resolve().parent.parent / "data" / "sample_fda_diabetes_data.csv"
    out = generate_dataset(default_out)
    print(f"Generated synthetic FDA-schema sample dataset at: {out}")
