"""Load and clean the Give Me Some Credit (GMSC) dataset."""

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "cs-training.csv"

TARGET = "SeriousDlqin2yrs"

FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]


def load_gmsc(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and do minimal, documented cleaning.

    GMSC is known to have two data-entry artifacts (both well documented
    in public analyses of this dataset):
    - age == 0 for one row (impossible, dropped).
    - NumberOfTime* columns contain a small number of sentinel values of
      96/98 (a small number of rows) that are known data-entry codes, not
      genuine counts of 96-98 late payments. Left as-is here since LightGBM
      is a tree model robust to these outliers, and dropping them silently
      would be exactly the kind of "quietly cleaning inconvenient rows"
      that undermines the compliance story of this tool. We surface, not
      hide, this so a reviewer can decide independently.
    """
    df = pd.read_csv(path)
    df = df.drop(columns=["Id"], errors="ignore")
    df = df[df["age"] > 0].reset_index(drop=True)
    # MonthlyIncome and NumberOfDependents have missing values in GMSC.
    # Median-impute income (skewed), zero-impute dependents (missing means
    # not reported, most common true value is 0 dependents).
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(0)
    return df


def sentinel_flag_count(df: pd.DataFrame) -> int:
    """Count rows with the known 96/98 sentinel artifact, for transparency."""
    late_cols = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
        "NumberOfTime60-89DaysPastDueNotWorse",
    ]
    return int((df[late_cols] >= 96).any(axis=1).sum())
