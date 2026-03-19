"""
Module 1: Data Ingestion & Cleaning
====================================
Pipeline Stage: Entry point of the end-to-end multimodal content generation system.

Responsibilities:
  - Load the raw COVID-19 news CSV dataset into a Pandas DataFrame.
  - Normalize column names (lowercase, strip whitespace, replace spaces with underscores).
  - Handle missing values: drop rows where critical text fields are absent; fill optional
    fields with sensible defaults.
  - Convert date strings to datetime objects for downstream temporal operations.
  - Remove duplicate records based on headline text.
  - Perform minimal exploratory data analysis (EDA): shape, dtypes, null counts, value
    distributions for categorical fields.
  - Write the cleaned DataFrame to a new CSV file for use by Module 2.

Input:  data/raw/covid_news.csv  (or path supplied via CLI / function argument)
Output: data/processed/cleaned_data.csv
"""

import argparse
import logging
import os
import re

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column configuration
# ---------------------------------------------------------------------------
# Expected columns after normalization (lower-snake-case).
REQUIRED_TEXT_COLS = ["headline", "description"]
OPTIONAL_COLS = {
    "covid_status": "unknown",
    "sentiment": "neutral",
    "source": "",
    "image": "",
}
DATE_COL = "date"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with column names converted to lower_snake_case."""
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^\w]", "_", regex=True)
    )
    return df


def drop_missing_critical(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where any required text column is null or empty."""
    before = len(df)
    for col in REQUIRED_TEXT_COLS:
        if col in df.columns:
            df = df[df[col].notna() & (df[col].astype(str).str.strip() != "")]
    after = len(df)
    logger.info("Dropped %d rows with missing critical text fields.", before - after)
    return df.reset_index(drop=True)


def fill_optional_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values in optional columns with defaults."""
    for col, default in OPTIONAL_COLS.items():
        if col in df.columns:
            df[col] = df[col].fillna(default)
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the date column to datetime, coercing unparseable values to NaT."""
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], infer_datetime_format=True, errors="coerce")
        n_nat = df[DATE_COL].isna().sum()
        if n_nat:
            logger.warning("%d rows have unparseable dates (set to NaT).", n_nat)
    else:
        logger.warning("Column '%s' not found; skipping date parsing.", DATE_COL)
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows keyed on the headline column."""
    before = len(df)
    if "headline" in df.columns:
        df = df.drop_duplicates(subset=["headline"]).reset_index(drop=True)
    after = len(df)
    logger.info("Removed %d duplicate headline rows.", before - after)
    return df


def clean_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Strip HTML tags, normalize whitespace, and strip leading/trailing spaces."""
    html_tag_re = re.compile(r"<[^>]+>")
    for col in REQUIRED_TEXT_COLS:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .apply(lambda t: html_tag_re.sub(" ", t))
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )
    return df


# ---------------------------------------------------------------------------
# EDA helpers
# ---------------------------------------------------------------------------

def minimal_eda(df: pd.DataFrame) -> None:
    """Log key dataset statistics."""
    logger.info("--- Minimal EDA ---")
    logger.info("Shape: %s", df.shape)
    logger.info("Dtypes:\n%s", df.dtypes.to_string())
    logger.info("Null counts:\n%s", df.isnull().sum().to_string())
    for col in ["covid_status", "sentiment"]:
        if col in df.columns:
            logger.info("Value counts for '%s':\n%s", col, df[col].value_counts().to_string())
    if DATE_COL in df.columns and pd.api.types.is_datetime64_any_dtype(df[DATE_COL]):
        logger.info(
            "Date range: %s  →  %s",
            df[DATE_COL].min(),
            df[DATE_COL].max(),
        )


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------

def ingest_and_clean(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Load, clean, and save the dataset.

    Parameters
    ----------
    input_path : str
        Path to the raw CSV file.
    output_path : str
        Destination path for the cleaned CSV file.

    Returns
    -------
    pd.DataFrame
        The cleaned DataFrame.
    """
    logger.info("Loading dataset from: %s", input_path)
    df = pd.read_csv(input_path, encoding="utf-8", low_memory=False)
    logger.info("Loaded %d rows, %d columns.", *df.shape)

    df = normalize_column_names(df)
    df = parse_dates(df)
    df = clean_text_fields(df)
    df = drop_missing_critical(df)
    df = fill_optional_missing(df)
    df = remove_duplicates(df)

    minimal_eda(df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Cleaned dataset written to: %s  (%d rows)", output_path, len(df))
    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Module 1 – Data Ingestion & Cleaning"
    )
    parser.add_argument(
        "--input",
        default=os.path.join("data", "raw", "covid_news.csv"),
        help="Path to raw input CSV (default: data/raw/covid_news.csv)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("data", "processed", "cleaned_data.csv"),
        help="Path for cleaned output CSV (default: data/processed/cleaned_data.csv)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    ingest_and_clean(args.input, args.output)
