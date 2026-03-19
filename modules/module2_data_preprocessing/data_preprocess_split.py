"""
Module 2: Data Preprocessing & Splits
========================================
Pipeline Stage: Receives the cleaned CSV from Module 1 and prepares it for
                model training and evaluation.

Responsibilities:
  - Further text normalization: remove HTML entities, non-ASCII characters, and
    excess whitespace from headline and description fields.
  - Noise removal: strip special characters unrelated to content, handle encoding
    artefacts, and filter near-duplicate records.
  - Time-based train / validation / test splitting:
      • Training set   – 70 % of chronologically ordered data
      • Validation set – 15 % (middle slice)
      • Test set       – 15 % (most-recent slice)
  - Persist the three split DataFrames as separate CSV files.

Input:  data/processed/cleaned_data.csv  (output of Module 1)
Output:
  data/splits/train.csv
  data/splits/val.csv
  data/splits/test.csv
"""

import argparse
import logging
import os
import re
import unicodedata

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
# Constants
# ---------------------------------------------------------------------------
TEXT_COLS = ["headline", "description"]
DATE_COL = "date"
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# TEST_RATIO is the remainder (0.15)

HTML_ENTITY_RE = re.compile(r"&[a-z]+;|&#\d+;", re.IGNORECASE)
SPECIAL_CHAR_RE = re.compile(r"[^\w\s.,!?'\-]")  # keep basic punctuation


# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------

def remove_html_entities(text: str) -> str:
    """Replace HTML entities (&amp;, &#39;, …) with a space."""
    return HTML_ENTITY_RE.sub(" ", text)


def normalize_unicode(text: str) -> str:
    """Normalise Unicode to NFC and drop non-ASCII characters."""
    text = unicodedata.normalize("NFC", text)
    return text.encode("ascii", errors="ignore").decode("ascii")


def remove_special_chars(text: str) -> str:
    """Strip characters that carry no semantic content for NLP."""
    return SPECIAL_CHAR_RE.sub(" ", text)


def collapse_whitespace(text: str) -> str:
    """Collapse consecutive whitespace to a single space and strip ends."""
    return re.sub(r"\s+", " ", text).strip()


def preprocess_text(text: str) -> str:
    """Apply the full text-normalisation pipeline to a single string."""
    text = str(text)
    text = remove_html_entities(text)
    text = normalize_unicode(text)
    text = remove_special_chars(text)
    text = collapse_whitespace(text)
    return text


# ---------------------------------------------------------------------------
# Noise / duplicate removal
# ---------------------------------------------------------------------------

def remove_near_duplicates(df: pd.DataFrame, col: str = "headline") -> pd.DataFrame:
    """
    Drop near-duplicate rows based on normalised headline text.
    Two headlines are considered duplicates if their lowercased, whitespace-
    collapsed forms are identical.
    """
    before = len(df)
    if col in df.columns:
        key = df[col].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
        df = df[~key.duplicated(keep="first")].reset_index(drop=True)
    after = len(df)
    logger.info("Near-duplicate removal: dropped %d rows.", before - after)
    return df


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def time_based_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the DataFrame into train / val / test sets in chronological order.

    If a parseable date column is present the data is sorted by date before
    splitting; otherwise the existing row order is used.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (should be cleaned).
    train_ratio : float
        Fraction of data assigned to the training set.
    val_ratio : float
        Fraction of data assigned to the validation set.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df)
    """
    if DATE_COL in df.columns and pd.api.types.is_datetime64_any_dtype(df[DATE_COL]):
        df = df.sort_values(DATE_COL).reset_index(drop=True)
        logger.info("Data sorted chronologically by '%s'.", DATE_COL)
    else:
        logger.info("No datetime column found; using existing row order for split.")

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    logger.info(
        "Split sizes – train: %d, val: %d, test: %d",
        len(train_df), len(val_df), len(test_df),
    )
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------

def preprocess_and_split(
    input_path: str,
    output_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load cleaned data, apply further preprocessing, split, and save CSVs.

    Parameters
    ----------
    input_path : str
        Path to cleaned_data.csv (Module 1 output).
    output_dir : str
        Directory where train.csv, val.csv, and test.csv will be written.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df)
    """
    logger.info("Loading cleaned dataset from: %s", input_path)
    df = pd.read_csv(input_path, parse_dates=[DATE_COL] if DATE_COL else None)
    logger.info("Loaded %d rows.", len(df))

    # Further text normalisation
    for col in TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].apply(preprocess_text)
            logger.info("Preprocessed text column: '%s'.", col)

    df = remove_near_duplicates(df)

    train_df, val_df, test_df = time_based_split(df)

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info("Splits written to: %s", output_dir)
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Module 2 – Data Preprocessing & Splits"
    )
    parser.add_argument(
        "--input",
        default=os.path.join("data", "processed", "cleaned_data.csv"),
        help="Path to cleaned input CSV (default: data/processed/cleaned_data.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("data", "splits"),
        help="Output directory for split CSVs (default: data/splits/)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    preprocess_and_split(args.input, args.output_dir)
