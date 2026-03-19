"""
Module 3: Metadata & Keyword Extraction
=========================================
Pipeline Stage: Enriches the training, validation, and test splits produced by
                Module 2 with structured metadata to support downstream LLM
                prompting and image generation.

Responsibilities:
  - Fit a TF-IDF vectorizer (max 5 000 features, unigrams + bigrams, English stop-
    words removed) **only on the training split** to prevent data leakage.
  - Extract the top-K keywords per article from the fitted TF-IDF matrix.
  - Apply Truncated SVD (100 components) to project the sparse TF-IDF matrix into
    a dense semantic space for downstream clustering (Module 4).
  - Attach extracted keywords and SVD-reduced vectors to each DataFrame row.
    SVD vectors are stored as comma-separated float strings for CSV compatibility
    with downstream modules; this trades some numeric precision and I/O speed
    for universal readability. For large datasets consider switching to Parquet.
  - Save keyword-enriched versions of train / val / test CSVs and the fitted
    TF-IDF + SVD artefacts (via joblib) for reproducibility.

Input:
  data/splits/train.csv
  data/splits/val.csv
  data/splits/test.csv
Output:
  data/metadata/train_keywords.csv
  data/metadata/val_keywords.csv
  data/metadata/test_keywords.csv
  data/metadata/tfidf_vectorizer.joblib
  data/metadata/svd_reducer.joblib
"""

import argparse
import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TEXT_COL = "headline"          # primary text source for TF-IDF
DESCRIPTION_COL = "description"
TOP_K_KEYWORDS = 5             # keywords extracted per article
TFIDF_MAX_FEATURES = 5_000
TFIDF_NGRAM_RANGE = (1, 2)
SVD_N_COMPONENTS = 100
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# TF-IDF training
# ---------------------------------------------------------------------------

def build_tfidf_vectorizer() -> TfidfVectorizer:
    """Instantiate and return a configured TF-IDF vectorizer."""
    return TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        stop_words="english",
        min_df=2,       # ignore very rare terms
        max_df=0.95,    # ignore very common terms
        sublinear_tf=True,
    )


def fit_tfidf(train_texts: pd.Series) -> TfidfVectorizer:
    """Fit TF-IDF vectorizer on training texts only."""
    vectorizer = build_tfidf_vectorizer()
    vectorizer.fit(train_texts.fillna("").astype(str))
    logger.info(
        "TF-IDF fitted on %d documents; vocabulary size: %d.",
        len(train_texts),
        len(vectorizer.vocabulary_),
    )
    return vectorizer


def fit_svd(tfidf_matrix) -> TruncatedSVD:
    """Fit Truncated SVD on the training TF-IDF matrix."""
    svd = TruncatedSVD(n_components=SVD_N_COMPONENTS, random_state=RANDOM_STATE)
    svd.fit(tfidf_matrix)
    explained = svd.explained_variance_ratio_.sum()
    logger.info(
        "SVD fitted with %d components (%.1f%% variance explained).",
        SVD_N_COMPONENTS,
        explained * 100,
    )
    return svd


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------

def extract_keywords(
    texts: pd.Series,
    vectorizer: TfidfVectorizer,
    top_k: int = TOP_K_KEYWORDS,
) -> list[str]:
    """
    Extract top-K TF-IDF keywords for each document.

    Parameters
    ----------
    texts : pd.Series
        Series of text strings to extract keywords from.
    vectorizer : TfidfVectorizer
        A *fitted* TF-IDF vectorizer.
    top_k : int
        Number of keywords to return per document.

    Returns
    -------
    list[str]
        Comma-separated keyword string per document.
    """
    feature_names = vectorizer.get_feature_names_out()
    tfidf_matrix = vectorizer.transform(texts.fillna("").astype(str))

    keywords_list: list[str] = []
    for row in tfidf_matrix:
        scores = row.toarray().flatten()
        top_indices = scores.argsort()[::-1][:top_k]
        top_words = [feature_names[i] for i in top_indices if scores[i] > 0]
        keywords_list.append(", ".join(top_words))

    return keywords_list


# ---------------------------------------------------------------------------
# SVD embedding
# ---------------------------------------------------------------------------

def embed_svd(texts: pd.Series, vectorizer: TfidfVectorizer, svd: TruncatedSVD) -> np.ndarray:
    """
    Transform texts into dense SVD-reduced semantic vectors.

    Returns
    -------
    np.ndarray
        Shape (n_docs, SVD_N_COMPONENTS).
    """
    tfidf_matrix = vectorizer.transform(texts.fillna("").astype(str))
    return svd.transform(tfidf_matrix)


# ---------------------------------------------------------------------------
# Per-split enrichment
# ---------------------------------------------------------------------------

def enrich_split(
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    svd: TruncatedSVD,
    split_name: str,
) -> pd.DataFrame:
    """Attach keyword and SVD columns to a single split DataFrame."""
    texts = df[TEXT_COL] if TEXT_COL in df.columns else pd.Series([""] * len(df))

    logger.info("Extracting keywords for %s split (%d rows)…", split_name, len(df))
    df = df.copy()
    df["keywords"] = extract_keywords(texts, vectorizer)

    logger.info("Computing SVD embeddings for %s split…", split_name)
    svd_vectors = embed_svd(texts, vectorizer, svd)
    # Store SVD vectors as a single JSON-like string column for CSV compatibility
    df["svd_vector"] = [",".join(f"{v:.6f}" for v in vec) for vec in svd_vectors]

    return df


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------

def extract_metadata_and_keywords(
    train_path: str,
    val_path: str,
    test_path: str,
    output_dir: str,
    artefact_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fit TF-IDF + SVD on training data, extract keywords for all splits, save outputs.

    Parameters
    ----------
    train_path, val_path, test_path : str
        Paths to the split CSVs (Module 2 output).
    output_dir : str
        Where to write keyword-enriched CSVs.
    artefact_dir : str
        Where to save the fitted vectorizer and SVD model (joblib).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_enriched, val_enriched, test_enriched)
    """
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    logger.info(
        "Loaded splits – train: %d, val: %d, test: %d",
        len(train_df), len(val_df), len(test_df),
    )

    # ---- Fit on TRAINING data only ----
    train_texts = train_df[TEXT_COL] if TEXT_COL in train_df.columns else pd.Series([""] * len(train_df))
    vectorizer = fit_tfidf(train_texts)
    train_tfidf = vectorizer.transform(train_texts.fillna("").astype(str))
    svd = fit_svd(train_tfidf)

    # ---- Enrich all splits ----
    train_enriched = enrich_split(train_df, vectorizer, svd, "train")
    val_enriched = enrich_split(val_df, vectorizer, svd, "val")
    test_enriched = enrich_split(test_df, vectorizer, svd, "test")

    # ---- Save enriched CSVs ----
    os.makedirs(output_dir, exist_ok=True)
    train_enriched.to_csv(os.path.join(output_dir, "train_keywords.csv"), index=False)
    val_enriched.to_csv(os.path.join(output_dir, "val_keywords.csv"), index=False)
    test_enriched.to_csv(os.path.join(output_dir, "test_keywords.csv"), index=False)
    logger.info("Keyword-enriched CSVs saved to: %s", output_dir)

    # ---- Save model artefacts ----
    os.makedirs(artefact_dir, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(artefact_dir, "tfidf_vectorizer.joblib"))
    joblib.dump(svd, os.path.join(artefact_dir, "svd_reducer.joblib"))
    logger.info("Model artefacts saved to: %s", artefact_dir)

    return train_enriched, val_enriched, test_enriched


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Module 3 – Metadata & Keyword Extraction"
    )
    parser.add_argument("--train", default=os.path.join("data", "splits", "train.csv"))
    parser.add_argument("--val", default=os.path.join("data", "splits", "val.csv"))
    parser.add_argument("--test", default=os.path.join("data", "splits", "test.csv"))
    parser.add_argument("--output-dir", default=os.path.join("data", "metadata"))
    parser.add_argument("--artefact-dir", default=os.path.join("data", "metadata"))
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    extract_metadata_and_keywords(
        train_path=args.train,
        val_path=args.val,
        test_path=args.test,
        output_dir=args.output_dir,
        artefact_dir=args.artefact_dir,
    )
