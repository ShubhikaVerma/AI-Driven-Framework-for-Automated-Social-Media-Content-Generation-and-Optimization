"""
Module 4: Topic Clustering & Prompt Dataset Building
=====================================================
Pipeline Stage: Uses the SVD-reduced TF-IDF vectors from Module 3 to group
                articles into semantic clusters, assign human-readable topic
                labels, and assemble the prompt-target CSV files consumed by
                Module 5 for LLM generation.

Responsibilities:
  - Load keyword-enriched splits and the fitted TF-IDF / SVD artefacts from
    Module 3.
  - Run KMeans clustering (k=8, random_state=42) on the SVD vectors to assign
    each article a cluster ID.
  - Derive a concise topic label for each cluster from the top TF-IDF centroid
    terms.
  - Attach cluster_id and topic_label columns to every split.
  - Build a prompt-target CSV for each split:
      • prompt  – a structured text block combining headline, keywords, topic,
                  sentiment, and covid_status.
      • target  – the original description used as the reference/ground-truth
                  summary.
  - Save all outputs to disk.

Input:
  data/metadata/train_keywords.csv
  data/metadata/val_keywords.csv
  data/metadata/test_keywords.csv
  data/metadata/tfidf_vectorizer.joblib
  data/metadata/svd_reducer.joblib
Output:
  data/clustered/train_clustered.csv
  data/clustered/val_clustered.csv
  data/clustered/test_clustered.csv
  data/prompt_dataset/train_prompts.csv
  data/prompt_dataset/val_prompts.csv
  data/prompt_dataset/test_prompts.csv
  data/clustered/kmeans_model.joblib
  data/clustered/cluster_labels.json
"""

import argparse
import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

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
N_CLUSTERS = 8
RANDOM_STATE = 42
TOP_TERMS_PER_CLUSTER = 5   # used to generate topic labels
SVD_COL = "svd_vector"      # column containing comma-separated SVD values


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def load_svd_vectors(df: pd.DataFrame) -> np.ndarray:
    """Parse the SVD vector column (comma-separated floats) into a 2-D array."""
    if SVD_COL not in df.columns:
        raise ValueError(
            f"Column '{SVD_COL}' not found. Run Module 3 first."
        )
    vectors = np.array(
        [list(map(float, row.split(","))) for row in df[SVD_COL]],
        dtype=np.float32,
    )
    return vectors


def fit_kmeans(train_vectors: np.ndarray) -> KMeans:
    """Fit KMeans on the training SVD vectors."""
    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    km.fit(train_vectors)
    logger.info(
        "KMeans fitted – inertia: %.2f, clusters: %d",
        km.inertia_,
        N_CLUSTERS,
    )
    return km


# ---------------------------------------------------------------------------
# Topic label generation
# ---------------------------------------------------------------------------

def derive_cluster_labels_with_svd(
    km: KMeans,
    svd,
    vectorizer,
) -> dict[int, str]:
    """
    Generate topic labels using SVD inverse_transform for better accuracy.

    Parameters
    ----------
    km : KMeans
        Fitted KMeans model (cluster centers in SVD space).
    svd : TruncatedSVD
        Fitted SVD reducer (provides inverse_transform).
    vectorizer : TfidfVectorizer
        Fitted TF-IDF vectorizer (provides feature names).

    Returns
    -------
    dict[int, str]
        Mapping from cluster ID to topic label string.
    """
    feature_names = vectorizer.get_feature_names_out()
    # Project centroids back to approximate TF-IDF space
    centroid_tfidf = svd.inverse_transform(km.cluster_centers_)

    labels: dict[int, str] = {}
    for cid, centroid in enumerate(centroid_tfidf):
        top_idx = np.argsort(centroid)[::-1][:TOP_TERMS_PER_CLUSTER]
        top_terms = [feature_names[i] for i in top_idx]
        labels[cid] = " | ".join(top_terms)
        logger.debug("Cluster %d label: %s", cid, labels[cid])

    logger.info("Generated labels for %d clusters (via SVD inverse).", len(labels))
    return labels


# ---------------------------------------------------------------------------
# Prompt-target dataset construction
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = (
    "### Context\n"
    "Headline: {headline}\n"
    "Keywords: {keywords}\n"
    "Topic: {topic_label}\n"
    "Sentiment: {sentiment}\n"
    "COVID Status: {covid_status}\n\n"
    "### Task\n"
    "Write a concise, factual social-media post (≤ 280 characters) that "
    "accurately summarises the news article above. Focus on the key facts, "
    "preserve named entities, and maintain a {sentiment} tone.\n\n"
    "### Summary"
)


def build_prompt(row: pd.Series) -> str:
    """Format the structured prompt template for a single DataFrame row."""
    return PROMPT_TEMPLATE.format(
        headline=row.get("headline", ""),
        keywords=row.get("keywords", ""),
        topic_label=row.get("topic_label", ""),
        sentiment=row.get("sentiment", "neutral"),
        covid_status=row.get("covid_status", "unknown"),
    )


def build_prompt_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a prompt-target DataFrame from a clustered split.

    Returns
    -------
    pd.DataFrame
        Columns: [id, prompt, target, headline, keywords, topic_label,
                  sentiment, covid_status, cluster_id]
    """
    prompt_df = pd.DataFrame()
    if "id" in df.columns:
        prompt_df["id"] = df["id"]
    else:
        prompt_df["id"] = df.index

    prompt_df["prompt"] = df.apply(build_prompt, axis=1)
    prompt_df["target"] = df["description"] if "description" in df.columns else pd.Series([""] * len(df))

    # Carry forward useful metadata columns
    for col in ["headline", "keywords", "topic_label", "sentiment", "covid_status", "cluster_id"]:
        if col in df.columns:
            prompt_df[col] = df[col]

    return prompt_df


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------

def cluster_and_build_prompts(
    train_keyword_path: str,
    val_keyword_path: str,
    test_keyword_path: str,
    vectorizer_path: str,
    svd_path: str,
    clustered_dir: str,
    prompt_dir: str,
) -> None:
    """
    Full Module 4 pipeline: cluster articles and build prompt-target CSVs.

    Parameters
    ----------
    train_keyword_path, val_keyword_path, test_keyword_path : str
        Paths to keyword-enriched split CSVs (Module 3 output).
    vectorizer_path : str
        Path to the saved TF-IDF vectorizer joblib.
    svd_path : str
        Path to the saved SVD reducer joblib.
    clustered_dir : str
        Output directory for clustered CSVs and KMeans artefact.
    prompt_dir : str
        Output directory for prompt-target CSVs.
    """
    # ---- Load artefacts ----
    vectorizer = joblib.load(vectorizer_path)
    svd = joblib.load(svd_path)
    logger.info("Loaded TF-IDF vectorizer and SVD reducer.")

    # ---- Load splits ----
    train_df = pd.read_csv(train_keyword_path)
    val_df = pd.read_csv(val_keyword_path)
    test_df = pd.read_csv(test_keyword_path)
    logger.info(
        "Loaded – train: %d, val: %d, test: %d rows.",
        len(train_df), len(val_df), len(test_df),
    )

    # ---- Parse SVD vectors ----
    train_vecs = load_svd_vectors(train_df)
    val_vecs = load_svd_vectors(val_df)
    test_vecs = load_svd_vectors(test_df)

    # ---- Fit KMeans on training vectors ----
    km = fit_kmeans(train_vecs)

    # ---- Derive cluster labels ----
    cluster_labels = derive_cluster_labels_with_svd(km, svd, vectorizer)

    # ---- Assign clusters ----
    for df, vecs, name in [
        (train_df, train_vecs, "train"),
        (val_df, val_vecs, "val"),
        (test_df, test_vecs, "test"),
    ]:
        df["cluster_id"] = km.predict(vecs)
        df["topic_label"] = df["cluster_id"].map(cluster_labels)
        logger.info("Assigned cluster IDs to %s split.", name)

    # ---- Save clustered CSVs ----
    os.makedirs(clustered_dir, exist_ok=True)
    train_df.to_csv(os.path.join(clustered_dir, "train_clustered.csv"), index=False)
    val_df.to_csv(os.path.join(clustered_dir, "val_clustered.csv"), index=False)
    test_df.to_csv(os.path.join(clustered_dir, "test_clustered.csv"), index=False)

    # Save KMeans model
    joblib.dump(km, os.path.join(clustered_dir, "kmeans_model.joblib"))

    # Save cluster label mapping as JSON
    with open(os.path.join(clustered_dir, "cluster_labels.json"), "w") as f:
        json.dump({str(k): v for k, v in cluster_labels.items()}, f, indent=2)

    logger.info("Clustered CSVs and KMeans artefact saved to: %s", clustered_dir)

    # ---- Build and save prompt-target datasets ----
    os.makedirs(prompt_dir, exist_ok=True)
    for df, split_name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        prompt_dataset = build_prompt_dataset(df)
        out_path = os.path.join(prompt_dir, f"{split_name}_prompts.csv")
        prompt_dataset.to_csv(out_path, index=False)
        logger.info(
            "Prompt dataset for '%s' saved to: %s  (%d rows)",
            split_name, out_path, len(prompt_dataset),
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Module 4 – Topic Clustering & Prompt Dataset Building"
    )
    parser.add_argument("--train-keywords", default=os.path.join("data", "metadata", "train_keywords.csv"))
    parser.add_argument("--val-keywords", default=os.path.join("data", "metadata", "val_keywords.csv"))
    parser.add_argument("--test-keywords", default=os.path.join("data", "metadata", "test_keywords.csv"))
    parser.add_argument("--vectorizer", default=os.path.join("data", "metadata", "tfidf_vectorizer.joblib"))
    parser.add_argument("--svd", default=os.path.join("data", "metadata", "svd_reducer.joblib"))
    parser.add_argument("--clustered-dir", default=os.path.join("data", "clustered"))
    parser.add_argument("--prompt-dir", default=os.path.join("data", "prompt_dataset"))
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cluster_and_build_prompts(
        train_keyword_path=args.train_keywords,
        val_keyword_path=args.val_keywords,
        test_keyword_path=args.test_keywords,
        vectorizer_path=args.vectorizer,
        svd_path=args.svd,
        clustered_dir=args.clustered_dir,
        prompt_dir=args.prompt_dir,
    )
