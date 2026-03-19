"""
Module 6: Evaluation Module
============================
Pipeline Stage: Final stage of the pipeline – aggregates per-sample generation
                results from Module 5, computes summary statistics across
                validation and test sets, and formats outputs for reporting.

Responsibilities:
  - Load generated CSVs (val_generated.csv, test_generated.csv) produced by
    Module 5.
  - Re-compute or collect per-sample metrics:
      • SBERT cosine similarity (all-MiniLM-L6-v2)
      • ROUGE-1, ROUGE-2, ROUGE-L
      • BERTScore Precision / Recall / F1 (RoBERTa-large)
  - Compute aggregate summary statistics (mean, std, min, max, median) for each
    metric across all samples in a split.
  - Compare summary statistics across prompt strategies (structured, detailed,
    role-based, metadata-aware) when multiple strategy results are present.
  - Format and export:
      • Per-sample metrics tables as CSV.
      • Summary statistics tables as CSV and as a human-readable Markdown report.
  - Log top-K and bottom-K samples ranked by BERTScore F1 for qualitative review.

Input:
  data/generated/val_generated.csv
  data/generated/test_generated.csv
  (Optional) data/generated/train_generated.csv
Output:
  data/evaluation/val_metrics.csv
  data/evaluation/test_metrics.csv
  data/evaluation/summary_stats.csv
  data/evaluation/evaluation_report.md
"""

import argparse
import logging
import os
from typing import Optional

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
METRIC_COLS = [
    "sbert_similarity",
    "rouge1",
    "rouge2",
    "rougeL",
    "bertscore_p",
    "bertscore_r",
    "bertscore_f1",
]
SUMMARY_STATS = ["mean", "std", "min", "median", "max"]
TOP_K = 5  # samples to display for qualitative review


# ---------------------------------------------------------------------------
# ============================================================
# SECTION 1: METRIC (RE-)COMPUTATION
# ============================================================
# ---------------------------------------------------------------------------


def ensure_metrics(
    df: pd.DataFrame,
    run_sbert: bool = True,
    run_rouge: bool = True,
    run_bertscore: bool = True,
) -> pd.DataFrame:
    """
    Ensure all metric columns are present in the DataFrame.

    If a metric column is missing (e.g. because it was not computed in Module 5),
    this function recomputes it on the fly from 'generated_text' and 'target'.

    Parameters
    ----------
    df : pd.DataFrame
        Generated results DataFrame.
    run_sbert, run_rouge, run_bertscore : bool
        Flags controlling which metrics to (re-)compute.

    Returns
    -------
    pd.DataFrame
        DataFrame with all requested metric columns present.
    """
    generated = df["generated_text"].fillna("").astype(str).tolist()
    references = df["target"].fillna("").astype(str).tolist()

    # SBERT
    if run_sbert and "sbert_similarity" not in df.columns:
        logger.info("Re-computing SBERT similarity…")
        from sentence_transformers import SentenceTransformer, util

        sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        gen_emb = sbert_model.encode(generated, convert_to_tensor=True)
        ref_emb = sbert_model.encode(references, convert_to_tensor=True)
        df["sbert_similarity"] = util.cos_sim(gen_emb, ref_emb).diagonal().tolist()

    # ROUGE
    if run_rouge and any(c not in df.columns for c in ["rouge1", "rouge2", "rougeL"]):
        logger.info("Re-computing ROUGE scores…")
        from rouge_score import rouge_scorer as rs_mod

        scorer = rs_mod.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        rouge1, rouge2, rougeL = [], [], []
        for gen, ref in zip(generated, references):
            s = scorer.score(ref, gen)
            rouge1.append(s["rouge1"].fmeasure)
            rouge2.append(s["rouge2"].fmeasure)
            rougeL.append(s["rougeL"].fmeasure)
        df["rouge1"] = rouge1
        df["rouge2"] = rouge2
        df["rougeL"] = rougeL

    # BERTScore
    if run_bertscore and any(
        c not in df.columns for c in ["bertscore_p", "bertscore_r", "bertscore_f1"]
    ):
        logger.info("Re-computing BERTScore…")
        from bert_score import score as bert_score_fn

        P, R, F1 = bert_score_fn(
            generated, references, model_type="roberta-large", lang="en", verbose=False
        )
        df["bertscore_p"] = [p.item() for p in P]
        df["bertscore_r"] = [r.item() for r in R]
        df["bertscore_f1"] = [f.item() for f in F1]

    return df


# ---------------------------------------------------------------------------
# ============================================================
# SECTION 2: SUMMARY STATISTICS
# ============================================================
# ---------------------------------------------------------------------------


def compute_summary_stats(
    df: pd.DataFrame,
    split_name: str,
    group_by_strategy: bool = True,
) -> pd.DataFrame:
    """
    Compute aggregate statistics for each metric column.

    Parameters
    ----------
    df : pd.DataFrame
        Per-sample metrics DataFrame.
    split_name : str
        Identifier for the data split (e.g. 'val', 'test').
    group_by_strategy : bool
        When True and 'prompt_strategy' column is present, compute stats
        separately per strategy.

    Returns
    -------
    pd.DataFrame
        Summary statistics table.
    """
    available_metrics = [c for c in METRIC_COLS if c in df.columns]

    if group_by_strategy and "prompt_strategy" in df.columns:
        summary = (
            df.groupby("prompt_strategy")[available_metrics]
            .agg(SUMMARY_STATS)
        )
        summary.columns = ["_".join(col) for col in summary.columns]
        summary.insert(0, "split", split_name)
        summary = summary.reset_index()
    else:
        summary = df[available_metrics].agg(SUMMARY_STATS).T
        summary.columns = SUMMARY_STATS
        summary.insert(0, "metric", summary.index)
        summary.insert(0, "split", split_name)
        summary = summary.reset_index(drop=True)

    return summary


# ---------------------------------------------------------------------------
# ============================================================
# SECTION 3: QUALITATIVE REVIEW
# ============================================================
# ---------------------------------------------------------------------------


def log_top_bottom_samples(
    df: pd.DataFrame,
    split_name: str,
    rank_col: str = "bertscore_f1",
    top_k: int = TOP_K,
) -> None:
    """
    Log the top-K and bottom-K samples ranked by the given metric column.

    Parameters
    ----------
    df : pd.DataFrame
        Per-sample metrics DataFrame.
    split_name : str
        Name of the split for logging context.
    rank_col : str
        Column to rank samples by (default: 'bertscore_f1').
    top_k : int
        Number of samples to log at each end.
    """
    if rank_col not in df.columns:
        logger.warning("Rank column '%s' not found; skipping qualitative review.", rank_col)
        return

    sorted_df = df.sort_values(rank_col, ascending=False).reset_index(drop=True)

    logger.info("=== [%s] Top-%d samples by %s ===", split_name.upper(), top_k, rank_col)
    for _, row in sorted_df.head(top_k).iterrows():
        logger.info(
            "Score: %.4f | Headline: %s | Generated: %s",
            row[rank_col],
            str(row.get("headline", ""))[:80],
            str(row.get("generated_text", ""))[:120],
        )

    logger.info("=== [%s] Bottom-%d samples by %s ===", split_name.upper(), top_k, rank_col)
    for _, row in sorted_df.tail(top_k).iterrows():
        logger.info(
            "Score: %.4f | Headline: %s | Generated: %s",
            row[rank_col],
            str(row.get("headline", ""))[:80],
            str(row.get("generated_text", ""))[:120],
        )


# ---------------------------------------------------------------------------
# ============================================================
# SECTION 4: MARKDOWN REPORT GENERATION
# ============================================================
# ---------------------------------------------------------------------------


def _stats_table_md(summary_df: pd.DataFrame) -> str:
    """Convert a summary DataFrame to a Markdown table string."""
    return summary_df.to_markdown(index=False)


def generate_markdown_report(
    all_summaries: list[tuple[str, pd.DataFrame]],
    output_path: str,
) -> None:
    """
    Write a human-readable Markdown evaluation report.

    Parameters
    ----------
    all_summaries : list[tuple[str, pd.DataFrame]]
        List of (split_name, summary_df) pairs.
    output_path : str
        File path for the Markdown report.
    """
    lines = [
        "# Evaluation Report",
        "",
        "Generated automatically by **Module 6 – Evaluation Module**.",
        "",
        "---",
        "",
    ]

    for split_name, summary_df in all_summaries:
        lines.append(f"## {split_name.capitalize()} Set Results")
        lines.append("")

        if not summary_df.empty:
            try:
                lines.append(_stats_table_md(summary_df))
            except Exception:
                lines.append(summary_df.to_string())
        else:
            lines.append("_No results available._")

        lines.append("")
        lines.append("---")
        lines.append("")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("Markdown evaluation report written to: %s", output_path)


# ---------------------------------------------------------------------------
# ============================================================
# SECTION 5: MAIN PIPELINE
# ============================================================
# ---------------------------------------------------------------------------


def evaluate_split(
    generated_csv_path: str,
    output_csv_path: str,
    split_name: str,
    run_sbert: bool = True,
    run_rouge: bool = True,
    run_bertscore: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Process a single split: ensure metrics, save per-sample CSV, log top/bottom.

    Parameters
    ----------
    generated_csv_path : str
        Path to the generated CSV from Module 5.
    output_csv_path : str
        Path to save the per-sample metrics CSV.
    split_name : str
        Label for the split ('val' or 'test').
    run_sbert, run_rouge, run_bertscore : bool
        Metric computation flags.

    Returns
    -------
    pd.DataFrame or None
        Per-sample metrics DataFrame, or None if the input file is missing.
    """
    if not os.path.exists(generated_csv_path):
        logger.warning("Generated file not found, skipping: %s", generated_csv_path)
        return None

    logger.info("Processing %s split from: %s", split_name, generated_csv_path)
    df = pd.read_csv(generated_csv_path)
    logger.info("Loaded %d rows.", len(df))

    df = ensure_metrics(df, run_sbert=run_sbert, run_rouge=run_rouge, run_bertscore=run_bertscore)
    log_top_bottom_samples(df, split_name)

    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    logger.info("Per-sample metrics saved to: %s", output_csv_path)
    return df


def run_evaluation(
    generated_dir: str,
    output_dir: str,
    splits: Optional[list[str]] = None,
    run_sbert: bool = True,
    run_rouge: bool = True,
    run_bertscore: bool = True,
) -> None:
    """
    Full Module 6 pipeline: evaluate all requested splits and produce a report.

    Parameters
    ----------
    generated_dir : str
        Directory containing *_generated.csv files from Module 5.
    output_dir : str
        Directory to write evaluation outputs.
    splits : list[str] or None
        Splits to evaluate; defaults to ['val', 'test'].
    run_sbert, run_rouge, run_bertscore : bool
        Metric computation flags.
    """
    if splits is None:
        splits = ["val", "test"]

    os.makedirs(output_dir, exist_ok=True)
    all_summaries: list[tuple[str, pd.DataFrame]] = []
    all_summary_dfs: list[pd.DataFrame] = []

    for split in splits:
        generated_path = os.path.join(generated_dir, f"{split}_generated.csv")
        output_path = os.path.join(output_dir, f"{split}_metrics.csv")

        df = evaluate_split(
            generated_csv_path=generated_path,
            output_csv_path=output_path,
            split_name=split,
            run_sbert=run_sbert,
            run_rouge=run_rouge,
            run_bertscore=run_bertscore,
        )
        if df is not None:
            summary = compute_summary_stats(df, split_name=split)
            all_summaries.append((split, summary))
            all_summary_dfs.append(summary)
            logger.info("Summary stats for '%s' split:\n%s", split, summary.to_string())

    # Save combined summary statistics
    if all_summary_dfs:
        combined = pd.concat(all_summary_dfs, ignore_index=True)
        combined.to_csv(os.path.join(output_dir, "summary_stats.csv"), index=False)
        logger.info("Combined summary stats saved to: %s", os.path.join(output_dir, "summary_stats.csv"))

    # Markdown report
    generate_markdown_report(
        all_summaries=all_summaries,
        output_path=os.path.join(output_dir, "evaluation_report.md"),
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Module 6 – Evaluation Module"
    )
    parser.add_argument(
        "--generated-dir",
        default=os.path.join("data", "generated"),
        help="Directory containing *_generated.csv files (default: data/generated/)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("data", "evaluation"),
        help="Output directory for evaluation results (default: data/evaluation/)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val", "test"],
        help="Splits to evaluate (default: val test)",
    )
    parser.add_argument("--no-sbert", action="store_true", help="Skip SBERT evaluation")
    parser.add_argument("--no-rouge", action="store_true", help="Skip ROUGE evaluation")
    parser.add_argument("--no-bertscore", action="store_true", help="Skip BERTScore evaluation")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_evaluation(
        generated_dir=args.generated_dir,
        output_dir=args.output_dir,
        splits=args.splits,
        run_sbert=not args.no_sbert,
        run_rouge=not args.no_rouge,
        run_bertscore=not args.no_bertscore,
    )
