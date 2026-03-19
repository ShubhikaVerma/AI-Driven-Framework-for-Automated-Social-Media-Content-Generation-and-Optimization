"""
Module 5: Prompt Engineering & Generation
==========================================
Pipeline Stage: Consumes prompt-target CSVs from Module 4 and orchestrates
                LLM-based social-media post generation plus inline quality
                evaluation.

Responsibilities:
  A. Prompt Templates
     - Structured Instruction Prompt  : separates Context / Task / Focus sections.
     - Detailed Guidelines Prompt     : explicit rules for fact retention and
                                        named-entity preservation.
     - Role-Based Expert Prompt       : instructs the model to adopt a senior news
                                        editor persona with journalism constraints.
     - Metadata-Aware Prompt          : integrates TF-IDF keywords, topic label,
                                        sentiment, and covid status into a single
                                        streamlined template.

  B. LLM Generation (LLaMA 3.1-8B Instruct via HuggingFace Transformers)
     - Tokenizer input limit  : 384 tokens
     - Max new tokens         : 72
     - Beam search            : 1 (greedy / deterministic)
     - Sampling               : disabled (do_sample=False)
     - Temperature            : 0.0
     - Repetition penalty     : no_repeat_ngram_size=3
     - Batch size             : configurable (default 8)

  C. Output Cleaning
     - Strip the prompt preamble from the generated text.
     - Normalise whitespace and remove artefacts.

  D. Evaluation Workflows (per generated output)
     - SBERT Cosine Similarity (all-MiniLM-L6-v2)
     - ROUGE-1 / ROUGE-2 / ROUGE-L
     - BERTScore Precision / Recall / F1 (RoBERTa-large)

Input:
  data/prompt_dataset/train_prompts.csv
  data/prompt_dataset/val_prompts.csv
  data/prompt_dataset/test_prompts.csv
Output:
  data/generated/train_generated.csv
  data/generated/val_generated.csv
  data/generated/test_generated.csv
"""

import argparse
import logging
import os
import re
from typing import Optional

import pandas as pd
import torch

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
DEFAULT_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MAX_INPUT_TOKENS = 384
MAX_NEW_TOKENS = 72
BATCH_SIZE = 8
NO_REPEAT_NGRAM_SIZE = 3

# ---------------------------------------------------------------------------
# ============================================================
# SECTION 1: PROMPT TEMPLATES
# ============================================================
# ---------------------------------------------------------------------------


def structured_instruction_prompt(
    headline: str,
    description: str,
    keywords: str = "",
    topic_label: str = "",
    sentiment: str = "neutral",
) -> str:
    """
    Structured Instruction Prompt.

    Separates Context, Task, and Focus into distinct labelled sections to
    reduce model ambiguity.
    """
    return (
        f"### Context\n{headline}\n\n"
        f"### Task\nSummarise the above news article into a concise social-media "
        f"post (≤ 280 characters).\n\n"
        f"### Focus\nRetain key facts and named entities. Tone: {sentiment}.\n\n"
        f"### Summary"
    )


def detailed_guidelines_prompt(
    headline: str,
    description: str,
    keywords: str = "",
    topic_label: str = "",
    sentiment: str = "neutral",
) -> str:
    """
    Detailed Guidelines Prompt.

    Provides explicit summarisation rules for fact retention and named-entity
    preservation.
    """
    return (
        f"You are a news summariser.\n\n"
        f"Article headline: {headline}\n\n"
        f"Article body: {description}\n\n"
        f"Instructions:\n"
        f"1. Write a single-sentence social-media post (≤ 280 characters).\n"
        f"2. Preserve all named entities (people, places, organisations).\n"
        f"3. Include at least one key fact or statistic from the article.\n"
        f"4. Use a {sentiment} tone.\n"
        f"5. Do NOT include hashtags or emojis.\n\n"
        f"Social-media post:"
    )


def role_based_expert_prompt(
    headline: str,
    description: str,
    keywords: str = "",
    topic_label: str = "",
    sentiment: str = "neutral",
    covid_status: str = "unknown",
) -> str:
    """
    Role-Based Expert Prompt.

    Instructs the model to act as a senior news editor with specific journalism
    constraints.
    """
    return (
        f"You are a senior news editor at a leading wire service.\n\n"
        f"Story type: COVID-19 news ({covid_status})\n"
        f"Topic category: {topic_label}\n"
        f"Tone: {sentiment}\n\n"
        f"Headline: {headline}\n\n"
        f"Body: {description}\n\n"
        f"Produce a factual, jargon-free social-media post (≤ 280 characters) "
        f"suitable for a general audience. Mention geography and key entities. "
        f"Do not start with 'Here is'.\n\n"
        f"Post:"
    )


def metadata_aware_prompt(
    headline: str,
    description: str,
    keywords: str = "",
    topic_label: str = "",
    sentiment: str = "neutral",
    covid_status: str = "unknown",
) -> str:
    """
    Metadata-Aware Prompt.

    Integrates TF-IDF keywords, topic label, sentiment, and covid_status into
    a single streamlined template to provide rich context while minimising
    token usage.
    """
    return (
        f"[METADATA]\n"
        f"Topic: {topic_label} | Keywords: {keywords} | "
        f"Sentiment: {sentiment} | COVID: {covid_status}\n\n"
        f"[ARTICLE]\n{headline}\n{description}\n\n"
        f"[INSTRUCTION]\n"
        f"Write a concise, factual social-media post (≤ 280 characters) that "
        f"captures the essential information above.\n\n"
        f"[POST]"
    )


PROMPT_STRATEGIES = {
    "structured": structured_instruction_prompt,
    "detailed": detailed_guidelines_prompt,
    "role_based": role_based_expert_prompt,
    "metadata_aware": metadata_aware_prompt,
}


def build_prompt(row: pd.Series, strategy: str = "metadata_aware") -> str:
    """Select and apply the specified prompt strategy to a DataFrame row."""
    fn = PROMPT_STRATEGIES.get(strategy, metadata_aware_prompt)
    return fn(
        headline=str(row.get("headline", "")),
        description=str(row.get("target", "")),
        keywords=str(row.get("keywords", "")),
        topic_label=str(row.get("topic_label", "")),
        sentiment=str(row.get("sentiment", "neutral")),
        covid_status=str(row.get("covid_status", "unknown")),
    )


# ---------------------------------------------------------------------------
# ============================================================
# SECTION 2: LLM LOADING & GENERATION
# ============================================================
# ---------------------------------------------------------------------------


def load_model_and_tokenizer(model_name: str = DEFAULT_MODEL_NAME):
    """
    Load the LLaMA tokenizer and model from HuggingFace.

    The model is placed on GPU if available, otherwise CPU.

    Returns
    -------
    tuple
        (tokenizer, model)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading tokenizer from: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading model on device: %s", device)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()
    logger.info("Model loaded successfully.")
    return tokenizer, model


def generate_batch(
    prompts: list[str],
    tokenizer,
    model,
    max_input_tokens: int = MAX_INPUT_TOKENS,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> list[str]:
    """
    Generate text for a batch of prompts using the loaded LLM.

    Parameters
    ----------
    prompts : list[str]
        List of formatted prompt strings.
    tokenizer : PreTrainedTokenizer
        Fitted tokenizer.
    model : PreTrainedModel
        Loaded causal-LM model.
    max_input_tokens : int
        Maximum number of input tokens per prompt.
    max_new_tokens : int
        Maximum number of tokens to generate.

    Returns
    -------
    list[str]
        Decoded generated texts (prompt preamble stripped).
    """
    device = next(model.parameters()).device

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    input_len = inputs["input_ids"].shape[1]
    decoded = tokenizer.batch_decode(
        output_ids[:, input_len:],
        skip_special_tokens=True,
    )
    return decoded


# ---------------------------------------------------------------------------
# ============================================================
# SECTION 3: OUTPUT CLEANING
# ============================================================
# ---------------------------------------------------------------------------


def clean_generated_text(text: str) -> str:
    """
    Clean a raw LLM generation by removing artefacts and normalising whitespace.

    Steps:
      1. Strip common preamble phrases the model may prepend.
      2. Collapse multiple spaces / newlines to a single space.
      3. Strip leading/trailing whitespace.
      4. Truncate to 280 characters (social-media post limit).
    """
    preamble_patterns = [
        r"^\s*(Here is|Sure[,!]|Certainly[,!]|Of course[,!])[^\n]*\n*",
        r"^\s*(Post|Summary|Social-media post|Generated post)\s*:\s*",
    ]
    for pat in preamble_patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)

    text = re.sub(r"\s+", " ", text).strip()
    return text[:280]


# ---------------------------------------------------------------------------
# ============================================================
# SECTION 4: EVALUATION WORKFLOWS
# ============================================================
# ---------------------------------------------------------------------------


def compute_sbert_similarity(
    generated_texts: list[str],
    reference_texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
) -> list[float]:
    """
    Compute SBERT cosine similarity between generated and reference texts.

    Parameters
    ----------
    generated_texts : list[str]
        Model-generated summaries.
    reference_texts : list[str]
        Ground-truth reference texts.
    model_name : str
        SentenceTransformer model identifier.

    Returns
    -------
    list[float]
        Cosine similarity score per pair.
    """
    from sentence_transformers import SentenceTransformer, util

    logger.info("Computing SBERT similarity with model: %s", model_name)
    sbert_model = SentenceTransformer(model_name)
    gen_embeddings = sbert_model.encode(generated_texts, convert_to_tensor=True)
    ref_embeddings = sbert_model.encode(reference_texts, convert_to_tensor=True)
    similarities = util.cos_sim(gen_embeddings, ref_embeddings).diagonal().tolist()
    return similarities


def compute_rouge_scores(
    generated_texts: list[str],
    reference_texts: list[str],
) -> list[dict]:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

    Returns
    -------
    list[dict]
        One dict per pair with keys 'rouge1', 'rouge2', 'rougeL'.
    """
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    results = []
    for gen, ref in zip(generated_texts, reference_texts):
        scores = scorer.score(ref, gen)
        results.append(
            {
                "rouge1": scores["rouge1"].fmeasure,
                "rouge2": scores["rouge2"].fmeasure,
                "rougeL": scores["rougeL"].fmeasure,
            }
        )
    return results


def compute_bertscore(
    generated_texts: list[str],
    reference_texts: list[str],
    model_type: str = "roberta-large",
) -> list[dict]:
    """
    Compute BERTScore Precision, Recall, and F1.

    Returns
    -------
    list[dict]
        One dict per pair with keys 'bertscore_p', 'bertscore_r', 'bertscore_f1'.
    """
    from bert_score import score as bert_score_fn

    logger.info("Computing BERTScore with model: %s", model_type)
    P, R, F1 = bert_score_fn(
        generated_texts,
        reference_texts,
        model_type=model_type,
        lang="en",
        verbose=False,
    )
    return [
        {"bertscore_p": p.item(), "bertscore_r": r.item(), "bertscore_f1": f.item()}
        for p, r, f in zip(P, R, F1)
    ]


def evaluate_generations(
    generated_texts: list[str],
    reference_texts: list[str],
    run_sbert: bool = True,
    run_rouge: bool = True,
    run_bertscore: bool = True,
) -> pd.DataFrame:
    """
    Aggregate all evaluation metrics into a single DataFrame.

    Parameters
    ----------
    generated_texts : list[str]
        Model-generated summaries.
    reference_texts : list[str]
        Ground-truth reference texts.
    run_sbert, run_rouge, run_bertscore : bool
        Flags to enable/disable individual metric groups.

    Returns
    -------
    pd.DataFrame
        One row per sample with all available metric columns.
    """
    metrics: dict[str, list] = {}

    if run_sbert:
        metrics["sbert_similarity"] = compute_sbert_similarity(
            generated_texts, reference_texts
        )

    if run_rouge:
        rouge_results = compute_rouge_scores(generated_texts, reference_texts)
        for key in ["rouge1", "rouge2", "rougeL"]:
            metrics[key] = [r[key] for r in rouge_results]

    if run_bertscore:
        bs_results = compute_bertscore(generated_texts, reference_texts)
        for key in ["bertscore_p", "bertscore_r", "bertscore_f1"]:
            metrics[key] = [r[key] for r in bs_results]

    return pd.DataFrame(metrics)


# ---------------------------------------------------------------------------
# ============================================================
# SECTION 5: MAIN PIPELINE
# ============================================================
# ---------------------------------------------------------------------------


def run_generation_pipeline(
    prompt_csv_path: str,
    output_path: str,
    strategy: str = "metadata_aware",
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = BATCH_SIZE,
    run_sbert: bool = True,
    run_rouge: bool = True,
    run_bertscore: bool = True,
    tokenizer=None,
    model=None,
) -> pd.DataFrame:
    """
    Load a prompt CSV, generate summaries, clean outputs, evaluate, and save.

    Parameters
    ----------
    prompt_csv_path : str
        Path to a *_prompts.csv produced by Module 4.
    output_path : str
        Destination CSV path for generated results with metrics.
    strategy : str
        Prompt strategy key: 'structured', 'detailed', 'role_based',
        'metadata_aware'.
    model_name : str
        HuggingFace model identifier for the LLM.
    batch_size : int
        Number of samples per inference batch.
    run_sbert, run_rouge, run_bertscore : bool
        Flags controlling which evaluation metrics are computed.
    tokenizer, model : optional
        Pre-loaded tokenizer and model (avoids reloading for multiple splits).

    Returns
    -------
    pd.DataFrame
        DataFrame with generated texts and evaluation scores.
    """
    df = pd.read_csv(prompt_csv_path)
    logger.info("Loaded %d rows from: %s", len(df), prompt_csv_path)

    # Build prompts using the chosen strategy
    prompts = df.apply(lambda row: build_prompt(row, strategy), axis=1).tolist()

    # Load model if not provided
    if tokenizer is None or model is None:
        tokenizer, model = load_model_and_tokenizer(model_name)

    # Batched generation
    all_generated: list[str] = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i: i + batch_size]
        logger.info(
            "Generating batch %d/%d (rows %d–%d)…",
            i // batch_size + 1,
            (len(prompts) + batch_size - 1) // batch_size,
            i,
            min(i + batch_size, len(prompts)) - 1,
        )
        generated = generate_batch(batch, tokenizer, model)
        cleaned = [clean_generated_text(g) for g in generated]
        all_generated.extend(cleaned)

    df["generated_text"] = all_generated
    df["prompt_strategy"] = strategy

    # Evaluation
    reference_texts = df["target"].fillna("").astype(str).tolist()
    metrics_df = evaluate_generations(
        all_generated,
        reference_texts,
        run_sbert=run_sbert,
        run_rouge=run_rouge,
        run_bertscore=run_bertscore,
    )
    result_df = pd.concat([df.reset_index(drop=True), metrics_df], axis=1)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    result_df.to_csv(output_path, index=False)
    logger.info("Results saved to: %s", output_path)
    return result_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Module 5 – Prompt Engineering & Generation"
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="val",
        help="Which split(s) to run generation on (default: val)",
    )
    parser.add_argument(
        "--strategy",
        choices=list(PROMPT_STRATEGIES.keys()),
        default="metadata_aware",
        help="Prompt strategy to use (default: metadata_aware)",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help=f"HuggingFace model identifier (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Inference batch size (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--prompt-dir",
        default=os.path.join("data", "prompt_dataset"),
        help="Directory containing *_prompts.csv files",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("data", "generated"),
        help="Output directory for generated result CSVs",
    )
    parser.add_argument("--no-sbert", action="store_true", help="Skip SBERT evaluation")
    parser.add_argument("--no-rouge", action="store_true", help="Skip ROUGE evaluation")
    parser.add_argument("--no-bertscore", action="store_true", help="Skip BERTScore evaluation")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    # Load model once for all splits
    _tokenizer, _model = load_model_and_tokenizer(args.model_name)

    for split in splits:
        prompt_path = os.path.join(args.prompt_dir, f"{split}_prompts.csv")
        output_path = os.path.join(args.output_dir, f"{split}_generated.csv")
        if not os.path.exists(prompt_path):
            logger.warning("Prompt file not found, skipping: %s", prompt_path)
            continue
        run_generation_pipeline(
            prompt_csv_path=prompt_path,
            output_path=output_path,
            strategy=args.strategy,
            batch_size=args.batch_size,
            run_sbert=not args.no_sbert,
            run_rouge=not args.no_rouge,
            run_bertscore=not args.no_bertscore,
            tokenizer=_tokenizer,
            model=_model,
        )
