# Modules – Pipeline Component Reference

This directory contains the six modular components of the **AI-Driven Framework for Automated Social-Media Content Generation and Optimization** pipeline. Each module is a self-contained Python script with a top-level docstring, CLI entry point, and clearly named functions. Run them in order (1 → 6) or call individual functions from your own orchestration code.

---

## Directory Structure

```
modules/
├── README.md                          ← this file
├── module1_data_ingestion/
│   └── data_ingest_clean.py           ← Module 1
├── module2_data_preprocessing/
│   └── data_preprocess_split.py       ← Module 2
├── module3_metadata_keywords/
│   └── metadata_keywords.py           ← Module 3
├── module4_topic_clustering/
│   └── topic_clustering_promptbuild.py← Module 4
├── module5_prompt_generation/
│   └── prompt_generation.py           ← Module 5
└── module6_evaluation/
    └── eval_validation_metrics.py     ← Module 6
```

---

## Module Summaries

| # | Directory | Script | One-Line Summary |
|---|-----------|--------|-----------------|
| 1 | [`module1_data_ingestion/`](module1_data_ingestion/) | [`data_ingest_clean.py`](module1_data_ingestion/data_ingest_clean.py) | Loads the raw COVID-19 news CSV, normalises columns, handles missing values and duplicates, and writes a cleaned dataset. |
| 2 | [`module2_data_preprocessing/`](module2_data_preprocessing/) | [`data_preprocess_split.py`](module2_data_preprocessing/data_preprocess_split.py) | Further cleans text (HTML entities, Unicode, special chars) and performs time-based 70/15/15 train/val/test splitting. |
| 3 | [`module3_metadata_keywords/`](module3_metadata_keywords/) | [`metadata_keywords.py`](module3_metadata_keywords/metadata_keywords.py) | Fits TF-IDF (5 000 features, bigrams) and Truncated SVD (100 components) on training data, extracts per-article keywords, and enriches all splits. |
| 4 | [`module4_topic_clustering/`](module4_topic_clustering/) | [`topic_clustering_promptbuild.py`](module4_topic_clustering/topic_clustering_promptbuild.py) | Clusters articles with KMeans (k=8) on SVD vectors, assigns human-readable topic labels, and builds structured prompt–target CSVs for LLM generation. |
| 5 | [`module5_prompt_generation/`](module5_prompt_generation/) | [`prompt_generation.py`](module5_prompt_generation/prompt_generation.py) | Implements four prompt strategies (structured, detailed, role-based, metadata-aware), runs batched LLaMA 3.1-8B Instruct generation, cleans outputs, and evaluates with SBERT / ROUGE / BERTScore. |
| 6 | [`module6_evaluation/`](module6_evaluation/) | [`eval_validation_metrics.py`](module6_evaluation/eval_validation_metrics.py) | Aggregates per-sample metrics from Module 5, computes summary statistics across prompt strategies, and exports a Markdown evaluation report. |

---

## Data Flow

```
Raw CSV
   │
   ▼
[Module 1] data_ingest_clean.py
   │  data/processed/cleaned_data.csv
   ▼
[Module 2] data_preprocess_split.py
   │  data/splits/{train,val,test}.csv
   ▼
[Module 3] metadata_keywords.py
   │  data/metadata/{train,val,test}_keywords.csv
   │  data/metadata/tfidf_vectorizer.joblib
   │  data/metadata/svd_reducer.joblib
   ▼
[Module 4] topic_clustering_promptbuild.py
   │  data/clustered/{train,val,test}_clustered.csv
   │  data/clustered/kmeans_model.joblib
   │  data/clustered/cluster_labels.json
   │  data/prompt_dataset/{train,val,test}_prompts.csv
   ▼
[Module 5] prompt_generation.py
   │  data/generated/{train,val,test}_generated.csv
   ▼
[Module 6] eval_validation_metrics.py
      data/evaluation/{val,test}_metrics.csv
      data/evaluation/summary_stats.csv
      data/evaluation/evaluation_report.md
```

---

## Quick Start

Install dependencies first:

```bash
pip install -r requirements.txt
```

Then run each module in sequence, adjusting paths as needed:

```bash
# Module 1 – Ingest & Clean
python modules/module1_data_ingestion/data_ingest_clean.py \
    --input data/raw/covid_news.csv \
    --output data/processed/cleaned_data.csv

# Module 2 – Preprocess & Split
python modules/module2_data_preprocessing/data_preprocess_split.py \
    --input data/processed/cleaned_data.csv \
    --output-dir data/splits

# Module 3 – Metadata & Keywords
python modules/module3_metadata_keywords/metadata_keywords.py \
    --train data/splits/train.csv \
    --val   data/splits/val.csv \
    --test  data/splits/test.csv \
    --output-dir  data/metadata \
    --artefact-dir data/metadata

# Module 4 – Topic Clustering & Prompt Build
python modules/module4_topic_clustering/topic_clustering_promptbuild.py \
    --train-keywords data/metadata/train_keywords.csv \
    --val-keywords   data/metadata/val_keywords.csv \
    --test-keywords  data/metadata/test_keywords.csv \
    --vectorizer     data/metadata/tfidf_vectorizer.joblib \
    --svd            data/metadata/svd_reducer.joblib \
    --clustered-dir  data/clustered \
    --prompt-dir     data/prompt_dataset

# Module 5 – Prompt Engineering & Generation (GPU recommended)
python modules/module5_prompt_generation/prompt_generation.py \
    --split all \
    --strategy metadata_aware \
    --prompt-dir  data/prompt_dataset \
    --output-dir  data/generated

# Module 6 – Evaluation
python modules/module6_evaluation/eval_validation_metrics.py \
    --generated-dir data/generated \
    --output-dir    data/evaluation \
    --splits val test
```

---

## Dependencies

All required libraries are listed in [`requirements.txt`](../requirements.txt) at the repository root. Key packages:

| Library | Purpose |
|---------|---------|
| `pandas`, `numpy` | Data loading, manipulation, and numerical operations |
| `scikit-learn` | TF-IDF vectorisation, SVD dimensionality reduction, KMeans clustering |
| `joblib` | Serialisation of fitted ML artefacts |
| `transformers`, `accelerate`, `sentencepiece` | LLaMA 3.1-8B Instruct inference |
| `sentence-transformers` | SBERT semantic similarity evaluation |
| `rouge-score` | ROUGE-1/2/L lexical overlap evaluation |
| `bert-score` | BERTScore semantic evaluation (RoBERTa-large) |
| `torch` | GPU-accelerated tensor operations |
