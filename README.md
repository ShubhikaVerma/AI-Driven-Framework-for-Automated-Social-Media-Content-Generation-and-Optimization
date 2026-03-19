# AI-Driven Framework for Automated Social-Media Content Generation and Optimization

An end-to-end multimodal content generation and evaluation system for COVID-19 news synthesis. The pipeline ingests raw news articles, extracts keywords and topics, generates social-media posts using **LLaMA 3.1-8B Instruct**, and evaluates output quality with SBERT, ROUGE, and BERTScore.

## Repository Structure

```
.
├── requirements.txt          # Python dependencies
├── modules/                  # Modular pipeline components (see modules/README.md)
│   ├── README.md             # Module index, data-flow diagram, quick-start guide
│   ├── module1_data_ingestion/
│   │   └── data_ingest_clean.py
│   ├── module2_data_preprocessing/
│   │   └── data_preprocess_split.py
│   ├── module3_metadata_keywords/
│   │   └── metadata_keywords.py
│   ├── module4_topic_clustering/
│   │   └── topic_clustering_promptbuild.py
│   ├── module5_prompt_generation/
│   │   └── prompt_generation.py
│   └── module6_evaluation/
│       └── eval_validation_metrics.py
└── data/                     # Created at runtime (not committed)
    ├── raw/                  # Place covid_news.csv here
    ├── processed/
    ├── splits/
    ├── metadata/
    ├── clustered/
    ├── prompt_dataset/
    ├── generated/
    └── evaluation/
```

## Pipeline Overview

| Module | Script | Purpose |
|--------|--------|---------|
| 1 | `data_ingest_clean.py` | Load CSV, normalise columns, handle missing values, remove duplicates |
| 2 | `data_preprocess_split.py` | Further text cleaning, time-based 70/15/15 train/val/test split |
| 3 | `metadata_keywords.py` | TF-IDF keyword extraction, SVD dimensionality reduction |
| 4 | `topic_clustering_promptbuild.py` | KMeans topic clustering, human-readable labels, prompt–target CSV build |
| 5 | `prompt_generation.py` | Four prompt strategies, LLM generation, SBERT/ROUGE/BERTScore evaluation |
| 6 | `eval_validation_metrics.py` | Aggregate metrics, summary statistics, Markdown evaluation report |

See **[modules/README.md](modules/README.md)** for detailed descriptions, data-flow diagrams, and a quick-start guide.

## Installation

```bash
pip install -r requirements.txt
```

> **Note:** LLM inference (Module 5) requires a CUDA-enabled GPU (≥ 16 GB VRAM recommended). All other modules run on CPU.
