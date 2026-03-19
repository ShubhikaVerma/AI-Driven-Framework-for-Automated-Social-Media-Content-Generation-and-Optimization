# AI-Driven Framework for Automated Social-Media Content Generation and Optimization

An end-to-end multimodal content generation and evaluation system for COVID-19 news synthesis. The pipeline ingests raw news articles, extracts keywords and topics, generates social-media posts using **LLaMA 3.1-8B Instruct**, and evaluates output quality with SBERT, ROUGE, and BERTScore.
## Table of Contents

1. [Project Overview](#project-overview)
2. [Objectives](#objectives)
3. [Hardware Requirements](#hardware-requirements)
4. [Software Requirements](#software-requirements)
5. [Quick Start](#quick-start)
6. [Project Structure](#project-structure)
7. [Pipeline Overview](#pipeline-overview)

---

## Project Overview

This system develops and evaluates a metadata-driven, multimodal content generation pipeline designed for COVID-19 news synthesis. It automates the creation of social media posts (text summaries and corresponding images) from raw news article metadata, evaluates the quality and semantic alignment of generated content, and refines outputs through user feedback.

The pipeline integrates:
- **LLaMA 3.1-8B Instruct** for text generation
- **Stable Diffusion** (SD 1.5, SD 2.1, SD Turbo) for image generation
- **CLIP** (ViT-B/32) for text–image alignment evaluation
- **SBERT**, **BERTScore**, and **ROUGE** for text quality evaluation
- **Streamlit** for the interactive user interface

---

## Objectives

- **Text Generation**: Develop a metadata-driven text generation pipeline using LLaMA 3.1-8B Instruct combined with TF-IDF keyword extraction to automate high-quality social media post creation from news articles.
- **Text Evaluation**: Deploy a semantic evaluation module integrating SBERT (all-MiniLM-L6-v2), BERTScore (RoBERTa-large), and ROUGE-1/2/L to systematically validate the quality, relevance, and fidelity of generated text.
- **Image Generation**: Use Stable Diffusion to create images with a multimodal alignment mechanism.
- **Image–Text Alignment**: Use CLIP embeddings to evaluate how well the generated images represent the semantics of the associated text.
- **Optimization Loop**: Create an optimization loop for improving cross-modal alignment and content quality that incorporates user feedback (ratings) and user behavior data.

---

## Hardware Requirements

The system can execute in a minimal configuration suitable for local development, UI deployment, and basic dataset processing without GPU acceleration.

| Component | Minimum | Implemented |
|-----------|---------|-------------|
| CPU | Intel i5 8th Gen / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 |
| RAM | 8 GB | 16 GB (Local); 30 GB (Kaggle) |
| GPU | Not required | NVIDIA T4 16 GB (Kaggle) |
| GPU VRAM | N/A | 16 GB (Kaggle) |
| Storage | 10 GB free | 50+ GB (Local + Cloud) |
| OS | Windows 10/11 | Windows 11 (Local) |

The minimum configuration requires an Intel i5 8th Generation or AMD Ryzen 5 processor for running the Streamlit interface, TF-IDF search operations, and basic text evaluation metrics. 8 GB of system RAM is sufficient for handling 4,000+ text samples. GPU support is optional at this tier, as text summarization and image generation can fall back to CPU inference with extended computation times.

---



## Software Requirements

### Text Generation

| Library | Purpose |
|---------|---------|
| `transformers` | LLaMA 3.1-8B Instruct model inference |
| `sentencepiece` | Tokenization backend for transformer models |
| `accelerate` | Optimized inference for faster transformer computations |

### Text Evaluation Metrics

| Metric | Library | Purpose |
|--------|---------|---------|
| ROUGE-1/2/L | `rouge-score` | Lexical overlap between generated and reference text |
| BERTScore | `bert-score` | Semantic similarity using RoBERTa-large contextual embeddings |
| SBERT Similarity | `sentence-transformers` | Query-to-summary semantic matching via all-MiniLM-L6-v2 |

### Image Generation

| Model | Inference Steps | VRAM Requirement |
|-------|:--------------:|:----------------:|
| Stable Diffusion 1.5 | 25–50 | 6 GB |
| Stable Diffusion 2.1 | 30–50 | 8 GB |
| Stable Diffusion Turbo | 4–8 | 4 GB |

| Library | Purpose |
|---------|---------|
| `diffusers` | Stable Diffusion pipeline; model loading and inference |
| `xformers` | Memory-efficient attention (30–40% VRAM reduction) |
| `torch` (CUDA) | GPU-accelerated tensor operations |
| `Pillow (PIL)` | Image I/O, format conversion |

### Image-Text Alignment

| Component | Details |
|-----------|---------|
| Model | CLIP ViT-B/32 (63M parameters) |
| Similarity | Cosine similarity between text and image embeddings |
| Robustness | Negative sampling (unrelated images) for alignment ratio |


Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/ShubhikaVerma/AI-Driven-Framework-for-Automated-Social-Media-Content-Generation-and-Optimization.git
cd AI-Driven-Framework-for-Automated-Social-Media-Content-Generation-and-Optimization
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare your dataset

Your input data must be a CSV file with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `Date` | string / date | Publication date of the news article (e.g., `2020-03-15`) |
| `Headline` | string | News headline; primary text source for summarization and image prompts |
| `Description` | string | Full article body / content |
| `Covid Status` | string | Classification of article relevance to COVID-19 (e.g., `covid`, `non-covid`) |
| `Image URL` | string | URL to the original image associated with the article |
| `Source` | string | News source website (e.g., `reuters.com`, `bbc.com`) |
| `Sentiment` | string | Sentiment classification: `positive`, `neutral`, or `negative` |

**Sample row:**

```
Date,Headline,Description,Covid Status,Image URL,Source,Sentiment
2020-03-15,"WHO declares COVID-19 a global pandemic","The World Health Organization officially declared the coronavirus outbreak a global pandemic on March 11...",covid,https://example.com/image.jpg,reuters.com,neutral
```

> **Note:** You can create your own CSV following the format above. Example datasets and sample CSV files will be available in the [`/data`](./data) directory once added — check there first, and if not yet present, use the sample row above as a template.

### 4. Run the Streamlit app


>
> ```bash
> streamlit run app.py
> ```
>
> See the [Pipeline Overview](#pipeline-overview) section for details on what each component will provide.

---


## Project Structure

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
