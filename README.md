# AI-Driven Framework for Automated Social Media Content Generation and Optimization

An end-to-end multimodal content generation and evaluation system for COVID-19 news synthesis, combining LLaMA 3.1-8B Instruct text generation, Stable Diffusion image generation, and comprehensive evaluation using SBERT, BERTScore, ROUGE, and CLIP metrics.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Results at a Glance](#results-at-a-glance)
  - [Prompt Strategy Comparison (Text Metrics)](#prompt-strategy-comparison-text-metrics)
  - [Image-Text Alignment (CLIP Scores)](#image-text-alignment-clip-scores)
- [System Architecture](#system-architecture)
- [Modules](#modules)
- [Hardware Requirements](#hardware-requirements)
- [Software Requirements](#software-requirements)
- [Getting Started](#getting-started)

---

## Project Overview

This project develops and evaluates an end-to-end multimodal content generation and evaluation system for COVID-19 news synthesis. Key capabilities include:

- **Metadata-driven text generation** using LLaMA 3.1-8B Instruct with TF-IDF keyword extraction to automate high-quality social media post creation from news articles.
- **Semantic evaluation** integrating SBERT (all-MiniLM-L6-v2), BERTScore (RoBERTa-large), and ROUGE-1/2/L to validate text quality, relevance, and fidelity.
- **Image generation** via Stable Diffusion (SD 1.5, SD 2.1, SD Turbo) with CLIP-based multimodal alignment evaluation.
- **Optimization loop** incorporating user feedback (ratings) to improve cross-modal alignment and content quality.

Dataset: 4,073 COVID-19 news headlines from InShorts (03-Mar-2020 to 11-Apr-2020).

---

## Results at a Glance

### Prompt Strategy Comparison (Text Metrics)

The table below compares three prompt strategies evaluated on SBERT cosine similarity, ROUGE-L F1, and BERTScore F1 (RoBERTa-large). Higher scores indicate better summary quality.

| Prompt Strategy        | SBERT Similarity | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | BERTScore F1 |
|------------------------|:----------------:|:----------:|:----------:|:----------:|:------------:|
| Baseline (Structured)  | 0.72             | 0.38       | 0.17       | 0.35       | 0.86         |
| Meta-Aware             | 0.76             | 0.41       | 0.20       | 0.38       | 0.88         |
| Best Guideline (Role-Based Expert) | **0.81** | **0.45** | **0.24** | **0.42** | **0.91** |

> **Key Takeaway:** The Role-Based Expert (Best Guideline) prompt strategy consistently outperforms the baseline across all metrics, demonstrating that structured metadata-aware prompting with professional journalism context significantly improves summarization quality.

---

### Image-Text Alignment (CLIP Scores)

The table below shows CLIP ViT-B/32 cosine similarity scores between generated images and their corresponding text summaries across Stable Diffusion model variants. Higher positive scores and lower negative (unrelated image) scores indicate better semantic alignment.

| SD Model              | CLIP Score (Positive) | CLIP Score (Negative) | Alignment Ratio |
|-----------------------|:---------------------:|:---------------------:|:---------------:|
| SD 1.5 Baseline       | 0.28                  | 0.18                  | 1.56            |
| SD 2.1                | 0.31                  | 0.17                  | 1.82            |
| SD Turbo              | **0.33**              | **0.16**              | **2.06**        |

> **Key Takeaway:** SD Turbo achieves the highest positive CLIP alignment score and the best alignment ratio, making it the most semantically consistent model despite using fewer inference steps (4–8 vs. 25–50). The higher alignment ratio confirms better discriminative alignment between generated images and their text descriptions.

---

## System Architecture

```
Raw News Data (CSV)
       │
       ▼
[Module 1] Data Collection & Loading
       │
       ▼
[Module 2] Data Preprocessing (cleaning, normalization, splitting)
       │
       ▼
[Module 3] Metadata Extraction (TF-IDF, KMeans topic clustering)
       │
       ▼
[Module 4] NLP Processing — Text Generation (LLaMA 3.1-8B Instruct)
       │                    Text Evaluation (SBERT, BERTScore, ROUGE)
       │
       ▼
[Module 5] Multimodal Alignment — Image Generation (Stable Diffusion)
       │                          CLIP-Based Semantic Similarity
       │
       ▼
[Module 6] Evaluation Module (quantitative comparison across prompt strategies)
       │
       ▼
[Module 7] User Feedback Collector (ratings → optimization loop)
       │
       ▼
Streamlit UI (real-time query, image display, feedback collection)
```

---

## Modules

| Module | Name | Description |
|--------|------|-------------|
| 1 | Data Collection | Ingests 4,073 COVID-19 news headlines; structures raw CSV data |
| 2 | Data Preprocessing | Text normalization, noise removal, 70/15/15 train/val/test split |
| 3 | Metadata Extraction | TF-IDF vectorization (5,000 features, bigrams), Truncated SVD (100 components), KMeans (8 clusters) |
| 4 | NLP Processing | LLaMA 3.1-8B Instruct generation; SBERT, BERTScore, ROUGE evaluation; three prompt strategies |
| 5 | Multimodal Alignment | Stable Diffusion image generation; CLIP ViT-B/32 text-image similarity; negative sampling robustness check |
| 6 | Evaluation | Quantitative comparison of prompt strategies and model variants |
| 7 | User Feedback Collector | Binary like/dislike + 1–5 ratings on relevance, correctness, realism, alignment |

---

## Hardware Requirements

| Component | Minimum | Implemented |
|-----------|---------|-------------|
| CPU | Intel i5 8th Gen / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 |
| RAM | 8 GB | 16 GB (Local); 30 GB (Kaggle) |
| GPU | Not required | NVIDIA T4 16 GB (Kaggle) |
| GPU VRAM | N/A | 16 GB (Kaggle) |
| Storage | 10 GB free | 50+ GB (Local + Cloud) |
| OS | Windows 10/11 | Windows 11 (Local) |

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

---

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### Running the UI

```bash
streamlit run app.py
```

### Environment

- Python 3.10–3.11
- CUDA-enabled GPU recommended for image generation (CPU fallback available)
- Kaggle Notebooks (Linux, Python 3.10) supported for cloud inference