# AI-Driven Framework for Automated Social Media Content Generation and Optimization

An end-to-end multimodal pipeline for COVID-19 news synthesis: generating high-quality social media posts (text + images) from news metadata, evaluating their quality, and iteratively improving via user feedback.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Objectives](#objectives)
3. [Hardware Requirements](#hardware-requirements)
4. [Software Requirements](#software-requirements)
5. [Quick Start](#quick-start)
6. [Dataset Format](#dataset-format)
7. [System Modules](#system-modules)
8. [Project Structure](#project-structure)
9. [Contributing](#contributing)
10. [License](#license)

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

**Python**: 3.10–3.11 (project uses Python 3.11 locally; Kaggle/Colab default: 3.10)

**Platform support**:
- Cloud GPU: Kaggle Notebooks (Linux Debian-based, Python 3.10)
- Local: Windows 10/11, Python 3.11

### Core Dependencies

| Library | Purpose |
|---------|---------|
| `transformers` | LLaMA model inference; loading pre-trained language models |
| `sentencepiece` | Tokenization backend for transformer models |
| `accelerate` | Optimized inference for faster transformer computations |
| `scikit-learn` | TF-IDF vectorization, cosine similarity retrieval |
| `numpy` / `pandas` | Dataset preprocessing, metric aggregation, results storage |
| `nltk` / `re` | Text cleaning, tokenization, special character removal |
| `diffusers` | Stable Diffusion pipeline implementation |
| `xformers` | Memory-efficient attention mechanisms |
| `torch` (CUDA) | GPU-accelerated tensor operations |
| `Pillow` | Image file I/O, format conversion |
| `streamlit` | Interactive web UI |
| `sentence-transformers` | SBERT semantic similarity (all-MiniLM-L6-v2) |
| `bert-score` | BERTScore evaluation (RoBERTa-large) |
| `rouge-score` | ROUGE-1/2/L lexical overlap metrics |

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

> **Coming soon:** The main `app.py` entry point has not yet been added to the repository. Once it is available, you will be able to launch the UI with:
>
> ```bash
> streamlit run app.py
> ```
>
> See the [System Modules](#system-modules) section for details on what each component will provide.

---

## Dataset Format

The system uses a dataset of 4,073 COVID-19 news headlines extracted from the InShorts website (03-Mar-2020 to 11-Apr-2020).

Each record includes the following fields:

| Field | Purpose |
|-------|---------|
| `Date` | Publication date; used for temporal filtering and organization |
| `Headline` | News headline; primary text source for summarization and image prompt generation |
| `Description` | Full article body and content |
| `Covid Status` | Classification of article relevance to COVID-19 |
| `Image URL` | URL to the original/ground truth image; enables multimodal alignment evaluation |
| `Source` | News source website reference (e.g., reuters.com, bbc.com) |
| `Sentiment` | Sentiment classification (positive, neutral, negative) of article tone |

Each record maintains a consistent ID throughout text generation, image generation, and evaluation stages, enabling traceability and aggregation of results.

---

## System Modules

> **Coming soon:** The `modules/` directory will contain the full implementation code for each module below, along with module-specific README files and example notebooks. The `docs/` directory will include API references, experiment logs, dataset samples, and step-by-step worked examples. Once these directories are available, they will be linked directly from each module section below.

### Module 1: Data Collection

- **Purpose:** Entry point of the system; organizes raw input data into structured form for downstream processing.
- **Key responsibilities:** Sourcing and ingesting raw metadata (headlines, category, content, image URLs, source URLs); loading CSV files; tracking processed records via unique IDs.

> *Coming soon: Implementation code for data ingestion and ID assignment will be provided in `modules/data_collection/`.*

### Module 2: Data Preprocessing

- **Purpose:** Cleans and standardizes raw data to ensure correct functionality of downstream NLP and image generation components.
- **Processing steps:**
  1. **Text Normalization:** Remove HTML tags, strip whitespace, normalize Unicode.
  2. **Noise Removal:** Remove special characters, handle malformed entries, filter duplicates.
  3. **Data Splitting:** 70% training / 15% validation / 15% test.

> *Coming soon: Data preprocessing scripts and examples will be provided in `modules/data_preprocessing/`. This section will include code examples for text normalization, noise removal, and train/val/test splits.*

### Module 3: Metadata Extraction

- **Purpose:** Adds structured information to unstructured headline and description text via TF-IDF keyword extraction, dimensionality reduction, and topic clustering.
- **Approach:**
  - TF-IDF vectorization (max 5,000 features, unigrams + bigrams, English stop words removed)
  - Truncated SVD (100 components) for dimensionality reduction
  - KMeans clustering (8 clusters, random state=42) for topic grouping

> *Coming soon: TF-IDF and clustering implementation will be provided in `modules/metadata_extraction/`.*

### Module 4: NLP Processing (Text Generation & Evaluation)

- **Purpose:** Summarizes raw news headlines and associated metadata into concise social media posts using an instruction-tuned language model, with multi-metric evaluation.
- **Model:** LLaMA 3.1-8B Instruct (via Hugging Face Transformers)
- **Key parameters:** Max 384 input tokens, max 72 new tokens, beam search (deterministic), temperature 0.0, repetition penalty (no_repeat_ngram_size=3), batch size 8–12
- **Prompt strategies evaluated:** Structured Instruction Prompt, Detailed Guidelines Prompt, Role-Based Expert Prompt

> *Coming soon: Text generation scripts and prompt templates will be provided in `modules/nlp_processing/`. Example notebooks demonstrating each prompt strategy and evaluation results will be included.*

### Module 5: Multimodal Alignment (Image Generation + CLIP)

- **Purpose:** Generates news-style images aligned with textual summaries and validates semantic consistency via CLIP-based evaluation.
- **Image generation:** Stable Diffusion (SD 1.5, SD 2.1, SD Turbo); 512×512, 25 inference steps, guidance scale 7.5
- **Alignment evaluation:** CLIP ViT-B/32 cosine similarity between text and image embeddings; contrastive robustness check with negative sampling

> *Coming soon: Image generation pipeline and CLIP evaluation code will be provided in `modules/multimodal_alignment/`.*

### Module 6: Evaluation

- **Purpose:** Quantitatively evaluates the quality of text summaries and generated images using industry-standard metrics.
- **Text metrics:**
  - **SBERT Cosine Similarity** (sentence-level semantic coherence)
  - **ROUGE-1/2/L** (lexical overlap with source documents)
  - **BERTScore** Precision/Recall/F1 (contextual token-level semantic similarity via RoBERTa-large)
- **Image–text metrics:** CLIP similarity score, negative mean score

> *Coming soon: Evaluation scripts and metric aggregation notebooks will be provided in `modules/evaluation/`. Experiment results and comparison tables will be documented in `docs/experiments/`.*

### Module 7: User Feedback Collector

- **Purpose:** Collects human ratings on generated content quality to refine prompt design and improve model behavior.
- **Feedback options:** Binary (👍 / 👎) + optional detailed scoring (Relevance, Factual correctness, Image realism, Semantic alignment — each 1–5)
- **Logged attributes per entry:** `id`, `summary`, `image_path`, `clip_score`, `user_rating`

> *Coming soon: Feedback collection UI components and logging code will be provided in `modules/feedback/`.*

---

## Project Structure

```
AI-Driven-Framework-for-Automated-Social-Media-Content-Generation-and-Optimization/
├── README.md
├── requirements.txt
├── data/                  # (Coming soon) Example datasets and sample CSV files
├── modules/               # (Coming soon) Implementation code for each module
│   ├── data_collection/
│   ├── data_preprocessing/
│   ├── metadata_extraction/
│   ├── nlp_processing/
│   ├── multimodal_alignment/
│   ├── evaluation/
│   └── feedback/
└── docs/                  # (Coming soon) API references, experiment logs, worked examples
```

---

## Contributing

Contributions are welcome! Here is what is planned and where your help would be most valuable:

- **`modules/`**: Implement any of the 7 modules described above. Each module directory will include a module-specific README describing inputs, outputs, and usage.
- **`data/`**: Add dataset samples, preprocessing scripts, or links to the InShorts COVID-19 dataset.
- **`docs/`**: Add API references, experiment results, worked examples, or deployment documentation.

> *Coming soon: Detailed contribution guidelines, code style guide, and issue templates will be added. In the meantime, please open an issue or pull request describing your proposed change.*

---

## License

> *Coming soon: License information will be added here. Please check back or open an issue if you have questions about usage.*