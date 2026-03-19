# AI-Driven Framework for Automated Social Media Content Generation and Optimization

An end-to-end multimodal content generation and evaluation system for COVID-19 news synthesis, combining large language models, stable diffusion image generation, and semantic evaluation metrics.

---

## 1. Objectives

- **Metadata-driven text generation**: Develop a pipeline using LLaMA 3.1-8B Instruct combined with TF-IDF keyword extraction to automate high-quality social media post creation from news articles.
- **Semantic evaluation**: Deploy an evaluation module integrating SBERT (all-MiniLM-L6-v2), BERTScore (RoBERTa-large), and ROUGE-1/2/L to systematically validate quality, relevance, and fidelity of generated text.
- **Multimodal image generation**: Use Stable Diffusion to create images aligned with generated text, evaluated with CLIP embeddings for semantic consistency.
- **Optimization loop**: Create a feedback-driven loop incorporating user ratings and behavior data to improve cross-modal alignment and content quality.

---

## 2. Key Hardware and Software Requirements

### Hardware

| Component | Minimum | Implemented |
|-----------|---------|-------------|
| CPU | Intel i5 8th Gen / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 |
| RAM | 8 GB | 16 GB (Local); 30 GB (Kaggle) |
| GPU | Not required | NVIDIA T4 16 GB (Kaggle) |
| Storage | 10 GB free | 50+ GB (Local + Cloud) |
| OS | Windows 10/11 | Windows 11 (Local) |

### Software

| Category | Library / Tool | Purpose |
|----------|---------------|---------|
| Platform | Python 3.10–3.11, Kaggle Notebooks | Execution environment |
| Text Generation | `transformers` | LLaMA model inference |
| Text Generation | `sentencepiece` | Tokenization backend |
| Text Generation | `accelerate` | Optimized GPU inference |
| Text Evaluation | `rouge-score` | Lexical overlap (ROUGE-1/2/L) |
| Text Evaluation | `bert-score` | Semantic similarity (RoBERTa-large) |
| Text Evaluation | `sentence-transformers` | SBERT sentence-level similarity |
| Text Processing | `scikit-learn` | TF-IDF vectorization, cosine similarity |
| Text Processing | `numpy`, `pandas` | Dataset preprocessing, metric aggregation |
| Text Processing | `nltk` | Text cleaning, tokenization |
| Image Generation | `diffusers` | Stable Diffusion pipeline |
| Image Generation | `xformers` | Memory-efficient attention |
| Image Generation | `torch` | GPU-accelerated tensor operations |
| Image Generation | `Pillow` | Image I/O and format conversion |
| User Interface | `streamlit` | Interactive web UI |

---

## 3. Implementation Overview

The system comprises seven modules:

1. **Data Collection**: Ingests 4,073 COVID-19 news headlines (Mar–Apr 2020) from InShorts, structured with fields: Date, Headline, Description, Covid Status, Image URL, Source, and Sentiment.

2. **Data Preprocessing**: Normalises raw text (HTML stripping, Unicode normalisation, deduplication) and splits data into 70% train / 15% validation / 15% test.

3. **Metadata Extraction**: Applies TF-IDF vectorisation (5,000 features, unigrams + bigrams), Truncated SVD (100 components), and KMeans clustering (8 clusters) to assign topic labels and keywords to each article.

4. **NLP Processing (Text Generation)**: LLaMA 3.1-8B Instruct generates summaries from structured prompts combining headlines, descriptions, and extracted metadata. Three prompt strategies (Structured Instruction, Detailed Guidelines, Role-Based Expert) are benchmarked.

5. **Multimodal Alignment**: Stable Diffusion (SD 1.5, SD 2.1, SD Turbo) generates 512×512 news-style images from combined text+keyword prompts. CLIP ViT-B/32 computes cosine similarity between text and image embeddings, with negative sampling for robustness.

6. **Evaluation**: Comprehensive assessment using SBERT cosine similarity, ROUGE-1/2/L, and BERTScore (Precision/Recall/F1) for text quality; CLIP alignment score for image-text consistency.

7. **User Feedback Collector**: Collects binary (👍/👎) and optional detailed ratings (relevance, factual correctness, image realism, semantic alignment, each 1–5) to guide iterative prompt refinement.

---

## 4. Results Summary

- **Text quality**: The Role-Based Expert prompt strategy achieved the highest ROUGE-L and BERTScore F1, outperforming the baseline Structured Instruction prompt across all three metrics.
- **Image-text alignment**: CLIP similarity scores consistently exceeded the negative-sample baseline, demonstrating discriminative semantic alignment between generated summaries and Stable Diffusion images.
- **Efficiency**: SD Turbo (4–8 inference steps, 4 GB VRAM) provided the best throughput while maintaining acceptable CLIP scores, making it suitable for production deployment.
- **Feedback loop**: Iterative prompt refinement guided by user ratings improved average SBERT similarity by reducing hallucinated or off-topic sentences.

---

## 5. Conclusion

This project demonstrates a fully automated multimodal pipeline that transforms raw COVID-19 news data into optimised social media content. By combining instruction-tuned LLMs with stable diffusion image generation and a rich suite of evaluation metrics, the system achieves both quantitative quality improvements and human-verified alignment. The modular architecture supports easy extension to other news domains and language models.