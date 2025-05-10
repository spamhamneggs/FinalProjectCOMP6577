# Subculture Term Analyzer Script

This script analyzes a text dataset to identify terms specifically associated with predefined subcultures (e.g., "weeb" and "furry") using a combination of topic modeling techniques (LDA and NMF) and heuristic scoring.

## Overview

The script processes a large text corpus to discover and rank terms relevant to specified subcultures. It leverages initial seed terms for each subculture to guide the topic modeling and term identification process. The core idea is to find terms that are specific to, similar in usage to, and contextually relevant within discussions related to these subcultures.

## Key Functionalities

1. **Data Loading & Preprocessing**:
    * Loads text data from a specified CSV file.
    * Performs text cleaning: lowercasing, removal of URLs, punctuation, and standalone numbers.
    * Tokenizes text and removes English stopwords using NLTK.

2. **Feature Extraction**:
    * Utilizes TF-IDF (Term Frequency-Inverse Document Frequency) to vectorize the preprocessed text, creating a numerical representation of the corpus.

3. **Dual Topic Modeling**:
    * For each defined subculture:
        * Filters the dataset to include documents containing at least one of the subculture's seed terms.
        * Independently trains Latent Dirichlet Allocation (LDA) and Non-negative Matrix Factorization (NMF) models on this filtered subset to discover underlying topics.
        * Combines the topic-word distributions from LDA and NMF using an adaptive weighting scheme based on model confidence.

4. **Term Identification & Scoring**:
    * For each term in the vocabulary, it calculates several metrics to determine its relevance to a subculture:
        * **Specificity**: How specific a term is to the subculture's documents compared to the general dataset.
        * **Similarity**: Cosine similarity between the term's topic distribution and the average topic distribution of the subculture's seed terms.
        * **Contextual Relevance**: Overall document prevalence of the term.
        * **Seed Closeness**: Co-occurrence strength (Jaccard similarity) with the subculture's seed terms.
        * **Uniqueness**: How distinctly the term is associated with specific topic clusters within the subculture.
    * Applies dynamic, quantile-based thresholds for similarity and specificity to filter candidate terms.
    * Normalizes the calculated scores and computes a `combined_score` using predefined weights to rank the identified terms.

5. **Model Evaluation**:
    * Evaluates the performance of the LDA and NMF models using metrics such as:
        * Topic Diversity
        * Model Score (e.g., perplexity for LDA)
        * Silhouette Score
        * Average Topic Concentration
        * Topic Distribution Entropy

6. **Saving Results**:
    * Saves the identified terms for each subculture, along with their scores, into separate CSV files (e.g., `weeb_terms.csv`, `furry_terms.csv`).
    * Saves the model evaluation metrics to a JSON file (`model_metrics.json`).
    * All outputs are stored in the `./output/terms-analysis/` directory.

## Dependencies

* **Python 3.x**
* **Python Libraries**:
  * `jax` and `jaxlib`
  * `nltk`
  * `numpy`
  * `pandas`
  * `scipy`
  * `scikit-learn`

* **NLTK Resources**: The script automatically attempts to download `punkt`, `punkt_tab`, and `stopwords` from NLTK.
* **Input Data**: Requires an input CSV file located at `output/dataset-filter/bluesky_ten_million_english_only.csv`. This file is expected to be the output of a preceding data filtering script (like `dataset-filter.py` if used in conjunction). The CSV should contain a column named `text` with the textual data.

## Output

* **`output/terms-analysis/weeb_terms.csv`**: CSV file containing identified terms for the "weeb" subculture, along with their various scores and the final combined score.
* **`output/terms-analysis/furry_terms.csv`**: CSV file containing identified terms for the "furry" subculture, with similar details.
* **`output/terms-analysis/model_metrics.json`**: JSON file containing evaluation metrics for the trained topic models.

The script includes logging that prints progress and information to the console during execution.
