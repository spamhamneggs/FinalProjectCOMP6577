# Terms Analysis with BERTopic

Extract and evaluate subculture-specific terms (e.g. “weeb” and “furry”) from a large text corpus using a hybrid BERTopic-based pipeline. This tool combines topic modeling, semantic expansion, co-occurrence analysis, n-gram extraction and clustering diagnostics to produce ranked term lists and model metrics.

## Features

- Preprocess raw text (cleanup, tokenization, stop-word removal)  
- Embed documents in chunks using a SentenceTransformer  
- UMAP dimensionality reduction + HDBSCAN / fallback clustering  
- BERTopic initial fit + incremental partial_fit + final refinement  
- Extract candidate terms per subculture via:
  - Cleaned BERTopic topic terms  
  - Semantic similarity expansion  
  - Co-occurrence within context windows  
  - Relevant n-gram mining  
- Compute multi-factor scores (specificity, seed-closeness, contextual relevance, etc.)  
- Normalize, weight and combine feature scores into a single `combined_score`  
- Save per-subculture term lists (`weeb_terms_bertopic.csv`, `furry_terms_bertopic.csv`)  
- Evaluate topic coherence & clustering via Silhouette, Davies–Bouldin, topic diversity  
- Output a JSON metrics file (`model_metrics_bertopic.json`)

## Requirements

- Python ≥ 3.12  
- GPU recommended for embedding/clustering but CPU fallback supported  

All Python dependencies are listed in `pyproject.toml`:

```toml
[project]
name = "terms-analysis-bertopic"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
  "bertopic>=0.17.0",
  "nltk>=3.9.1",
  "numpy>=1.26.4",
  "pandas>=2.2.3",
  "scikit-learn>=1.6.1",
  "scipy>=1.13.1",
  "sentence-transformers>=4.1.0",
  "torch>=2.7.0",
]
```

### Optional torch index configuration (via [tool.uv.sources] in `pyproject.toml`)

```toml
[tool.uv.sources]
torch = [
  { index = "pytorch-cpu", marker = "sys_platform == 'darwin'" },
  { index = "pytorch-cu128", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]
```

## Installation (using uv)

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd terms-analysis-bertopic

# 2. Sync dependencies (creates and uses a virtualenv automatically)
uv sync
```

## Usage

All commands should be run via `uv run` to ensure the correct environment:

```bash
uv run python terms_analysis_bertopic.py \
  --input path/to/posts.csv \
  [--n_topics_hint N] \
  [--max_features_tfidf M] \
  [--seed S]
```

### Arguments

- `--input`  
  Path to a CSV file containing at least a `text` column.  
- `--n_topics_hint` (default 50)  
  Initial hint for number of topics in MiniBatchKMeans.  
- `--max_features_tfidf` (default 10000)  
  Maximum vocabulary size for TF-IDF feature extraction.  
- `--seed` (default 42)  
  Random seed for reproducibility.

### Example

```bash
uv run python terms_analysis_bertopic.py \
  --input ./data/bluesky_posts.csv \
  --n_topics_hint 80 \
  --max_features_tfidf 15000 \
  --seed 2025
```

## Output

1. `output/terms-analysis-bertopic/weeb_terms_bertopic.csv`  
2. `output/terms-analysis-bertopic/furry_terms_bertopic.csv`  
   • Columns:  
   - `term`, `specificity`, `similarity`, `contextual_relevance`,  
   - `seed_closeness`, `subculture_relevance`, `uniqueness`, `combined_score`  

3. `metrics/terms-analysis-bertopic/model_metrics_bertopic.json`  
   • Contains:
   - `n_topics`  
   - Silhouette & Davies–Bouldin scores for BERTopic & final clustering  
   - Topic diversity metrics  

## Directory Structure

```
.
├── terms_analysis_bertopic.py     # Main analysis script
├── pyproject.toml                 # UV config and dependencies
├── thirdparty/
│   └── stopwords-custom.txt       # Custom stop-word list (optional)
├── output/
│   └── terms-analysis-bertopic/
│       ├── weeb_terms_bertopic.csv
│       └── furry_terms_bertopic.csv
└── metrics/
    └── terms-analysis-bertopic/
        └── model_metrics_bertopic.json
```

## Customization

- **Stop words**: place your own list in `thirdparty/stopwords-custom.txt`.  
- **Seed terms**: adjust `weeb_seed_terms` and `furry_seed_terms` lists in the script.  
- **Clustering parameters**: tune UMAP, HDBSCAN, fallback thresholds in `main()`.  
- **Scoring weights**: modify the `weights` dict in `identify_subculture_terms_bertopic()`.
