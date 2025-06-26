# Bluesky User Interest Classifier

This project provides a complete pipeline for classifying Bluesky users based on their post content, primarily identifying interests related to "weeb" (anime/manga) and "furry" subcultures.

The system uses a two-stage approach:

1. **Heuristic Analysis:** A rule-based system calculates "weeb" and "furry" scores for each post based on weighted term lists. This stage is used to generate a large, auto-labeled dataset and to analyze score distributions.
2. **LLM Fine-Tuning (Continuous Scoring):** A small, efficient language model (e.g., `unsloth/Qwen3-0.6B-unsloth-bnb-4bit`) is fine-tuned on the heuristically labeled data. The model learns to replicate and generalize the classification logic, using **continuous scores** (not fixed cutoffs) for nuanced, context-aware classification of new users. **Thresholds for "weeb" and "furry" are independent and configurable, not percentile-based.**

The project is split into two main scripts:

* `heuristic_analyzer.py`: Tool for analyzing score distributions and exploring percentile-based cutoffs for labeling (for research/analysis only).
* `classifier.py`: Main application for preprocessing data, fine-tuning the model (using continuous scoring), evaluating its performance, and classifying live Bluesky users.

## Table of Contents

- [Bluesky User Interest Classifier](#bluesky-user-interest-classifier)
  - [Table of Contents](#table-of-contents)
  - [Project Workflow](#project-workflow)
  - [Thresholds and Scoring](#thresholds-and-scoring)
  - [File Structure](#file-structure)
  - [Usage](#usage)
    - [1. (Optional) Analyze Heuristics (`heuristic_analyzer.py`)](#1-optional-analyze-heuristics-heuristic_analyzerpy)
    - [2. Preprocess Data (`classifier.py preprocess`)](#2-preprocess-data-classifierpy-preprocess)
    - [3. Fine-tune the Model (`classifier.py finetune`)](#3-fine-tune-the-model-classifierpy-finetune)
    - [4. Evaluate the Model (`classifier.py evaluate`)](#4-evaluate-the-model-classifierpy-evaluate)
    - [5. Classify a User (`classifier.py classify`)](#5-classify-a-user-classifierpy-classify)
  - [Configuration](#configuration)

## Project Workflow

A typical end-to-end workflow for this project looks like this:

1. **Prepare Term Databases:** Create `weeb_terms_bertopic.csv` and `furry_terms_bertopic.csv` in `output/terms-analysis-bertopic/`. These files must contain `term` and `combined_score` columns.
2. **(Optional) Analyze Score Distributions:** Use `heuristic_analyzer.py` on a large, representative dataset of posts to analyze the score distributions and explore percentile-based cutoffs. This is for research/analysis only; the classifier uses continuous scoring and does **not** require you to set cutoffs.
3. **Preprocess Raw Data:** Use the `classifier.py preprocess` command to clean and filter a raw CSV dump of Bluesky posts, preparing it for training.
4. **Fine-tune the LLM:** Use the `classifier.py finetune` command, feeding it the preprocessed data. The model will use continuous scoring logic for labeling and training.
5. **Evaluate Performance:** Use the `classifier.py evaluate` command to test your fine-tuned model against a hold-out set, generating a classification report and confusion matrix.
6. **Classify Live Users:** Use the `classifier.py classify` command with your fine-tuned model to fetch a user's posts from Bluesky and determine their primary and secondary interest classifications.

## Thresholds and Scoring

**Classifier thresholds are independent and configurable.**

- The classifier uses **continuous scoring** for both "weeb" and "furry" categories.
- Thresholds for "slight" and "strong" classification are **not percentile-based** and can be set independently for each category.
- **Default threshold values** (can be overridden via command-line arguments):

  - `--min_threshold_weeb`: `0.0031`
  - `--strong_threshold_weeb`: `0.0047`
  - `--min_threshold_furry`: `0.0034`
  - `--strong_threshold_furry`: `0.0051`

- You can override these defaults for all relevant commands (`finetune`, `evaluate`, `classify`).

**Example:**
```bash
uv run classifier.py finetune \
  --data_csv processed_posts.csv \
  --output_dir finetuned_weeb-furry_model_v1 \
  --min_threshold_weeb 0.0032 \
  --strong_threshold_weeb 0.0048 \
  --min_threshold_furry 0.0035 \
  --strong_threshold_furry 0.0052
```

## File Structure

```txt
.
├── classifier.py                 # Main script for training, evaluation, and classification (continuous scoring)
├── heuristic_analyzer.py         # Script for analyzing heuristics and finding cutoffs (for research/analysis)
├── output/
│   ├── terms-analysis-bertopic/
│   │   ├── weeb_terms_bertopic.csv # Weeb term list with scores
│   │   └── furry_terms_bertopic.csv # Furry term list with scores
│   └── heuristic_analysis_results_.../ # Output from heuristic_analyzer.py
│       ├── category_thresholds.json    # Calculated score cutoffs (for reference)
│       └── ...                         # Distribution plots and reports
├── finetuned_model/                # Default output directory for fine-tuned models
│   ├── adapter_config.json
│   ├── README.md
│   └── ...
├── .env                            # For storing Bluesky credentials (optional)
└── README.md                       # This file
```

## Usage

Below are detailed instructions for each script and command.

### 1. (Optional) Analyze Heuristics (`heuristic_analyzer.py`)

This script helps you explore **score distributions** and percentile-based cutoffs for labeling by analyzing a large dataset. This is for research/analysis only; the classifier uses continuous scoring and does **not** require you to set cutoffs.

**Example:**
Analyze score percentiles for both categories.

```bash
uv run heuristic_analyzer.py \
  --data_csv path/to/large_dataset.csv \
  --output_dir heuristic_analysis_results \
  --weeb-normie-percentile 80 \
  --weeb-strong-percentile 95 \
  --furry-normie-percentile 80 \
  --furry-strong-percentile 95
```

This will create a directory under `heuristic_analysis_results/` containing plots, reports, and a `category_thresholds.json` file. The JSON file contains the absolute score values at the specified percentiles (for reference).

### 2. Preprocess Data (`classifier.py preprocess`)

This command cleans a raw data CSV, keeping only valid posts and saving them to a new file.

**Example:**

```bash
uv run classifier.py preprocess \
  --input raw_bluesky_data.csv \
  --output processed_posts.csv
```

### 3. Fine-tune the Model (`classifier.py finetune`)

This is the core training step. It uses the heuristic rules with **continuous scoring** to generate prompt/response pairs on the fly and fine-tune the LLM. **You do not need to provide cutoff values.**  
**Thresholds for "weeb" and "furry" are independent and can be set via command-line arguments.**

**Example:**

```bash
uv run classifier.py finetune \
  --data_csv processed_posts.csv \
  --output_dir finetuned_weeb-furry_model_v1 \
  --base_model_name "unsloth/Qwen3-0.6B-unsloth-bnb-4bit" \
  --epochs 3 \
  --batch_size 8 \
  --learning_rate 1e-5 \
  --max_training_samples 400000 \
  --test_size 10000 \
  --min_threshold_weeb 0.0031 \
  --strong_threshold_weeb 0.0047 \
  --min_threshold_furry 0.0034 \
  --strong_threshold_furry 0.0051
```

This will train the model and save the LoRA adapters to the `finetuned_weeb-furry_model_v1` directory. It will also automatically run an evaluation at the end.

### 4. Evaluate the Model (`classifier.py evaluate`)

If you want to re-evaluate a trained model on a different dataset, use this command.  
**Thresholds are configurable as above.**

**Example:**

```bash
uv run classifier.py evaluate \
  --model_path finetuned_weeb-furry_model_v1 \
  --eval_data_csv path/to/evaluation_data.csv \
  --metrics_output_dir evaluation_results \
  --min_threshold_weeb 0.0031 \
  --strong_threshold_weeb 0.0047 \
  --min_threshold_furry 0.0034 \
  --strong_threshold_furry 0.0051
```

This generates a `comprehensive_report.txt` and `confusion_matrix_combined.png` in the specified output directory.

### 5. Classify a User (`classifier.py classify`)

Use your fine-tuned model to classify a live Bluesky user.  
**Thresholds are configurable as above.**

**Prerequisites:**

* You must have a Bluesky account.
* Provide your credentials either via a `.env` file (see [Configuration](#configuration)) or command-line arguments.

**Example:**

```bash
uv run classifier.py classify \
  --model_path finetuned_weeb-furry_model_v1 \
  --username pfau.bsky.social \
  --bluesky_user "your-handle.bsky.social" \
  --bluesky_pass "your-app-password" \
  --min_threshold_weeb 0.0031 \
  --strong_threshold_weeb 0.0047 \
  --min_threshold_furry 0.0034 \
  --strong_threshold_furry 0.0051
```

The script will fetch the user's recent posts, classify each one using the LLM (with continuous scoring), and determine a final classification based on the most confident prediction.

## Configuration

For the `classify` command, you can store your Bluesky credentials in a `.env` file in the project's root directory to avoid passing them on the command line.

**`.env` file:**

```env
BLUESKY_USERNAME="your-handle.bsky.social"
BLUESKY_PASSWORD="your-app-password-xxxx-xxxx-xxxx"
```
