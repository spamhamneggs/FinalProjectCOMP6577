# Bluesky User Interest Classifier

This project provides a complete pipeline for classifying Bluesky users based on their post content, primarily identifying interests related to "weeb" (anime/manga) and "furry" subcultures.

The system uses a two-stage approach:

1. **Heuristic Analysis:** A rule-based system calculates "weeb" and "furry" scores for each post based on weighted term lists. This stage is used to generate a large, auto-labeled dataset.
2. **LLM Fine-Tuning:** A small, efficient language model (e.g., `unsloth/Qwen3-0.6B-unsloth-bnb-4bit`) is fine-tuned on the heuristically labeled data. This model learns to replicate and generalize the classification logic, allowing for nuanced, context-aware classification of new users without relying directly on the heuristic rules at inference time.

The project is split into two main scripts:

* `heuristic_analyzer.py`: A tool for analyzing score distributions and determining the optimal score cutoffs for labeling (e.g., what score constitutes a "Slight Weeb" vs. a "Strong Weeb").
* `classifier.py`: The main application for preprocessing data, fine-tuning the model with fixed cutoffs, evaluating its performance, and classifying live Bluesky users.

## Table of Contents

- [Bluesky User Interest Classifier](#bluesky-user-interest-classifier)
  - [Table of Contents](#table-of-contents)
  - [Project Workflow](#project-workflow)
  - [File Structure](#file-structure)
  - [Installation (using uv)](#installation-using-uv)
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
2. **Find Optimal Cutoffs:** Use `heuristic_analyzer.py` on a large, representative dataset of posts to analyze the score distributions and find the ideal score values that correspond to different classification strengths (e.g., the 80th percentile of weeb scores). This will output a `category_thresholds.json` file.
3. **Preprocess Raw Data:** Use the `classifier.py preprocess` command to clean and filter a raw CSV dump of Bluesky posts, preparing it for training.
4. **Fine-tune the LLM:** Use the `classifier.py finetune` command, feeding it the preprocessed data and the **fixed cutoff values** you determined in step 2. This will train and save a new model adapter.
5. **Evaluate Performance:** Use the `classifier.py evaluate` command to test your fine-tuned model against a hold-out set, generating a classification report and confusion matrix.
6. **Classify Live Users:** Use the `classifier.py classify` command with your fine-tuned model to fetch a user's posts from Bluesky and determine their primary and secondary interest classifications.

## File Structure

```txt
.
├── classifier.py                 # Main script for training, evaluation, and classification
├── heuristic_analyzer.py         # Script for analyzing heuristics and finding cutoffs
├── output/
│   ├── terms-analysis-bertopic/
│   │   ├── weeb_terms_bertopic.csv # Weeb term list with scores
│   │   └── furry_terms_bertopic.csv # Furry term list with scores
│   └── heuristic_analysis_results_.../ # Output from heuristic_analyzer.py
│       ├── category_thresholds.json    # Calculated score cutoffs
│       └── ...                         # Distribution plots and reports
├── finetuned_model/                # Default output directory for fine-tuned models
│   ├── adapter_config.json
│   ├── README.md
│   └── ...
├── .env                            # For storing Bluesky credentials (optional)
└── README.md                       # This file
```

## Installation (using uv)

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management.

1. Clone the repository:

    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2. Install the required Python packages using `uv`:

    ```bash
    # Install base dependencies
    uv sync
    ```

## Usage

Below are detailed instructions for each script and command.

### 1. (Optional) Analyze Heuristics (`heuristic_analyzer.py`)

This script helps you find the best **absolute score cutoffs** to use for training by analyzing the score distribution across a large dataset and calculating values based on percentiles.

**Example:**
Find the score values at the 80th percentile for "Normie" and 95th for "Strong" for both categories.

```bash
uv run heuristic_analyzer.py \
  --data_csv path/to/large_dataset.csv \
  --output_dir heuristic_analysis_results \
  --weeb-normie-percentile 80 \
  --weeb-strong-percentile 95 \
  --furry-normie-percentile 80 \
  --furry-strong-percentile 95
```

This will create a directory under `heuristic_analysis_results/` containing plots, reports, and a `category_thresholds.json` file. The JSON file will contain the absolute score values you should use in the `finetune` step.

**Example `category_thresholds.json` output:**

```json
{
  "weeb_normie_cutoff": 0.0115,
  "weeb_strong_cutoff": 0.0521,
  "furry_normie_cutoff": 0.0145,
  "furry_strong_cutoff": 0.0633,
  "weeb_normie_percentile": 77,
  "weeb_strong_percentile": 87,
  "furry_normie_percentile": 78,
  "furry_strong_percentile": 88
}
}
```

### 2. Preprocess Data (`classifier.py preprocess`)

This command cleans a raw data CSV, keeping only valid posts and saving them to a new file.

**Example:**

```bash
uv run classifier.py preprocess \
  --input raw_bluesky_data.csv \
  --output processed_posts.csv
```

### 3. Fine-tune the Model (`classifier.py finetune`)

This is the core training step. It uses the heuristic rules with **fixed, absolute cutoffs** to generate prompt/response pairs on the fly and fine-tune the LLM.

**Example:**
Use the cutoff values discovered with `heuristic_analyzer.py`.

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
  --weeb-normie-cutoff 0.0115 \
  --weeb-strong-cutoff 0.0521 \
  --furry-normie-cutoff 0.0145 \
  --furry-strong-cutoff 0.0633
```

This will train the model and save the LoRA adapters to the `finetuned_weeb-furry_model_v1` directory. It will also automatically run an evaluation at the end.

### 4. Evaluate the Model (`classifier.py evaluate`)

If you want to re-evaluate a trained model on a different dataset, use this command.

**Example:**

```bash
uv run classifier.py evaluate \
  --model_path finetuned_weeb-furry_model_v1 \
  --eval_data_csv path/to/evaluation_data.csv \
  --metrics_output_dir evaluation_results \
  --weeb-normie-cutoff 0.0115 \
  --weeb-strong-cutoff 0.0521 \
  --furry-normie-cutoff 0.0145 \
  --furry-strong-cutoff 0.0633
```

This generates a `classification_report.txt` and `confusion_matrix_combined.png` in the specified output directory.

### 5. Classify a User (`classifier.py classify`)

Use your fine-tuned model to classify a live Bluesky user.

**Prerequisites:**

* You must have a Bluesky account.
* Provide your credentials either via a `.env` file (see [Configuration](#configuration)) or command-line arguments.

**Example:**

```bash
uv run classifier.py classify \
  --model_path finetuned_weeb-furry_model_v1 \
  --username pfau.bsky.social \
  --bluesky_user "your-handle.bsky.social" \
  --bluesky_pass "your-app-password"
```

The script will fetch the user's recent posts, classify each one using the LLM, and determine a final classification based on the most frequent prediction.

**Example Output:**

```txt
==================================================
LLM-Only Classification Results for @pfau.bsky.social
==================================================
Primary Classification: Weeb
Secondary Classification: Slight Furry
Model Post Classifications: 250 posts analyzed
Prediction distribution:
  Weeb-Slight Furry: 120 (48.0%)
  Weeb-None: 80 (32.0%)
  Normie-None: 50 (20.0%)
==================================================
```

## Configuration

For the `classify` command, you can store your Bluesky credentials in a `.env` file in the project's root directory to avoid passing them on the command line.

**`.env` file:**

```env
BLUESKY_USERNAME="your-handle.bsky.social"
BLUESKY_PASSWORD="your-app-password-xxxx-xxxx-xxxx"
```
