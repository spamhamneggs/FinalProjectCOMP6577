# Bluesky User Classifier

This project implements a machine learning model to classify Bluesky users as Weeb, Furry, or Normie based on their posts.

## Features

- Fine-tunes Unsloth's Gemma 3 1B model for efficient training
- Fetches Bluesky user posts using the atproto package
- Classifies users based on weeb and furry score metrics
- Provides command-line interface for training and classification
- Visualizes training metrics and confusion matrix

## Requirements

```txt
torch
transformers
unsloth
bitsandbytes>=0.41.0
atproto
pandas
numpy
tqdm
matplotlib
seaborn
scikit-learn
```

## Usage

### Training the model

```bash
uv run bluesky_classifier.py train --data path/to/training_data.csv
```

The training data CSV should have at least these columns:

- `text`: The text content to analyze
- `weeb_score`: Score indicating weeb characteristics (float between 0-1)
- `furry_score`: Score indicating furry characteristics (float between 0-1)

The model will be saved to the `bluesky_classifier_model` directory.

### Classifying a user

```bash
uv run bluesky_classifier.py classify --username username.bsky.social --bluesky-user your_username.bsky.social --bluesky-pass your_password
```

Or classify text directly:

```bash
uv run bluesky_classifier.py classify --text "Text to analyze"
```

## Model Architecture

The model fine-tunes Unsloth's optimized Gemma 3 1B model (4-bit quantized) using efficient fine-tuning methods:

- Uses LoRA (Low-Rank Adaptation) for efficient parameter updates
- Adds a classification head for the three categories (Normie, Weeb, Furry)
- Optimizes with AdamW and Cross-Entropy Loss

## Scoring System

The model returns confidence scores for each category:

- Normie score: Likelihood the user is a normie
- Weeb score: Likelihood the user is a weeb
- Furry score: Likelihood the user is a furry

The classification is based on the highest score.

## Example Output

```txt
==================================================
Classification: Weeb
Normie Score: 0.0254
Weeb Score: 0.9546
Furry Score: 0.0200
==================================================
```

## Customization

You can adjust the following parameters in the script:

- `THRESHOLD_WEEB`: Minimum weeb score to classify as weeb (default: 0.4)
- `THRESHOLD_FURRY`: Minimum furry score to classify as furry (default: 0.4)
- `MAX_LENGTH`: Maximum text length for tokenization (default: 512)
- `BATCH_SIZE`: Batch size for training (default: 16)
- `EPOCHS`: Number of training epochs (default: 5)
- `LEARNING_RATE`: Learning rate for optimizer (default: 2e-5)
