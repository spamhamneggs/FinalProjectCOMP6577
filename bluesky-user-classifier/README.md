# Bluesky User Classifier

This tool classifies Bluesky users as Weeb, Furry, or Normie based on their post history. It uses a fine-tuned language model to analyze post content and identify patterns associated with different user types.

## Features

- Fine-tunes the Gemma 3 1B-IT model with 4-bit quantization for efficient classification
- Uses a scoring system based on weighted term databases
- Fetches user posts directly from Bluesky using the ATProto API
- Provides classification with confidence scores for each category

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/bluesky-classifier
cd bluesky-classifier
```

2. Install dependencies:

```bash
pip install torch unsloth transformers trl pandas numpy tqdm atproto
```

3. Place the term databases in the root directory:
   - `weeb_terms.csv`
   - `furry_terms.csv`

## Usage

The tool provides three main commands:

### 1. Preprocess Bluesky Data

Before training, you need to preprocess the Bluesky data:

```bash
python BlueskyUserClassifier.py preprocess --input bluesky_data.csv --output processed_data.csv
```

### 2. Fine-tune the Model

Train the model using the preprocessed data:

```bash
python BlueskyUserClassifier.py finetune --data processed_data.csv --output_dir finetuned_model --epochs 3
```

Additional options:

- `--batch_size`: Training batch size (default: 8)
- `--learning_rate`: Learning rate (default: 2e-4)

### 3. Classify a User

Once the model is trained, you can classify any Bluesky user:

```bash
python BlueskyUserClassifier.py classify --model finetuned_model --username target_user.bsky.social --bluesky_user your_username.bsky.social --bluesky_pass your_password
```

## How It Works

1. **Term Database**: The system uses weighted term databases for both Weeb and Furry categories that contain relevant terms and their scores.

2. **Fine-tuning**: The model is fine-tuned on a dataset of Bluesky posts with labeled classifications.

3. **Classification Process**:
   - Fetches user posts from Bluesky
   - Calculates initial scores using term databases
   - Refines classification using the fine-tuned model
   - Combines scores with proper weighting
   - Determines the final classification

4. **Score Interpretation**:
   - High score in one category (>0.6): Clear classification
   - Moderate score (0.3-0.6): "Slight" classification
   - Low scores in all categories: Normie

## Example Output

```txt
=================================================
Classification Results for @example.bsky.social
=================================================
Classification: Weeb
Weeb Score: 0.725
Furry Score: 0.152
Normie Score: 0.275
=================================================
```

## Notes

- The accuracy of classification depends on the quality of the term databases and the diversity of the training data.
- You need a valid Bluesky account to fetch user posts.
- The model is fine-tuned using a small subset of features extracted from the original CSV files to keep the process efficient.
