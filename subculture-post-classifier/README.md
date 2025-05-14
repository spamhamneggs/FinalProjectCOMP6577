# Subculture Post Classifier

This script classifies text posts into subcultural categories ("normie", "furry", "weeb") using a Text Convolutional Neural Network (CNN) built with JAX and Flax. It leverages pre-identified subculture-specific term lists for initial data labeling and feature enrichment.

## Overview

The script performs end-to-end text classification:

1. **Loads Data**: Ingests subculture-specific term lists (previously generated, e.g., by a term identification script) and a main dataset of text posts.
2. **Initial Labeling**: Assigns preliminary labels to posts based on the presence and scores of terms from the loaded lists.
3. **Text Preprocessing**: Cleans and prepares text data for the CNN model.
4. **Model Training**: Trains a JAX/Flax-based TextCNN model using the initially labeled data, incorporating class weights to handle imbalances.
5. **Prediction & Refinement**: Predicts subculture probabilities for all posts using the trained model. It then assigns a primary category and potentially a secondary category using a dynamic thresholding logic to better capture nuanced classifications.
6. **Output Generation**: Saves the classified posts along with their assigned categories, model scores, and a list of identified subculture terms found in each post.

## Key Features

* **CNN for Text Classification**: Employs a Convolutional Neural Network with multiple filter sizes for feature extraction from text.
* **JAX/Flax Implementation**: Utilizes JAX for numerical computation and Flax for neural network modeling.
* **Term-Based Initial Labeling**: Uses external lists of scored "weeb" and "furry" terms to heuristically label data for training.
* **Dynamic Secondary Category Assignment**: After primary classification by the CNN, a secondary subculture category can be assigned based on configurable absolute and relative probability thresholds.
* **Weighted Loss**: Addresses class imbalance during training using weighted cross-entropy loss.
* **Configurable Parameters**: Offers various settings for file paths, model architecture (vocabulary size, embedding dimension, sequence length, filter sizes, dropout), optimizer, batching, and thresholding.
* **Stop Word Filtering**: Uses a custom stop word list for term loading and vocabulary building.

## Dependencies

* **Python 3.x**
* **Python Libraries**:
  * `pandas`
  * `scikit-learn`
  * `jax`
  * `jaxlib`
  * `flax`
  * `optax`
  * `numpy`
  * `tqdm`
    You can typically install these using pip:

    ```bash
    pip install pandas scikit-learn jax jaxlib flax optax numpy tqdm
    ```

* **Input Files**:
  * **Term Lists**:
    * `output/terms-analysis/furry_terms.csv` (default: `FURRY_TERMS_FILE`)
    * `output/terms-analysis/weeb_terms.csv` (default: `WEEB_TERMS_FILE`)
        These files should contain at least 'term' and a score column (default: 'combined_score'). They are expected to be outputs from a preceding term identification script.
  * **Posts Data**:
    * `output/dataset-filter/bluesky_ten_million_english_only.csv` (default: `POSTS_FILE`)
        This CSV should contain a column with the text of the posts (default: 'text'). It is expected to be the output of a data filtering script.
  * **Stopwords List**:
    * `./subculture-post-classifier/stopwords-en.txt`
        A plain text file with one stopword per line.

## Configuration

Key parameters can be adjusted at the top of the `script.py` file:

* `FURRY_TERMS_FILE`, `WEEB_TERMS_FILE`, `POSTS_FILE`: Paths to input data.
* `OUTPUT_CSV_FILE`: Path for the output classification results.
* `SCORE_COLUMN`, `SCORE_PERCENTILE_THRESHOLD`, `MIN_SUM_SCORE_LABELING`: Parameters for term filtering and initial labeling.
* `MIN_SECONDARY_ABS_THRESHOLD`, `SECONDARY_RELATIVE_FACTOR`: Thresholds for assigning secondary categories.
* `VOCAB_SIZE`, `EMBED_DIM`, `MAX_SEQ_LENGTH`, `NUM_FILTERS`, `FILTER_SIZES`, `DROPOUT_RATE`: CNN model architecture parameters.
* `LEARNING_RATE`, `WEIGHT_DECAY`: Optimizer parameters.
* `TRAIN_BATCH_SIZE`, `PRED_BATCH_SIZE`: Batch sizes for training and prediction.

## Output

The script generates a CSV file (default: `output/subculture-classifier/subculture_posts_classified.csv`) with the following columns:

* `text`: The original text of the post.
* `primary_category`: The main subculture category assigned by the model and refinement logic (e.g., "normie", "furry", "weeb").
* `secondary_category`: A potential secondary subculture category (e.g., "weeb" if primary was "furry", or "None").
* `weeb_score`: The raw probability score from the CNN model for the "weeb" class.
* `furry_score`: The raw probability score from the CNN model for the "furry" class.
* `top_weeb_terms`: A comma-separated string of pre-identified "weeb" terms found in the post.
* `top_furry_terms`: A comma-separated string of pre-identified "furry" terms found in the post.

The script also prints the head of the output CSV and value counts for primary and secondary categories to the console upon completion.
