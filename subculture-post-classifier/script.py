#!/usr/bin/env python3
import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight  # To calculate weights
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import numpy as np
import os
from tqdm.auto import tqdm  # Import tqdm for progress bars
import gc  # Garbage collector
from collections import Counter  # For vocabulary building
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import json  # For saving metrics

# --- Configuration ---
FURRY_TERMS_FILE = "furry_terms.csv"
WEEB_TERMS_FILE = "weeb_terms.csv"
POSTS_FILE = "bluesky_ten_million_english_only.csv"
OUTPUT_DIR = "output/post-classifier"
OUTPUT_METRICS_DIR = "metrics/post-classifier"
OUTPUT_CSV_FILENAME = "subculture_posts_classified_3class.csv"
OUTPUT_METRICS_TRAIN_FILENAME = "training_metrics_3class.txt"
OUTPUT_METRICS_TEST_FILENAME = "test_metrics_3class.txt"
OUTPUT_CM_TRAIN_FILENAME = "training_cm_3class.png"
OUTPUT_CM_TEST_FILENAME = "test_cm_3class.png"

POST_TEXT_COLUMN = "text"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- Secondary Category Thresholding (DYNAMIC) ---
MIN_SECONDARY_ABS_THRESHOLD = 0.10  # Secondary prob must be at least this value
SECONDARY_RELATIVE_FACTOR = (
    0.3  # Secondary prob must be at least this fraction of primary prob
)

# --- Term Filtering & Labeling Configuration ---
SCORE_COLUMN = "combined_score"
SCORE_PERCENTILE_THRESHOLD = 0.85  # Keep top 15% of terms by score
MIN_SUM_SCORE_LABELING = (
    0.7  # Min sum of term scores to activate a category for labeling
)

# --- Model & Tokenization Configuration ---
VOCAB_SIZE = 10000
EMBED_DIM = 64
MAX_SEQ_LENGTH = 100
NUM_FILTERS = 64
FILTER_SIZES = [3, 4, 5]
DROPOUT_RATE = 0.5

# --- Optimizer Configuration ---
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 2e-4

# --- Batching Configuration ---
TRAIN_BATCH_SIZE = 64
PRED_BATCH_SIZE = 512

# --- Special Tokens ---
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

# --- MODIFIED: Label Mapping (3 classes for training) ---
label_map = {"normie": 0, "furry": 1, "weeb": 2}
num_classes = len(label_map)  # Will be 3
id2label = {v: k for k, v in label_map.items()}


# --- Load Stop Words ---
def load_stop_words(file_path):
    """
    Load stop words from a text file with one word per line.
    Returns a set of stop words.
    """
    stop_words = set()
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                word = line.strip()
                if word:
                    stop_words.add(word)  # Assuming stopwords file is already processed
        print(f"Successfully loaded {len(stop_words)} stop words from {file_path}.")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Using empty set.")
    except Exception as e:
        print(f"Error loading stop words: {e}. Using empty set.")
    return stop_words


STOP_WORDS = load_stop_words("./shared/stopwords-en.txt")
print(f"Total stop words being used: {len(STOP_WORDS)}.")


# --- JAX/Flax CNN Model Definition --- (Output layer will adapt to num_classes)
class TextCNN(nn.Module):
    vocab_size: int
    embed_dim: int
    num_classes: int  # This will be 3
    num_filters: int
    filter_sizes: list[int]
    dropout_rate: float

    @nn.compact
    def __call__(self, x, deterministic: bool):
        embed = nn.Embed(num_embeddings=self.vocab_size, features=self.embed_dim)(x)
        embed = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(embed)
        pooled_outputs = []
        for filter_size in self.filter_sizes:
            conv = nn.Conv(
                features=self.num_filters, kernel_size=(filter_size,), padding="VALID"
            )(embed)
            conv = nn.relu(conv)
            pooled = jnp.max(conv, axis=1)
            pooled_outputs.append(pooled)
        concatenated = jnp.concatenate(pooled_outputs, axis=1)
        concatenated = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(
            concatenated
        )
        logits = nn.Dense(features=self.num_classes)(
            concatenated
        )  # Output features = 3
        return logits


# --- Helper Functions ---
def load_terms_with_scores(
    filepath, score_column, score_percentile_threshold, stop_words
):
    """Loads terms and their scores, filters by score percentile, and excludes stop words."""
    print(f"Loading terms and scores from {filepath}...")
    try:
        df = pd.read_csv(filepath, usecols=["term", score_column])
        df[score_column] = pd.to_numeric(df[score_column], errors="coerce")
        df = df.dropna(subset=["term", score_column])
        if df.empty:
            print(
                f"  Warning: No valid terms or scores found in {filepath} before filtering."
            )
            return {}
        if not df[score_column].empty:
            actual_score_threshold = df[score_column].quantile(
                score_percentile_threshold
            )
        else:
            print(
                f"  Warning: No scores available in {score_column} to calculate percentile for {filepath}. Keeping all terms before stopword filter."
            )
            actual_score_threshold = -np.inf
        print(
            f"  Calculated score threshold for {score_percentile_threshold * 100:.0f}th percentile: {actual_score_threshold:.4f}"
        )
        original_count = len(df)
        df_filtered = df[df[score_column] >= actual_score_threshold].copy()
        score_filtered_count = len(df_filtered)
        print(
            f"  Kept {score_filtered_count} of {original_count} terms after applying percentile threshold on '{score_column}'."
        )
        term_score_map = {}
        for _, row in df_filtered.iterrows():
            term = str(row["term"]).lower().strip()
            if term and term not in stop_words:
                term_score_map[term] = row[score_column]
        final_count = len(term_score_map)
        print(f"  Kept {final_count} terms (with scores) after stop word filtering.")
        if final_count == 0:
            print(
                f"  Warning: No terms left for {filepath} after all filtering. Check percentile, scores, and stop words."
            )
        return term_score_map
    except FileNotFoundError:
        print(f"  Error: File not found at {filepath}")
        return {}
    except KeyError:
        print(
            f"  Error: Column 'term' or '{score_column}' not found in {filepath}. Check CSV header."
        )
        return {}
    except Exception as e:
        print(f"  Error loading or processing terms from {filepath}: {e}")
        return {}


def preprocess_text_for_cnn(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Remove URLs, replace with a space
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text, flags=re.MULTILINE)
    # Remove mentions, replace with a space
    text = re.sub(r"@\w+", " ", text)
    # Remove only '#' from hashtags, keep the tag word, replace with the word and a space
    text = re.sub(
        r"#(\w+)",
        r"\1 ",
        text,  # Added a space after \1 to ensure separation
    )
    # Remove remaining punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse multiple spaces and strip leading/trailing whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# MODIFIED: generate_label_sum_score for 3 classes
def generate_label_sum_score(
    post_text, furry_terms_map, weeb_terms_map, min_sum_score_threshold
):
    if not isinstance(post_text, str):
        post_text = ""
    words_in_post = set(re.findall(r"\b\w+\b", post_text.lower()))
    current_furry_sum = sum(
        furry_terms_map[word] for word in words_in_post if word in furry_terms_map
    )
    current_weeb_sum = sum(
        weeb_terms_map[word] for word in words_in_post if word in weeb_terms_map
    )
    is_furry_active = current_furry_sum >= min_sum_score_threshold
    is_weeb_active = current_weeb_sum >= min_sum_score_threshold

    if is_furry_active and is_weeb_active:
        # If both are active, assign to the one with the higher sum for training
        # If sums are equal, prioritize 'weeb' (arbitrary choice, can be changed)
        if current_weeb_sum >= current_furry_sum:
            return label_map["weeb"]
        else:
            return label_map["furry"]
    elif is_furry_active:
        return label_map["furry"]
    elif is_weeb_active:
        return label_map["weeb"]
    else:
        return label_map["normie"]


def find_terms_in_post_optimized(post_text, terms_map):
    if not isinstance(post_text, str) or not terms_map:
        return ""
    words_in_post = set(re.findall(r"\b\w+\b", post_text.lower()))
    term_keys = set(terms_map.keys())
    found_terms = words_in_post.intersection(term_keys)
    return ", ".join(sorted(list(found_terms)))


def build_vocab(texts, vocab_size, min_freq=2):
    print("Building vocabulary...")
    word_counts = Counter()
    for text in tqdm(texts, desc="Counting words", mininterval=10.0):
        words = [word for word in text.split() if word not in STOP_WORDS]
        word_counts.update(words)
    most_common_words = [
        word
        for word, count in word_counts.most_common(vocab_size - 2)
        if count >= min_freq
    ]
    word_to_index = {word: i + 2 for i, word in enumerate(most_common_words)}
    word_to_index[PAD_TOKEN] = 0
    word_to_index[UNK_TOKEN] = 1
    print(f"Vocabulary size: {len(word_to_index)}")
    return word_to_index


def tokenize_and_pad(texts, word_to_index, max_length):
    tokenized_sequences = []
    unk_index = word_to_index[UNK_TOKEN]
    pad_index = word_to_index[PAD_TOKEN]
    for text in texts:
        tokens = [word for word in text.split() if word in word_to_index]
        indices = [word_to_index.get(word, unk_index) for word in tokens]
        if len(indices) < max_length:
            padded_indices = indices + [pad_index] * (max_length - len(indices))
        else:
            padded_indices = indices[:max_length]
        tokenized_sequences.append(padded_indices)
    return np.array(tokenized_sequences, dtype=np.int32)


def weighted_cross_entropy_loss(logits, labels, class_weights):
    # Logits shape: (batch_size, num_classes=3)
    # Labels shape: (batch_size,)
    # Class_weights shape: (num_classes=3,)
    one_hot_labels = jax.nn.one_hot(labels, num_classes=logits.shape[-1])
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32))
    unweighted_loss = -jnp.sum(one_hot_labels * log_probs, axis=-1)
    weights_per_sample = class_weights[labels]
    weighted_loss = unweighted_loss * weights_per_sample
    return weighted_loss


def compute_metrics(logits, labels, class_weights):
    loss = jnp.mean(weighted_cross_entropy_loss(logits, labels, class_weights))
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {"loss": loss, "accuracy": accuracy}
    return metrics


# --- Training Step ---
@jax.jit
def train_step(state, batch_sequences, batch_labels, dropout_rng, class_weights):
    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

    def loss_fn(params):
        logits = state.apply_fn(
            {"params": params},
            batch_sequences,
            deterministic=False,
            rngs={"dropout": dropout_rng},
        )
        loss_per_sample = weighted_cross_entropy_loss(
            logits, batch_labels, class_weights
        )
        loss = jnp.mean(loss_per_sample)
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits, batch_labels, class_weights)
    return state, metrics, new_dropout_rng


# --- Prediction Step ---
@jax.jit
def predict_step(state, batch_sequences):
    logits = state.apply_fn(
        {"params": state.params}, batch_sequences, deterministic=True
    )
    return logits


# --- Metrics Functions ---
def plot_confusion_matrix(
    y_true, y_pred, labels, title="Confusion Matrix", filepath="confusion_matrix.png"
):
    cm = confusion_matrix(
        y_true, y_pred, labels=np.arange(len(labels))
    )  # Ensure labels match indices
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def calculate_metrics(y_true, y_pred, target_names_list):
    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names_list,
        output_dict=True,
        zero_division=0,
    )
    return report


def save_metrics_report(metrics_dict, filename):
    with open(filename, "w") as f:
        f.write("Classification Metrics Report\n")
        f.write("===========================\n\n")
        f.write("Overall Metrics:\n")
        f.write(f"Accuracy: {metrics_dict.get('accuracy', 'N/A'):.4f}\n")
        if "macro avg" in metrics_dict:
            f.write(
                f"Macro Avg F1-Score: {metrics_dict['macro avg'].get('f1-score', 'N/A'):.4f}\n"
            )
        if "weighted avg" in metrics_dict:
            f.write(
                f"Weighted Avg F1-Score: {metrics_dict['weighted avg'].get('f1-score', 'N/A'):.4f}\n\n"
            )
        f.write("Per-Class Metrics:\n")
        # Use the global id2label to iterate in defined order
        for label_idx, class_name_str in id2label.items():
            if class_name_str in metrics_dict:
                metrics = metrics_dict[class_name_str]
                f.write(f"\n{class_name_str.upper()}:\n")
                f.write(f"Precision: {metrics.get('precision', 'N/A'):.4f}\n")
                f.write(f"Recall: {metrics.get('recall', 'N/A'):.4f}\n")
                f.write(f"F1-Score: {metrics.get('f1-score', 'N/A'):.4f}\n")
                f.write(f"Support: {metrics.get('support', 'N/A')}\n")


# --- Main Execution ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_METRICS_DIR, exist_ok=True)
# Define full paths for outputs
output_csv_path = os.path.join(OUTPUT_DIR, OUTPUT_CSV_FILENAME)
output_metrics_train_path = os.path.join(OUTPUT_METRICS_DIR, OUTPUT_METRICS_TRAIN_FILENAME)
output_metrics_test_path = os.path.join(OUTPUT_METRICS_DIR, OUTPUT_METRICS_TEST_FILENAME)
output_cm_train_path = os.path.join(OUTPUT_METRICS_DIR, OUTPUT_CM_TRAIN_FILENAME)
output_cm_test_path = os.path.join(OUTPUT_METRICS_DIR, OUTPUT_CM_TEST_FILENAME)


print("Loading terms...")
furry_terms_path = os.path.join("output", "terms-analysis", FURRY_TERMS_FILE)
weeb_terms_path = os.path.join("output", "terms-analysis", WEEB_TERMS_FILE)
furry_terms_map = load_terms_with_scores(
    furry_terms_path, SCORE_COLUMN, SCORE_PERCENTILE_THRESHOLD, STOP_WORDS
)
weeb_terms_map = load_terms_with_scores(
    weeb_terms_path, SCORE_COLUMN, SCORE_PERCENTILE_THRESHOLD, STOP_WORDS
)
if not furry_terms_map or not weeb_terms_map:
    print("Warning: One or both term maps are empty after filtering.")
    if not furry_terms_map and not weeb_terms_map:
        print("Error: Both term maps are empty after filtering. Exiting.")
        exit()

print("\nLoading and preparing data...")
try:
    full_posts_path = os.path.join("output", "dataset-filter", POSTS_FILE)
    print(f"Attempting to load posts from: {full_posts_path}")
    posts_df = pd.read_csv(full_posts_path, usecols=[POST_TEXT_COLUMN])
    posts_df["original_text"] = posts_df[POST_TEXT_COLUMN]
    print(f"Loaded {len(posts_df)} posts from {full_posts_path}")
    posts_df[POST_TEXT_COLUMN] = posts_df[POST_TEXT_COLUMN].astype(str)

    print("Generating labels based on summed term scores...")
    tqdm.pandas(desc="Generating Labels", mininterval=10.0)
    posts_df["label"] = posts_df[POST_TEXT_COLUMN].progress_apply(
        lambda x: generate_label_sum_score(
            x, furry_terms_map, weeb_terms_map, MIN_SUM_SCORE_LABELING
        )
    )

    print("Preprocessing text for CNN...")
    tqdm.pandas(desc="Preprocessing Text", mininterval=10.0)
    posts_df["processed_text"] = posts_df[POST_TEXT_COLUMN].progress_apply(
        preprocess_text_for_cnn
    )

    print("\nLabel Distribution (for 3-class training):")
    label_counts = posts_df["label"].map(id2label).value_counts()
    print(label_counts)
    y_labels_full = posts_df["label"].values

    print("Splitting data indices for training...")
    indices = np.arange(len(posts_df))
    train_indices, test_indices = train_test_split(
        indices, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_labels_full
    )
    print(f"Training set size (indices): {len(train_indices)}")
    print(f"Test set size (indices): {len(test_indices)}")

    print("Calculating class weights for training data...")
    train_labels = y_labels_full[train_indices]
    unique_train_labels = np.unique(train_labels)
    print(f"Unique labels found in training data: {unique_train_labels}")
    if len(unique_train_labels) > 1:
        weights_calculated = compute_class_weight(
            class_weight="balanced", classes=unique_train_labels, y=train_labels
        )
        class_weights_np = np.ones(
            num_classes, dtype=np.float32
        )  # num_classes is now 3
        for class_label, weight in zip(unique_train_labels, weights_calculated):
            if class_label < num_classes:  # Ensure index is valid for the 3 classes
                class_weights_np[class_label] = weight
    else:
        print("Warning: Only one class found in training data. Using uniform weights.")
        class_weights_np = np.ones(num_classes, dtype=np.float32)
    class_weights_jax = jnp.array(class_weights_np, dtype=jnp.float32)
    print(f"Final Class Weights (for indices 0-{num_classes - 1}): {class_weights_np}")
    weight_map = {
        id2label.get(i, f"Unknown_{i}"): weight
        for i, weight in enumerate(class_weights_np)
    }
    print(f"Weight Map: {weight_map}")
    del train_labels
    gc.collect()

    print("Building vocabulary from training data...")
    train_texts = posts_df.loc[train_indices, "processed_text"].tolist()
    word_to_index = build_vocab(train_texts, VOCAB_SIZE)
    actual_vocab_size = len(word_to_index)
    del train_texts
    gc.collect()

    print("\nInitializing CNN model and optimizer...")
    key = jax.random.PRNGKey(RANDOM_STATE)
    init_key, dropout_init_key = jax.random.split(key)
    model = TextCNN(
        vocab_size=actual_vocab_size,
        embed_dim=EMBED_DIM,
        num_classes=num_classes,  # This is now 3
        num_filters=NUM_FILTERS,
        filter_sizes=FILTER_SIZES,
        dropout_rate=DROPOUT_RATE,
    )
    dummy_sequences = jnp.ones([1, MAX_SEQ_LENGTH], dtype=jnp.int32)
    params = model.init(init_key, dummy_sequences, deterministic=True)["params"]
    optimizer = optax.adamw(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )

    print("\nStarting CNN training (with class weights & regularization)...")
    num_epochs = 5
    dropout_rng = dropout_init_key
    final_epoch_metrics = {}

    for epoch in range(num_epochs):
        perm = np.random.permutation(train_indices)
        epoch_loss, epoch_accuracy, num_batches = 0.0, 0.0, 0
        with tqdm(
            range(0, len(perm), TRAIN_BATCH_SIZE),
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            mininterval=10.0,
        ) as pbar:
            for i_batch in pbar:
                batch_indices = perm[i_batch : i_batch + TRAIN_BATCH_SIZE]
                if len(batch_indices) == 0:
                    continue
                batch_texts = posts_df.loc[batch_indices, "processed_text"].tolist()
                batch_labels_np = y_labels_full[batch_indices]
                batch_sequences_np = tokenize_and_pad(
                    batch_texts, word_to_index, MAX_SEQ_LENGTH
                )
                batch_sequences_jax = jnp.array(batch_sequences_np)
                batch_labels_jax = jnp.array(batch_labels_np)
                state, metrics, dropout_rng = train_step(
                    state,
                    batch_sequences_jax,
                    batch_labels_jax,
                    dropout_rng,
                    class_weights_jax,
                )
                current_loss = metrics["loss"]
                current_accuracy = metrics["accuracy"]
                if jnp.isnan(current_loss):
                    print(
                        f"\nWarning: NaN loss detected at batch {i_batch // TRAIN_BATCH_SIZE} in epoch {epoch + 1}. Stopping training."
                    )
                    epoch_loss = jnp.nan
                    break
                epoch_loss += current_loss
                epoch_accuracy += current_accuracy
                num_batches += 1
                pbar.set_postfix(
                    loss=f"{current_loss:.4f}", acc=f"{current_accuracy:.4f}"
                )
                del (
                    batch_texts,
                    batch_labels_np,
                    batch_sequences_np,
                    batch_sequences_jax,
                    batch_labels_jax,
                )
                gc.collect()
            if jnp.isnan(epoch_loss):
                break
        if num_batches > 0:
            avg_epoch_loss = epoch_loss / num_batches
            avg_epoch_accuracy = epoch_accuracy / num_batches
            print(
                f"Epoch {epoch + 1}/{num_epochs} Summary - Avg Weighted Loss: {avg_epoch_loss:.4f}, Avg Accuracy: {avg_epoch_accuracy:.4f}"
            )
            if epoch == num_epochs - 1:  # Store metrics from the last completed epoch
                final_epoch_metrics["final_training_avg_weighted_loss"] = float(
                    avg_epoch_loss
                )
                final_epoch_metrics["final_training_avg_accuracy"] = float(
                    avg_epoch_accuracy
                )
        else:
            print(f"Epoch {epoch + 1}/{num_epochs} - No batches processed.")
    print("Training finished.")

    # --- Calculate and Save Training & Test Metrics ---
    # (Using the trained model state)
    target_names_for_report = [
        id2label[i] for i in range(num_classes)
    ]  # normie, furry, weeb

    print("\nCalculating training metrics...")
    train_predictions_list = []
    train_true_labels_list = y_labels_full[
        train_indices
    ]  # Get all true labels for training set
    for i in tqdm(
        range(0, len(train_indices), PRED_BATCH_SIZE),
        desc="Getting training predictions",
        mininterval=10.0,
    ):
        batch_indices = train_indices[i : i + PRED_BATCH_SIZE]
        batch_texts = posts_df.loc[batch_indices, "processed_text"].tolist()
        batch_sequences = tokenize_and_pad(batch_texts, word_to_index, MAX_SEQ_LENGTH)
        batch_sequences_jax = jnp.array(batch_sequences)
        batch_logits = predict_step(state, batch_sequences_jax)
        batch_probabilities = jax.nn.softmax(batch_logits.astype(jnp.float32), axis=-1)
        train_predictions_list.extend(np.argmax(np.array(batch_probabilities), axis=1))
        del (
            batch_texts,
            batch_sequences,
            batch_sequences_jax,
            batch_logits,
            batch_probabilities,
        )
        gc.collect()

    train_report_dict = calculate_metrics(
        train_true_labels_list, train_predictions_list, target_names_for_report
    )
    save_metrics_report(train_report_dict, output_metrics_train_path)
    plot_confusion_matrix(
        train_true_labels_list,
        train_predictions_list,
        labels=target_names_for_report,
        title="Training Confusion Matrix",
        filepath=output_cm_train_path,
    )
    del train_predictions_list, train_true_labels_list  # Free memory
    gc.collect()

    print("\nCalculating test metrics...")
    test_predictions_list = []
    test_true_labels_list = y_labels_full[
        test_indices
    ]  # Get all true labels for test set
    for i in tqdm(
        range(0, len(test_indices), PRED_BATCH_SIZE),
        desc="Getting test predictions",
        mininterval=10.0,
    ):
        batch_indices = test_indices[i : i + PRED_BATCH_SIZE]
        batch_texts = posts_df.loc[batch_indices, "processed_text"].tolist()
        batch_sequences = tokenize_and_pad(batch_texts, word_to_index, MAX_SEQ_LENGTH)
        batch_sequences_jax = jnp.array(batch_sequences)
        batch_logits = predict_step(state, batch_sequences_jax)
        batch_probabilities = jax.nn.softmax(batch_logits.astype(jnp.float32), axis=-1)
        test_predictions_list.extend(np.argmax(np.array(batch_probabilities), axis=1))
        del (
            batch_texts,
            batch_sequences,
            batch_sequences_jax,
            batch_logits,
            batch_probabilities,
        )
        gc.collect()

    test_report_dict = calculate_metrics(
        test_true_labels_list, test_predictions_list, target_names_for_report
    )
    save_metrics_report(test_report_dict, output_metrics_test_path)
    plot_confusion_matrix(
        test_true_labels_list,
        test_predictions_list,
        labels=target_names_for_report,
        title="Test Confusion Matrix",
        filepath=output_cm_test_path,
    )
    print("\nMetrics have been saved to specified output paths.")
    print("\nTest Set Metrics Summary:")
    print(f"Accuracy: {test_report_dict.get('accuracy', 'N/A'):.4f}")
    if "macro avg" in test_report_dict:
        print(
            f"Macro Avg F1-Score: {test_report_dict['macro avg'].get('f1-score', 'N/A'):.4f}"
        )
    if "weighted avg" in test_report_dict:
        print(
            f"Weighted Avg F1-Score: {test_report_dict['weighted avg'].get('f1-score', 'N/A'):.4f}"
        )
    del test_predictions_list, test_true_labels_list  # Free memory
    gc.collect()

    # --- Prediction on Full Dataset for CSV output ---
    print("\nRunning predictions on the full dataset for CSV output (CNN)...")
    all_probabilities_list = []
    full_indices = np.arange(len(posts_df))
    for i_batch in tqdm(
        range(0, len(full_indices), PRED_BATCH_SIZE),
        desc="Predicting for CSV",
        mininterval=10.0,
    ):
        batch_indices = full_indices[i_batch : i_batch + PRED_BATCH_SIZE]
        if len(batch_indices) == 0:
            continue
        batch_texts = posts_df.loc[batch_indices, "processed_text"].tolist()
        batch_sequences_np = tokenize_and_pad(
            batch_texts, word_to_index, MAX_SEQ_LENGTH
        )
        batch_sequences_jax = jnp.array(batch_sequences_np)
        batch_logits = predict_step(state, batch_sequences_jax)
        batch_probabilities = jax.nn.softmax(batch_logits.astype(jnp.float32), axis=-1)
        all_probabilities_list.append(np.array(jax.device_get(batch_probabilities)))
        del (
            batch_texts,
            batch_sequences_np,
            batch_sequences_jax,
            batch_logits,
            batch_probabilities,
        )
        gc.collect()
    print("Concatenating batch probabilities for CSV...")
    probabilities_np_full = np.concatenate(
        all_probabilities_list, axis=0
    )  # For full dataset
    del all_probabilities_list
    gc.collect()

    # --- MODIFIED: Step 9 - Determine Categories (for 3-class model output) ---
    print(
        "Determining primary and secondary categories for CSV..."
    )
    primary_categories_final = []
    secondary_categories_final = []

    # Probabilities from the 3-class model
    prob_normie_all = probabilities_np_full[:, label_map["normie"]]
    prob_furry_all = probabilities_np_full[:, label_map["furry"]]
    prob_weeb_all = probabilities_np_full[:, label_map["weeb"]]

    for i in tqdm(
        range(len(probabilities_np_full)),
        desc="Assigning Final Categories for CSV",
        mininterval=10.0,
    ):
        p_normie = prob_normie_all[i]
        p_furry = prob_furry_all[i]
        p_weeb = prob_weeb_all[i]

        # Determine primary category
        # Order of checks matters if probabilities are equal; here normie -> furry -> weeb
        if p_normie >= p_furry and p_normie >= p_weeb:
            assigned_primary = "normie"
            primary_prob_raw = p_normie
            # Potential secondaries are furry or weeb
            if p_furry >= p_weeb:
                potential_secondary_category = "furry"
                potential_secondary_prob = p_furry
            else:
                potential_secondary_category = "weeb"
                potential_secondary_prob = p_weeb
        elif p_furry >= p_normie and p_furry >= p_weeb:
            assigned_primary = "furry"
            primary_prob_raw = p_furry
            # Potential secondary is weeb (normie is not a subculture secondary)
            potential_secondary_category = "weeb"
            potential_secondary_prob = p_weeb
        else:  # Weeb must be highest
            assigned_primary = "weeb"
            primary_prob_raw = p_weeb
            # Potential secondary is furry
            potential_secondary_category = "furry"
            potential_secondary_prob = p_furry

        assigned_secondary = None
        if (
            potential_secondary_category is not None
            and potential_secondary_category != "normie"
        ):  # Don't assign normie as secondary
            if potential_secondary_prob > MIN_SECONDARY_ABS_THRESHOLD:
                if (
                    potential_secondary_prob
                    >= primary_prob_raw * SECONDARY_RELATIVE_FACTOR
                ):
                    if (
                        assigned_primary != potential_secondary_category
                    ):  # Should always be true if primary != normie
                        assigned_secondary = potential_secondary_category

        primary_categories_final.append(assigned_primary)
        secondary_categories_final.append(assigned_secondary)

    del prob_normie_all, prob_furry_all, prob_weeb_all
    gc.collect()

    # 10. Find Matching Terms
    print(
        "Finding specific terms in original posts (using score-percentile-filtered term lists)..."
    )
    tqdm.pandas(desc="Finding Furry Terms", mininterval=10.0)
    posts_df["found_furry_terms"] = posts_df["original_text"].progress_apply(
        lambda x: find_terms_in_post_optimized(x, furry_terms_map)
    )
    tqdm.pandas(desc="Finding Weeb Terms", mininterval=10.0)
    posts_df["found_weeb_terms"] = posts_df["original_text"].progress_apply(
        lambda x: find_terms_in_post_optimized(x, weeb_terms_map)
    )

    # 11. Create Output DataFrame
    print("Creating final output DataFrame...")
    output_df = pd.DataFrame(
        {
            "text": posts_df["original_text"],
            "primary_category": primary_categories_final,
            "secondary_category": secondary_categories_final,
            # Scores for weeb and furry from the 3-class model's direct output
            "weeb_score": probabilities_np_full[:, label_map["weeb"]]
            if "weeb" in label_map
            else np.nan,  # Ensure key exists
            "furry_score": probabilities_np_full[:, label_map["furry"]]
            if "furry" in label_map
            else np.nan,  # Ensure key exists
            "top_weeb_terms": posts_df["found_weeb_terms"],
            "top_furry_terms": posts_df["found_furry_terms"],
        }
    )
    del (
        posts_df,
        primary_categories_final,
        secondary_categories_final,
        probabilities_np_full,
    )
    gc.collect()
    output_df["weeb_score"] = output_df["weeb_score"].round(4)
    output_df["furry_score"] = output_df["furry_score"].round(4)
    output_df["secondary_category"] = output_df["secondary_category"].fillna("None")

    # 12. Save Results and Final Training Metrics (if not saved already)
    print(f"Saving results to {output_csv_path}...")
    output_df.to_csv(output_csv_path, index=False)
    print("Output CSV saved successfully.")

    # Save final epoch metrics (if not already captured and saved during training loop)
    # This part is slightly redundant if training completed all epochs, but good for robustness
    if final_epoch_metrics:  # Check if it has content
        print(
            f"Saving final training epoch metrics to {os.path.join(OUTPUT_METRICS_DIR, 'final_epoch_training_metrics.json')}..."
        )
        try:
            with open(
                os.path.join(OUTPUT_DIR, "final_epoch_training_metrics.json"), "w"
            ) as f:
                json.dump(final_epoch_metrics, f, indent=4)
            print("Final epoch training metrics saved successfully.")
        except Exception as e:
            print(f"Error saving final epoch training metrics: {e}")

    print("\n--- Head of Output CSV ---")
    print(output_df.head())
    print("\n--- Value Counts for Final Primary Category ---")
    print(output_df["primary_category"].value_counts(dropna=False))
    print("\n--- Value Counts for Final Secondary Category ---")
    print(output_df["secondary_category"].value_counts(dropna=False))


except FileNotFoundError as e:
    print(f"Error: File not found - {e}")
    print(
        "Please ensure the file paths in the script configuration section are correct for:"
    )
    print(f"- FURRY_TERMS_FILE ('{furry_terms_path}')")
    print(f"- WEEB_TERMS_FILE ('{weeb_terms_path}')")
    print(f"- POSTS_FILE ('{full_posts_path}')")
except KeyError as e:
    print(f"Error: Missing expected column in a CSV file: {e}. Check headers.")
except Exception as e:
    print(f"An error occurred during execution: {e}")
    import traceback

    traceback.print_exc()
