#!/usr/bin/env python3
import json
import os
import re
from functools import lru_cache

import jax.numpy as jnp
import jax.random as random
import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from scipy.stats import entropy
from sklearn.decomposition import LatentDirichletAllocation, MiniBatchNMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Download necessary NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# Create output directories
os.makedirs("output/terms-analysis", exist_ok=True)
os.makedirs("metrics/terms-analysis", exist_ok=True)

# Define lists of known weeb and furry terms for initial seeding
weeb_seed_terms = [
    "anime",
    "manga",
    "otaku",
    "waifu",
    "kawaii",
    "isekai",
    "nani",
    "baka",
    "cosplay",
    "senpai",
    "moe",
    "tsundere",
    "weeb",
]

furry_seed_terms = [
    "furry",
    "fursona",
    "anthro",
    "floof",
    "fursuit",
    "paws",
    "uwu",
    "owo",
    "furcon",
    "furries",
    "fursuits",
    "pawsome",
    "tailwag",
]


# Load and preprocess the data
def load_data(file_path):
    print("Loading and preprocessing data...")
    df = pd.read_csv(file_path)

    # Remove rows with empty text
    df = df[df["text"].notna()]
    return df


def get_term_docs(term_idx, X):
    """Get documents containing a specific term (cached)"""
    return set(X[:, term_idx].nonzero()[0])


# Text preprocessing
@lru_cache(maxsize=10000)
def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text, flags=re.MULTILINE)

    # Remove all punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # Remove all independent numbers
    text = re.sub(r"\b\d+\b", "", text)

    return text


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


# Load stopwords
STOP_WORDS = load_stop_words("./shared/stopwords-en.txt")


# Tokenization function
def tokenize_text(text):
    if not isinstance(text, str):
        return []
    tokens = word_tokenize(text)
    # Don't filter out short words as they might be relevant terms
    tokens = [token for token in tokens if token not in STOP_WORDS]
    return tokens


# Feature extraction
def extract_features(texts, max_features=5000, min_df=20, max_df=0.90):
    print("Extracting features...")
    # Use TF-IDF instead of raw counts for better NMF performance
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        tokenizer=tokenize_text,
        preprocessor=preprocess_text,
        lowercase=False,
        ngram_range=(1, 2),
        sublinear_tf=True,  # Apply sublinear scaling to term frequencies
        norm="l2",  # Use L2 normalization
        use_idf=True,
        smooth_idf=True,
        strip_accents="unicode",
    )

    print(f"Using max_df={max_df}, min_df={min_df}")
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    # Print statistics about feature extraction
    n_samples, n_features = X.shape
    print(f"Extracted {n_features} features from {n_samples} documents")

    return X, vectorizer, feature_names


def combine_topic_matrices(lda_matrix, nmf_matrix, alpha=0.5):
    """Combine LDA and NMF topic-word matrices with adaptive weighting"""
    # Add confidence scoring
    lda_confidence = jnp.mean(jnp.max(jnp.asarray(lda_matrix), axis=1))
    nmf_confidence = jnp.mean(jnp.max(jnp.asarray(nmf_matrix), axis=1))

    # Extract scalar values using .item()
    lda_conf_val = lda_confidence.item()
    nmf_conf_val = nmf_confidence.item()

    # Adjust alpha based on confidence scores
    alpha = lda_conf_val / (lda_conf_val + nmf_conf_val + 1e-10)

    return alpha * lda_matrix + (1 - alpha) * nmf_matrix


# Calculate topic diversity
def calculate_topic_diversity(topic_word_matrix, top_n=10):
    """Calculate topic diversity"""
    print("Calculating topic diversity...")

    num_topics = topic_word_matrix.shape[0]
    top_words_indices = []

    for topic_idx in range(num_topics):
        # Get word probabilities for this topic
        word_probs = topic_word_matrix[topic_idx]

        # Get top N indices
        top_indices = jnp.argsort(word_probs)[-top_n:]

        # Convert JAX array to NumPy array for hashable type
        top_indices_np = np.asarray(top_indices)

        # Add to list of sets
        top_words_indices.append(set(top_indices_np))

    # Calculate Jaccard similarity between topics
    similarity_matrix = jnp.zeros((num_topics, num_topics))

    for i in range(num_topics):
        for j in range(i + 1, num_topics):
            intersection = len(top_words_indices[i].intersection(top_words_indices[j]))
            union = len(top_words_indices[i].union(top_words_indices[j]))
            similarity = intersection / union if union > 0 else 0
            similarity_matrix = similarity_matrix.at[i, j].set(similarity)
            similarity_matrix = similarity_matrix.at[j, i].set(similarity)

    # Topic diversity is the average pairwise distance
    diversity = 1 - jnp.mean(similarity_matrix)

    return diversity


# Helper functions for term identification
def calculate_term_specificity(term_idx, X, seed_docs_indices=None):
    """Calculate how specific a term is to subculture vs. general usage in the dataset"""
    # Get all documents containing this term
    term_docs = get_term_docs(term_idx, X)

    if not term_docs:
        return 0.0

    # If no seed docs provided, fall back to document frequency as a proxy
    if seed_docs_indices is None:
        return np.log(X.shape[0] / (len(term_docs) + 1))

    # Calculate the term's frequency in seed documents vs. overall
    seed_docs_set = set(seed_docs_indices)
    seed_docs_with_term = seed_docs_set.intersection(term_docs)

    # Avoid division by zero
    if not seed_docs_set or not seed_docs_with_term:
        return 0.0

    # Calculate concentration ratio (how much more common in seed docs vs overall)
    seed_concentration = len(seed_docs_with_term) / len(seed_docs_set)
    overall_concentration = len(term_docs) / X.shape[0]

    # Return ratio of concentrations (higher = more specific to subculture)
    return seed_concentration / (overall_concentration + 1e-10)


def calculate_seed_term_closeness(term_idx, feature_names_dict, X, seed_terms):
    """Calculate direct closeness to seed terms based on co-occurrence patterns"""
    seed_indices = [
        feature_names_dict[seed] for seed in seed_terms if seed in feature_names_dict
    ]

    if not seed_indices:
        return 0.0

    # Get documents where this term appears (as a set once)
    term_docs = get_term_docs(term_idx, X)

    if not term_docs:
        return 0.0

    # Pre-compute all seed docs sets at once
    seed_docs_sets = [set(X[:, seed_idx].nonzero()[0]) for seed_idx in seed_indices]

    # Calculate jaccard similarity with each seed term
    closeness_scores = []
    for seed_docs in seed_docs_sets:
        intersection = len(term_docs.intersection(seed_docs))
        union = len(term_docs.union(seed_docs))
        closeness_scores.append(intersection / union if union > 0 else 0)

    return sum(closeness_scores) / len(seed_indices)


def calculate_term_uniqueness(term_idx, X, lda_doc_topics):
    """Calculate how uniquely a term belongs to specific topic clusters within a subculture."""
    # Get documents containing this term
    term_docs = get_term_docs(term_idx, X)

    if len(term_docs) <= 1:
        return 0.0

    # Convert set to list for proper indexing
    term_docs_list = list(term_docs)

    # Get topic distribution for these documents
    term_doc_topics = lda_doc_topics[term_docs_list]

    # Find the dominant topic for each document
    dominant_topics = np.argmax(term_doc_topics, axis=1)

    # Calculate topic concentration using normalized entropy
    topic_counts = np.bincount(dominant_topics, minlength=lda_doc_topics.shape[1])
    topic_probs = topic_counts / np.sum(topic_counts)
    topic_probs = topic_probs[topic_probs > 0]  # Remove zeros

    # Calculate entropy (lower entropy = higher concentration = more unique)
    if len(topic_probs) <= 1:
        return 1.0  # Maximum uniqueness if term appears in only one topic

    # Calculate normalized entropy (0-1 scale)
    max_entropy = np.log(len(topic_probs))
    if max_entropy == 0:
        return 0.0

    topic_entropy = entropy(topic_probs)
    uniqueness = 1 - (topic_entropy / max_entropy)

    return uniqueness


# Save results to files
def save_results(df_weeb, df_furry, output_dir="output/terms-analysis"):
    print("Saving results to files...")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Select and order columns according to the required output format
    columns = [
        "term",
        "specificity",
        "similarity",
        "contextual_relevance",
        "seed_closeness",
        "uniqueness",
        "normalized_specificity",
        "normalized_similarity",
        "normalized_contextual_relevance",
        "normalized_seed_closeness",
        "normalized_uniqueness",
        "combined_score",
    ]

    # Save weeb terms
    weeb_file = os.path.join(output_dir, "weeb_terms.csv")
    if not df_weeb.empty:
        df_weeb[columns].to_csv(weeb_file, index=False)

    # Save furry terms
    furry_file = os.path.join(output_dir, "furry_terms.csv")
    if not df_furry.empty:
        df_furry[columns].to_csv(furry_file, index=False)

    print(f"Results saved to {weeb_file} and {furry_file}")


def save_metrics(metrics, metrics_output_dir="metrics/terms-analysis"):
    """Save evaluation metrics to a JSON file."""
    print("Saving metrics to file...")
    metrics_file = os.path.join(metrics_output_dir, "model_metrics.json")

    # Format floating point numbers and convert JAX arrays to NumPy arrays
    formatted_metrics = {}
    for key, value in metrics.items():
        # Convert JAX array to NumPy array and then to a float
        value = np.asarray(value).item()
        if isinstance(value, float):
            formatted_metrics[key] = round(value, 4)
        else:
            formatted_metrics[key] = value

    with open(metrics_file, "w") as f:
        json.dump(formatted_metrics, f, indent=4)

    print(f"Metrics saved to {metrics_file}")


def evaluate_model(
    model, key, X_train, X_test, feature_names, topic_word_matrix, model_name=""
):
    """Comprehensive model evaluation."""
    metrics = {}
    prefix = f"{model_name}_" if model_name else ""

    # Basic metrics
    metrics[f"{prefix}n_topics"] = model.n_components
    metrics[f"{prefix}topic_diversity"] = calculate_topic_diversity(topic_word_matrix)

    # Model-specific metrics with error handling and sample-based approach
    if hasattr(model, "score"):  # Only for LDA
        try:
            # Sample data for faster calculation
            sample_size = min(1000, X_test.shape[0])
            indices = random.choice(
                key, jnp.arange(X_test.shape[0]), shape=(sample_size,), replace=False
            )
            X_sample = X_test[indices]

            # Use score instead of score for efficiency
            metrics[f"{prefix}score"] = model.score(X_sample)
            print(f"Calculated {model_name} score: {metrics[f'{prefix}score']:.4f}")
        except Exception as e:
            metrics[f"{prefix}score"] = None
            print(f"Warning: Could not calculate score for {model_name} - {e}")
    else:
        # Skip score for non-LDA models
        metrics[f"{prefix}score"] = None
        if model_name != "lda":
            print(f"Skipping score calculation for {model_name} (not supported)")

    # Topic-document distribution
    doc_topic_dist = model.transform(X_train)

    # Calculate silhouette score using topic distributions
    try:
        # Sample data for faster calculation if needed
        if doc_topic_dist.shape[0] > 5000:
            sample_size = 5000
            indices = random.choice(
                key,
                jnp.arange(doc_topic_dist.shape[0]),
                shape=(sample_size,),
                replace=False,
            )
            sample_dist = doc_topic_dist[indices]
            sample_labels = jnp.argmax(sample_dist, axis=1)
        else:
            sample_dist = doc_topic_dist
            sample_labels = jnp.argmax(sample_dist, axis=1)

        # Convert JAX arrays to NumPy arrays
        sample_dist_np = np.asarray(sample_dist)
        sample_labels_np = np.asarray(sample_labels)

        # Calculate silhouette score
        metrics[f"{prefix}silhouette_score"] = silhouette_score(
            sample_dist_np, sample_labels_np
        )
    except Exception as e:
        metrics[f"{prefix}silhouette_score"] = None
        print(f"Warning: Could not calculate silhouette score for {model_name} - {e}")

    # Calculate topic distribution statistics
    try:
        # Sample data for faster calculation if needed
        if X_test.shape[0] > 1000:
            sample_size = 1000
            indices = random.choice(
                key, jnp.arange(X_test.shape[0]), shape=(sample_size,), replace=False
            )
            X_sample = X_test[indices]
            topic_distributions = model.transform(X_sample)
        else:
            topic_distributions = model.transform(X_test)

        # Avoid log(0) by adding small epsilon
        metrics[f"{prefix}avg_topic_concentration"] = jnp.mean(
            jnp.max(topic_distributions, axis=1)
        )
        metrics[f"{prefix}topic_distribution_entropy"] = -jnp.mean(
            jnp.sum(topic_distributions * jnp.log(topic_distributions + 1e-10), axis=1)
        )
    except Exception as e:
        metrics[f"{prefix}avg_topic_concentration"] = None
        metrics[f"{prefix}topic_distribution_entropy"] = None
        print(
            f"Warning: Could not calculate topic distribution metrics for {model_name} - {e}"
        )

    return metrics


def train_independent_models(
    X, feature_names, seed_terms_dict, n_topics=15, batch_size=2048
):
    """Train separate models for each subculture"""
    models_dict = {}
    topic_matrices_dict = {}
    feature_names_list = list(feature_names)

    for subculture, seed_terms in seed_terms_dict.items():
        print(f"Training model for {subculture} subculture...")
        # Filter documents that contain at least one seed term
        seed_indices = [
            i for i, term in enumerate(feature_names_list) if term in seed_terms
        ]
        if not seed_indices:
            print(f"Warning: No seed terms found for {subculture}")
            continue

        # Find documents with seed terms
        seed_docs = set()
        for term_idx in seed_indices:
            docs = set(X[:, term_idx].nonzero()[0])
            seed_docs.update(docs)

        if not seed_docs:
            print(f"Warning: No documents found containing seed terms for {subculture}")
            continue

        # Create filtered matrix
        X_filtered = X[list(seed_docs), :]

        print(f"Training LDA and NMF models for {subculture}...")
        # Train both LDA and NMF models
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            learning_method="batch",
            max_iter=20,
            random_state=42,
            n_jobs=-1,
            verbose=1,
            evaluate_every=2,
        )

        nmf = MiniBatchNMF(
            n_components=n_topics,
            init="nndsvd",
            batch_size=batch_size,
            random_state=42,
            max_iter=200,
            verbose=0,
        )

        print(f"Fitting models for {subculture}...")
        lda.fit(X_filtered)
        nmf.fit(X_filtered)

        # Store models and their topic matrices
        models_dict[subculture] = {"lda": lda, "nmf": nmf}

        # Combine topic matrices as before
        lda_matrix = lda.components_
        nmf_matrix = nmf.components_
        nmf_matrix = nmf_matrix / jnp.sum(nmf_matrix, axis=1)[:, jnp.newaxis]

        # Combine matrices with adaptive weighting
        topic_matrices_dict[subculture] = combine_topic_matrices(lda_matrix, nmf_matrix)

    return models_dict, topic_matrices_dict


def set_dynamic_thresholds(df_terms, subculture):
    """Set subculture-specific thresholds based on distribution"""
    subculture_terms = df_terms[df_terms["subculture"] == subculture]

    similarity_threshold = float(subculture_terms["similarity"].quantile(0.35))
    specificity_threshold = float(subculture_terms["specificity"].quantile(0.35))

    print(f"Dynamic thresholds for {subculture}:")
    print(f"- Similarity: {similarity_threshold:.4f}")
    print(f"- Specificity: {specificity_threshold:.4f}")

    return similarity_threshold, specificity_threshold


def identify_subculture_terms(
    models,
    topic_matrix,
    feature_names,
    X,
    seed_terms,
    subculture_name,
):
    """Identify terms for a specific subculture using both LDA and NMF models"""
    print(f"Identifying terms for {subculture_name} subculture...")

    feature_names_list = list(feature_names)
    feature_names_dict = {term: idx for idx, term in enumerate(feature_names_list)}

    # Get document-topic distributions from both models - this is expensive, so do it once
    print(f"Computing document-topic distributions for {subculture_name}...")
    lda_doc_topics = models["lda"].transform(X)
    nmf_doc_topics = models["nmf"].transform(X)

    # Combine document-topic distributions with the same weighting as topic matrices
    lda_confidence = np.mean(np.max(lda_doc_topics, axis=1))
    nmf_confidence = np.mean(np.max(nmf_doc_topics, axis=1))
    alpha = lda_confidence / (lda_confidence + nmf_confidence + 1e-10)
    doc_topics = alpha * lda_doc_topics + (1 - alpha) * nmf_doc_topics

    # Create seed vector in topic space - do this once
    num_topics = topic_matrix.shape[0]
    seed_vector = np.zeros(num_topics)
    seed_count = 0
    seed_docs = set()

    # Pre-compute seed docs in one pass
    for seed_term in seed_terms:
        if seed_term in feature_names_dict:
            term_idx = feature_names_dict[seed_term]

            for topic_idx in range(num_topics):
                seed_vector[topic_idx] += topic_matrix[topic_idx][term_idx]
            seed_count += 1

            docs = set(X[:, term_idx].nonzero()[0])
            seed_docs.update(docs)

    if seed_count > 0:
        seed_vector /= seed_count

    # Prepare data structures we'll reuse
    processed_terms = []
    total_docs = X.shape[0]
    doc_counts = np.array(X.astype(bool).sum(axis=0))[0]
    seed_docs_list = list(seed_docs)

    # Pre-filter terms with minimum frequency and length
    viable_term_indices = []

    for term_idx, term in enumerate(feature_names_list):
        # Skip very short terms and stopwords early
        if term in STOP_WORDS or len(term) <= 1:
            continue

        # Skip extremely rare terms
        if doc_counts[term_idx] < 5:
            continue

        viable_term_indices.append(term_idx)

    print(
        f"Processing {len(viable_term_indices)} viable terms for {subculture_name}..."
    )

    # Process terms in batches to avoid memory issues
    batch_size = 1024
    for i in range(0, len(viable_term_indices), batch_size):
        batch_indices = viable_term_indices[i : i + batch_size]

        for term_idx in batch_indices:
            term = feature_names_list[term_idx]

            # Calculate term specificity - skip terms with low specificity early
            term_specificity = calculate_term_specificity(term_idx, X, seed_docs_list)
            if term_specificity < 0.1:
                continue

            # Get documents where this term appears
            term_docs = get_term_docs(term_idx, X)

            if len(term_docs) > 0:
                # Convert set to list for proper indexing
                term_docs_list = list(term_docs)
                # Calculate term's topic distribution as average of its documents
                term_topic_dist = np.mean(doc_topics[term_docs_list], axis=0)
                term_vector = term_topic_dist.reshape(1, -1)
            else:
                continue

            # Calculate similarity with seed terms
            seed_closeness = calculate_seed_term_closeness(
                term_idx, feature_names_dict, X, seed_terms
            )

            # Calculate metrics
            term_vector = term_vector.reshape(1, -1)
            seed_vector_reshaped = seed_vector.reshape(1, -1)
            similarity = cosine_similarity(term_vector, seed_vector_reshaped)[0][0]

            # Skip terms with very low similarity
            if similarity < 0.3:
                continue

            # Calculate other metrics more efficiently
            doc_count = doc_counts[term_idx]
            doc_prevalence = float(doc_count) / total_docs

            # Add topic distribution similarity
            topic_sim = similarity  # Already calculated above
            similarity = (similarity + topic_sim) / 2  # Combine similarity measures

            # Calculate uniqueness within the subculture
            term_uniqueness = calculate_term_uniqueness(term_idx, X, lda_doc_topics)

            processed_terms.append(
                {
                    "term": term,
                    "subculture": subculture_name,
                    "specificity": term_specificity * 1e9,
                    "similarity": similarity,
                    "contextual_relevance": doc_prevalence,
                    "seed_closeness": seed_closeness,
                    "uniqueness": term_uniqueness,
                }
            )

        # Progress update
        print(
            f"Processed {min(i + batch_size, len(viable_term_indices))} of {len(viable_term_indices)} terms"
        )

    if not processed_terms:
        return pd.DataFrame()

    df_terms = pd.DataFrame(processed_terms)

    # Apply dynamic thresholds
    similarity_threshold, specificity_threshold = set_dynamic_thresholds(
        df_terms, subculture_name
    )
    df_terms = df_terms[
        (df_terms["similarity"] >= similarity_threshold)
        & (df_terms["specificity"] >= specificity_threshold)
    ]

    # Apply normalization and scoring
    if len(df_terms) > 1:
        features = [
            "specificity",
            "similarity",
            "contextual_relevance",
            "seed_closeness",
            "uniqueness",
        ]
        scaler = MinMaxScaler()
        normalized_features = scaler.fit_transform(df_terms[features])

        for i, feature in enumerate(features):
            df_terms[f"normalized_{feature}"] = normalized_features[:, i]

        # Calculate combined score with existing weights
        weights = {
            "specificity": 0.25,
            "similarity": 0.20,
            "contextual_relevance": 0.15,
            "seed_closeness": 0.15,
            "uniqueness": 0.25,
        }

        df_terms["combined_score"] = (
            weights["specificity"] * df_terms["normalized_specificity"]
            + weights["similarity"] * df_terms["normalized_similarity"]
            + weights["contextual_relevance"]
            * (1 - df_terms["normalized_contextual_relevance"])  # Inverted contribution
            + weights["seed_closeness"] * df_terms["normalized_seed_closeness"]
            + weights["uniqueness"] * df_terms["normalized_uniqueness"]
        )

    return df_terms.sort_values("combined_score", ascending=False)


# Main function
def main(file_path, n_topics=15, max_features=10000, batch_size=2048, seed=42):
    # Set random seed for reproducibility
    key = random.PRNGKey(seed)

    # Load and preprocess data
    df = load_data(file_path)

    # Extract features
    X, _, feature_names = extract_features(df["text"], max_features=max_features)

    # Split data for training and evaluation
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=seed)

    # Define seed terms dictionary
    seed_terms_dict = {"weeb": weeb_seed_terms, "furry": furry_seed_terms}

    # Train independent models
    models_dict, topic_matrices_dict = train_independent_models(
        X_train, feature_names, seed_terms_dict, n_topics, batch_size
    )

    # Identify terms for each subculture independently
    df_weeb = identify_subculture_terms(
        models_dict["weeb"],
        topic_matrices_dict["weeb"],
        feature_names,
        X_train,
        weeb_seed_terms,
        "weeb",
    )

    df_furry = identify_subculture_terms(
        models_dict["furry"],
        topic_matrices_dict["furry"],
        feature_names,
        X_train,
        furry_seed_terms,
        "furry",
    )

    # Evaluate models and combine metrics
    metrics = {}
    for subculture, models in models_dict.items():
        for model_name, model in models.items():
            subculture_metrics = evaluate_model(
                model,
                key,
                X_train,
                X_test,
                feature_names,
                topic_matrices_dict[subculture],
                f"{subculture}_{model_name}",
            )
            metrics.update(subculture_metrics)

    save_metrics(metrics)
    save_results(df_weeb, df_furry)

    return df_weeb, df_furry, metrics


# Entry point
if __name__ == "__main__":
    # Define the path to the CSV file
    input_file = "output/dataset-filter/bluesky_ten_million_english_only.csv"

    # Run the main function with appropriate batch size for large dataset
    main(input_file, batch_size=2048)
