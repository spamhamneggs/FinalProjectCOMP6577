import json
import os
import re
from collections import defaultdict
from functools import lru_cache

import jax.numpy as jnp
import jax.random as random
import nltk
import numpy as np
import pandas as pd
from nltk import FreqDist
from nltk.corpus import stopwords, twitter_samples, webtext
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
nltk.download("stopwords", quiet=True)
nltk.download("webtext", quiet=True)
nltk.download("twitter_samples", quiet=True)

# Create output directories
os.makedirs("output", exist_ok=True)

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


# Text preprocessing
@lru_cache(maxsize=10000)
def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove all punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # Remove all independent numbers
    text = re.sub(r"\b\d+\b", "", text)

    return text


# Tokenization function
def tokenize_text(text):
    if not isinstance(text, str):
        return []
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    # Don't filter out short words as they might be relevant terms
    tokens = [token for token in tokens if token not in stop_words]
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
        stop_words="english",
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


def train_models(X, feature_names, n_topics=20, batch_size=2048):
    print("Training LDA and NMF models...")

    # Train LDA model
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        learning_method="batch",
        max_iter=20,
        random_state=42,
        n_jobs=-1,
        verbose=1,
        evaluate_every=2,
    )

    # Train MiniBatch NMF model
    nmf = MiniBatchNMF(
        n_components=n_topics,
        init="nndsvd",
        batch_size=batch_size,
        random_state=42,
        max_iter=200,
        verbose=0,
    )

    print("Fitting LDA...")
    lda.fit(X)
    print("Fitting MiniBatch NMF...")
    nmf.fit(X)

    # Get topic-word matrices
    lda_topic_word = lda.components_
    nmf_topic_word = nmf.components_

    # Normalize NMF components for comparison with LDA
    nmf_topic_word = nmf_topic_word / jnp.sum(nmf_topic_word, axis=1)[:, jnp.newaxis]

    return (lda, nmf), (lda_topic_word, nmf_topic_word)


def combine_topic_matrices(lda_matrix, nmf_matrix, alpha=0.5):
    """Combine LDA and NMF topic-word matrices with adaptive weighting"""
    # Add confidence scoring
    lda_confidence = np.mean(np.max(lda_matrix, axis=1))
    nmf_confidence = np.mean(np.max(nmf_matrix, axis=1))

    # Adjust alpha based on confidence scores
    alpha = lda_confidence / (lda_confidence + nmf_confidence)

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
def calculate_term_specificity(term_idx, X, general_corpus=None):
    """Compare term frequency in dataset vs general language corpus"""
    # Higher scores mean more specific to the dataset
    if general_corpus is not None:
        dataset_freq = X[:, term_idx].sum() / X.sum()
        return dataset_freq / (general_corpus[term_idx] + 1e-10)
    else:
        # Fallback to document frequency as a proxy for specificity
        return np.log(X.shape[0] / (X[:, term_idx].getnnz() + 1))


def calculate_domain_specificity(term, similarity_score, general_corpus):
    """Calculate domain-specific similarity with penalties for common terms"""
    # Get the term's frequency in general corpus (defaults to 0 if not present)
    general_freq = general_corpus.get(term, 0)

    # Apply graduated penalty based on frequency in general corpus
    # For very common terms (high frequency), penalty is stronger
    # For rare terms (low frequency), penalty is minimal
    penalty_factor = max(0.3, 1.0 - (general_freq * 200))

    return similarity_score * penalty_factor


def calculate_entropy(term_idx, X):
    """Calculate entropy of term distribution across documents using sparse operations"""
    # Get term frequency across documents (keeping sparse)
    term_docs = X[:, term_idx].nonzero()[0]

    # If no occurrences, return 0
    if len(term_docs) == 0:
        return 0

    # Count occurrences in each document (still sparse)
    term_freqs = X[term_docs, term_idx].toarray().flatten()

    # Normalize and calculate entropy
    term_freqs = term_freqs / term_freqs.sum()
    return entropy(term_freqs)


# Identify furry and weeb terms from topics
def identify_subculture_terms(topic_word_matrix, feature_names, X):
    print("Identifying subculture terms...")
    num_topics = topic_word_matrix.shape[0]

    # Convert feature_names to list for efficient lookups
    feature_names_list = list(feature_names)
    feature_names_dict = {term: idx for idx, term in enumerate(feature_names_list)}

    # Create vectors for seed terms in topic space
    weeb_seed_vector = np.zeros(num_topics)
    furry_seed_vector = np.zeros(num_topics)

    # Calculate average topic distribution for seed terms
    weeb_seed_count = 0
    furry_seed_count = 0

    for seed_term in weeb_seed_terms:
        if seed_term in feature_names_dict:
            term_idx = feature_names_dict[seed_term]
            # Find which topics this seed term appears in significantly
            for topic_idx in range(num_topics):
                weeb_seed_vector[topic_idx] += topic_word_matrix[topic_idx][term_idx]
            weeb_seed_count += 1

    for seed_term in furry_seed_terms:
        if seed_term in feature_names_dict:
            term_idx = feature_names_dict[seed_term]
            # Find which topics this seed term appears in significantly
            for topic_idx in range(num_topics):
                furry_seed_vector[topic_idx] += topic_word_matrix[topic_idx][term_idx]
            furry_seed_count += 1

    # Normalize seed vectors
    if weeb_seed_count > 0:
        weeb_seed_vector /= weeb_seed_count
    if furry_seed_count > 0:
        furry_seed_vector /= furry_seed_count

    # Calculate seed vector distance for measuring distinctness
    seed_vector_distance = cosine_similarity([weeb_seed_vector], [furry_seed_vector])[
        0
    ][0]
    print(f"Seed vector similarity: {seed_vector_distance:.4f}")

    # Create a general usage vector in topic space instead of term space
    print("Calculating general usage vector...")
    general_usage_vector = np.zeros(num_topics)
    doc_topic_distributions = np.array(X.mean(axis=0))[0]
    for topic_idx in range(num_topics):
        general_usage_vector[topic_idx] = np.sum(
            topic_word_matrix[topic_idx] * doc_topic_distributions
        )
    general_usage_vector = general_usage_vector / np.sum(general_usage_vector)

    # Initialize containers for terms
    all_terms = {}

    # Get document counts efficiently using sparse matrix
    print("Calculating document counts...")
    total_docs = X.shape[0]
    doc_counts = np.array(X.astype(bool).sum(axis=0))[0]

    # Calculate general language frequencies from NLTK webtext and twitter_samples corpus
    print("Calculating general language frequencies from NLTK corpora...")

    # Get words from webtext
    webtext_words = [
        word.lower() for text in webtext.raw().split() for word in word_tokenize(text)
    ]

    # Get words from twitter samples (excluding non-English)
    twitter_words = [
        word.lower()
        for text in twitter_samples.strings()
        for word in word_tokenize(text)
        if any(c.isalpha() for c in word)  # Filter non-alphabetic tokens
    ]

    # Combine both corpora
    combined_words = webtext_words + twitter_words

    # Process combined words
    processed_words = [
        word
        for word in combined_words
        if len(word) > 1
        and word not in stopwords.words("english")
        and word.isalnum()  # Only allow alphanumeric terms
    ]

    # Calculate frequencies
    frequency_list = FreqDist(processed_words)
    general_corpus_frequencies = defaultdict(int)

    # Convert to dictionary and normalize frequencies
    total_words = sum(frequency_list.values())
    for term in feature_names_list:
        general_corpus_frequencies[term] = frequency_list[term] / total_words

    general_language_scores = general_corpus_frequencies  # Already normalized

    # Calculate metrics for each term
    print("Processing terms...")
    for term_idx, term in enumerate(feature_names_list):
        # Skip stopwords and very short terms
        if term in stopwords.words("english") or len(term) <= 1:
            continue

        # Calculate term specificity using helper function
        term_specificity = calculate_term_specificity(
            term_idx, X, general_language_scores
        )

        # Skip terms with low specificity
        if term_specificity < 0.1:
            continue

        # Create a vector for this term in topic space
        term_vector = np.zeros(num_topics)
        for topic_idx in range(num_topics):
            term_vector[topic_idx] = topic_word_matrix[topic_idx][term_idx]

        def calculate_seed_term_closeness(
            term_idx, term, feature_names_dict, X, seed_terms
        ):
            """Calculate direct closeness to seed terms based on co-occurrence patterns"""
            seed_indices = [
                feature_names_dict[seed]
                for seed in seed_terms
                if seed in feature_names_dict
            ]

            if not seed_indices:
                return 0.0

            # Get documents where this term appears (as a set once)
            term_docs = set(X[:, term_idx].nonzero()[0])

            if not term_docs:
                return 0.0

            # Pre-compute all seed docs sets at once
            seed_docs_sets = [
                set(X[:, seed_idx].nonzero()[0]) for seed_idx in seed_indices
            ]

            # Calculate jaccard similarity with each seed term
            closeness_scores = []
            for seed_docs in seed_docs_sets:
                intersection = len(term_docs.intersection(seed_docs))
                union = len(term_docs.union(seed_docs))
                closeness_scores.append(intersection / union if union > 0 else 0)

            return sum(closeness_scores) / len(seed_indices)

        weeb_closeness = calculate_seed_term_closeness(
            term_idx, term, feature_names_dict, X, weeb_seed_terms
        )
        furry_closeness = calculate_seed_term_closeness(
            term_idx, term, feature_names_dict, X, furry_seed_terms
        )

        # Reshape term_vector for cosine similarity
        term_vector = term_vector.reshape(1, -1)
        weeb_seed_vector = weeb_seed_vector.reshape(1, -1)
        furry_seed_vector = furry_seed_vector.reshape(1, -1)
        general_usage_vector = general_usage_vector.reshape(1, -1)

        # Calculate cosine similarity with seed vectors
        weeb_similarity = cosine_similarity(term_vector, weeb_seed_vector)[0][0]
        furry_similarity = cosine_similarity(term_vector, furry_seed_vector)[0][0]

        min_similarity_threshold = 0.3

        if (
            weeb_similarity < min_similarity_threshold
            and weeb_similarity < min_similarity_threshold
        ):
            # Skip terms that have low similarity to both communities
            continue

        # Apply domain specificity adjustments
        weeb_similarity = calculate_domain_specificity(
            term, weeb_similarity, general_language_scores
        )
        furry_similarity = calculate_domain_specificity(
            term, furry_similarity, general_language_scores
        )

        # Calculate entropy for term distribution
        term_entropy = calculate_entropy(term_idx, X)

        # Small constant to avoid division by zero
        epsilon = 1e-10

        # Determine community based on similarity and distinctiveness
        if (
            weeb_similarity > furry_similarity
            and weeb_similarity >= min_similarity_threshold
        ):
            community = "weeb"
            distinctiveness = (weeb_similarity - furry_similarity) / (
                seed_vector_distance + epsilon
            )
            primary_similarity = weeb_similarity
        elif furry_similarity >= min_similarity_threshold:
            community = "furry"
            distinctiveness = (furry_similarity - weeb_similarity) / (
                seed_vector_distance + epsilon
            )
            primary_similarity = furry_similarity
        else:
            # Skip terms that don't have sufficient similarity to either community
            continue

        # Calculate general similarity and uniqueness
        general_similarity = cosine_similarity(term_vector, general_usage_vector)[0][0]
        uniqueness = 1.0 - general_similarity

        # Document prevalence (using precomputed doc_counts)
        doc_count = doc_counts[term_idx]
        doc_prevalence = float(doc_count) / total_docs

        # Store the calculated metrics
        term_data = {
            "term": term,
            "community": community,
            "specificity": term_specificity * 1e9,  # Scale up for readability
            "weeb_similarity": weeb_similarity,
            "furry_similarity": furry_similarity,
            "primary_similarity": primary_similarity,
            "general_similarity": general_similarity,
            "uniqueness": uniqueness,
            "distinctiveness": distinctiveness,
            "contextual_relevance": doc_prevalence,
            "entropy": term_entropy,
            "seed_closeness": weeb_closeness
            if community == "weeb"
            else furry_closeness,
        }

        all_terms[term] = term_data

    # Convert to list for DataFrame creation
    processed_terms = list(all_terms.values())

    # Create dataframes
    df_terms = pd.DataFrame(processed_terms)
    if df_terms.empty or len(df_terms) <= 1:
        return pd.DataFrame(), pd.DataFrame()

    # Set distinctiveness threshold adaptively based on data distribution
    if len(df_terms) > 10:
        weeb_terms = df_terms[df_terms["community"] == "weeb"]
        furry_terms = df_terms[df_terms["community"] == "furry"]

        if not weeb_terms.empty:
            weeb_distinctiveness_threshold = weeb_terms["distinctiveness"].quantile(0.5)
        else:
            weeb_distinctiveness_threshold = 0.05

        if not furry_terms.empty:
            furry_distinctiveness_threshold = furry_terms["distinctiveness"].quantile(
                0.5
            )
        else:
            furry_distinctiveness_threshold = 0.05
    else:
        weeb_distinctiveness_threshold = 0.05
        furry_distinctiveness_threshold = 0.05

    print(f"Weeb distinctiveness threshold: {weeb_distinctiveness_threshold:.4f}")
    print(f"Furry distinctiveness threshold: {furry_distinctiveness_threshold:.4f}")

    # Split into weeb and furry dataframes
    df_weeb = df_terms[df_terms["community"] == "weeb"].copy()
    df_furry = df_terms[df_terms["community"] == "furry"].copy()

    # Apply distinctiveness filters
    df_weeb = df_weeb[df_weeb["distinctiveness"] >= weeb_distinctiveness_threshold]
    df_furry = df_furry[df_furry["distinctiveness"] >= furry_distinctiveness_threshold]

    # Rename similarity column for consistency with expected output format
    df_weeb = df_weeb.rename(columns={"primary_similarity": "similarity"})
    df_furry = df_furry.rename(columns={"primary_similarity": "similarity"})

    # Define weights for scoring
    weight_specificity = 0.10
    weight_similarity = 0.25
    weight_uniqueness = 0.15
    weight_distinctiveness = 0.15
    weight_context = 0.05
    weight_entropy = 0.05
    weight_seed_closeness = 0.25

    # Define features for normalization and scoring
    features = [
        "specificity",
        "similarity",
        "uniqueness",
        "distinctiveness",
        "contextual_relevance",
        "entropy",
        "seed_closeness",
    ]

    # Update normalization and scoring sections for both dataframes
    if len(df_weeb) > 1:
        scaler = MinMaxScaler()
        normalized_features = scaler.fit_transform(df_weeb[features])
        df_weeb["normalized_spec"] = normalized_features[:, 0]
        df_weeb["normalized_sim"] = normalized_features[:, 1]
        df_weeb["normalized_uniq"] = normalized_features[:, 2]
        df_weeb["normalized_dist"] = normalized_features[:, 3]
        df_weeb["normalized_context"] = normalized_features[:, 4]
        df_weeb["normalized_entropy"] = normalized_features[:, 5]
        df_weeb["normalized_seed_closeness"] = normalized_features[:, 6]

        # Combined score with weights
        df_weeb["combined_score"] = (
            weight_specificity * df_weeb["normalized_spec"]
            + weight_similarity * df_weeb["normalized_sim"]
            + weight_uniqueness * df_weeb["normalized_uniq"]
            + weight_distinctiveness * df_weeb["normalized_dist"]
            + weight_context * df_weeb["normalized_context"]
            + weight_entropy * df_weeb["normalized_entropy"]
            + weight_seed_closeness * df_weeb["normalized_seed_closeness"]
        )

    if len(df_furry) > 1:
        scaler = MinMaxScaler()
        normalized_features = scaler.fit_transform(df_furry[features])
        df_furry["normalized_spec"] = normalized_features[:, 0]
        df_furry["normalized_sim"] = normalized_features[:, 1]
        df_furry["normalized_uniq"] = normalized_features[:, 2]
        df_furry["normalized_dist"] = normalized_features[:, 3]
        df_furry["normalized_context"] = normalized_features[:, 4]
        df_furry["normalized_entropy"] = normalized_features[:, 5]
        df_furry["normalized_seed_closeness"] = normalized_features[:, 6]

        # Combined score with weights
        df_furry["combined_score"] = (
            weight_specificity * df_furry["normalized_spec"]
            + weight_similarity * df_furry["normalized_sim"]
            + weight_uniqueness * df_furry["normalized_uniq"]
            + weight_distinctiveness * df_furry["normalized_dist"]
            + weight_context * df_furry["normalized_context"]
            + weight_entropy * df_furry["normalized_entropy"]
            + weight_seed_closeness * df_furry["normalized_seed_closeness"]
        )

    # Sort by combined score
    df_weeb = df_weeb.sort_values("combined_score", ascending=False)
    df_furry = df_furry.sort_values("combined_score", ascending=False)

    return df_weeb, df_furry


# Save results to files
def save_results(df_weeb, df_furry, output_dir="output"):
    print("Saving results to files...")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Select and order columns according to the required output format
    columns = [
        "term",
        "specificity",
        "similarity",
        "contextual_relevance",
        "normalized_spec",
        "normalized_sim",
        "normalized_context",
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


def save_metrics(metrics, output_dir="output"):
    """Save evaluation metrics to a JSON file."""
    print("Saving metrics to file...")
    metrics_file = os.path.join(output_dir, "model_metrics.json")

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
    if hasattr(model, "score") and model_name == "lda":  # Only for LDA
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
            jnp.sum(topic_distributions * jnp.log(topic_distributions + 1e-12), axis=1)
        )
    except Exception as e:
        metrics[f"{prefix}avg_topic_concentration"] = None
        metrics[f"{prefix}topic_distribution_entropy"] = None
        print(
            f"Warning: Could not calculate topic distribution metrics for {model_name} - {e}"
        )

    return metrics


# Main function
def main(file_path, n_topics=15, max_features=10000, batch_size=2048, seed=42):
    # Set random seed for reproducibility
    key = random.PRNGKey(seed)

    # Load and preprocess data
    df = load_data(file_path)

    # Extract features
    X, vectorizer, feature_names = extract_features(
        df["text"], max_features=max_features
    )

    # Split data for training and evaluation
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=seed)

    # Train both models with batch processing
    models, topic_matrices = train_models(
        X_train, feature_names, n_topics=n_topics, batch_size=batch_size
    )
    lda_model, nmf_model = models
    lda_topic_matrix, nmf_topic_matrix = topic_matrices

    # Combine topic matrices
    combined_topic_matrix = combine_topic_matrices(lda_topic_matrix, nmf_topic_matrix)

    # Evaluate both models separately
    lda_metrics = evaluate_model(
        lda_model, key, X_train, X_test, feature_names, lda_topic_matrix, "lda"
    )
    nmf_metrics = evaluate_model(
        nmf_model, key, X_train, X_test, feature_names, nmf_topic_matrix, "nmf"
    )

    # Create a combined evaluation that uses both models appropriately
    combined_metrics = {
        "combined_n_topics": n_topics,
        "combined_topic_diversity": calculate_topic_diversity(combined_topic_matrix),
    }

    # Merge all metrics
    metrics = {**lda_metrics, **nmf_metrics, **combined_metrics}

    # Print metrics
    print("\nModel Evaluation Metrics:")
    for metric, value in metrics.items():
        print(
            f"{metric}: {value:.4f}"
            if isinstance(value, float)
            else f"{metric}: {value}"
        )

    # Save metrics
    save_metrics(metrics)

    # Use combined topic matrix for term identification
    df_weeb, df_furry = identify_subculture_terms(
        combined_topic_matrix, feature_names, X_train
    )

    # Save results
    save_results(df_weeb, df_furry)

    return df_weeb, df_furry, metrics


# Entry point
if __name__ == "__main__":
    # Define the path to the CSV file
    input_file = "output/dataset-filter/bluesky_ten_million_english_only.csv"

    # Run the main function with appropriate batch size for large dataset
    main(input_file, batch_size=2048)
