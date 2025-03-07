import json
import os
import re
import warnings

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.decomposition import LatentDirichletAllocation, MiniBatchNMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# warnings.filterwarnings("ignore")

# Download necessary NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# Create output directories
os.makedirs("output", exist_ok=True)

# Define lists of known weeb and furry terms for initial seeding
weeb_seed_terms = [
    "anime",
    "manga",
    "otaku",
    "waifu",
    "kawaii",
    "chibi",
    "isekai",
    "nani",
    "baka",
    "cosplay",
    "animeart",
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
]


# Load and preprocess the data
def load_data(file_path):
    print("Loading and preprocessing data...")
    df = pd.read_csv(file_path)

    # Remove rows with empty text
    df = df[df["text"].notna()]
    return df


# Text preprocessing
def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Replace URLs with a placeholder
    text = re.sub(r"http\S+", "URL", text)

    # Keep emoticons (not removing punctuation)
    # Keep hashtags
    text = re.sub(r"#(\w+)", r"\1", text)

    # Remove special characters except those used in emoticons
    emoticon_chars = r"()[]\{\}:;,.!?/\\|~-_<>^*+#@$%&="
    emoticon_pattern = f"[^a-zA-Z0-9\\s{re.escape(emoticon_chars)}]"
    text = re.sub(emoticon_pattern, "", text)

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
def extract_features(texts, max_features=5000, min_df=3, max_df=0.95):
    print("Extracting features...")
    # Use TF-IDF instead of raw counts for better NMF performance
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        tokenizer=tokenize_text,
        preprocessor=preprocess_text,
    )
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    return X, vectorizer, feature_names


def train_models(X, feature_names, n_topics=10, batch_size=2048):
    print("Training LDA and NMF models...")

    # Train LDA model
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        learning_method="batch",
        learning_decay=0.7,
        max_iter=20,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    # Train MiniBatch NMF model
    nmf = MiniBatchNMF(
        n_components=n_topics,
        init="nndsvd",
        batch_size=batch_size,
        random_state=42,
        max_iter=200,
        verbose=1,
    )

    print("Fitting LDA...")
    lda.fit(X)
    print("Fitting MiniBatch NMF...")
    nmf.fit(X)

    # Get topic-word matrices
    lda_topic_word = lda.components_
    nmf_topic_word = nmf.components_

    # Normalize NMF components for comparison with LDA
    nmf_topic_word = nmf_topic_word / nmf_topic_word.sum(axis=1)[:, np.newaxis]

    return (lda, nmf), (lda_topic_word, nmf_topic_word)


def combine_topic_matrices(lda_matrix, nmf_matrix, alpha=0.5):
    """Combine LDA and NMF topic-word matrices with a weighted average"""
    return alpha * lda_matrix + (1 - alpha) * nmf_matrix


# Calculate topic coherence
def calculate_coherence(model, X, feature_names, method="c_v"):
    print("Calculating topic coherence...")
    coherence_scores = []
    topic_word_dists = model.components_
    
    # Get dimensions of sparse matrix without converting to dense
    n_docs, n_terms = X.shape
    
    for topic_idx, topic_dist in enumerate(topic_word_dists):
        top_word_indices = topic_dist.argsort()[:-11:-1]
        top_words = [feature_names[i] for i in top_word_indices]
        
        # Calculate NPMI-based coherence
        word_cooccurrence = np.zeros((len(top_words), len(top_words)))
        
        # Process columns (words) of sparse matrix one at a time
        word_docs = {}
        for i, word_idx in enumerate(top_word_indices):
            # Extract column as sparse array and convert to binary occurrence
            col = X.getcol(word_idx)
            word_docs[i] = set(col.nonzero()[0])
        
        # Calculate co-occurrence without materializing the full matrix
        for i in range(len(top_word_indices)):
            for j in range(i+1, len(top_word_indices)):
                if i < j:
                    docs_i = word_docs[i]
                    docs_j = word_docs[j]
                    
                    co_docs = len(docs_i.intersection(docs_j))
                    word1_docs = len(docs_i)
                    word2_docs = len(docs_j)
                    total_docs = n_docs
                    
                    # Calculate NPMI (Normalized Pointwise Mutual Information)
                    if co_docs > 0:
                        pmi = np.log((co_docs * total_docs) / (word1_docs * word2_docs))
                        npmi = pmi / -np.log(co_docs / total_docs)
                        word_cooccurrence[i, j] = npmi
                        word_cooccurrence[j, i] = npmi
        
        # Average coherence for this topic
        coherence_scores.append(np.mean(word_cooccurrence))
    
    return np.mean(coherence_scores)


# Calculate perplexity
def calculate_perplexity(model, X_test):
    print("Calculating perplexity...")
    return model.perplexity(X_test)


# Calculate topic diversity
def calculate_topic_diversity(topic_word_matrix, top_n=10):
    print("Calculating topic diversity...")

    num_topics = topic_word_matrix.shape[0]
    top_words_indices = []

    for topic_idx in range(num_topics):
        top_indices = topic_word_matrix[topic_idx].argsort()[: -top_n - 1 : -1]
        top_words_indices.append(set(top_indices))

    # Calculate Jaccard similarity between topics
    similarity_matrix = np.zeros((num_topics, num_topics))

    for i in range(num_topics):
        for j in range(i + 1, num_topics):
            intersection = len(top_words_indices[i].intersection(top_words_indices[j]))
            union = len(top_words_indices[i].union(top_words_indices[j]))
            similarity = intersection / union if union > 0 else 0
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity

    # Topic diversity is the average pairwise distance
    diversity = 1 - np.mean(similarity_matrix)

    return diversity


# Identify furry and weeb terms from topics
def identify_subculture_terms(lda_model, feature_names, X):
    print("Identifying subculture terms...")
    topic_word_matrix = lda_model.components_
    num_topics = topic_word_matrix.shape[0]

    # Initialize containers for terms
    all_terms = {}

    # Calculate various metrics for each term
    for topic_idx in range(num_topics):
        # Get word probabilities for this topic
        word_probs = topic_word_matrix[topic_idx]

        # Get all words with their probabilities (not just top N)
        for term_idx, term in enumerate(feature_names):
            prob = word_probs[term_idx]

            if prob > 0.0001:  # Filter very low probability terms
                # Initialize term data if not already present
                if term not in all_terms:
                    all_terms[term] = {
                        "specificity": [],
                        "topic_assignments": [],
                        "doc_prevalence": 0,
                    }

                # Add specificity (how strongly this term is associated with this topic)
                all_terms[term]["specificity"].append(prob)
                all_terms[term]["topic_assignments"].append(topic_idx)

    # Calculate document prevalence for each term efficiently
    # Process one term at a time to avoid memory issues
    n_docs = X.shape[0]
    for term_idx, term in enumerate(feature_names):
        if term in all_terms:
            # Get the column for this term (sparse vector)
            col = X.getcol(term_idx)
            # Count documents where term appears (nonzero entries)
            doc_count = col.nnz
            # Calculate prevalence
            all_terms[term]["doc_prevalence"] = doc_count / n_docs

    # Process and categorize terms
    processed_terms = []

    for term, data in all_terms.items():
        # Skip if no specificity data
        if not data["specificity"]:
            continue

        # Calculate metrics
        specificity = np.sum(data["specificity"]) * 1e9  # Scale up for readability
        specificity_std = np.std(data["specificity"]) * 1e9
        specificity_ci = 1.96 * specificity_std / np.sqrt(len(data["specificity"]))

        # Calculate similarity scores with seed terms
        weeb_similarity = 0
        furry_similarity = 0

        # Simple similarity based on co-occurrence in topics
        for topic_idx in data["topic_assignments"]:
            # Check if weeb or furry seed terms are prominent in this topic
            topic_probs = topic_word_matrix[topic_idx]
            weeb_count = 0
            furry_count = 0

            for seed_term in weeb_seed_terms:
                if seed_term in feature_names:
                    seed_idx = np.where(feature_names == seed_term)[0]
                    if len(seed_idx) > 0 and topic_probs[seed_idx[0]] > 0.001:
                        weeb_count += 1

            for seed_term in furry_seed_terms:
                if seed_term in feature_names:
                    seed_idx = np.where(feature_names == seed_term)[0]
                    if len(seed_idx) > 0 and topic_probs[seed_idx[0]] > 0.001:
                        furry_count += 1

            # Assign similarity based on which seed terms are more prevalent
            if weeb_count > furry_count:
                weeb_similarity += 1
            elif furry_count > weeb_count:
                furry_similarity += 1
            else:
                # Equal counts, split the similarity
                weeb_similarity += 0.5
                furry_similarity += 0.5

        # Normalize similarities
        total_assignments = len(data["topic_assignments"])
        if total_assignments > 0:
            weeb_similarity /= total_assignments
            furry_similarity /= total_assignments

        # Calculate contextual relevance - how specific the term is to certain contexts
        contextual_relevance = data["doc_prevalence"]

        # Store the calculated metrics
        term_data = {
            "term": term,
            "specificity": specificity,
            "specificity_std": specificity_std,
            "specificity_ci": specificity_ci,
            "weeb_similarity": weeb_similarity,
            "furry_similarity": furry_similarity,
            "contextual_relevance": contextual_relevance,
        }

        processed_terms.append(term_data)

    # Create dataframes for weeb and furry terms
    df_terms = pd.DataFrame(processed_terms)

    # Calculate ensemble scores
    if not df_terms.empty:
        # Normalize values across all terms
        scaler = MinMaxScaler()

        # Make sure we have enough data to scale
        if len(df_terms) > 1:
            df_terms["normalized_spec"] = scaler.fit_transform(
                df_terms[["specificity"]]
            )

            # Calculate ensemble scores - weighted combination of metrics
            df_terms["ensemble_score"] = 1 - 1 / (1 + df_terms["specificity"])
            df_terms["normalized_ensemble"] = scaler.fit_transform(
                df_terms[["ensemble_score"]]
            )

            # Initialize the result dataframes
            df_weeb = df_terms.copy()
            df_furry = df_terms.copy()

            # Rename the columns to match output format
            df_weeb = df_weeb.rename(columns={"weeb_similarity": "similarity"})
            df_furry = df_furry.rename(columns={"furry_similarity": "similarity"})

            # Normalize similarity columns
            if len(df_weeb) > 1:
                df_weeb["normalized_sim"] = scaler.fit_transform(
                    df_weeb[["similarity"]]
                )
                df_weeb["normalized_context"] = scaler.fit_transform(
                    df_weeb[["contextual_relevance"]]
                )

                # Combined score for weeb terms
                df_weeb["combined_score"] = (
                    0.4 * df_weeb["normalized_spec"]
                    + 0.4 * df_weeb["normalized_sim"]
                    + 0.2 * df_weeb["normalized_context"]
                )
            else:
                df_weeb["normalized_sim"] = df_weeb["similarity"]
                df_weeb["normalized_context"] = df_weeb["contextual_relevance"]
                df_weeb["combined_score"] = df_weeb["normalized_spec"]

            if len(df_furry) > 1:
                df_furry["normalized_sim"] = scaler.fit_transform(
                    df_furry[["similarity"]]
                )
                df_furry["normalized_context"] = scaler.fit_transform(
                    df_furry[["contextual_relevance"]]
                )

                # Combined score for furry terms
                df_furry["combined_score"] = (
                    0.4 * df_furry["normalized_spec"]
                    + 0.4 * df_furry["normalized_sim"]
                    + 0.2 * df_furry["normalized_context"]
                )
            else:
                df_furry["normalized_sim"] = df_furry["similarity"]
                df_furry["normalized_context"] = df_furry["contextual_relevance"]
                df_furry["combined_score"] = df_furry["normalized_spec"]

            # Sort by combined score
            df_weeb = df_weeb.sort_values("combined_score", ascending=False)
            df_furry = df_furry.sort_values("combined_score", ascending=False)

            return df_weeb, df_furry

    # Return empty dataframes if processing failed
    return pd.DataFrame(), pd.DataFrame()


# Save results to files
def save_results(df_weeb, df_furry, output_dir="output"):
    print("Saving results to files...")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Select and order columns according to the required output format
    columns = [
        "term",
        "specificity",
        "specificity_std",
        "specificity_ci",
        "similarity",
        "contextual_relevance",
        "normalized_spec",
        "normalized_sim",
        "normalized_context",
        "ensemble_score",
        "normalized_ensemble",
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

    # Format floating point numbers
    formatted_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted_metrics[key] = round(value, 4)
        else:
            formatted_metrics[key] = value

    with open(metrics_file, "w") as f:
        json.dump(formatted_metrics, f, indent=4)

    print(f"Metrics saved to {metrics_file}")


def evaluate_model(
    model, X_train, X_test, feature_names, topic_word_matrix, model_name=""
):
    """Comprehensive model evaluation."""
    metrics = {}
    prefix = f"{model_name}_" if model_name else ""

    # Basic metrics
    metrics[f"{prefix}n_topics"] = model.n_components
    metrics[f"{prefix}topic_diversity"] = calculate_topic_diversity(topic_word_matrix)

    # Model-specific metrics
    if hasattr(model, "perplexity"):  # Only LDA has perplexity
        metrics[f"{prefix}perplexity"] = calculate_perplexity(model, X_test)

    metrics[f"{prefix}coherence"] = calculate_coherence(model, X_test, feature_names)

    # Topic-document distribution
    doc_topic_dist = model.transform(X_train)

    # Calculate silhouette score using topic distributions
    try:
        metrics[f"{prefix}silhouette_score"] = silhouette_score(
            doc_topic_dist, np.argmax(doc_topic_dist, axis=1)
        )
    except ValueError as e:
        metrics[f"{prefix}silhouette_score"] = None
        print(f"Warning: Could not calculate silhouette score for {model_name} - {e}")

    # Calculate topic distribution statistics
    topic_distributions = model.transform(X_test)
    metrics[f"{prefix}avg_topic_concentration"] = np.mean(
        np.max(topic_distributions, axis=1)
    )
    metrics[f"{prefix}topic_distribution_entropy"] = -np.mean(
        np.sum(topic_distributions * np.log(topic_distributions + 1e-12), axis=1)
    )

    return metrics


# Main function
def main(file_path, n_topics=15, max_features=10000, batch_size=1024, seed=42):
    # Set random seed for reproducibility
    np.random.seed(seed)

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
        lda_model, X_train, X_test, feature_names, lda_topic_matrix, "lda"
    )
    nmf_metrics = evaluate_model(
        nmf_model, X_train, X_test, feature_names, nmf_topic_matrix, "nmf"
    )

    # Evaluate combined model
    combined_metrics = evaluate_model(
        lda_model, X_train, X_test, feature_names, combined_topic_matrix, "combined"
    )

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

    # Identify terms using combined matrix
    df_weeb, df_furry = identify_subculture_terms(lda_model, feature_names, X)

    # Save results
    save_results(df_weeb, df_furry)

    return df_weeb, df_furry, metrics


# Entry point
if __name__ == "__main__":
    # Define the path to the CSV file
    input_file = "output/dataset-filter/bluesky_ten_million_english_only.csv"

    # Run the main function with appropriate batch size for large dataset
    main(input_file, batch_size=2048)  # Increased batch size for efficiency
