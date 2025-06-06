#!/usr/bin/env python3
import json
import os
import re

import nltk
import numpy as np
import pandas as pd
import torch
from umap import UMAP
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from hdbscan import HDBSCAN, approximate_predict
from joblib import Memory

# BERTopic imports
from bertopic import BERTopic
from bertopic.vectorizers import OnlineCountVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans

# Download necessary NLTK resources
nltk.download("punkt", quiet=True)

# Create output directories
os.makedirs("output/terms-analysis-bertopic", exist_ok=True)
os.makedirs("metrics/terms-analysis-bertopic", exist_ok=True)

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
    "weeaboo",
    "animu",
    "sugoi",
    "desu",
    "chan",
    "kun",
    "sama",
    "yokai",
    "shounen",
    "shoujo",
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
    "yiff",
    "murr",
    "awoo",
    "feral",
    "anthropomorphic",
    "scalies",
    "protogen",
]


# --- Preprocessing and Utility Functions ---
def load_data(file_path):
    print("Loading data...")
    df = pd.read_csv(file_path)
    df = df[df["text"].notna()]
    df["original_index"] = df.index
    return df


def get_term_docs(term_idx, X_tfidf_sparse):
    return set(X_tfidf_sparse.getcol(term_idx).nonzero()[0])


def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text, flags=re.MULTILINE)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\b\d+\b", "", text)
    return text


def load_stop_words(file_path):
    stop_words = set()
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                word = line.strip()
                if word:
                    stop_words.add(word)
        print(f"Successfully loaded {len(stop_words)} stop words from {file_path}.")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Using empty set.")
    except Exception as e:
        print(f"Error loading stop words: {e}. Using empty set.")
    return stop_words


STOP_WORDS = load_stop_words("./thirdparty/stopwords-custom.txt")


def tokenize_text(text):
    if not isinstance(text, str):
        return []
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in STOP_WORDS and len(token) > 1]
    return tokens


def extract_tfidf_features(texts, max_features=5000, min_df=20, max_df=0.90):
    print("Extracting TF-IDF features (for metrics)...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        tokenizer=tokenize_text,
        preprocessor=None,
        lowercase=False,
        ngram_range=(1, 2),
        sublinear_tf=True,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        strip_accents="unicode",
    )
    print(f"Using TF-IDF with max_df={max_df}, min_df={min_df}")
    X_tfidf = vectorizer.fit_transform(texts)
    feature_names_tfidf = vectorizer.get_feature_names_out()
    n_samples, n_features = X_tfidf.shape
    print(f"Extracted {n_features} TF-IDF features from {n_samples} documents")
    return X_tfidf, vectorizer, feature_names_tfidf


def clean_and_validate_term(term):
    """Clean and validate individual terms to ensure one term per row"""
    if not isinstance(term, str):
        return None
    term = term.strip().lower()
    if not term:
        return None
    words = term.split()
    if len(words) > 2:
        return None
    if any(sep in term for sep in [",", ";", "|", "\t", "/", "\\", "_"]):
        return None
    if len(term) < 2:
        return None
    if not any(c.isalpha() for c in term):
        return None
    if any(c.isdigit() for c in term):
        return None
    if term in STOP_WORDS:
        return None
    return term


def extract_individual_terms_from_topic_terms(topic_terms, max_terms_per_topic=50):
    """Extract and clean individual terms from BERTopic topic terms"""
    individual_terms = []
    seen_terms = set()
    for term, score in topic_terms[:max_terms_per_topic]:
        cleaned_term = clean_and_validate_term(term)
        if cleaned_term and cleaned_term not in seen_terms:
            words = cleaned_term.split()
            if len(words) == 1:
                individual_terms.append((cleaned_term, score))
                seen_terms.add(cleaned_term)
            elif len(words) == 2:
                word1, word2 = words
                if (
                    len(word1) > 2
                    and len(word2) > 2
                    and word1 not in STOP_WORDS
                    and word2 not in STOP_WORDS
                ):
                    individual_terms.append((cleaned_term, score))
                    seen_terms.add(cleaned_term)
                    for word in words:
                        clean_word = clean_and_validate_term(word)
                        if (
                            clean_word
                            and clean_word not in seen_terms
                            and len(clean_word) > 3
                            and clean_word not in STOP_WORDS
                        ):
                            individual_terms.append((clean_word, score * 0.7))
                            seen_terms.add(clean_word)
    term_scores = {}
    for term, score in individual_terms:
        if term not in term_scores or score > term_scores[term]:
            term_scores[term] = score
    result = [(term, score) for term, score in term_scores.items()]
    result.sort(key=lambda x: x[1], reverse=True)
    return result


def validate_dataframe_for_csv(df):
    """Ensure dataframe is properly formatted for CSV output"""
    if df.empty:
        return df
    if "term" in df.columns:
        df["term"] = df["term"].apply(
            lambda x: clean_and_validate_term(str(x)) if pd.notna(x) else None
        )
        df = df.dropna(subset=["term"])
        df = df[df["term"] != ""]
        df = df.sort_values("combined_score", ascending=False)
        df = df.drop_duplicates(subset=["term"], keep="first")
    return df


# --- Metric Calculation Functions ---
def calculate_term_specificity(term_idx_tfidf, X_tfidf, seed_docs_indices_tfidf):
    term_docs_tfidf = get_term_docs(term_idx_tfidf, X_tfidf)
    if not term_docs_tfidf:
        return 0.0
    if not seed_docs_indices_tfidf:
        return np.log(X_tfidf.shape[0] / (len(term_docs_tfidf) + 1))
    seed_docs_set_tfidf = set(seed_docs_indices_tfidf)
    seed_docs_with_term_tfidf = seed_docs_set_tfidf.intersection(term_docs_tfidf)
    if not seed_docs_set_tfidf or not seed_docs_with_term_tfidf:
        return 0.0
    seed_concentration = len(seed_docs_with_term_tfidf) / len(seed_docs_set_tfidf)
    overall_concentration = len(term_docs_tfidf) / X_tfidf.shape[0]
    return seed_concentration / (overall_concentration + 1e-10)


def calculate_seed_term_closeness(
    term_idx_tfidf, feature_names_dict_tfidf, X_tfidf, seed_terms
):
    seed_indices_tfidf = [
        feature_names_dict_tfidf[seed]
        for seed in seed_terms
        if seed in feature_names_dict_tfidf
    ]
    if not seed_indices_tfidf:
        return 0.0
    term_docs_tfidf = get_term_docs(term_idx_tfidf, X_tfidf)
    if not term_docs_tfidf:
        return 0.0
    closeness_scores = []
    for seed_idx_tfidf in seed_indices_tfidf:
        seed_docs_for_one_seed_term = get_term_docs(seed_idx_tfidf, X_tfidf)
        intersection = len(term_docs_tfidf.intersection(seed_docs_for_one_seed_term))
        union = len(term_docs_tfidf.union(seed_docs_for_one_seed_term))
        closeness_scores.append(intersection / union if union > 0 else 0)
    return (
        sum(closeness_scores) / len(seed_indices_tfidf) if seed_indices_tfidf else 0.0
    )


# --- BERTopic Specific Functions ---
def create_guidance_labels(texts_series, weeb_seeds, furry_seeds):
    print("Creating guidance labels for BERTopic...")
    y_labels = []
    y_numeric_labels = []
    weeb_seeds_set = set(weeb_seeds)
    furry_seeds_set = set(furry_seeds)
    for text in texts_series:
        text_str = str(text).lower()
        is_weeb = any(seed in text_str for seed in weeb_seeds_set)
        is_furry = any(seed in text_str for seed in furry_seeds_set)
        if is_weeb and is_furry:
            y_labels.append("both")
            y_numeric_labels.append(2)
        elif is_weeb:
            y_labels.append("weeb")
            y_numeric_labels.append(0)
        elif is_furry:
            y_labels.append("furry")
            y_numeric_labels.append(1)
        else:
            y_labels.append("other")
            y_numeric_labels.append(-1)
    print(
        f"Guidance label counts: Weeb: {y_numeric_labels.count(0)}, "
        f"Furry: {y_numeric_labels.count(1)}, Both: {y_numeric_labels.count(2)}, "
        f"Other: {y_numeric_labels.count(-1)}"
    )
    return y_labels, y_numeric_labels


# --- Enhanced Term Extraction Functions ---
def expand_terms_with_semantic_similarity(
    topic_model,
    initial_terms_df,
    seed_terms,
    all_processed_texts,
    similarity_threshold=0.6,
    max_additional_terms=200,
):
    """Expand term list by finding semantically similar terms using embeddings"""
    print(f"Expanding {len(initial_terms_df)} terms with semantic similarity...")

    if initial_terms_df.empty:
        return initial_terms_df

    # Get embedding model from topic model
    embedding_model = topic_model.embedding_model
    if not hasattr(embedding_model, "encode"):
        print("Cannot access embedding model for semantic expansion")
        return initial_terms_df

    # Get all unique terms from all topics
    all_topics = topic_model.get_topics()
    all_topic_terms = set()
    for topic_id, terms in all_topics.items():
        if topic_id != -1:
            for term, score in terms[:100]:
                if len(term) > 2 and term not in STOP_WORDS:
                    all_topic_terms.add(term)

    # Filter out terms we already have
    existing_terms = set(initial_terms_df["term"].tolist())
    candidate_terms = list(all_topic_terms - existing_terms)

    if not candidate_terms:
        print("No additional candidate terms found")
        return initial_terms_df

    print(f"Checking {len(candidate_terms)} candidate terms for semantic similarity...")

    # Encode seed terms and candidate terms
    try:
        seed_embeddings = embedding_model.encode(seed_terms, show_progress_bar=False)
        candidate_embeddings = embedding_model.encode(
            candidate_terms, show_progress_bar=False
        )

        # Calculate similarities
        similarities = cosine_similarity(candidate_embeddings, seed_embeddings)
        max_similarities = np.max(similarities, axis=1)

        # Find terms above threshold
        similar_indices = np.where(max_similarities >= similarity_threshold)[0]

        expanded_terms = []
        for idx in similar_indices[:max_additional_terms]:
            term = candidate_terms[idx]
            similarity_score = max_similarities[idx]

            expanded_terms.append(
                {
                    "term": term,
                    "subculture": initial_terms_df["subculture"].iloc[0]
                    if not initial_terms_df.empty
                    else "unknown",
                    "similarity": similarity_score,
                    "specificity": 0.0,
                    "contextual_relevance": 0.0,
                    "seed_closeness": similarity_score,
                    "subculture_relevance": similarity_score * 0.8,
                    "uniqueness": 0.5,
                    "combined_score": similarity_score * 0.7,
                }
            )

        if expanded_terms:
            expanded_df = pd.DataFrame(expanded_terms)
            print(f"Found {len(expanded_df)} semantically similar terms")

            # Combine with original terms
            combined_df = pd.concat([initial_terms_df, expanded_df], ignore_index=True)
            return combined_df.sort_values("combined_score", ascending=False)

    except Exception as e:
        print(f"Error in semantic expansion: {e}")

    return initial_terms_df


def find_cooccurring_terms(
    all_processed_texts,
    seed_terms,
    subculture_name,
    window_size=10,
    min_cooccurrence=5,
    max_terms=150,
):
    """Find terms that frequently co-occur with seed terms"""
    print(f"Finding terms that co-occur with {subculture_name} seed terms...")

    cooccurrence_counts = {}
    seed_terms_lower = [term.lower() for term in seed_terms]

    for text in all_processed_texts:
        if not isinstance(text, str):
            continue

        words = text.lower().split()

        # Find positions of seed terms
        seed_positions = []
        for i, word in enumerate(words):
            if word in seed_terms_lower:
                seed_positions.append(i)

        # Count co-occurring terms within window
        for seed_pos in seed_positions:
            start = max(0, seed_pos - window_size)
            end = min(len(words), seed_pos + window_size + 1)

            for i in range(start, end):
                if i != seed_pos:
                    term = words[i]
                    if (
                        len(term) > 2
                        and term not in STOP_WORDS
                        and term not in seed_terms_lower
                        and term.isalpha()
                    ):
                        cooccurrence_counts[term] = cooccurrence_counts.get(term, 0) + 1

    # Filter by minimum co-occurrence and convert to dataframe
    cooccur_terms = []
    for term, count in cooccurrence_counts.items():
        if count >= min_cooccurrence:
            relevance_score = min(1.0, count / 50.0)

            cooccur_terms.append(
                {
                    "term": term,
                    "subculture": subculture_name,
                    "similarity": relevance_score * 0.5,
                    "specificity": 0.0,
                    "contextual_relevance": relevance_score,
                    "seed_closeness": relevance_score,
                    "subculture_relevance": relevance_score * 0.6,
                    "uniqueness": 0.5,
                    "combined_score": relevance_score * 0.4,
                    "cooccurrence_count": count,
                }
            )

    if cooccur_terms:
        cooccur_df = pd.DataFrame(cooccur_terms)
        cooccur_df = cooccur_df.sort_values("cooccurrence_count", ascending=False).head(
            max_terms
        )
        print(f"Found {len(cooccur_df)} co-occurring terms")
        return cooccur_df

    return pd.DataFrame()


def extract_relevant_ngrams(
    all_processed_texts, seed_terms, subculture_name, max_ngrams=100, min_freq=3
):
    """Extract relevant n-grams that contain seed terms or are contextually related"""
    print(f"Extracting relevant n-grams for {subculture_name}...")

    # Create n-gram vectorizer
    ngram_vectorizer = CountVectorizer(
        ngram_range=(2, 3),
        min_df=min_freq,
        max_df=0.3,
        tokenizer=tokenize_text,
        lowercase=True,
        max_features=5000,
    )

    try:
        ngram_matrix = ngram_vectorizer.fit_transform(all_processed_texts)
        ngram_names = ngram_vectorizer.get_feature_names_out()

        # Find n-grams that contain seed terms
        relevant_ngrams = []
        seed_terms_lower = [term.lower() for term in seed_terms]

        for ngram in ngram_names:
            # Check if n-gram contains any seed term
            contains_seed = any(seed_term in ngram for seed_term in seed_terms_lower)

            if contains_seed:
                # Calculate frequency
                ngram_idx = list(ngram_names).index(ngram)
                frequency = ngram_matrix[:, ngram_idx].sum()

                relevance_score = min(1.0, frequency / 100.0)

                relevant_ngrams.append(
                    {
                        "term": ngram,
                        "subculture": subculture_name,
                        "similarity": relevance_score * 0.6,
                        "specificity": 0.0,
                        "contextual_relevance": relevance_score,
                        "seed_closeness": 1.0 if contains_seed else 0.0,
                        "subculture_relevance": relevance_score * 0.8,
                        "uniqueness": 0.5,
                        "combined_score": relevance_score * 0.5,
                        "frequency": frequency,
                    }
                )

        if relevant_ngrams:
            ngram_df = pd.DataFrame(relevant_ngrams)
            ngram_df = ngram_df.sort_values("frequency", ascending=False).head(
                max_ngrams
            )
            print(f"Found {len(ngram_df)} relevant n-grams")
            return ngram_df

    except Exception as e:
        print(f"Error extracting n-grams: {e}")

    return pd.DataFrame()


def identify_subculture_terms_bertopic(
    topic_model,
    subculture_name,
    subculture_numeric_label,
    X_tfidf,
    feature_names_tfidf,
    seed_terms_for_subculture,
    all_doc_texts_list,
    seed_docs_indices_in_Xtfidf,
    doc_indices_map,
    top_terms_per_topic=50,
    threshold_method="auto",
):
    print(f"Identifying terms for {subculture_name} subculture using BERTopic...")
    processed_terms = []

    # Get all topics and their terms
    all_topics = topic_model.get_topics()
    valid_topic_ids = [tid for tid in all_topics.keys() if tid != -1]

    print(f"Found {len(valid_topic_ids)} valid topics to analyze")

    feature_names_dict_tfidf = {
        term: idx for idx, term in enumerate(feature_names_tfidf)
    }
    total_docs_tfidf = X_tfidf.shape[0]
    doc_counts_tfidf = np.array(X_tfidf.sum(axis=0)).flatten()
    term_data = {}

    # Extract terms from all topics with aggressive cleaning
    for topic_id in valid_topic_ids:
        raw_terms_in_topic = topic_model.get_topic(topic_id)
        if raw_terms_in_topic:
            cleaned_terms = extract_individual_terms_from_topic_terms(
                raw_terms_in_topic, max_terms_per_topic=top_terms_per_topic
            )
            for term, score in cleaned_terms:
                if term not in STOP_WORDS and len(term) > 2 and len(term) < 20:
                    subculture_relevance = 0.0
                    term_lower = term.lower()
                    if term_lower in [s.lower() for s in seed_terms_for_subculture]:
                        subculture_relevance = 1.0
                    elif any(
                        seed.lower() in term_lower for seed in seed_terms_for_subculture
                    ):
                        subculture_relevance = 0.7
                    elif any(
                        term_lower in seed.lower() for seed in seed_terms_for_subculture
                    ):
                        subculture_relevance = 0.5
                    if term not in term_data:
                        term_data[term] = {
                            "bertopic_score": 0.0,
                            "count": 0,
                            "subculture_relevance": 0.0,
                        }
                    term_data[term]["bertopic_score"] = max(
                        term_data[term]["bertopic_score"], score
                    )
                    term_data[term]["count"] += 1
                    term_data[term]["subculture_relevance"] = max(
                        term_data[term]["subculture_relevance"], subculture_relevance
                    )

    if not term_data:
        print(f"No terms extracted for {subculture_name} from BERTopic topics.")
        return pd.DataFrame()

    # Process terms with enhanced scoring
    for term, data in term_data.items():
        term_specificity = 0.0
        seed_closeness = 0.0
        doc_prevalence = 0.0

        if term in feature_names_dict_tfidf:
            term_idx_tfidf = feature_names_dict_tfidf[term]
            term_specificity = calculate_term_specificity(
                term_idx_tfidf, X_tfidf, seed_docs_indices_in_Xtfidf
            )
            seed_closeness = calculate_seed_term_closeness(
                term_idx_tfidf,
                feature_names_dict_tfidf,
                X_tfidf,
                seed_terms_for_subculture,
            )
            if doc_counts_tfidf.size > term_idx_tfidf:
                doc_prevalence = (
                    float(X_tfidf[:, term_idx_tfidf].count_nonzero()) / total_docs_tfidf
                )

        processed_terms.append(
            {
                "term": term,
                "subculture": subculture_name,
                "similarity": data["bertopic_score"],
                "specificity": term_specificity * 1e9 if term_specificity else 0.0,
                "contextual_relevance": doc_prevalence,
                "seed_closeness": seed_closeness,
                "subculture_relevance": data["subculture_relevance"],
                "uniqueness": 0.5,
            }
        )

    if not processed_terms:
        return pd.DataFrame()

    df_terms = pd.DataFrame(processed_terms)

    df_terms = df_terms[
        (df_terms["subculture_relevance"] > 0)
        | (df_terms["seed_closeness"] > 0.1)
        | (df_terms["specificity"] > 0)
    ]

    if len(df_terms) > 1:
        features_to_normalize = [
            "specificity",
            "similarity",
            "contextual_relevance",
            "seed_closeness",
            "subculture_relevance",
        ]
        valid_features = []
        for f in features_to_normalize:
            if f in df_terms.columns and df_terms[f].sum() > 0:
                if df_terms[f].nunique() > 1:
                    valid_features.append(f)
                else:
                    df_terms[f"normalized_{f}"] = (
                        0.5 if df_terms[f].iloc[0] != 0 else 0.0
                    )

        if valid_features:
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(df_terms[valid_features])
            for i, feature in enumerate(valid_features):
                df_terms[f"normalized_{feature}"] = normalized_data[:, i]

        for f in features_to_normalize:
            if f"normalized_{f}" not in df_terms.columns:
                df_terms[f"normalized_{f}"] = 0.0

        weights = {
            "specificity": 0.20,
            "similarity": 0.15,
            "contextual_relevance": 0.10,
            "seed_closeness": 0.25,
            "subculture_relevance": 0.30,
        }

        df_terms["combined_score"] = (
            weights["specificity"] * df_terms["normalized_specificity"]
            + weights["similarity"] * df_terms["normalized_similarity"]
            + weights["contextual_relevance"]
            * (1 - df_terms["normalized_contextual_relevance"])
            + weights["seed_closeness"] * df_terms["normalized_seed_closeness"]
            + weights["subculture_relevance"]
            * df_terms["normalized_subculture_relevance"]
        )
    elif len(df_terms) == 1:
        df_terms["combined_score"] = 0.5

    df_terms = df_terms.sort_values("combined_score", ascending=False)

    # --- Output all terms, skip thresholding ---
    final_terms = df_terms
    print(f"Outputting all {len(final_terms)} terms (no thresholding applied)")
    return final_terms


def identify_subculture_terms_comprehensive(
    topic_model,
    subculture_name,
    subculture_numeric_label,
    X_tfidf,
    feature_names_tfidf,
    seed_terms_for_subculture,
    all_doc_texts_list,
    seed_docs_indices_in_Xtfidf,
    doc_indices_map,
):
    """Comprehensive term identification using multiple methods"""
    print(f"\n=== Comprehensive Term Identification for {subculture_name} ===")

    # Method 1: Original BERTopic-based extraction
    df_topic_terms = identify_subculture_terms_bertopic(
        topic_model,
        subculture_name,
        subculture_numeric_label,
        X_tfidf,
        feature_names_tfidf,
        seed_terms_for_subculture,
        all_doc_texts_list,
        seed_docs_indices_in_Xtfidf,
        doc_indices_map,
        top_terms_per_topic=50,
        threshold_method="auto",
    )
    print(f"BERTopic method: {len(df_topic_terms)} terms")

    # Method 2: Semantic similarity expansion
    df_semantic = expand_terms_with_semantic_similarity(
        topic_model,
        df_topic_terms,
        seed_terms_for_subculture,
        all_doc_texts_list,
        similarity_threshold=0.5,
        max_additional_terms=200,
    )
    print(f"After semantic expansion: {len(df_semantic)} terms")

    # Method 3: Co-occurrence analysis
    df_cooccur = find_cooccurring_terms(
        all_doc_texts_list,
        seed_terms_for_subculture,
        subculture_name,
        window_size=15,
        min_cooccurrence=3,
        max_terms=150,
    )

    # Method 4: N-gram extraction
    df_ngrams = extract_relevant_ngrams(
        all_doc_texts_list,
        seed_terms_for_subculture,
        subculture_name,
        max_ngrams=100,
        min_freq=2,
    )

    # Combine all methods
    all_dfs = [df for df in [df_semantic, df_cooccur, df_ngrams] if not df.empty]

    if len(all_dfs) > 1:
        combined_df = pd.concat(all_dfs, ignore_index=True)

        # Remove duplicates, keeping the one with highest combined_score
        combined_df = combined_df.sort_values("combined_score", ascending=False)
        combined_df = combined_df.drop_duplicates(subset=["term"], keep="first")

        # Ensure all required columns exist
        required_columns = [
            "term",
            "specificity",
            "similarity",
            "contextual_relevance",
            "seed_closeness",
            "subculture_relevance",
            "uniqueness",
            "combined_score",
        ]
        for col in required_columns:
            if col not in combined_df.columns:
                combined_df[col] = 0.0

        # Output all combined terms, skip thresholding
        final_df = combined_df

        print(f"\n=== Final Results for {subculture_name} ===")
        print(f"Total unique terms: {len(final_df)}")
        print(f"Top 10 terms: {final_df['term'].head(10).tolist()}")

        return final_df

    elif len(all_dfs) == 1:
        return all_dfs[0]
    else:
        return pd.DataFrame()


def calculate_topic_diversity_bertopic(topic_model, top_n=10):
    print("Calculating BERTopic topic diversity...")
    topic_representations = topic_model.get_topics()
    valid_topic_ids = [tid for tid in topic_representations if tid != -1]
    if len(valid_topic_ids) < 2:
        return 0.0
    all_words = set(
        word
        for topic_id in valid_topic_ids
        for word, score in topic_representations[topic_id][:top_n]
    )
    word_to_idx = {word: i for i, word in enumerate(all_words)}
    top_words_indices_sets = []
    for topic_id in valid_topic_ids:
        current_set = set(
            word_to_idx[word]
            for word, score in topic_representations[topic_id][:top_n]
            if word in word_to_idx
        )
        if current_set:
            top_words_indices_sets.append(current_set)
    num_valid_sets = len(top_words_indices_sets)
    if num_valid_sets < 2:
        return 0.0
    similarity_sum = 0
    pair_count = 0
    for i in range(num_valid_sets):
        for j in range(i + 1, num_valid_sets):
            intersection = len(
                top_words_indices_sets[i].intersection(top_words_indices_sets[j])
            )
            union = len(top_words_indices_sets[i].union(top_words_indices_sets[j]))
            similarity_sum += intersection / union if union > 0 else 0
            pair_count += 1
    if pair_count == 0:
        return 1.0
    return 1 - (similarity_sum / pair_count)


def evaluate_bertopic_model(topic_model, test_texts_list, model_name_prefix=""):
    metrics = {}
    prefix = f"{model_name_prefix}_" if model_name_prefix else ""
    topic_info = topic_model.get_topic_info()
    metrics[f"{prefix}n_topics"] = len(topic_info[topic_info["Topic"] != -1])

    print(
        f"Evaluating BERTopic model ({prefix})... Test set size: {len(test_texts_list)}"
    )
    sample_size_eval = min(2000, len(test_texts_list))
    if len(test_texts_list) > sample_size_eval:
        eval_indices = np.random.choice(
            len(test_texts_list), sample_size_eval, replace=False
        )
        sampled_test_texts = [test_texts_list[i] for i in eval_indices]
    else:
        sampled_test_texts = test_texts_list

    if not sampled_test_texts:
        print("Warning: No texts to evaluate for BERTopic.")
        return metrics

    try:
        print("Transforming test texts for BERTopic evaluation...")
        topic_assignments_test, _ = topic_model.transform(sampled_test_texts)

        print("Extracting embeddings for BERTopic evaluation...")

        embedding_model = None
        test_embeddings_np = None

        if (
            hasattr(topic_model, "embedding_model")
            and topic_model.embedding_model is not None
        ):
            embedding_model = topic_model.embedding_model
            if hasattr(embedding_model, "encode"):
                test_embeddings_np = np.asarray(
                    embedding_model.encode(sampled_test_texts, show_progress_bar=False)
                )

        if test_embeddings_np is None and hasattr(topic_model, "embedding_model"):
            if isinstance(topic_model.embedding_model, str):
                print(f"Loading embedding model: {topic_model.embedding_model}")
                embedding_model = SentenceTransformer(topic_model.embedding_model)
                test_embeddings_np = np.asarray(
                    embedding_model.encode(sampled_test_texts, show_progress_bar=False)
                )

        if test_embeddings_np is None:
            print("Creating fallback embedding model for evaluation...")
            embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            test_embeddings_np = np.asarray(
                embedding_model.encode(sampled_test_texts, show_progress_bar=False)
            )

        if test_embeddings_np is not None:
            topic_assignments_test_np = np.asarray(topic_assignments_test)
            valid_indices = topic_assignments_test_np != -1
            num_valid_points = np.sum(valid_indices)
            num_unique_labels = len(np.unique(topic_assignments_test_np[valid_indices]))

            if num_valid_points > 1 and num_unique_labels > 1:
                try:
                    metrics[f"{prefix}silhouette_score"] = silhouette_score(
                        test_embeddings_np[valid_indices],
                        topic_assignments_test_np[valid_indices],
                    )
                except Exception as e:
                    print(f"Silhouette score error: {e}")
                try:
                    metrics[f"{prefix}davies_bouldin_score"] = davies_bouldin_score(
                        test_embeddings_np[valid_indices],
                        topic_assignments_test_np[valid_indices],
                    )
                except Exception as e:
                    print(f"Davies-Bouldin score error: {e}")
            else:
                print(
                    "Skipping Silhouette/DB due to insufficient clusters/points after outlier removal."
                )
        else:
            print("Warning: Could not generate embeddings for evaluation.")

    except Exception as e:
        print(f"Error during BERTopic evaluation: {e}")

    return metrics


def get_terms_by_quality_tier(df_terms, tier="all"):
    """Return terms based on quality tier"""
    if df_terms.empty:
        return df_terms

    if tier == "high":
        return df_terms[
            (df_terms["combined_score"] >= 0.4)
            | (df_terms["subculture_relevance"] >= 0.8)
        ]
    elif tier == "medium":
        return df_terms[
            (df_terms["combined_score"] >= 0.2)
            & (df_terms["subculture_relevance"] >= 0.3)
        ]
    elif tier == "low":
        return df_terms[df_terms["combined_score"] >= 0.05]
    else:  # 'all'
        return df_terms


def save_results_tiered(df_weeb, df_furry, output_dir="output/terms-analysis-bertopic"):
    """Save all terms to a single CSV file per subculture (no tiers)"""
    print("Saving results to files...")
    os.makedirs(output_dir, exist_ok=True)

    columns = [
        "term",
        "specificity",
        "similarity",
        "contextual_relevance",
        "seed_closeness",
        "subculture_relevance",
        "uniqueness",
        "combined_score",
    ]

    for df, name in [(df_weeb, "weeb"), (df_furry, "furry")]:
        if df is None or df.empty:
            continue

        # Validate and clean the dataframe
        df = validate_dataframe_for_csv(df)

        if df.empty:
            print(f"Warning: No valid terms remaining for {name} after validation")
            continue

        # Ensure all columns exist
        for col in columns:
            if col not in df.columns:
                df[col] = 0.0

        # Save all terms to a single CSV file
        df.to_csv(os.path.join(output_dir, f"{name}_terms_bertopic.csv"), index=False)
        print(f"Saved {name} terms - All: {len(df)}")


def save_metrics(metrics, metrics_output_dir="metrics/terms-analysis-bertopic"):
    print("Saving metrics to file...")
    os.makedirs(metrics_output_dir, exist_ok=True)
    metrics_file = os.path.join(metrics_output_dir, "model_metrics_bertopic.json")
    formatted_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (np.float32, np.float64, float)):
            formatted_metrics[key] = round(float(value), 4)
        elif isinstance(value, (np.int32, np.int64, int)):
            formatted_metrics[key] = int(value)
        else:
            formatted_metrics[key] = value
    with open(metrics_file, "w") as f:
        json.dump(formatted_metrics, f, indent=4)
    print(f"Metrics saved to {metrics_file}")


def embed_and_save_in_chunks(
    df_full,
    text_column="processed_text",
    output_embedding_file="all_document_embeddings.npy",
    chunk_size=100000,
    embedding_model_name="all-MiniLM-L6-v2",
    device=None,
):
    print(f"Loading embedding model: {embedding_model_name}")
    model = SentenceTransformer(embedding_model_name, device=device)
    print(f"Using device: {model.device} for embeddings.")
    all_embeddings_list = []
    num_chunks = int(np.ceil(len(df_full) / chunk_size))
    print(
        f"Total documents: {len(df_full)}, Chunk size: {chunk_size}, Num chunks: {num_chunks}"
    )

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(df_full))
        chunk_texts = df_full[text_column][start_idx:end_idx].tolist()
        print(
            f"Processing chunk {i + 1}/{num_chunks} (docs {start_idx}-{end_idx - 1})..."
        )

        if not chunk_texts:
            print(f"Skipping empty chunk {i + 1}")
            continue

        try:
            chunk_embeddings = model.encode(
                chunk_texts, show_progress_bar=True, batch_size=128
            )
            all_embeddings_list.append(chunk_embeddings)
            print(f"Chunk {i + 1} embedded. Shape: {chunk_embeddings.shape}")
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) and model.device.type == "cuda":
                print(
                    f"CUDA OOM error in chunk {i + 1}. Try smaller batch or CPU for this chunk."
                )
                raise e
            else:
                raise e
        except Exception as e:
            print(f"An error occurred during embedding chunk {i + 1}: {e}")
            raise e

    print("Concatenating all chunk embeddings...")
    if not all_embeddings_list:
        print("No embeddings were generated.")
        return None
    final_embeddings = np.vstack(all_embeddings_list)
    print(f"Final embeddings shape: {final_embeddings.shape}")
    print(f"Saving embeddings to {output_embedding_file}...")
    np.save(output_embedding_file, final_embeddings)
    print("Embeddings saved.")
    return final_embeddings


def transform_embeddings_in_chunks(umap_model, embeddings, chunk_size=50000):
    """Transform embeddings in chunks to manage memory"""
    transformed_chunks = []
    num_chunks = int(np.ceil(len(embeddings) / chunk_size))
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(embeddings))
        chunk = embeddings[start_idx:end_idx]
        transformed_chunk = umap_model.transform(chunk)
        transformed_chunks.append(transformed_chunk)
    return np.vstack(transformed_chunks)


def perform_kmeans_fallback(reduced_embeddings_all, min_cluster_size_base):
    """Fallback clustering using MiniBatchKMeans"""
    n_samples = len(reduced_embeddings_all)
    n_clusters = max(10, min(200, n_samples // min_cluster_size_base))

    print(f"Using MiniBatchKMeans fallback with {n_clusters} clusters...")

    kmeans_fallback = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=4096,
        random_state=42,
        n_init="auto",
        verbose=0,
    )

    final_labels = kmeans_fallback.fit_predict(reduced_embeddings_all)
    print(f"MiniBatchKMeans completed with {len(set(final_labels))} clusters")

    return final_labels, kmeans_fallback


def perform_final_clustering(
    reduced_embeddings_all, min_cluster_size_base=20, max_samples_hdbscan=100000
):
    """Perform final clustering with memory-aware HDBSCAN or fallback methods"""
    n_samples = len(reduced_embeddings_all)
    print(f"Performing final clustering on {n_samples} samples...")

    # Strategy 1: Try HDBSCAN on full dataset if reasonable size
    if n_samples <= max_samples_hdbscan:
        print("Attempting HDBSCAN on full dataset...")
        try:
            hdb = HDBSCAN(
                min_cluster_size=max(min_cluster_size_base, n_samples // 1000),
                min_samples=5,
                metric="euclidean",
                algorithm="ball_tree",
                memory=Memory(location=None),
                n_jobs=1,
            )
            final_labels = hdb.fit_predict(reduced_embeddings_all)
            print(
                f"HDBSCAN completed successfully with {len(set(final_labels)) - (1 if -1 in final_labels else 0)} clusters"
            )
            return final_labels, hdb
        except (MemoryError, np.core._exceptions._ArrayMemoryError) as e:
            print(f"HDBSCAN failed due to memory: {e}")
            print("Falling back to sampling strategy...")

    # Strategy 2: Sample-based HDBSCAN with label propagation
    if n_samples > max_samples_hdbscan:
        print("Dataset too large for direct HDBSCAN. Using sampling strategy...")

        sample_size = max_samples_hdbscan
        sample_indices = np.random.choice(n_samples, sample_size, replace=False)
        sample_embeddings = reduced_embeddings_all[sample_indices]

        print(f"Running HDBSCAN on {sample_size} sampled points...")
        try:
            hdb = HDBSCAN(
                min_cluster_size=max(min_cluster_size_base, sample_size // 500),
                min_samples=5,
                metric="euclidean",
                algorithm="ball_tree",
                memory=Memory(location=None),
                n_jobs=1,
                prediction_data=True,
            )
            sample_labels = hdb.fit_predict(sample_embeddings)

            # Try approximate_predict first
            try:
                print("Attempting hdbscan.approximate_predict for label propagation...")
                pred_labels, _ = approximate_predict(hdb, reduced_embeddings_all)
                print(
                    f"approximate_predict completed. {len(set(pred_labels)) - (1 if -1 in pred_labels else 0)} clusters"
                )
                return pred_labels, hdb
            except Exception as e:
                print(f"approximate_predict failed: {e}")
                print("Falling back to NearestNeighbors cluster-centers propagation...")

            # Fallback: NearestNeighbors to cluster centers
            print("Propagating cluster labels to all points using NearestNeighbors...")

            unique_labels = set(sample_labels) - {-1}
            if len(unique_labels) == 0:
                print("No clusters found in sample. Using MiniBatchKMeans fallback.")
                return perform_kmeans_fallback(
                    reduced_embeddings_all, min_cluster_size_base
                )

            cluster_centers = []
            cluster_label_mapping = []
            for label in unique_labels:
                cluster_mask = sample_labels == label
                if np.sum(cluster_mask) > 0:
                    center = np.mean(sample_embeddings[cluster_mask], axis=0)
                    cluster_centers.append(center)
                    cluster_label_mapping.append(label)

            if len(cluster_centers) == 0:
                print("No valid cluster centers found. Using MiniBatchKMeans fallback.")
                return perform_kmeans_fallback(
                    reduced_embeddings_all, min_cluster_size_base
                )

            cluster_centers = np.array(cluster_centers)

            nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
            nn.fit(cluster_centers)

            chunk_size = 50000
            final_labels = np.full(n_samples, -1, dtype=int)

            for i in range(0, n_samples, chunk_size):
                end_idx = min(i + chunk_size, n_samples)
                chunk_embeddings = reduced_embeddings_all[i:end_idx]

                distances, indices = nn.kneighbors(chunk_embeddings)
                max_distance = np.percentile(distances, 95)

                for j, (dist, idx) in enumerate(
                    zip(distances.flatten(), indices.flatten())
                ):
                    if dist <= max_distance:
                        final_labels[i + j] = cluster_label_mapping[idx]

            print(
                f"Label propagation completed. {len(set(final_labels)) - (1 if -1 in final_labels else 0)} clusters"
            )
            return final_labels, hdb

        except (MemoryError, np.core._exceptions._ArrayMemoryError) as e:
            print(f"Sampled HDBSCAN also failed: {e}")
            print("Using MiniBatchKMeans fallback...")

    # Strategy 3: MiniBatchKMeans fallback
    return perform_kmeans_fallback(reduced_embeddings_all, min_cluster_size_base)


def main(file_path, n_topics_hint=50, max_features_tfidf=10000, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

    df = load_data(file_path)
    print("Preprocessing text data...")
    df["processed_text"] = df["text"].apply(preprocess_text)
    all_processed_texts = df["processed_text"].tolist()

    embedding_file = "all_document_embeddings.npy"
    if not os.path.exists(embedding_file):
        print("Pre-computed embeddings not found. Generating and saving them...")
        embedding_device = "cuda" if torch.cuda.is_available() else "cpu"
        precomputed_embeddings = embed_and_save_in_chunks(
            df,
            text_column="processed_text",
            output_embedding_file=embedding_file,
            chunk_size=100000,
            device=embedding_device,
        )
        if precomputed_embeddings is None:
            exit(1)
    else:
        print(f"Loading pre-computed embeddings from {embedding_file}...")
        precomputed_embeddings = np.load(embedding_file)

    print(f"Loaded embeddings with shape: {precomputed_embeddings.shape}")
    if len(precomputed_embeddings) != len(df):
        exit(1)

    _, y_numeric_labels_full = create_guidance_labels(
        df["processed_text"], weeb_seed_terms, furry_seed_terms
    )
    y_numeric_labels_full_np = np.array(y_numeric_labels_full)
    all_indices = np.arange(len(precomputed_embeddings))

    # Stratified split for train/test indices
    train_indices, test_indices, _, _ = train_test_split(
        all_indices,
        y_numeric_labels_full_np,
        test_size=0.2,
        random_state=seed,
        stratify=y_numeric_labels_full_np
        if len(set(y_numeric_labels_full_np)) > 1
        else None,
    )

    train_embeddings = precomputed_embeddings[train_indices]
    train_texts_for_bertopic = [all_processed_texts[i] for i in train_indices]
    y_train = y_numeric_labels_full_np[train_indices].tolist()
    test_texts_for_eval = [all_processed_texts[i] for i in test_indices]

    X_tfidf, _, feature_names_tfidf = extract_tfidf_features(
        all_processed_texts, max_features=max_features_tfidf
    )

    weeb_seed_docs_indices_tfidf = set()
    furry_seed_docs_indices_tfidf = set()
    feature_names_dict_tfidf = {name: i for i, name in enumerate(feature_names_tfidf)}

    for sl, ts in [
        (weeb_seed_terms, weeb_seed_docs_indices_tfidf),
        (furry_seed_terms, furry_seed_docs_indices_tfidf),
    ]:
        for st in sl:
            if st in feature_names_dict_tfidf:
                ts.update(get_term_docs(feature_names_dict_tfidf[st], X_tfidf))

    print(
        f"TF-IDF Weeb seed docs: {len(weeb_seed_docs_indices_tfidf)}, "
        f"Furry: {len(furry_seed_docs_indices_tfidf)}"
    )

    # Improved UMAP parameters
    umap_model = UMAP(
        n_neighbors=30,
        n_components=10,
        min_dist=0.1,
        metric="cosine",
        random_state=seed,
        low_memory=True,
        verbose=True,
        n_jobs=1,
    )
    print(
        f"UMAP: n_neighbors={umap_model.n_neighbors}, n_components={umap_model.n_components}"
    )

    # --- HYBRID INCREMENTAL PIPELINE ---
    kmeans_n_clusters = 100
    online_min_df = 5
    bertopic_min_topic_size = 30

    kmeans = MiniBatchKMeans(
        n_clusters=kmeans_n_clusters,
        batch_size=4096,
        random_state=seed,
        n_init="auto",
        verbose=0,
    )
    print(f"MiniBatchKMeans: n_clusters={kmeans_n_clusters}")

    online_vectorizer = OnlineCountVectorizer(
        tokenizer=tokenize_text,
        ngram_range=(1, 3),
        min_df=online_min_df,
        max_df=0.85,
    )
    print(f"OnlineCountVectorizer: min_df={online_min_df}")

    topic_model_km = BERTopic(
        embedding_model="all-MiniLM-L6-v2",
        umap_model=umap_model,
        hdbscan_model=kmeans,
        vectorizer_model=online_vectorizer,
        min_topic_size=bertopic_min_topic_size,
        verbose=True,
        calculate_probabilities=False,
        low_memory=True,
    )

    # Improved initial fit size
    initial_fit_ratio = 0.30
    initial_fit_size = int(len(train_embeddings) * initial_fit_ratio)
    initial_fit_size = min(initial_fit_size, 300000)
    initial_fit_size = max(initial_fit_size, 75000)
    initial_fit_size = min(initial_fit_size, len(train_embeddings))

    print(
        f"Total training samples: {len(train_embeddings)}, Initial fit size: {initial_fit_size}"
    )

    initial_fit_texts = train_texts_for_bertopic[:initial_fit_size]
    initial_fit_embeddings = train_embeddings[:initial_fit_size]
    initial_fit_y = y_train[:initial_fit_size] if y_train else None

    print(f"Starting initial BERTopic fit with {len(initial_fit_texts)} documents...")
    try:
        topic_model_km.fit(
            documents=initial_fit_texts,
            embeddings=initial_fit_embeddings,
            y=initial_fit_y,
        )
        print("Initial BERTopic fit completed.")
    except Exception as e:
        print(f"Error during initial BERTopic training: {e}")
        exit(1)

    partial_fit_texts = train_texts_for_bertopic[initial_fit_size:]
    partial_fit_embeddings = train_embeddings[initial_fit_size:]
    partial_fit_y = y_train[initial_fit_size:] if y_train else None

    if len(partial_fit_texts) > 0:
        print(f"Proceeding with partial_fit for {len(partial_fit_texts)} documents...")
        partial_fit_chunk_size = 50000
        num_partial_fit_chunks = int(
            np.ceil(len(partial_fit_texts) / partial_fit_chunk_size)
        )

        for i in range(num_partial_fit_chunks):
            start_idx = i * partial_fit_chunk_size
            end_idx = min((i + 1) * partial_fit_chunk_size, len(partial_fit_texts))
            chunk_texts = partial_fit_texts[start_idx:end_idx]
            chunk_embeddings = partial_fit_embeddings[start_idx:end_idx]
            chunk_y = partial_fit_y[start_idx:end_idx] if partial_fit_y else None

            if not chunk_texts:
                continue

            print(
                f"Partial_fit chunk {i + 1}/{num_partial_fit_chunks} with {len(chunk_texts)} documents..."
            )
            try:
                topic_model_km.partial_fit(
                    documents=chunk_texts, embeddings=chunk_embeddings, y=chunk_y
                )
            except Exception as e:
                print(f"Error in partial_fit chunk {i + 1}: {e}")
                raise e
        print("BERTopic partial_fit completed.")
    else:
        print("No additional data for partial_fit.")

    # --- Step B: final refinement with memory-aware clustering ---
    print("\n--- Transforming embeddings with UMAP for final clustering ---")

    # UMAP transformer selection logic
    umap_transformer = None
    if (
        hasattr(topic_model_km, "umap_model_")
        and topic_model_km.umap_model_ is not None
    ):
        umap_transformer = topic_model_km.umap_model_
    elif (
        hasattr(topic_model_km, "umap_model") and topic_model_km.umap_model is not None
    ):
        umap_transformer = topic_model_km.umap_model
    else:
        print("No UMAP model available. Skipping final refinement.")
        final_topic_model = topic_model_km
        final_labels = None

    if umap_transformer is not None:
        try:
            # Use chunked UMAP transformation if needed
            if len(precomputed_embeddings) > 200000:
                print("Large dataset detected. Transforming embeddings in chunks...")
                reduced_embeddings_all = transform_embeddings_in_chunks(
                    umap_transformer, precomputed_embeddings, chunk_size=50000
                )
            else:
                reduced_embeddings_all = umap_transformer.transform(
                    precomputed_embeddings
                )

            print(
                f"UMAP transformation completed. Shape: {reduced_embeddings_all.shape}"
            )

            # Perform memory-aware final clustering
            final_labels, clustering_model = perform_final_clustering(
                reduced_embeddings_all,
                min_cluster_size_base=20,
                max_samples_hdbscan=75000,
            )

            # Create refined topic model
            if final_labels is not None:
                print("Creating refined BERTopic model with original embeddings...")

                final_topic_model = BERTopic(
                    embedding_model="all-MiniLM-L6-v2",
                    vectorizer_model=online_vectorizer,
                    min_topic_size=10,
                    verbose=True,
                    calculate_probabilities=False,
                    low_memory=True,
                )

                docs = df["processed_text"].tolist()
                final_topic_model.fit_transform(
                    docs, embeddings=precomputed_embeddings, y=final_labels.tolist()
                )

                print("Refined BERTopic model created successfully.")
            else:
                print("Using original model as final_topic_model.")
                final_topic_model = topic_model_km

        except Exception as e:
            print(f"Error during UMAP transformation or clustering: {e}")
            print("Using original model without refinement.")
            final_topic_model = topic_model_km
            final_labels = None
    else:
        final_topic_model = topic_model_km
        final_labels = None

    # Post-training diagnostics
    print("\n--- Final Model Topics ---")
    freq_topics = final_topic_model.get_topic_freq()
    if not freq_topics.empty:
        print("Topic Frequencies (Top 15):")
        print(freq_topics.head(15))
        for topic_id in freq_topics["Topic"].head(15).tolist():
            if topic_id != -1:
                terms = final_topic_model.get_topic(topic_id)
                print(f"Topic {topic_id}: {terms[:8] if terms else 'No terms'}")
    else:
        print("No topics found in final model.")

    # Extract terms using comprehensive approach
    df_weeb = identify_subculture_terms_comprehensive(
        final_topic_model,
        "weeb",
        0,
        X_tfidf,
        feature_names_tfidf,
        weeb_seed_terms,
        all_processed_texts,
        list(weeb_seed_docs_indices_tfidf),
        None,
    )

    df_furry = identify_subculture_terms_comprehensive(
        final_topic_model,
        "furry",
        1,
        X_tfidf,
        feature_names_tfidf,
        furry_seed_terms,
        all_processed_texts,
        list(furry_seed_docs_indices_tfidf),
        None,
    )

    # Evaluation and saving
    eval_sample_size = min(2000, len(test_texts_for_eval))
    eval_texts = test_texts_for_eval[:eval_sample_size] if test_texts_for_eval else []

    bertopic_metrics = evaluate_bertopic_model(
        final_topic_model, eval_texts, "bertopic_hybrid_refined"
    )

    # Add clustering metrics if available
    if final_labels is not None:
        valid_indices = final_labels != -1
        if np.sum(valid_indices) > 1 and len(set(final_labels[valid_indices])) > 1:
            try:
                # Sample for metrics calculation if too large
                if np.sum(valid_indices) > 10000:
                    valid_sample_indices = np.random.choice(
                        np.where(valid_indices)[0], 10000, replace=False
                    )
                    sample_embeddings = precomputed_embeddings[valid_sample_indices]
                    sample_labels = final_labels[valid_sample_indices]
                else:
                    sample_embeddings = precomputed_embeddings[valid_indices]
                    sample_labels = final_labels[valid_indices]

                sil = silhouette_score(sample_embeddings, sample_labels)
                bertopic_metrics["final_clustering_silhouette_score"] = round(
                    float(sil), 4
                )

                db = davies_bouldin_score(sample_embeddings, sample_labels)
                bertopic_metrics["final_clustering_davies_bouldin_score"] = round(
                    float(db), 4
                )

            except Exception as e:
                print(f"Error calculating clustering metrics: {e}")

    save_results_tiered(df_weeb, df_furry, output_dir="output/terms-analysis-bertopic")
    save_metrics(
        bertopic_metrics,
        metrics_output_dir="metrics/terms-analysis-bertopic",
    )

    return df_weeb, df_furry, bertopic_metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run BERTopic subculture term analysis."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file with a 'text' column.",
    )
    parser.add_argument(
        "--n_topics_hint",
        type=int,
        default=50,
        help="Hint for the number of topics (default: 50).",
    )
    parser.add_argument(
        "--max_features_tfidf",
        type=int,
        default=10000,
        help="Maximum number of TF-IDF features (default: 10000).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)."
    )
    args = parser.parse_args()

    main(
        file_path=args.input,
        n_topics_hint=args.n_topics_hint,
        max_features_tfidf=args.max_features_tfidf,
        seed=args.seed,
    )
