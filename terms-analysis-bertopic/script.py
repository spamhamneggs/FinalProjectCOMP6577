#!/usr/bin/env python3
import json
import os
import re

import nltk
import numpy as np
import pandas as pd
import torch
import scipy.sparse
from umap import UMAP
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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


STOP_WORDS = load_stop_words("./thirdparty/stopwords-en.txt")


def tokenize_text(text):
    if not isinstance(text, str):
        return []
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in STOP_WORDS and len(token) > 1]
    return tokens


def extract_tfidf_features_chunked(texts, max_features=5000, min_df=50, max_df=0.85, chunk_size=500000):
    """
    Extract TF-IDF features using chunked processing
    """
    print(f"Extracting TF-IDF features in chunks of {chunk_size}...")

    vocab_sample_size = min(200000, len(texts))
    vocab_indices = np.random.choice(len(texts), vocab_sample_size, replace=False)
    vocab_texts = [texts[i] for i in vocab_indices]

    print(f"Building vocabulary from {len(vocab_texts)} sample documents...")
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

    vectorizer.fit(vocab_texts)
    feature_names_tfidf = vectorizer.get_feature_names_out()
    print(f"Vocabulary built with {len(feature_names_tfidf)} features")

    print("Transforming all documents in chunks...")
    chunk_matrices = []
    num_chunks = int(np.ceil(len(texts) / chunk_size))

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(texts))
        chunk_texts = texts[start_idx:end_idx]
        print(f"Processing chunk {i + 1}/{num_chunks} ({len(chunk_texts)} documents)...")
        
        # Add safety check for empty texts
        non_empty_texts = [t for t in chunk_texts if t and len(t.strip()) > 0]
        if not non_empty_texts:
            print(f"Warning: Chunk {i + 1} has no valid texts, skipping...")
            empty_matrix = scipy.sparse.csr_matrix((len(chunk_texts), len(feature_names_tfidf)))
            chunk_matrices.append(empty_matrix)
            continue
        
        try:
            chunk_matrix = vectorizer.transform(chunk_texts)
            chunk_matrices.append(chunk_matrix)
        except Exception as e:
            print(f"Error processing chunk {i + 1}: {e}")
            empty_matrix = scipy.sparse.csr_matrix((len(chunk_texts), len(feature_names_tfidf)))
            chunk_matrices.append(empty_matrix)

    print("Combining all chunks...")
    X_tfidf = scipy.sparse.vstack(chunk_matrices)
    print(f"Final TF-IDF matrix shape: {X_tfidf.shape}")
    return X_tfidf, vectorizer, feature_names_tfidf


def calculate_doc_counts_chunked(X_tfidf, chunk_size=100000):
    """
    Calculate document counts efficiently using chunked processing
    """
    print("Calculating document counts with chunked processing...")
    n_docs, n_features = X_tfidf.shape
    doc_counts = np.zeros(n_features, dtype=np.int32)
    num_chunks = int(np.ceil(n_docs / chunk_size))
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_docs)
        try:
            chunk = X_tfidf[start_idx:end_idx]
            chunk_counts = np.array((chunk > 0).sum(axis=0)).flatten()
            doc_counts += chunk_counts
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{num_chunks} chunks")
        except Exception as e:
            print(f"Error in chunk {i + 1}: {e}")
            continue
    return doc_counts


def hierarchical_topic_modeling(
    all_processed_texts,
    precomputed_embeddings,
    y_train,
    initial_sample_size=800000,
    refinement_batch_size=200000
):
    """
    Hierarchical approach: build quality model on large sample, then refine with remaining data
    """
    print("Using hierarchical topic modeling approach...")

    print(f"Step 1: Building initial model on {initial_sample_size} samples...")
    if len(all_processed_texts) > initial_sample_size:
        sample_indices = np.random.choice(len(all_processed_texts), initial_sample_size, replace=False)
        sample_texts = [all_processed_texts[i] for i in sample_indices]
        sample_embeddings = precomputed_embeddings[sample_indices]
        sample_y = [y_train[i] for i in sample_indices] if y_train else None
    else:
        sample_texts = all_processed_texts
        sample_embeddings = precomputed_embeddings
        sample_y = y_train
        sample_indices = np.arange(len(all_processed_texts))

    umap_model = UMAP(
        n_neighbors=30,
        n_components=10,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
        low_memory=True,
        n_jobs=1,
    )

    clustering_model = MiniBatchKMeans(
        n_clusters=100,
        batch_size=4096,
        random_state=42,
        n_init="auto",
        verbose=0,
    )

    vectorizer = OnlineCountVectorizer(
        tokenizer=tokenize_text,
        ngram_range=(1, 2),
        min_df=10,
        max_df=0.8,
    )

    initial_model = BERTopic(
        embedding_model="all-MiniLM-L6-v2",
        umap_model=umap_model,
        hdbscan_model=clustering_model,
        vectorizer_model=vectorizer,
        min_topic_size=30,
        verbose=True,
        calculate_probabilities=False,
        low_memory=True,
    )

    print("Fitting initial model...")
    initial_model.fit(sample_texts, embeddings=sample_embeddings, y=sample_y)

    remaining_indices = np.setdiff1d(np.arange(len(all_processed_texts)), sample_indices)
    if len(remaining_indices) > 0:
        print(f"Step 2: Refining with {len(remaining_indices)} remaining documents...")
        num_batches = int(np.ceil(len(remaining_indices) / refinement_batch_size))
        for i in range(num_batches):
            start_idx = i * refinement_batch_size
            end_idx = min((i + 1) * refinement_batch_size, len(remaining_indices))
            batch_indices = remaining_indices[start_idx:end_idx]
            batch_texts = [all_processed_texts[idx] for idx in batch_indices]
            batch_embeddings = precomputed_embeddings[batch_indices]
            batch_y = [y_train[idx] for idx in batch_indices] if y_train else None
            print(f"Refinement batch {i + 1}/{num_batches} ({len(batch_texts)} documents)...")
            try:
                initial_model.partial_fit(batch_texts, embeddings=batch_embeddings, y=batch_y)
                print(f"Batch {i + 1} completed successfully")
            except Exception as e:
                print(f"Error in refinement batch {i + 1}: {e}")
                continue

    return initial_model


def embed_and_save_in_chunks(
    df_full,
    text_column="processed_text",
    output_embedding_file="all_embeddings.npy",
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
    max_return_terms=500,
    threshold_method="auto",
):
    print(f"Identifying terms for {subculture_name} subculture using BERTopic...")
    processed_terms = []

    all_topics = topic_model.get_topics()
    valid_topic_ids = [tid for tid in all_topics.keys() if tid != -1]

    print(f"Found {len(valid_topic_ids)} valid topics to analyze")

    feature_names_dict_tfidf = {
        term: idx for idx, term in enumerate(feature_names_tfidf)
    }
    total_docs_tfidf = X_tfidf.shape[0]

    try:
        doc_counts_tfidf = calculate_doc_counts_chunked(X_tfidf)
    except Exception as e:
        print(f"Error computing document counts: {e}")
        print("Using simplified approach without document prevalence...")
        doc_counts_tfidf = np.zeros(X_tfidf.shape[1])

    term_data = {}

    for topic_id in valid_topic_ids:
        terms_in_topic = topic_model.get_topic(topic_id)
        if terms_in_topic:
            for term, score in terms_in_topic[:top_terms_per_topic]:
                if term not in STOP_WORDS and len(term) > 1:
                    subculture_relevance = 0.0
                    if term.lower() in [s.lower() for s in seed_terms_for_subculture]:
                        subculture_relevance = 1.0
                    elif any(
                        seed.lower() in term.lower()
                        for seed in seed_terms_for_subculture
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

    for term, data in term_data.items():
        term_specificity = 0.0
        seed_closeness = 0.0
        doc_prevalence = 0.0

        if term in feature_names_dict_tfidf and len(doc_counts_tfidf) > 0:
            term_idx_tfidf = feature_names_dict_tfidf[term]
            try:
                if seed_docs_indices_in_Xtfidf:
                    term_specificity = calculate_term_specificity(
                        term_idx_tfidf, X_tfidf, seed_docs_indices_in_Xtfidf
                    )
                seed_closeness = calculate_seed_term_closeness(
                    term_idx_tfidf,
                    feature_names_dict_tfidf,
                    X_tfidf,
                    seed_terms_for_subculture,
                )
                if term_idx_tfidf < len(doc_counts_tfidf):
                    doc_prevalence = float(doc_counts_tfidf[term_idx_tfidf]) / total_docs_tfidf
            except Exception as e:
                print(f"Error processing term {term}: {e}")

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
    # Optionally call get_threshold_summary(df_terms) here if desired
    return df_terms


def evaluate_bertopic_model(topic_model, test_texts_list, model_name_prefix=""):
    metrics = {}
    prefix = f"{model_name_prefix}_" if model_name_prefix else ""
    topic_info = topic_model.get_topic_info()
    metrics[f"{prefix}n_topics"] = len(topic_info[topic_info["Topic"] != -1])
    # Optionally add topic_diversity or other metrics here

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


def save_results(df_weeb, df_furry, output_dir="output/terms-analysis-bertopic-hierarchical"):
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
        "normalized_specificity",
        "normalized_similarity",
        "normalized_contextual_relevance",
        "normalized_seed_closeness",
        "normalized_subculture_relevance",
        "combined_score",
    ]

    for df in [df_weeb, df_furry]:
        if df is None or df.empty:
            continue
        for col in columns:
            if col not in df.columns:
                df[col] = 0.0

    if df_weeb is not None and not df_weeb.empty:
        df_weeb.to_csv(
            os.path.join(output_dir, "weeb_terms_bertopic.csv"),
            index=False,
            columns=columns,
        )
        print(f"Saved {len(df_weeb)} weeb terms")

    if df_furry is not None and not df_furry.empty:
        df_furry.to_csv(
            os.path.join(output_dir, "furry_terms_bertopic.csv"),
            index=False,
            columns=columns,
        )
        print(f"Saved {len(df_furry)} furry terms")

    print(f"Results saved to {output_dir}")


def save_metrics(metrics, metrics_output_dir="metrics/terms-analysis-bertopic-hierarchical"):
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


def main_recommended(file_path, n_topics_hint=50, max_features_tfidf=8000, seed=42):
    """
    Recommended main function using hierarchical + chunked processing
    """
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
            df, text_column="processed_text", output_embedding_file=embedding_file,
            chunk_size=100000, device=embedding_device
        )
        if precomputed_embeddings is None:
            exit(1)
    else:
        print(f"Loading pre-computed embeddings from {embedding_file}...")
        precomputed_embeddings = np.load(embedding_file)

    print(f"Processing {len(precomputed_embeddings)} documents with hierarchical approach")

    _, y_numeric_labels_full = create_guidance_labels(
        df["processed_text"], weeb_seed_terms, furry_seed_terms
    )
    y_numeric_labels_full_np = np.array(y_numeric_labels_full)
    all_indices = np.arange(len(precomputed_embeddings))

    train_indices, test_indices, _, _ = train_test_split(
        all_indices, y_numeric_labels_full_np, test_size=0.2, random_state=seed,
        stratify=y_numeric_labels_full_np if len(set(y_numeric_labels_full_np)) > 1 else None
    )

    train_embeddings = precomputed_embeddings[train_indices]
    train_texts_for_bertopic = [all_processed_texts[i] for i in train_indices]
    y_train = y_numeric_labels_full_np[train_indices].tolist()
    test_texts_for_eval = [all_processed_texts[i] for i in test_indices]

    print("Extracting TF-IDF features with chunked processing...")
    X_tfidf, _, feature_names_tfidf = extract_tfidf_features_chunked(
        all_processed_texts,
        max_features=max_features_tfidf,
        min_df=50,
        chunk_size=500000
    )

    print("Calculating seed document indices...")
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

    print(f"Found {len(weeb_seed_docs_indices_tfidf)} weeb seed docs, "
          f"{len(furry_seed_docs_indices_tfidf)} furry seed docs")

    print("Creating topic model with hierarchical processing...")
    final_topic_model = hierarchical_topic_modeling(
        train_texts_for_bertopic,
        train_embeddings,
        y_train,
        initial_sample_size=800000,
        refinement_batch_size=200000
    )

    print("Extracting subculture terms...")
    df_weeb = identify_subculture_terms_bertopic(
        final_topic_model, "weeb", 0, X_tfidf, feature_names_tfidf,
        weeb_seed_terms, all_processed_texts,
        list(weeb_seed_docs_indices_tfidf), None
    )

    df_furry = identify_subculture_terms_bertopic(
        final_topic_model, "furry", 1, X_tfidf, feature_names_tfidf,
        furry_seed_terms, all_processed_texts,
        list(furry_seed_docs_indices_tfidf), None
    )

    eval_sample_size = min(2000, len(test_texts_for_eval))
    eval_texts = test_texts_for_eval[:eval_sample_size] if test_texts_for_eval else []
    bertopic_metrics = evaluate_bertopic_model(final_topic_model, eval_texts, "bertopic_hierarchical")

    save_results(df_weeb, df_furry, output_dir="output/terms-analysis-bertopic-hierarchical")
    save_metrics(bertopic_metrics, metrics_output_dir="metrics/terms-analysis-bertopic-hierarchical")

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
        default=8000,
        help="Maximum number of TF-IDF features (default: 8000).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)."
    )
    args = parser.parse_args()

    main_recommended(
        file_path=args.input,
        n_topics_hint=args.n_topics_hint,
        max_features_tfidf=args.max_features_tfidf,
        seed=args.seed,
    )
