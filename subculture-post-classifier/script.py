import logging
import math
import os
import re
from dataclasses import dataclass
from enum import Enum

import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import CountVectorizer
from spacy.lang.en import stop_words
from tqdm.auto import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def setup_directories(base_dir):
    """Setup and return common directory paths"""
    directories = {
        "temp": os.path.join(base_dir, "temp"),
        "output": os.path.join(base_dir, "output"),
    }

    # Create all directories at once
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)

    return directories


class PostCategory(Enum):
    NORMIE = "normie"
    WEEB = "weeb"
    FURRY = "furry"


@dataclass
class CategoryScores:
    category: PostCategory
    score: float
    contributing_terms: dict


class SubculturePostClassifier:
    def __init__(
        self,
        weeb_terms_data,
        furry_terms_data,
        threshold=0.3,
        subthreshold=0.5,
        batch_size=5000,
    ):
        logger.info("Initializing classifier...")
        self.stop_words = set(stop_words.STOP_WORDS)
        self.tokenize_pattern = re.compile(r"\b\w+\b")
        self.threshold = threshold
        self.subthreshold = subthreshold
        self.batch_size = batch_size

        self.weeb_terms = self._process_terms(weeb_terms_data, PostCategory.WEEB)
        self.furry_terms = self._process_terms(furry_terms_data, PostCategory.FURRY)

        # Combine vocabularies for vectorizer
        all_terms = set(self.weeb_terms.keys()) | set(self.furry_terms.keys())

        # Initialize vectorizer with combined vocabulary
        self.vectorizer = CountVectorizer(
            vocabulary=all_terms, ngram_range=(1, 2), lowercase=True
        )

        logger.info(f"Initialized with {len(all_terms)} terms")

    def _process_terms(self, terms_data, category):
        """Process terms data into a score dictionary."""
        filtered_terms = terms_data[terms_data["combined_score"] > 0].copy()
        return dict(zip(filtered_terms["term"], filtered_terms["combined_score"]))

    def preprocess_text(self, text):
        """Simple and fast text preprocessing using regex."""
        try:
            text = str(text).lower()
            # Find all valid words
            tokens = self.tokenize_pattern.findall(text)
            # Remove stop words
            tokens = [t for t in tokens if t not in self.stop_words and len(t) > 1]
            return " ".join(tokens)
        except Exception as e:
            logger.warning(f"Error preprocessing text: {e}")
            return ""

    def _process_batch(self, texts):
        """Process a batch of texts."""
        # Preprocess texts
        cleaned_texts = [self.preprocess_text(text) for text in texts]

        # Vectorize all texts at once
        term_counts = self.vectorizer.transform(cleaned_texts)
        feature_names = self.vectorizer.get_feature_names_out()

        results = []
        # Process each document in the batch
        for i in range(term_counts.shape[0]):
            # Get present terms and their counts for this document
            doc_vector = term_counts[i]
            if doc_vector.nnz == 0:  # Skip empty documents
                results.append(
                    {
                        "primary_category": PostCategory.NORMIE.value,
                        "subcategory": None,
                        "weeb_score": 0.0,
                        "furry_score": 0.0,
                        "top_weeb_terms": [],
                        "top_furry_terms": [],
                    }
                )
                continue

            # Get non-zero indices and values
            indices = doc_vector.indices
            data = doc_vector.data

            # Compute weeb score
            weeb_terms = {}
            for idx, count in zip(indices, data):
                term = feature_names[idx]
                if term in self.weeb_terms:
                    weeb_terms[term] = self.weeb_terms[term] * count
            weeb_score = sum(weeb_terms.values()) / max(1, len(weeb_terms))

            # Compute furry score
            furry_terms = {}
            for idx, count in zip(indices, data):
                term = feature_names[idx]
                if term in self.furry_terms:
                    furry_terms[term] = self.furry_terms[term] * count
            furry_score = sum(furry_terms.values()) / max(1, len(furry_terms))

            # Sort terms by contribution
            weeb_terms = dict(
                sorted(weeb_terms.items(), key=lambda x: x[1], reverse=True)
            )
            furry_terms = dict(
                sorted(furry_terms.items(), key=lambda x: x[1], reverse=True)
            )

            # Determine categories
            if weeb_score > furry_score:
                if weeb_score > self.threshold:
                    primary_category = PostCategory.WEEB
                    subcategory = (
                        PostCategory.FURRY if furry_score > self.subthreshold else None
                    )
                else:
                    primary_category = PostCategory.NORMIE
                    subcategory = None
            else:
                if furry_score > self.threshold:
                    primary_category = PostCategory.FURRY
                    subcategory = (
                        PostCategory.WEEB if weeb_score > self.subthreshold else None
                    )
                else:
                    primary_category = PostCategory.NORMIE
                    subcategory = None

            results.append(
                {
                    "primary_category": primary_category.value,
                    "subcategory": subcategory.value if subcategory else None,
                    "weeb_score": weeb_score,
                    "furry_score": furry_score,
                    "top_weeb_terms": list(weeb_terms.keys())[:5],
                    "top_furry_terms": list(furry_terms.keys())[:5],
                }
            )

        return results

    def classify_posts(
        self,
        df,
        temp_dir,
        text_column="text",
        n_jobs=-1,
        chunk_size=100000,
    ):
        """
        Classify posts in chunks to avoid memory issues and provide better progress tracking.
        """
        logger.info(f"Starting classification of {len(df)} posts...")

        total_rows = len(df)
        chunks = math.ceil(total_rows / chunk_size)
        all_chunk_files = []

        # Create a directory for temporary results
        os.makedirs(temp_dir, exist_ok=True)

        # Format for progress bars
        bar_format = (
            "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )

        # Main progress bar for chunks
        with tqdm(
            total=chunks,
            desc="Processing chunks",
            position=0,
            bar_format=bar_format,
            dynamic_ncols=True,
        ) as pbar_chunks:
            # Process each chunk
            for chunk_idx in range(chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, total_rows)

                logger.info(
                    f"Processing chunk {chunk_idx + 1}/{chunks} (rows {start_idx}-{end_idx})"
                )

                # Get chunk data
                chunk_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

                # Calculate batches
                num_batches = math.ceil(len(chunk_df) / self.batch_size)
                batch_indices = [
                    (i * self.batch_size, min((i + 1) * self.batch_size, len(chunk_df)))
                    for i in range(num_batches)
                ]

                def process_one_batch(start_idx, end_idx):
                    batch_texts = chunk_df[text_column].iloc[start_idx:end_idx].tolist()
                    return self._process_batch(batch_texts)

                # Process batches with nested progress bar
                all_results = []
                with tqdm(
                    total=len(chunk_df),
                    desc=f"Current chunk ({chunk_idx + 1}/{chunks})",
                    position=1,
                    leave=False,
                    bar_format=bar_format,
                    dynamic_ncols=True,
                ) as pbar_batch:
                    batch_results = Parallel(n_jobs=n_jobs, verbose=0)(
                        delayed(process_one_batch)(start_idx, end_idx)
                        for start_idx, end_idx in batch_indices
                    )

                    for batch_idx, batch_result in enumerate(batch_results):
                        all_results.extend(batch_result)
                        batch_size = (
                            batch_indices[batch_idx][1] - batch_indices[batch_idx][0]
                        )
                        pbar_batch.update(batch_size)

                # Create and save chunk results
                results_df = pd.DataFrame(all_results)

                # Only include necessary columns with proper text column handling
                final_chunk_df = pd.DataFrame(
                    {
                        "text": chunk_df[text_column],
                        "primary_category": results_df["primary_category"],
                        "subcategory": results_df["subcategory"],
                        "weeb_score": results_df["weeb_score"],
                        "furry_score": results_df["furry_score"],
                        "top_weeb_terms": results_df["top_weeb_terms"],
                        "top_furry_terms": results_df["top_furry_terms"],
                    }
                )

                chunk_file = os.path.join(temp_dir, f"classified_chunk_{chunk_idx}.csv")
                final_chunk_df.to_csv(chunk_file, index=False)
                all_chunk_files.append(chunk_file)

                logger.info(
                    f"Chunk {chunk_idx + 1}/{chunks} completed and saved to {chunk_file}"
                )
                pbar_chunks.update(1)

        # Combine chunks
        logger.info("Combining all chunks into final result...")
        final_df = pd.concat(
            [pd.read_csv(f) for f in all_chunk_files], ignore_index=True
        )

        return final_df


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dirs = setup_directories(base_dir)

    temp_dir = dirs["temp"]
    output_dir = dirs["output"]

    logger.info("Starting classification process...")

    # Create output directories
    classifier_output_dir = f"{output_dir}/subculture-classifier"
    os.makedirs(classifier_output_dir, exist_ok=True)

    # Load term analysis results
    logger.info("Loading term analysis data...")
    weeb_terms_data = pd.read_csv(
        f"{output_dir}/terms-analysis/weeb_terms_analysis.csv"
    )
    furry_terms_data = pd.read_csv(
        f"{output_dir}/terms-analysis/furry_terms_analysis.csv"
    )

    # Initialize classifier with smaller batch size for better memory management
    classifier = SubculturePostClassifier(
        weeb_terms_data, furry_terms_data, batch_size=2000
    )

    # Load posts from dataset
    logger.info("Loading dataset...")
    df = pd.read_csv(
        f"{output_dir}/dataset-filter/bluesky_ten_million_english_only.csv"
    )
    logger.info(f"Loaded {len(df)} English posts")

    # Classify posts with chunking and parallel processing
    results = classifier.classify_posts(
        df, temp_dir, n_jobs=os.cpu_count() - 1 or 1, chunk_size=250000
    )  # Limit cores and use chunks

    # Save results with only necessary columns
    logger.info("Saving results...")
    results.to_csv(
        f"{output_dir}/subculture-classifier/subculture_posts_classified.csv",
        columns=[
            "text",
            "primary_category",
            "subcategory",
            "weeb_score",
            "furry_score",
            "top_weeb_terms",
            "top_furry_terms",
        ],
        index=False,
    )

    # Clean up temporary chunk files
    logger.info("Cleaning up temporary files...")
    if os.path.exists(temp_dir):
        try:
            import shutil

            shutil.rmtree(temp_dir)
            logger.debug("Removed temporary directory and its contents")
        except Exception as e:
            logger.warning(f"Failed to remove temporary directory: {str(e)}")
    logger.info("Cache cleared")

    logger.info("Classification process completed successfully")
