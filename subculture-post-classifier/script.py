import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import spacy
import logging
from enum import Enum
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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
        self, weeb_terms_data, furry_terms_data, threshold=0.6, batch_size=1000
    ):
        logger.info("Initializing classifier...")
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        self.threshold = threshold
        self.batch_size = batch_size

        # Process terms for each category
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
        """Preprocess text using spaCy."""
        try:
            doc = self.nlp(str(text).lower())
            return " ".join(
                [
                    token.text
                    for token in doc
                    if not token.is_stop and not token.is_punct and token.text.strip()
                ]
            )
        except Exception as e:
            logger.warning(f"Error preprocessing text: {e}")
            return ""

    def _process_batch(self, texts):
        """Process a batch of texts."""
        # Preprocess all texts in batch
        cleaned_texts = [self.preprocess_text(text) for text in texts]

        # Vectorize all texts at once
        term_counts = self.vectorizer.transform(cleaned_texts)

        results = []
        for idx, text_counts in enumerate(term_counts):
            # Get present terms for both categories
            weeb_scores = self._compute_category_score_vectorized(
                text_counts, self.weeb_terms
            )
            furry_scores = self._compute_category_score_vectorized(
                text_counts, self.furry_terms
            )

            # Determine categories
            categories = [
                CategoryScores(PostCategory.WEEB, weeb_scores[0], weeb_scores[1]),
                CategoryScores(PostCategory.FURRY, furry_scores[0], furry_scores[1]),
            ]

            categories.sort(key=lambda x: x.score, reverse=True)
            highest_score = categories[0].score

            if highest_score > self.threshold:
                primary_category = categories[0].category
                secondary_score = categories[1].score
                subcategory = (
                    categories[1].category
                    if secondary_score > self.threshold * 0.8
                    else None
                )
            else:
                primary_category = PostCategory.NORMIE
                subcategory = None

            results.append(
                {
                    "primary_category": primary_category.value,
                    "subcategory": subcategory.value if subcategory else None,
                    "weeb_score": weeb_scores[0],
                    "furry_score": furry_scores[0],
                    "top_weeb_terms": list(weeb_scores[1].keys())[:5],
                    "top_furry_terms": list(furry_scores[1].keys())[:5],
                }
            )

        return results

    def _compute_category_score_vectorized(self, term_counts, term_scores):
        """Compute score for a category using vectorized operations."""
        # Get non-zero terms
        terms = self.vectorizer.get_feature_names_out()
        counts = term_counts.toarray()[0]
        present_terms = {}

        for idx in counts.nonzero()[0]:
            term = terms[idx]
            if term in term_scores:
                score_contribution = term_scores[term] * counts[idx]
                present_terms[term] = score_contribution

        total_score = sum(present_terms.values())
        word_count = np.count_nonzero(counts) + 1  # Add 1 to avoid division by zero
        normalized_score = total_score / word_count

        return normalized_score, dict(
            sorted(present_terms.items(), key=lambda x: x[1], reverse=True)
        )

    def classify_posts(self, df, text_column="text"):
        """
        Classify posts in batches with progress tracking.
        """
        logger.info(f"Starting classification of {len(df)} posts...")

        # Calculate number of batches
        num_batches = math.ceil(len(df) / self.batch_size)
        all_results = []

        # Process batches with progress bar
        with tqdm(total=len(df), desc="Classifying posts") as pbar:
            for i in range(num_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, len(df))

                batch_texts = df[text_column].iloc[start_idx:end_idx]
                batch_results = self._process_batch(batch_texts)
                all_results.extend(batch_results)

                pbar.update(len(batch_texts))

                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {end_idx}/{len(df)} posts")

        logger.info("Classification complete. Creating results DataFrame...")
        results_df = pd.DataFrame(all_results)
        final_df = pd.concat([df["text"], results_df], axis=1)
        logger.info("Results DataFrame created successfully")

        return final_df


if __name__ == "__main__":
    logger.info("Starting classification process...")

    # Load term analysis results
    logger.info("Loading term analysis data...")
    weeb_terms_data = pd.read_csv("./output/terms-analysis/weeb_terms_analysis.csv")
    furry_terms_data = pd.read_csv("./output/terms-analysis/furry_terms_analysis.csv")

    # Initialize classifier
    classifier = SubculturePostClassifier(weeb_terms_data, furry_terms_data)

    # Load posts from dataset
    logger.info("Loading dataset...")
    df = pd.read_csv("./output/dataset-filter/bluesky_ten_million_english_only.csv")
    logger.info(f"Loaded {len(df)} English posts")

    # Classify posts
    results = classifier.classify_posts(df)

    # Save results
    logger.info("Saving results...")
    results.to_csv(
        "./output/subculture-classifier/subculture_posts_classified.csv", index=False
    )
    logger.info("Classification process completed successfully")
