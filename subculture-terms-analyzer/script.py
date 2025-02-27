import logging
import os
import re

import numpy as np
import pandas as pd
import spacy
from gensim.models import Word2Vec
from joblib import Memory
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

# Configure logging and enable tqdm for pandas
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
tqdm.pandas()  # Enable progress_apply for pandas


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


class SubcultureTermAnalyzer:
    def __init__(self, culture_type, temp_dir, output_dir):
        """Initialize the analyzer with the specified culture type."""
        self.culture_type = culture_type
        self.temp_dir = temp_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Set seed terms based on culture type
        if culture_type == "weeb":
            self.seed_terms = {"anime", "manga", "otaku", "waifu", "weeb"}
            self.culture_column = "is_weeb"
        elif culture_type == "furry":
            self.seed_terms = {"furry", "fursuit", "uwu", "owo", "paws", "fursona"}
            self.culture_column = "is_furry"
        else:
            raise ValueError(f"Unsupported culture type: {culture_type}")

        # Load spaCy with only necessary components and disable the pipeline we don't need
        self.nlp = spacy.load(
            "en_core_web_sm", disable=["parser", "ner", "tagger", "lemmatizer"]
        )
        # Increase batch size for pipe processing
        self.nlp.max_length = 2000000  # Increase max document length

        # Pre-compile regex pattern with word boundaries for more precise matching
        pattern_terms = [rf"\b{term}\b" for term in self.seed_terms]
        self.pattern = re.compile("|".join(pattern_terms), re.IGNORECASE)

        # Initialize cache for text preprocessing
        temp_tp_dir = f"{temp_dir}/text_preprocessing"
        os.makedirs(temp_tp_dir, exist_ok=True)
        self.memory = Memory(temp_tp_dir, verbose=0, mmap_mode=None)
        self.cached_preprocess = self.memory.cache(self._preprocess_text)

        # Determine optimal number of workers
        self.n_workers = os.cpu_count() - 1 or 1  # Leave one CPU free for system

    def _preprocess_text(self, text):
        """Internal preprocessing function that will be cached"""
        try:
            if pd.isna(text) or not isinstance(text, str):
                return ""

            # Skip tokenization for very short texts
            if len(text) < 3:
                return ""

            doc = self.nlp(text.lower())
            return " ".join(
                [
                    token.text
                    for token in doc
                    if not token.is_stop and not token.is_punct and token.text.strip()
                ]
            )
        except Exception as e:
            logger.warning(f"Error preprocessing text: {str(e)}")
            return ""

    def preprocess_text(self, text):
        """Wrapper for cached preprocessing"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        return self.cached_preprocess(text)

    def load_data(self):
        """Load data from CSV file"""
        logger.info("Loading dataset...")
        # Only load needed columns
        df = pd.read_csv(
            f"{self.output_dir}/dataset-filter/bluesky_ten_million_english_only.csv",
            usecols=["text"],  # Only load the text column initially
        )
        logger.info(f"Loaded {len(df)} English posts")
        return df

    def identify_culture_posts(self, df):
        """Identify posts that contain seed terms using compiled regex"""
        logger.info(f"Identifying {self.culture_type}-related posts...")

        # Use vectorized operations instead of apply
        # Create mask using str.contains with regex
        df[self.culture_column] = df["text"].str.contains(self.pattern, na=False)

        return df

    def _preprocess_batch(self, texts):
        """Process a batch of texts at once"""
        try:
            # Filter out invalid texts
            valid_texts = [
                text.lower()
                for text in texts
                if isinstance(text, str) and not pd.isna(text)
            ]

            if not valid_texts:
                return [""] * len(texts)

            # Process batch using spaCy's pipe with batch_size
            docs = list(self.nlp.pipe(valid_texts, batch_size=1000))
            processed = [
                " ".join(
                    token.text
                    for token in doc
                    if not token.is_stop and not token.is_punct and token.text.strip()
                )
                for doc in docs
            ]

            # Handle any missing results
            if len(processed) < len(texts):
                processed.extend([""] * (len(texts) - len(processed)))

            return processed

        except Exception as e:
            logger.warning(f"Error in batch preprocessing: {str(e)}")
            return [""] * len(texts)

    def batch_process_texts(self, texts, batch_size=5000):
        """Process texts in larger batches with optimized caching and parallelization"""
        # Adjust batch size based on available memory
        if len(texts) <= batch_size:
            # Small dataset - process in one batch
            return pd.Series(self._preprocess_batch(texts.tolist()), index=texts.index)

        # Calculate optimal batch size based on text length
        avg_len = texts.str.len().mean()
        if avg_len > 1000:
            batch_size = max(1000, batch_size // 4)

        results = []
        total_batches = len(texts) // batch_size + (
            1 if len(texts) % batch_size > 0 else 0
        )

        # Process batches sequentially
        with tqdm(total=total_batches, desc="Processing text batches") as pbar:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size].tolist()
                processed = self._preprocess_batch(batch)
                results.extend(processed)
                pbar.update(1)

                # Clear some memory periodically
                if i % (batch_size * 5) == 0:
                    import gc

                    gc.collect()

        return pd.Series(results, index=texts.index)

    def compute_term_specificity(self, df):
        """Compute how specific terms are to culture posts vs non-culture posts"""
        logger.info("Computing term specificity...")

        # Use sampling for large datasets
        if len(df) > 100000:
            # Sample equal numbers from both categories for balanced comparison
            culture_count = df[df[self.culture_column]].shape[0]
            non_culture_count = df[~df[self.culture_column]].shape[0]

            # Take all culture posts if they're the minority
            if culture_count < non_culture_count:
                culture_sample = df[df[self.culture_column]]
                # Sample the same number of non-culture posts
                non_culture_sample = df[~df[self.culture_column]].sample(
                    min(culture_count * 2, non_culture_count), random_state=42
                )
            else:
                non_culture_sample = df[~df[self.culture_column]]
                culture_sample = df[df[self.culture_column]].sample(
                    min(non_culture_count * 2, culture_count), random_state=42
                )

            # Combine samples
            df_sample = pd.concat([culture_sample, non_culture_sample])
            logger.info(
                f"Using balanced sample of {len(df_sample)} posts for TF-IDF analysis"
            )
        else:
            df_sample = df

        # Create separate corpora
        culture_corpus = df_sample[df_sample[self.culture_column]]["cleaned_text"]
        non_culture_corpus = df_sample[~df_sample[self.culture_column]]["cleaned_text"]

        # Vectorize both corpora
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=10,  # Require terms to appear in at least 10 documents
            lowercase=False,  # Text already lowercased
        )

        # Fit on sampled corpus to get shared vocabulary
        logger.info("Fitting TF-IDF vectorizer...")
        vectorizer.fit(df_sample["cleaned_text"])

        # Transform both corpora
        logger.info("Transforming corpora and computing specificity...")
        culture_matrix = vectorizer.transform(culture_corpus)
        non_culture_matrix = vectorizer.transform(non_culture_corpus)

        # Use sparse matrix operations
        culture_freqs = np.asarray(culture_matrix.sum(axis=0)).flatten() / max(
            1, culture_corpus.shape[0]
        )
        non_culture_freqs = np.asarray(non_culture_matrix.sum(axis=0)).flatten() / max(
            1, non_culture_corpus.shape[0]
        )

        # Compute specificity score (ratio of frequencies)
        epsilon = 1e-10  # Small constant to avoid division by zero
        specificity = np.log((culture_freqs + epsilon) / (non_culture_freqs + epsilon))

        # Create DataFrame with results
        terms = vectorizer.get_feature_names_out()
        term_stats = pd.DataFrame(
            {
                "term": terms,
                f"{self.culture_type}_freq": culture_freqs,
                f"non_{self.culture_type}_freq": non_culture_freqs,
                "specificity": specificity,
            }
        )

        # Filter for terms that appear more in culture posts
        term_stats = term_stats[term_stats["specificity"] > 0]
        return term_stats.sort_values("specificity", ascending=False)

    def filter_terms(self, terms_df):
        """Filter redundant compound terms from results"""
        filtered_terms = []

        # Get list of single-word terms for checking compounds
        all_single_terms = set([t for t in terms_df["term"] if " " not in t])

        for term in terms_df["term"]:
            # Keep all single words
            if " " not in term:
                filtered_terms.append(term)
                continue

            # Split compound terms
            parts = term.split()

            # Skip if any part is a seed term
            if any(part in self.seed_terms for part in parts):
                continue

            # Skip if both parts appear individually in the dataset
            # This catches cases like "furryart art" where both terms appear separately
            if all(part in all_single_terms for part in parts):
                continue

            # Skip terms where one part is contained in the other part
            # This catches cases like "furryart furryartist"
            if any(
                part in other_part and part != other_part
                for i, part in enumerate(parts)
                for other_part in parts[:i] + parts[i + 1 :]
            ):
                continue

            filtered_terms.append(term)

        # Return filtered dataframe
        return terms_df[terms_df["term"].isin(filtered_terms)]

    def train_word2vec(self, df):
        """Train Word2Vec and find terms similar to seed terms"""
        logger.info("Training Word2Vec model...")

        # Sample data for Word2Vec if dataset is very large
        if len(df) > 200000:
            # Ensure we have enough culture posts in the sample
            culture_posts = df[df[self.culture_column]]
            non_culture_posts = df[~df[self.culture_column]]

            # Keep all culture posts if they're less than 50k
            if len(culture_posts) < 50000:
                culture_sample = culture_posts
            else:
                culture_sample = culture_posts.sample(50000, random_state=42)

            # Sample from non-culture posts
            non_culture_sample = non_culture_posts.sample(
                min(len(non_culture_posts), 150000), random_state=42
            )

            # Combine samples
            df_sample = pd.concat([culture_sample, non_culture_sample])
            logger.info(f"Using sample of {len(df_sample)} posts for Word2Vec training")
        else:
            df_sample = df

        # More efficient sentence preparation using list comprehension
        sentences = []
        for text in tqdm(
            df_sample["cleaned_text"].dropna(), desc="Preparing Word2Vec data"
        ):
            if isinstance(text, str) and text.strip():
                sentences.append(text.split())

        logger.info(f"Training Word2Vec model with {len(sentences)} sentences...")

        # Faster Word2Vec training
        model = Word2Vec(
            sentences,
            vector_size=300,
            window=5,
            min_count=5,
            workers=self.n_workers,
            sg=1,
            compute_loss=False,  # Speed up training by not computing loss
        )

        # Use a dictionary to keep track of highest similarity scores
        similar_terms_dict = {}

        # More efficient similar term finding
        seed_terms_in_vocab = [seed for seed in self.seed_terms if seed in model.wv]

        if not seed_terms_in_vocab:
            logger.warning(
                "No seed terms found in vocabulary. Using most common words instead."
            )
            # Use most common words as fallback
            similar_terms = pd.DataFrame(
                {"term": model.wv.index_to_key[:20], "similarity": [1.0] * 20}
            )
            return similar_terms

        # Find similar terms to seed terms
        for seed in tqdm(seed_terms_in_vocab, desc="Finding similar terms"):
            similar = model.wv.most_similar(seed, topn=20)
            for term, score in similar:
                # Skip compound terms that directly contain seed terms
                if " " in term and any(
                    seed_term in term.split() for seed_term in self.seed_terms
                ):
                    continue

                # Only keep highest similarity score for each term
                if term not in similar_terms_dict or score > similar_terms_dict[term]:
                    similar_terms_dict[term] = score

        # Convert to DataFrame
        similar_terms = [(term, score) for term, score in similar_terms_dict.items()]
        return pd.DataFrame(similar_terms, columns=["term", "similarity"])

    def analyze(self):
        """Complete analysis pipeline"""
        # Load and preprocess data
        df = self.load_data()

        # Generate cache based on data
        temp_file = f"{self.temp_dir}/processed_data.parquet"

        if os.path.exists(temp_file):
            logger.info("Loading preprocessed data from cache...")
            df = pd.read_parquet(temp_file)
        else:
            logger.info("Processing data and creating cache...")
            logger.info(f"Total rows to process: {len(df)}")

            # Faster NaN handling with vectorized operations
            df["text"] = df["text"].fillna("")

            logger.info("Starting batch processing...")
            df["cleaned_text"] = self.batch_process_texts(df["text"])

            # Save processed data to cache
            logger.info("Saving processed data to cache...")
            os.makedirs(self.temp_dir, exist_ok=True)
            df.to_parquet(temp_file)
            logger.info(f"Cached processed data to {temp_file}")

        # Continue with analysis
        df = self.identify_culture_posts(df)

        # Get term specificity
        term_stats = self.compute_term_specificity(df)

        # Apply term filtering
        term_stats = self.filter_terms(term_stats)

        # Get word embeddings similarities
        embedding_similarities = self.train_word2vec(df)

        # Combine results
        final_terms = pd.merge(
            term_stats, embedding_similarities, on="term", how="outer"
        ).fillna(0)

        # Compute combined score
        final_terms["combined_score"] = (
            final_terms["specificity"] * 0.7 + final_terms["similarity"] * 0.3
        )

        # Save results
        terms_analysis_dir = f"{self.output_dir}/terms-analysis"
        output_file = f"{terms_analysis_dir}/{self.culture_type}_terms_analysis.csv"
        os.makedirs(terms_analysis_dir, exist_ok=True)
        final_terms.sort_values("combined_score", ascending=False).to_csv(
            output_file, index=False
        )
        logger.info(f"Results saved to {output_file}")

        return final_terms.sort_values("combined_score", ascending=False)

    def clear_cache(self):
        """Clear all cached preprocessing data"""
        logger.info("Clearing preprocessing cache...")
        self.memory.clear()
        temp_file = f"{self.temp_dir}/processed_data.parquet"
        if os.path.exists(temp_file):
            os.remove(temp_file)
        logger.info("Cache cleared")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dirs = setup_directories(base_dir)

    temp_dir = dirs["temp"]
    output_dir = dirs["output"]

    for culture in tqdm(["weeb", "furry"], desc="Analyzing cultures"):
        logger.info(f"Starting {culture} term analysis...")
        analyzer = SubcultureTermAnalyzer(
            culture_type=culture, temp_dir=temp_dir, output_dir=output_dir
        )
        analyzer.analyze()
        logger.info(f"Completed {culture} term analysis")

    analyzer.clear_cache()
    logger.info("All term analysis completed successfully")
