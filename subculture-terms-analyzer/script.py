import spacy
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from tqdm import tqdm
import re
import os
from joblib import Memory

# Configure logging and enable tqdm for pandas
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
tqdm.pandas()  # Re-enable progress_apply for pandas


class SubcultureTermAnalyzer:
    def __init__(self, culture_type="weeb"):
        """Initialize the analyzer with the specified culture type."""
        self.culture_type = culture_type

        # Set seed terms based on culture type
        if culture_type == "weeb":
            self.seed_terms = {"anime", "manga", "otaku", "waifu", "weeb"}
            self.culture_column = "is_weeb"
        elif culture_type == "furry":
            self.seed_terms = {"furry", "fursuit", "uwu", "owo", "paws", "fursona"}
            self.culture_column = "is_furry"
        else:
            raise ValueError(f"Unsupported culture type: {culture_type}")

        # Load spaCy with only necessary components
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

        # Pre-compile regex pattern
        self.pattern = re.compile("|".join(self.seed_terms), re.IGNORECASE)

        # Initialize cache
        cache_dir = "./cache/text_preprocessing"
        os.makedirs(cache_dir, exist_ok=True)
        self.memory = Memory(cache_dir, verbose=0)
        self.cached_preprocess = self.memory.cache(self._preprocess_text)

    def _preprocess_text(self, text):
        """Internal preprocessing function that will be cached"""
        try:
            if pd.isna(text) or not isinstance(text, str):
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
        # Create a hash of the text to use as cache key
        if pd.isna(text) or not isinstance(text, str):
            return ""
        return self.cached_preprocess(text)

    def load_data(self):
        """Load data from CSV file"""
        logger.info("Loading dataset...")
        df = pd.read_csv("./output/dataset-filter/bluesky_ten_million_english_only.csv")
        logger.info(f"Loaded {len(df)} English posts")
        return df

    def identify_culture_posts(self, df):
        """Identify posts that contain seed terms using compiled regex"""
        logger.info(f"Identifying {self.culture_type}-related posts...")
        # Use tqdm to show progress
        tqdm.pandas(desc=f"Identifying {self.culture_type} posts")
        df[self.culture_column] = df["text"].progress_apply(
            lambda x: bool(self.pattern.search(x)) if isinstance(x, str) else False
        )
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

            # Process batch using spaCy's pipe
            docs = self.nlp.pipe(valid_texts)
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

    def batch_process_texts(self, texts, batch_size=10000):
        """Process texts in larger batches with optimized caching"""
        results = []
        total_batches = len(texts) // batch_size + (
            1 if len(texts) % batch_size > 0 else 0
        )

        with tqdm(total=total_batches, desc="Processing batches") as pbar:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size].tolist()

                # Process batch
                if hasattr(self, "memory"):
                    processed = self.memory.cache(self._preprocess_batch)(batch)
                else:
                    processed = self._preprocess_batch(batch)

                results.extend(processed)
                pbar.update(1)

                # Clear some memory
                del processed
                if i % (batch_size * 10) == 0:  # GC every 10 batches
                    import gc

                    gc.collect()

        return pd.Series(results, index=texts.index)

    def compute_term_specificity(self, df):
        """Compute how specific terms are to culture posts vs non-culture posts"""
        logger.info("Computing term specificity...")

        # Create separate corpora
        culture_corpus = df[df[self.culture_column]]["cleaned_text"]
        non_culture_corpus = df[~df[self.culture_column]]["cleaned_text"]

        # Vectorize both corpora
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=10,  # Require terms to appear in at least 10 documents
        )

        # Fit on entire corpus to get shared vocabulary
        logger.info("Fitting TF-IDF vectorizer...")
        vectorizer.fit(df["cleaned_text"])

        # Transform both corpora with progress indication
        logger.info("Transforming culture corpus...")
        culture_matrix = vectorizer.transform(culture_corpus)
        logger.info("Transforming non-culture corpus...")
        non_culture_matrix = vectorizer.transform(non_culture_corpus)

        # Compute normalized frequencies
        culture_freqs = np.array(culture_matrix.sum(axis=0)).flatten() / len(
            culture_corpus
        )
        non_culture_freqs = np.array(non_culture_matrix.sum(axis=0)).flatten() / len(
            non_culture_corpus
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

    def train_word2vec(self, df):
        """Train Word2Vec and find terms similar to seed terms"""
        logger.info("Training Word2Vec model...")
        # Prepare sentences with progress bar
        logger.info("Preparing sentences for Word2Vec...")
        sentences = [
            text.split()
            for text in tqdm(df["cleaned_text"], desc="Preparing Word2Vec data")
            if text and isinstance(text, str)
        ]

        logger.info(f"Training Word2Vec model with {len(sentences)} sentences...")
        model = Word2Vec(
            sentences,
            vector_size=1000,
            window=5,
            min_count=5,
            workers=os.cpu_count(),
            sg=1,
        )

        # Use a dictionary to keep track of highest similarity scores
        similar_terms_dict = {}

        # Find similar terms to seed terms with progress bar
        for seed in tqdm(self.seed_terms, desc="Finding similar terms"):
            if seed in model.wv:
                similar = model.wv.most_similar(seed, topn=10)
                for term, score in similar:
                    # Only keep highest similarity score for each term
                    if (
                        term not in similar_terms_dict
                        or score > similar_terms_dict[term]
                    ):
                        similar_terms_dict[term] = score

        # Convert to DataFrame
        similar_terms = [(term, score) for term, score in similar_terms_dict.items()]
        return pd.DataFrame(similar_terms, columns=["term", "similarity"])

    def analyze(self):
        """Complete analysis pipeline"""
        # Load and preprocess data
        df = self.load_data()

        # Generate cache key based on data
        cache_file = f"./cache/processed_data_{self.culture_type}.parquet"

        if os.path.exists(cache_file):
            logger.info("Loading preprocessed data from cache...")
            df = pd.read_parquet(cache_file)
        else:
            logger.info("Processing data and creating cache...")
            logger.info(f"Total rows to process: {len(df)}")

            # Handle NaN values and convert to string before preprocessing
            nan_count = df["text"].isna().sum()
            if nan_count > 0:
                logger.warning(f"Found {nan_count} NaN values in text column")
                df["text"] = df["text"].fillna("")

            df["text"] = df["text"].astype(str)
            logger.info("Starting batch processing...")
            df["cleaned_text"] = self.batch_process_texts(df["text"])

            # Save processed data to cache
            logger.info("Saving processed data to cache...")
            os.makedirs("./cache", exist_ok=True)
            df.to_parquet(cache_file)
            logger.info(f"Cached processed data to {cache_file}")

        # Continue with analysis
        df = self.identify_culture_posts(df)

        # Get term specificity
        term_stats = self.compute_term_specificity(df)

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
        output_file = f"./output/terms-analysis/{self.culture_type}_terms_analysis.csv"
        final_terms.sort_values("combined_score", ascending=False).to_csv(
            output_file, index=False
        )
        logger.info(f"Results saved to {output_file}")

        return final_terms.sort_values("combined_score", ascending=False)

    def clear_cache(self):
        """Clear all cached preprocessing data"""
        logger.info("Clearing preprocessing cache...")
        self.memory.clear()
        cache_file = f"./cache/processed_data_{self.culture_type}.parquet"
        if os.path.exists(cache_file):
            os.remove(cache_file)
        logger.info("Cache cleared")


if __name__ == "__main__":
    for culture in tqdm(["weeb", "furry"], desc="Analyzing cultures"):
        logger.info(f"Starting {culture} term analysis...")
        analyzer = SubcultureTermAnalyzer(culture_type=culture)
        analyzer.analyze()
        analyzer.clear_cache()
        logger.info(f"Completed {culture} term analysis")
