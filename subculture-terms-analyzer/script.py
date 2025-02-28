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
            self.seed_terms = {
                "anime",
                "manga",
                "otaku",
                "waifu",
                "weeb",
                "kawaii",
                "アニメ",
            }
            self.culture_column = "is_weeb"
        elif culture_type == "furry":
            self.seed_terms = {
                "furry",
                "fursuit",
                "uwu",
                "owo",
                "paws",
                "fursona",
                ":3",
            }
            self.culture_column = "is_furry"
        else:
            raise ValueError(f"Unsupported culture type: {culture_type}")

        # Load spaCy with only necessary components and disable the pipeline we don't need
        self.nlp = spacy.load(
            "en_core_web_sm",
            disable=[
                "parser",
                "ner",
                "tagger",
                "lemmatizer",
                "attribute_ruler",
                "tok2vec",
            ],
        )
        self.nlp.max_length = 2000000

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

            text = text.lower()

            # Patterns for different types of cultural markers
            patterns = {
                "emoticons": r"[:;><=][-~]?[3DOoPp)(|\\/]",  # :3, >w<, etc
                "kaomoji": r"[\(\[]\s*[;:=]\s*[_-]?\s*[\)\]]",  # (: :) etc
                "text_faces": r"[◕ᴥ◕✿]",  # Unicode face components
                "cjk": r"[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]+",  # CJK characters
            }

            # Check if text consists entirely of special patterns
            full_pattern = "|".join(patterns.values())
            if re.match(f"^({full_pattern})$", text):
                return text

            # For longer texts, process normally but preserve special patterns
            doc = self.nlp(text)
            processed_tokens = []

            for token in doc:
                # Keep token if it matches any pattern or is a regular word
                if any(
                    re.match(pattern, token.text) for pattern in patterns.values()
                ) or (not token.is_stop and not token.is_punct and token.text.strip()):
                    processed_tokens.append(token.text)

            return " ".join(processed_tokens)

        except Exception as e:
            logger.warning(f"Error preprocessing text: {str(e)}")
            return ""

    def preprocess_text(self, text):
        """Wrapper for cached preprocessing"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        return self.cached_preprocess(text)

    def load_and_preprocess_data(self):
        """Optimized data loading and preprocessing"""
        logger.info("Loading dataset...")

        # Load full dataset initially
        df = pd.read_csv(
            f"{self.output_dir}/dataset-filter/bluesky_ten_million_english_only.csv",
            usecols=["text"],
        )

        # Identify culture posts first
        df = self.identify_culture_posts(df)

        # Keep ALL culture-related posts
        culture_posts = df[df[self.culture_column]]

        # For non-culture posts, use stratified sampling
        if len(df) > 500000:
            # Take 5x the number of culture posts, or all if less
            non_culture_sample_size = min(
                len(culture_posts) * 5, len(df[~df[self.culture_column]])
            )
            non_culture_sample = df[~df[self.culture_column]].sample(
                non_culture_sample_size, random_state=42
            )
            df_to_process = pd.concat([culture_posts, non_culture_sample])
            logger.info(
                f"Processing {len(culture_posts)} culture posts and {len(non_culture_sample)} non-culture posts"
            )
        else:
            df_to_process = df

        # Preprocess with optimized batch processing
        logger.info("Preprocessing text data...")
        df_to_process["cleaned_text"] = self.batch_process_texts(df_to_process["text"])

        return df_to_process

    def identify_culture_posts(self, df):
        """Identify posts that contain seed terms using compiled regex"""
        logger.info(f"Identifying {self.culture_type}-related posts...")

        # Use vectorized operations instead of apply
        # Create mask using str.contains with regex
        df[self.culture_column] = df["text"].str.contains(self.pattern, na=False)

        return df

    def _preprocess_batch(self, texts):
        """Process a batch of texts"""
        processed = []

        # Patterns for cultural markers
        patterns = {
            "emoticons": r"[:;><=][-~]?[3DOoPp)(|\\/]",  # :3, >w<, etc
            "kaomoji": r"[\(\[]\s*[;:=]\s*[_-]?\s*[\)\]]",  # (: :) etc
            "text_faces": r"[◕ᴥ◕✿]",  # Unicode face components
            "cjk": r"[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]+",  # CJK characters
        }
        full_pattern = "|".join(patterns.values())

        # Pre-filter invalid texts but preserve cultural markers
        valid_texts = []
        for text in texts:
            if not isinstance(text, str) or pd.isna(text):
                valid_texts.append("")
                continue

            text = text.lower()
            # Keep text if it's either:
            # 1. Contains cultural markers (emoticons, CJK, etc.)
            # 2. Is a normal text of sufficient length
            if re.search(full_pattern, text) or len(text) > 3:
                valid_texts.append(text)
            else:
                valid_texts.append("")

        if not any(valid_texts):
            return [""] * len(texts)

        # Process with spaCy
        docs = self.nlp.pipe(
            valid_texts,
            batch_size=min(len(valid_texts), 1000),
            n_process=1,  # Disable multiprocessing within spaCy
        )

        # Create mapping of original indices to processed texts
        processed = []

        for doc in docs:
            # Keep cultural markers and regular tokens
            tokens = []
            for token in doc:
                if re.match(full_pattern, token.text) or (
                    not token.is_stop and not token.is_punct and token.text.strip()
                ):
                    tokens.append(token.text)
            processed.append(" ".join(tokens) if tokens else "")

        return processed

    def batch_process_texts(self, texts, batch_size=10000):
        """Process texts in parallel batches with optimized memory management"""
        logger.info("Processing texts in batches...")

        # Convert to list for easier processing
        texts_list = texts.tolist() if isinstance(texts, pd.Series) else list(texts)

        # Process in smaller chunks to manage memory
        chunk_size = 100000
        if len(texts_list) > chunk_size:
            results = []
            for i in range(0, len(texts_list), chunk_size):
                chunk = texts_list[i : i + chunk_size]
                processed_chunk = self._process_chunk(chunk, batch_size)
                results.extend(processed_chunk)

                # Clear memory
                import gc

                gc.collect()

            return pd.Series(results, index=texts.index)
        else:
            processed = self._process_chunk(texts_list, batch_size)
            return pd.Series(processed, index=texts.index)

    def _process_chunk(self, texts, batch_size=10000):
        """Process a chunk of texts sequentially in batches"""
        processed = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch = texts[i : i + batch_size]
            batch_processed = self._preprocess_batch(batch)
            processed.extend(batch_processed)

        return processed

    def _parallel_process_texts(self, texts, batch_size):
        """Process texts in parallel using joblib"""
        from joblib import Parallel, delayed

        # Calculate optimal number of batches
        n_batches = min(self.n_workers * 4, max(1, len(texts) // batch_size))
        batch_size = len(texts) // n_batches + (1 if len(texts) % n_batches > 0 else 0)

        # Create batches
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

        # Process batches in parallel
        results = Parallel(n_jobs=self.n_workers)(
            delayed(self._preprocess_batch)(batch.tolist(), batch_size)
            for batch in batches
        )

        # Combine results
        flattened = []
        for result in results:
            flattened.extend(result)

        return pd.Series(flattened, index=texts.index)

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

        # Vectorize both corpora with improved token pattern
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=10,  # Require terms to appear in at least 10 documents
            lowercase=False,  # Text already lowercased
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-_]+[a-zA-Z]\b",  # Better pattern for subcultural terms
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

        # Filter for terms that appear more in culture posts with frequency threshold
        term_stats = term_stats[
            (term_stats["specificity"] > 0)
            & (term_stats[f"{self.culture_type}_freq"] > 0.0001)
        ]

        # Save intermediate results to temp directory
        os.makedirs(f"{self.temp_dir}/intermediate", exist_ok=True)
        term_stats.to_csv(
            f"{self.temp_dir}/intermediate/{self.culture_type}_specificity.csv",
            index=False,
        )

        return term_stats.sort_values("specificity", ascending=False)

    def filter_terms(self, terms_df):
        """Filter redundant compound terms from results with improved approach"""
        # Sort terms by specificity score
        terms_sorted = terms_df.sort_values("specificity", ascending=False)
        filtered_terms = []
        seen_terms = set()

        # Get list of single-word terms for checking compounds
        all_single_terms = set([t for t in terms_df["term"] if " " not in t])

        for term in terms_sorted["term"]:
            # Skip if term is substring of higher-ranked term
            if any(term in seen_term for seen_term in seen_terms):
                continue

            # Keep all single words
            if " " not in term:
                filtered_terms.append(term)
                seen_terms.add(term)
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
            seen_terms.add(term)

        # Return filtered dataframe
        return terms_df[terms_df["term"].isin(filtered_terms)]

    def evaluate_contextual_relevance(self, df, terms):
        """Evaluate terms in context by checking co-occurrence with seed terms"""
        logger.info("Evaluating contextual relevance of terms...")

        relevance_scores = {}
        for term in tqdm(terms, desc="Calculating contextual relevance"):
            # Count how often term appears in posts with seed terms vs without
            in_context = df[
                df["cleaned_text"].str.contains(rf"\b{term}\b", na=False, regex=True)
                & df[self.culture_column]
            ].shape[0]
            out_context = df[
                df["cleaned_text"].str.contains(rf"\b{term}\b", na=False, regex=True)
                & ~df[self.culture_column]
            ].shape[0]
            relevance_scores[term] = in_context / (in_context + out_context + 1e-10)

        return pd.Series(relevance_scores)

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
            return similar_terms, model

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

        # Save intermediate results to temp directory
        sim_terms_df = pd.DataFrame(similar_terms, columns=["term", "similarity"])
        sim_terms_df.to_csv(
            f"{self.temp_dir}/intermediate/{self.culture_type}_similarity.csv",
            index=False,
        )

        return sim_terms_df, model

    def cluster_similar_terms(self, term_df, model, threshold=0.8):
        """Group similar terms to reduce redundancy"""
        logger.info("Clustering similar terms...")
        clusters = {}

        terms = term_df["term"].tolist()
        for i, term1 in enumerate(tqdm(terms, desc="Clustering terms")):
            if term1 not in model.wv:
                continue
            if term1 not in clusters:
                clusters[term1] = [term1]
            for term2 in terms[i + 1 :]:
                if term2 not in model.wv:
                    continue
                if model.wv.similarity(term1, term2) > threshold:
                    clusters[term1].append(term2)

        # Return representative terms from each cluster
        representative_terms = []
        for cluster_key, cluster_terms in clusters.items():
            if len(cluster_terms) > 1:
                logger.debug(f"Cluster: {cluster_terms}")
            representative_terms.append(cluster_terms[0])

        return representative_terms

    def analyze(self):
        """Complete analysis pipeline"""
        # Load and preprocess data
        df = self.load_and_preprocess_data()

        # Continue with analysis
        df = self.identify_culture_posts(df)

        # Get term specificity
        term_stats = self.compute_term_specificity(df)
        term_stats.to_csv(
            f"{self.temp_dir}/intermediate/{self.culture_type}_specificity_raw.csv",
            index=False,
        )

        # Apply term filtering
        term_stats = self.filter_terms(term_stats)
        term_stats.to_csv(
            f"{self.temp_dir}/intermediate/{self.culture_type}_specificity_filtered.csv",
            index=False,
        )

        # Get word embeddings similarities
        embedding_similarities, word2vec_model = self.train_word2vec(df)

        # Get contextual relevance
        context_relevance = self.evaluate_contextual_relevance(df, term_stats["term"])
        context_df = pd.DataFrame(
            {
                "term": context_relevance.index,
                "contextual_relevance": context_relevance.values,
            }
        )
        context_df.to_csv(
            f"{self.temp_dir}/intermediate/{self.culture_type}_contextual.csv",
            index=False,
        )

        # Cluster similar terms
        representative_terms = self.cluster_similar_terms(term_stats, word2vec_model)
        pd.DataFrame({"term": representative_terms}).to_csv(
            f"{self.temp_dir}/intermediate/{self.culture_type}_clustered.csv",
            index=False,
        )

        # Combine results
        final_terms = pd.merge(
            term_stats, embedding_similarities, on="term", how="outer"
        ).fillna(0)

        # Add contextual relevance
        final_terms = pd.merge(final_terms, context_df, on="term", how="left").fillna(0)

        # Normalize scores for better combined scoring
        final_terms["normalized_spec"] = (
            final_terms["specificity"] - final_terms["specificity"].min()
        ) / (
            final_terms["specificity"].max() - final_terms["specificity"].min() + 1e-10
        )
        final_terms["normalized_sim"] = (
            final_terms["similarity"] - final_terms["similarity"].min()
        ) / (final_terms["similarity"].max() - final_terms["similarity"].min() + 1e-10)
        final_terms["normalized_context"] = (
            final_terms["contextual_relevance"]
            - final_terms["contextual_relevance"].min()
        ) / (
            final_terms["contextual_relevance"].max()
            - final_terms["contextual_relevance"].min()
            + 1e-10
        )

        # Compute combined score with normalized values
        final_terms["combined_score"] = (
            final_terms["normalized_spec"] * 0.6
            + final_terms["normalized_sim"] * 0.2
            + final_terms["normalized_context"] * 0.2
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

        if os.path.exists(self.temp_dir):
            try:
                import shutil

                shutil.rmtree(self.temp_dir)
                logger.debug("Removed temporary directory and its contents")
            except Exception as e:
                logger.warning(f"Failed to remove temporary directory: {str(e)}")
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
