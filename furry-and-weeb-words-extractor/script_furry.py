import spacy
import pandas as pd
import numpy as np
from datasets import load_dataset
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FurryTermAnalyzer:
    def __init__(self):
        # Seed terms that are definitively furry-related
        self.seed_terms = {
            "furry",
            "fursuitfriday",
            "fursuit",
            "uwu",
            "owo",
            "paws",
            "yiff",
            "fursona",
        }

        self.nlp = spacy.load("en_core_web_sm")

    def load_data(self):
        """Load data from Hugging Face dataset"""
        logger.info("Loading dataset...")
        dataset = load_dataset("Roronotalt/bluesky-ten-million", split="train")
        logger.info("Converting to DataFrame...")
        df = pd.DataFrame(dataset)
        df["langs_str"] = df["langs"].astype(str)
        df = df[df["langs_str"].str.contains("en", na=False)]
        return df

    def identify_furry_posts(self, df):
        """Identify posts that contain seed terms"""
        logger.info("Identifying furry-related posts...")
        pattern = "|".join(self.seed_terms)
        df["is_furry"] = df["text"].str.lower().str.contains(pattern, na=False)
        return df

    def preprocess_text(self, text):
        doc = self.nlp(text.lower())
        return " ".join(
            [
                token.text
                for token in doc
                if not token.is_stop and not token.is_punct and token.text.strip()
            ]
        )

    def compute_term_specificity(self, df):
        """Compute how specific terms are to furry posts vs non-furry posts"""
        logger.info("Computing term specificity...")

        # Create separate corpora
        furry_corpus = df[df["is_furry"]]["cleaned_text"]
        non_furry_corpus = df[~df["is_furry"]]["cleaned_text"]

        # Vectorize both corpora
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=10,  # Require terms to appear in at least 10 documents
        )

        # Fit on entire corpus to get shared vocabulary
        vectorizer.fit(df["cleaned_text"])

        # Transform both corpora
        furry_matrix = vectorizer.transform(furry_corpus)
        non_furry_matrix = vectorizer.transform(non_furry_corpus)

        # Compute normalized frequencies
        furry_freqs = np.array(furry_matrix.sum(axis=0)).flatten() / len(furry_corpus)
        non_furry_freqs = np.array(non_furry_matrix.sum(axis=0)).flatten() / len(
            non_furry_corpus
        )

        # Compute specificity score (ratio of frequencies)
        epsilon = 1e-10  # Small constant to avoid division by zero
        specificity = np.log((furry_freqs + epsilon) / (non_furry_freqs + epsilon))

        # Create DataFrame with results
        terms = vectorizer.get_feature_names_out()
        term_stats = pd.DataFrame(
            {
                "term": terms,
                "furry_freq": furry_freqs,
                "non_furry_freq": non_furry_freqs,
                "specificity": specificity,
            }
        )

        # Filter for terms that appear more in furry posts
        term_stats = term_stats[term_stats["specificity"] > 0]
        return term_stats.sort_values("specificity", ascending=False)

    def train_word2vec(self, df):
        """Train Word2Vec and find terms similar to seed terms"""
        logger.info("Training Word2Vec model...")
        sentences = [text.split() for text in df["cleaned_text"]]
        model = Word2Vec(
            sentences, vector_size=1000, window=5, min_count=5, workers=4, sg=1
        )

        # Find similar terms to seed terms
        similar_terms = []
        for seed in self.seed_terms:
            if seed in model.wv:
                similar = model.wv.most_similar(seed, topn=10)
                similar_terms.extend(similar)

        return pd.DataFrame(similar_terms, columns=["term", "similarity"])

    def analyze(self):
        # Load and preprocess data
        df = self.load_data()
        logger.info(f"Loaded {len(df)} posts")

        df["cleaned_text"] = df["text"].apply(self.preprocess_text)
        df = self.identify_furry_posts(df)

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

        return final_terms.sort_values("combined_score", ascending=False)


if __name__ == "__main__":
    analyzer = FurryTermAnalyzer()
    results = analyzer.analyze()
    results.to_csv("../output/terms-analysis/furry_terms_analysis.csv", index=False)
    logger.info("Results saved to furry_terms_analysis.csv")
