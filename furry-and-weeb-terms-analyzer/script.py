import spacy
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CommunityTermAnalyzer:
    def __init__(self, communities):
        """
        Initialize analyzer with multiple communities to analyze

        Args:
            communities (dict): Dictionary mapping community names to seed terms
        """
        self.communities = communities
        self.nlp = spacy.load("en_core_web_sm")

    def load_data(self):
        """Load data from dataset"""
        logger.info("Loading dataset...")
        df = pd.read_csv("./output/dataset-filter/bluesky_ten_million_english_only.csv")
        logger.info(f"Loaded {len(df)} English posts")
        return df

    def identify_community_posts(self, df):
        """Identify posts that contain seed terms for each community"""
        for community, seed_terms in self.communities.items():
            logger.info(f"Identifying {community}-related posts...")
            pattern = "|".join(seed_terms)
            col_name = f"is_{community}"
            df[col_name] = df["text"].str.lower().str.contains(pattern, na=False)
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

    def compute_term_specificity(self, df, community):
        """Compute how specific terms are to community posts vs non-community posts"""
        logger.info(f"Computing term specificity for {community}...")

        # Create separate corpora
        col_name = f"is_{community}"
        community_corpus = df[df[col_name]]["cleaned_text"]
        non_community_corpus = df[~df[col_name]]["cleaned_text"]

        # Vectorize both corpora
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=10,  # Require terms to appear in at least 10 documents
        )

        # Fit on entire corpus to get shared vocabulary
        vectorizer.fit(df["cleaned_text"])

        # Transform both corpora
        community_matrix = vectorizer.transform(community_corpus)
        non_community_matrix = vectorizer.transform(non_community_corpus)

        # Compute normalized frequencies
        community_freqs = np.array(community_matrix.sum(axis=0)).flatten() / len(
            community_corpus
        )
        non_community_freqs = np.array(
            non_community_matrix.sum(axis=0)
        ).flatten() / len(non_community_corpus)

        # Compute specificity score (ratio of frequencies)
        epsilon = 1e-10  # Small constant to avoid division by zero
        specificity = np.log(
            (community_freqs + epsilon) / (non_community_freqs + epsilon)
        )

        # Create DataFrame with results
        terms = vectorizer.get_feature_names_out()
        term_stats = pd.DataFrame(
            {
                "term": terms,
                f"{community}_freq": community_freqs,
                f"non_{community}_freq": non_community_freqs,
                "specificity": specificity,
            }
        )

        # Filter for terms that appear more in community posts
        term_stats = term_stats[term_stats["specificity"] > 0]
        return term_stats.sort_values("specificity", ascending=False)

    def train_word2vec(self, df, community):
        """Train Word2Vec and find terms similar to seed terms for a specific community"""
        logger.info(f"Training Word2Vec model for {community}...")
        sentences = [text.split() for text in df["cleaned_text"]]
        model = Word2Vec(
            sentences, vector_size=1000, window=5, min_count=5, workers=4, sg=1
        )

        # Find similar terms to seed terms
        similar_terms = []
        for seed in self.communities[community]:
            if seed in model.wv:
                similar = model.wv.most_similar(seed, topn=10)
                similar_terms.extend(similar)

        return pd.DataFrame(similar_terms, columns=["term", "similarity"])

    def analyze(self):
        # Load and preprocess data
        df = self.load_data()
        logger.info(f"Loaded {len(df)} posts")

        df["cleaned_text"] = df["text"].apply(self.preprocess_text)
        df = self.identify_community_posts(df)

        results = {}

        # Analyze each community
        for community in self.communities:
            # Get term specificity
            term_stats = self.compute_term_specificity(df, community)

            # Get word embeddings similarities
            embedding_similarities = self.train_word2vec(df, community)

            # Combine results
            final_terms = pd.merge(
                term_stats, embedding_similarities, on="term", how="outer"
            ).fillna(0)

            # Compute combined score
            final_terms["combined_score"] = (
                final_terms["specificity"] * 0.7 + final_terms["similarity"] * 0.3
            )

            results[community] = final_terms.sort_values(
                "combined_score", ascending=False
            )

            # Save individual community results
            results[community].to_csv(
                f"./output/terms-analysis/{community}_terms_analysis.csv", index=False
            )
            logger.info(f"Results saved to {community}_terms_analysis.csv")

        return results


if __name__ == "__main__":
    # Define communities and their seed terms
    communities = {
        "furry": {
            "furry",
            "fursuitfriday",
            "fursuit",
            "uwu",
            "owo",
            "paws",
            "yiff",
            "fursona",
        },
        "weeb": {"anime", "manga", "otaku", "waifu", "weeb"},
    }

    analyzer = CommunityTermAnalyzer(communities)
    results = analyzer.analyze()
