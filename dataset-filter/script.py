import logging
import os

import pandas as pd
from datasets import load_dataset

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


def load_and_filter_data(dataset_name, split="train"):
    """Load data from Hugging Face dataset"""
    logger.info("Loading dataset...")
    dataset = load_dataset(dataset_name, split=f"{split}")
    logger.info("Converting to DataFrame...")
    df = pd.DataFrame(dataset)
    logger.info("Filtering DataFrame for English posts...")
    df["langs_str"] = df["langs"].astype(str)
    df = df[df["langs_str"].str.contains("en", na=False)]
    logger.info("Finished Filtering DataFrame for English posts!")
    logger.info(f"Got {len(df)} English posts")
    df.drop(columns=["langs_str"], inplace=True)
    return df


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dirs = setup_directories(base_dir)

    output_dir = dirs["output"]

    dataset_filter_dir = f"{output_dir}/dataset-filter"
    os.makedirs(dataset_filter_dir, exist_ok=True)

    results = load_and_filter_data("Roronotalt/bluesky-ten-million")

    # Save results
    logger.info("Saving results...")
    results.to_csv(
        f"{dataset_filter_dir}/bluesky_ten_million_english_only.csv", index=False
    )
    logger.info("Results saved to bluesky_ten_million_english_only.csv")
