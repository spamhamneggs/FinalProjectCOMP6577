#!/usr/bin/env python3
import os
import random
import argparse
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm
import torch

# For fine-tuning
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer

# For Bluesky/ATProto
from atproto import Client

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


class BlueskyClassifier:
    def __init__(
        self, model_name="unsloth/gemma-3-1b-it-unsloth-bnb-4bit", batch_size=8
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None
        self.tokenizer = None

        # Load term databases with scores
        self.weeb_terms = self._load_term_database("weeb_terms.csv")
        self.furry_terms = self._load_term_database("furry_terms.csv")

        # ATProto client for Bluesky
        self.client = None

    def _load_term_database(self, csv_file: str) -> pd.DataFrame:
        """Load and process a terms CSV file"""
        if not os.path.exists(csv_file):
            print(f"Warning: {csv_file} not found. Using empty dataframe.")
            return pd.DataFrame(columns=["term", "combined_score"])

        df = pd.read_csv(csv_file)
        df = df[["term", "combined_score"]].sort_values(
            by="combined_score", ascending=False
        )
        return df

    def setup_model(self):
        """Initialize the model and tokenizer"""
        print(f"Loading model: {self.model_name}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=2048,
            dtype=None,  # None will use model's default dtype
            load_in_4bit=True,
        )

        # Adapter for fine-tuning
        self.model = FastLanguageModel.get_peft_model(
            model=self.model,
            r=16,  # LoRA rank
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
        )

        print("Model loaded successfully.")

    def login_bluesky(self, username: str, password: str):
        """Log in to Bluesky"""
        try:
            self.client = Client()
            self.client.login(username, password)
            print(f"Logged in to Bluesky as {username}")
            return True
        except Exception as e:
            print(f"Failed to log in to Bluesky: {e}")
            return False

    def _prepare_training_data(self, bluesky_data_csv: str) -> List[Dict[str, str]]:
        """Prepare training data from Bluesky posts CSV and term databases"""
        print(f"Loading Bluesky data from {bluesky_data_csv}...")

        if not os.path.exists(bluesky_data_csv):
            raise FileNotFoundError(f"Bluesky data file {bluesky_data_csv} not found")

        # Load Bluesky posts
        posts_df = pd.read_csv(bluesky_data_csv)
        posts_df = posts_df[posts_df["type"] == "post"]
        posts_df = posts_df.dropna(subset=["text"])

        training_data = []

        # Analyze each post
        for _, row in tqdm(
            posts_df.iterrows(), total=len(posts_df), desc="Analyzing posts"
        ):
            text = str(row["text"]).lower()

            # Calculate weeb and furry scores for each post
            weeb_score = self._calculate_category_score(text, self.weeb_terms)
            furry_score = self._calculate_category_score(text, self.furry_terms)

            # Determine the label based on scores
            if weeb_score > 0.7 and weeb_score > furry_score:
                label = "Weeb"
            elif furry_score > 0.7 and furry_score > weeb_score:
                label = "Furry"
            elif max(weeb_score, furry_score) > 0.4:
                # Some indication but not strong enough
                if weeb_score > furry_score:
                    label = "Slight Weeb"
                else:
                    label = "Slight Furry"
            else:
                label = "Normie"

            # Create training example
            prompt = f"Analyze this Bluesky post for weeb and furry traits: {text}"
            response = f"Classification: {label}\nWeeb Score: {weeb_score:.2f}\nFurry Score: {furry_score:.2f}"

            training_data.append({"prompt": prompt, "response": response})

        print(f"Created {len(training_data)} training examples")
        return training_data

    def _calculate_category_score(self, text: str, terms_df: pd.DataFrame) -> float:
        """Calculate category score for a given text using terms database"""
        if terms_df.empty:
            return 0.0

        # Initialize score
        score = 0.0
        max_potential_score = 0.0

        # Check each term
        for _, row in terms_df.iterrows():
            term = row["term"].lower()
            term_score = row["combined_score"]

            # Add to max potential score
            max_potential_score += term_score

            # Check if term appears in text
            if term in text:
                score += term_score

        # Normalize score
        if max_potential_score > 0:
            return score / max_potential_score
        return 0.0

    def finetune_model(
        self,
        bluesky_data_csv: str,
        output_dir: str = "finetuned_model",
        epochs: int = 3,
        learning_rate: float = 2e-4,
    ):
        """Fine-tune the model using the prepared data"""
        if self.model is None or self.tokenizer is None:
            self.setup_model()

        # Prepare training data
        train_data = self._prepare_training_data(bluesky_data_csv)

        # Create dataset formatter function
        def formatting_func(example):
            return f"<|user|>\n{example['prompt']}\n<|assistant|>\n{example['response']}</s>"

        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=1,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            remove_unused_columns=False,
            report_to="none",
            optim="adamw_torch",
            fp16=torch.cuda.is_available(),
        )

        # Set up SFT trainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=train_data,
            formatting_func=formatting_func,
            max_seq_length=1024,
            packing=True,
        )

        # Train the model
        print("Starting model fine-tuning...")
        trainer.train()

        # Save the fine-tuned model
        print(f"Saving fine-tuned model to {output_dir}")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        return output_dir

    def load_finetuned_model(self, model_path: str):
        """Load a fine-tuned model"""
        print(f"Loading fine-tuned model from {model_path}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path, max_seq_length=2048, dtype=None, load_in_4bit=True
        )
        print("Fine-tuned model loaded successfully")

    async def fetch_user_posts(self, user_handle: str, limit: int = 100) -> List[str]:
        """Fetch user posts from Bluesky"""
        if self.client is None:
            raise ValueError("Not logged in to Bluesky. Call login_bluesky first.")

        try:
            # Get user DID
            user_info = self.client.get_profile(user_handle)
            user_did = user_info.did

            # Fetch posts
            posts = []
            cursor = None

            while len(posts) < limit:
                response = self.client.get_author_feed(
                    actor=user_did, limit=min(100, limit - len(posts)), cursor=cursor
                )

                if not response.feed:
                    break

                for feed_item in response.feed:
                    if hasattr(feed_item.post, "record") and hasattr(
                        feed_item.post.record, "text"
                    ):
                        posts.append(feed_item.post.record.text)

                cursor = response.cursor
                if cursor is None:
                    break

            print(f"Fetched {len(posts)} posts for user {user_handle}")
            return posts

        except Exception as e:
            print(f"Error fetching posts for {user_handle}: {e}")
            return []

    def classify_user(self, posts: List[str]) -> Dict[str, Any]:
        """Classify a user based on their posts"""
        if not posts:
            return {
                "classification": "Unknown",
                "weeb_score": 0,
                "furry_score": 0,
                "normie_score": 1,
            }

        if self.model is None or self.tokenizer is None:
            raise ValueError(
                "Model not loaded. Call setup_model or load_finetuned_model first."
            )

        # Combined text from all posts
        combined_text = " ".join(posts).lower()

        # Calculate raw scores from term databases
        weeb_score = self._calculate_category_score(combined_text, self.weeb_terms)
        furry_score = self._calculate_category_score(combined_text, self.furry_terms)

        # Use model to refine classification
        model_inputs = []
        for post in posts[:10]:  # Use at most 10 posts to avoid excessive processing
            prompt = f"Analyze this Bluesky post for weeb and furry traits: {post}"
            model_inputs.append(
                self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            )

        # Make model predictions
        model_weeb_score = 0
        model_furry_score = 0

        for input_ids in model_inputs:
            with torch.no_grad():
                generation = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=50,
                    temperature=0.1,
                    do_sample=False,
                )

                response = self.tokenizer.decode(
                    generation[0], skip_special_tokens=True
                )

                # Extract scores from model response
                if "Weeb Score:" in response:
                    try:
                        weeb_part = (
                            response.split("Weeb Score:")[1].split("\n")[0].strip()
                        )
                        model_weeb_score += float(weeb_part)
                    except (ValueError, IndexError) as e:
                        print(f"Error processing Weeb Score: {e}")

                if "Furry Score:" in response:
                    try:
                        furry_part = (
                            response.split("Furry Score:")[1].split("\n")[0].strip()
                        )
                        model_furry_score += float(furry_part)
                    except (ValueError, IndexError) as e:
                        print(f"Error processing Furry Score: {e}")

        # Average model scores
        if model_inputs:
            model_weeb_score /= len(model_inputs)
            model_furry_score /= len(model_inputs)

        # Combine raw scores with model predictions (weighted average)
        combined_weeb_score = 0.4 * weeb_score + 0.6 * model_weeb_score
        combined_furry_score = 0.4 * furry_score + 0.6 * model_furry_score

        # Calculate normie score as inverse of weeb and furry scores
        normie_score = 1.0 - max(combined_weeb_score, combined_furry_score)
        normie_score = max(0, normie_score)  # Ensure non-negative

        # Determine final classification
        if combined_weeb_score > 0.6 and combined_weeb_score > combined_furry_score:
            classification = "Weeb"
        elif combined_furry_score > 0.6 and combined_furry_score > combined_weeb_score:
            classification = "Furry"
        elif max(combined_weeb_score, combined_furry_score) > 0.3:
            if combined_weeb_score > combined_furry_score:
                classification = "Slight Weeb"
            else:
                classification = "Slight Furry"
        else:
            classification = "Normie"

        return {
            "classification": classification,
            "weeb_score": round(combined_weeb_score, 3),
            "furry_score": round(combined_furry_score, 3),
            "normie_score": round(normie_score, 3),
        }


class BlueskyUserDataset:
    """Helper class to prepare Bluesky data for training"""

    @staticmethod
    def preprocess_and_save(input_csv: str, output_csv: str):
        """Preprocess Bluesky data and save to CSV"""
        if not os.path.exists(input_csv):
            raise FileNotFoundError(f"Input file {input_csv} not found")

        # Load data
        print(f"Loading data from {input_csv}...")
        df = pd.read_csv(input_csv)

        # Basic preprocessing
        df = df[df["type"] == "post"]
        df = df.dropna(subset=["text"])

        # Replace any 0------------0 values with empty strings
        df["text"] = df["text"].apply(lambda x: "" if x == "0------------0" else x)
        df = df[df["text"] != ""]

        # Save processed data
        print(f"Saving processed data to {output_csv}...")
        df.to_csv(output_csv, index=False)
        print(f"Saved {len(df)} processed posts")


def main():
    parser = argparse.ArgumentParser(description="Bluesky User Classifier")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Preprocessing command
    preprocess_parser = subparsers.add_parser(
        "preprocess", help="Preprocess Bluesky data"
    )
    preprocess_parser.add_argument("--input", required=True, help="Input CSV file")
    preprocess_parser.add_argument("--output", required=True, help="Output CSV file")

    # Fine-tuning command
    finetune_parser = subparsers.add_parser("finetune", help="Fine-tune the model")
    finetune_parser.add_argument(
        "--data", required=True, help="Processed Bluesky data CSV"
    )
    finetune_parser.add_argument(
        "--output_dir",
        default="finetuned_model",
        help="Output directory for fine-tuned model",
    )
    finetune_parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    finetune_parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training"
    )
    finetune_parser.add_argument(
        "--learning_rate", type=float, default=2e-4, help="Learning rate"
    )

    # Classify command
    classify_parser = subparsers.add_parser("classify", help="Classify a Bluesky user")
    classify_parser.add_argument(
        "--model", required=True, help="Path to fine-tuned model"
    )
    classify_parser.add_argument("--username", required=True, help="Bluesky username")
    classify_parser.add_argument(
        "--bluesky_user", required=True, help="Your Bluesky username"
    )
    classify_parser.add_argument(
        "--bluesky_pass", required=True, help="Your Bluesky password"
    )

    args = parser.parse_args()

    if args.command == "preprocess":
        BlueskyUserDataset.preprocess_and_save(args.input, args.output)

    elif args.command == "finetune":
        classifier = BlueskyClassifier(batch_size=args.batch_size)
        classifier.setup_model()
        classifier.finetune_model(
            bluesky_data_csv=args.data,
            output_dir=args.output_dir,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
        )

    elif args.command == "classify":
        import asyncio

        async def run_classification():
            classifier = BlueskyClassifier()
            classifier.load_finetuned_model(args.model)

            # Login to Bluesky
            success = classifier.login_bluesky(args.bluesky_user, args.bluesky_pass)
            if not success:
                print("Failed to login to Bluesky")
                return

            # Fetch user posts
            posts = await classifier.fetch_user_posts(args.username)
            if not posts:
                print(f"No posts found for user {args.username}")
                return

            # Classify user
            result = classifier.classify_user(posts)

            # Display results
            print("\n" + "=" * 50)
            print(f"Classification Results for @{args.username}")
            print("=" * 50)
            print(f"Classification: {result['classification']}")
            print(f"Weeb Score: {result['weeb_score']:.3f}")
            print(f"Furry Score: {result['furry_score']:.3f}")
            print(f"Normie Score: {result['normie_score']:.3f}")
            print("=" * 50 + "\n")

        asyncio.run(run_classification())

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
