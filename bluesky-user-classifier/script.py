#!/usr/bin/env python3
import os
import random
import argparse
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import torch
from multiprocessing import Pool
import matplotlib.pyplot as plt
import seaborn as sns

# For fine-tuning
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer

# For Bluesky/ATProto
from atproto import Client

# For metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


def process_post_for_training(args) -> Dict[str, Any]:
    """Process a single post for training data generation."""
    text, weeb_terms, furry_terms = args

    if not text.strip():  # Skip empty posts
        return None

    # Helper functions that were previously methods of BlueskyClassifier
    def calculate_category_score(text: str, terms_df: pd.DataFrame) -> float:
        if terms_df.empty:
            return 0.0
        text = text.lower()
        matched_terms = terms_df[terms_df["term"].str.lower().apply(lambda x: x in text)]
        score = matched_terms["combined_score"].sum()
        max_potential_score = terms_df["combined_score"].sum()
        return score / max_potential_score if max_potential_score > 0 else 0.0

    def determine_classification_labels(w_score: float, f_score: float) -> Tuple[str, str]:
        primary_label = "Normie"
        secondary_label = "None"

        # Rule: Normie if max score <= 0.4
        if max(w_score, f_score) <= 0.4:
            return "Normie", "None"

        # Determine dominant and secondary scores/types
        is_weeb_dominant = w_score >= f_score  # Default to weeb in case of exact tie

        dominant_score = w_score if is_weeb_dominant else f_score
        secondary_s = f_score if is_weeb_dominant else w_score

        dominant_type_strong = "Weeb" if is_weeb_dominant else "Furry"
        dominant_type_slight = "Slight Weeb" if is_weeb_dominant else "Slight Furry"

        secondary_type_strong = "Furry" if is_weeb_dominant else "Weeb"
        secondary_type_slight = "Slight Furry" if is_weeb_dominant else "Slight Weeb"

        # Case 1: Strong primary (dominant score > 0.7)
        if dominant_score > 0.7:
            primary_label = dominant_type_strong
            if secondary_s > 0.7:  # Strong secondary
                secondary_label = secondary_type_strong
            elif secondary_s > 0.4:  # Slight secondary
                secondary_label = secondary_type_slight
            else:  # No significant secondary
                secondary_label = "None"
        # Case 2: Slight primary (dominant score is > 0.4 and <= 0.7)
        elif dominant_score > 0.4:
            primary_label = dominant_type_slight
            # Secondary can only be slight or none if primary is slight
            if secondary_s > 0.4 and secondary_s <= 0.7:  # Slight secondary
                secondary_label = secondary_type_slight
            # If secondary_s > 0.7, it would have been dominant, handled by Case 1 logic inversion.
            # This ensures primary is truly the 'slight dominant' one.
            # If secondary_s is also slight, it's a valid slight-slight pair.
            else:  # No significant or only very weak secondary
                secondary_label = "None"
        # Fallback, should be covered by the first condition (max_score <= 0.4)
        else:
            primary_label = "Normie"
            secondary_label = "None"

        return primary_label, secondary_label

    weeb_score = calculate_category_score(text, weeb_terms)
    furry_score = calculate_category_score(text, furry_terms)

    primary_label, secondary_label = determine_classification_labels(
        weeb_score, furry_score
    )

    prompt = f"Analyze this Bluesky post for weeb and furry traits: {text}"
    response = (
        f"Primary Classification: {primary_label}\n"
        f"Secondary Classification: {secondary_label}\n"
        f"Weeb Score: {weeb_score:.2f}\n"
        f"Furry Score: {furry_score:.2f}"
    )

    return {
        "prompt": prompt,
        "response": response,
        "true_label_heuristic": f"{primary_label}-{secondary_label}",
    }


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

        # Define the canonical combined labels for classification (Primary-Secondary)
        # "None" as a secondary label means no significant secondary category.
        self.defined_combined_labels = sorted(
            [
                "Normie-None",
                "Weeb-None",
                "Weeb-Slight Furry",
                "Weeb-Furry",
                "Furry-None",
                "Furry-Slight Weeb",
                "Furry-Weeb",
                "Slight Weeb-None",
                "Slight Weeb-Slight Furry",
                "Slight Furry-None",
                "Slight Furry-Slight Weeb",
            ]
        )
        # For parsing individual primary/secondary labels if needed elsewhere
        self.primary_labels_set = {
            "Weeb",
            "Furry",
            "Slight Weeb",
            "Slight Furry",
            "Normie",
        }
        self.secondary_labels_set = {
            "Weeb",
            "Furry",
            "Slight Weeb",
            "Slight Furry",
            "None",
        }

        # Load term databases with scores
        self.weeb_terms = self._load_term_database(
            "output/terms-analysis/weeb_terms.csv"
        )
        self.furry_terms = self._load_term_database(
            "output/terms-analysis/furry_terms.csv"
        )

        # ATProto client for Bluesky
        self.client = None

    def _load_term_database(self, csv_file: str) -> pd.DataFrame:
        """Load and process a terms CSV file"""
        if not os.path.exists(csv_file):
            print(f"Error: {csv_file} not found. Exiting")
            exit()

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

        # Important for Gemma: Ensure pad token is set if not already (Unsloth usually handles this)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

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
        """Prepare training data from Bluesky posts CSV and term databases using multiprocessing"""
        print(f"Loading Bluesky data from {bluesky_data_csv}...")

        if not os.path.exists(bluesky_data_csv):
            raise FileNotFoundError(f"Bluesky data file {bluesky_data_csv} not found")

        posts_df = pd.read_csv(bluesky_data_csv)
        posts_df = posts_df[posts_df["type"] == "post"]
        posts_df = posts_df.dropna(subset=["text"])
        posts_df["text"] = posts_df["text"].astype(str)  # Ensure text is string

        # Prepare arguments for multiprocessing
        process_args = [
            (text, self.weeb_terms, self.furry_terms)
            for text in posts_df["text"]
        ]

        # Process posts in parallel
        with Pool(processes=4) as pool:
            results = list(
                tqdm(
                    pool.imap(process_post_for_training, process_args),
                    total=len(process_args),
                    desc="Analyzing posts for training data",
                    mininterval=10.0,
                )
            )

        # Filter out None results
        training_data = [r for r in results if r is not None]
        print(f"Created {len(training_data)} training examples")
        return training_data

    def _evaluate_model_and_print_metrics(
        self, eval_data: List[Dict[str, str]], output_dir: str
    ):
        """Evaluates the model on the evaluation data and prints metrics for combined labels."""
        print(f"\nEvaluating model on {len(eval_data)} samples...")

        y_true_combined = []
        y_pred_combined = []

        # Ensure model and tokenizer are on the correct device
        self.model.to(self.device)

        for item in tqdm(eval_data, desc="Evaluating model", mininterval=10.0):
            prompt_text = item["prompt"]
            # true_label_heuristic is already a combined "Primary-Secondary" string
            true_combined_label = item["true_label_heuristic"]

            # Prepare prompt for inference (Unsloth/Gemma specific with ChatML)
            # The model expects the format: <|user|>\n{prompt}\n<|assistant|>
            # It will then generate the assistant's response.
            # Note: Unsloth's FastLanguageModel.generate handles chat templates internally if available.
            # For explicit control or if issues arise, manual formatting is safer.
            # Let's use the manual formatting to be sure.

            # Gemma uses specific tokens <start_of_turn> and <end_of_turn>
            # For Unsloth fine-tuned Gemma, it often maps to a common chat template like ChatML.
            # The SFTTrainer formatting_func uses <|user|> and <|assistant|>. We should match this for inference.
            formatted_prompt_for_inference = f"<|user|>\n{prompt_text}\n<|assistant|>"
            # Ensure max_length accounts for prompt and potential response length
            inputs = self.tokenizer(
                formatted_prompt_for_inference,
                return_tensors="pt",
                truncation=True,
                max_length=self.tokenizer.model_max_length - 70,
            ).to(
                self.device
            )  # max_length to prevent overflow, -70 for generated response

            with torch.no_grad():
                # Generate response. max_new_tokens should be enough for the classification output.
                # pad_token_id is crucial for batching and proper generation.
                outputs = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,  # Pass attention_mask
                    max_new_tokens=70,  # Increased slightly for safety
                    pad_token_id=self.tokenizer.pad_token_id,  # Use the tokenizer's pad token id
                    eos_token_id=self.tokenizer.eos_token_id,  # Explicitly set EOS
                    do_sample=False,  # For deterministic output for eval
                    temperature=0.1,  # Low temperature for less randomness
                )

            # Decode only the generated part
            # outputs[0] includes the input_ids. We need to slice them off.
            input_ids_length = inputs.input_ids.shape[1]
            generated_tokens = outputs[0][input_ids_length:]
            model_response_part = self.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            ).strip()

            predicted_combined_label = self._parse_combined_label_from_text(
                model_response_part, source_type="model output"
            )

            y_true_combined.append(true_combined_label)
            y_pred_combined.append(predicted_combined_label)

        print("\n--- Classification Metrics (Combined Labels) ---")

        current_labels_in_data = sorted(list(set(y_true_combined + y_pred_combined)))

        # Filter self.defined_combined_labels to only those present in the actual data for the report
        # and ensure "Unknown-Unknown" is included if it appeared.
        report_labels = [
            label
            for label in self.defined_combined_labels
            if label in current_labels_in_data
        ]
        if (
            "Unknown-Unknown" in current_labels_in_data
            and "Unknown-Unknown" not in report_labels
        ):
            report_labels.append("Unknown-Unknown")
        report_labels = sorted(list(set(report_labels)))  # Ensure uniqueness and order

        print(
            classification_report(
                y_true_combined, y_pred_combined, labels=report_labels, zero_division=0
            )
        )
        print("\n--- Confusion Matrix (Combined Labels) ---")
        cm = confusion_matrix(y_true_combined, y_pred_combined, labels=report_labels)
        print(cm)

        plt.figure(
            figsize=(
                max(12, len(report_labels) * 0.8),
                max(10, len(report_labels) * 0.6),
            )
        )  # Dynamic sizing
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=report_labels,
            yticklabels=report_labels,
        )
        plt.xlabel("Predicted Label (Primary-Secondary)")
        plt.ylabel("True Label (Primary-Secondary)")
        plt.title("Confusion Matrix (Combined Labels)")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()  # Adjust layout to prevent labels from overlapping

        cm_path = os.path.join(output_dir, "confusion_matrix_combined.png")
        try:
            plt.savefig(cm_path)
            print(f"Confusion matrix saved to {cm_path}")
        except Exception as e:
            print(f"Error saving confusion matrix: {e}")
        plt.close()  # Close the plot to free memory

    def finetune_model(
        self,
        bluesky_data_csv: str,
        output_dir: str = "finetuned_model",
        epochs: int = 3,
        learning_rate: float = 2e-4,
        eval_split_size: float = 0.20,
    ):
        """Fine-tune the model using the prepared data and evaluate it."""
        if self.model is None or self.tokenizer is None:
            self.setup_model()

        # Prepare all data
        all_prepared_data = self._prepare_training_data(bluesky_data_csv)
        if not all_prepared_data:
            print("No training data prepared. Skipping fine-tuning and evaluation.")
            return None

        # Extract prompts, responses, and true labels for splitting
        # The 'true_label_heuristic' is what we'll use as ground truth for evaluation

        # Stratification needs labels. We use 'true_label_heuristic'.
        labels_for_stratification = [
            item["true_label_heuristic"] for item in all_prepared_data
        ]
        train_data_dicts, eval_data_dicts = [], []

        if eval_split_size > 0 and len(all_prepared_data) > 1:
            # Check for stratification feasibility for sklearn's train_test_split
            # It requires at least 2 samples for each class present in `labels_for_stratification`
            # if stratify parameter is used.
            min_samples_per_class_for_split = 2
            counts = pd.Series(labels_for_stratification).value_counts()
            # Stratification is possible if all unique labels have at least min_samples_per_class_for_split
            # and there's more than one unique label.
            stratify_possible_sklearn = (
                all(counts >= min_samples_per_class_for_split) and len(counts) > 1
            )

            # Ensure enough samples for both train and eval sets after splitting
            enough_total_samples_for_split = (
                len(all_prepared_data) * (1 - eval_split_size) >= 1
                and len(all_prepared_data) * eval_split_size >= 1
            )

            if stratify_possible_sklearn and enough_total_samples_for_split:
                train_data_dicts, eval_data_dicts = train_test_split(
                    all_prepared_data,
                    test_size=eval_split_size,
                    random_state=RANDOM_SEED,
                    stratify=labels_for_stratification,
                )
            else:  # Not enough samples for a split
                if not stratify_possible_sklearn:
                    print(
                        "Warning: Stratification not possible due to insufficient samples in some classes or only one class. Splitting without stratification."
                    )
                elif not enough_total_samples_for_split:
                    print(
                        "Warning: Not enough total samples to create both training and evaluation sets with the current split size. Splitting without stratification or adjusting."
                    )
                # Fallback to non-stratified split if conditions aren't met but splitting is desired
                if enough_total_samples_for_split:
                    train_data_dicts, eval_data_dicts = train_test_split(
                        all_prepared_data,
                        test_size=eval_split_size,
                        random_state=RANDOM_SEED,
                    )
                else:  # Not enough for a split at all, use all for training
                    print(
                        "Warning: Not enough samples for a train/eval split. Using all data for training and skipping evaluation."
                    )
                    train_data_dicts = all_prepared_data
                    eval_data_dicts = []
        else:
            print(
                "Evaluation split size is 0 or not enough data. Using all data for training and skipping evaluation."
            )
            train_data_dicts = all_prepared_data
            eval_data_dicts = []

        print(
            f"Training with {len(train_data_dicts)} samples, evaluating with {len(eval_data_dicts)} samples."
        )

        # The SFTTrainer expects data with 'prompt' and 'response' or a single text field.
        # Our train_data_dicts items have 'prompt', 'response', and 'true_label_heuristic'.
        # We need to ensure the formatting_func only uses 'prompt' and 'response'.
        sft_train_dataset = [
            {"prompt": d["prompt"], "response": d["response"]} for d in train_data_dicts
        ]

        # Create dataset formatter function
        # Capture tokenizer's EOS token for use in the formatting function
        eos_token = self.tokenizer.eos_token

        def formatting_func(example):
            prompt_text = str(example.get("prompt", ""))  # Robust access
            response_text = str(example.get("response", ""))  # Robust access
            # Standard ChatML format that Unsloth often uses for Gemma fine-tuning
            return f"<|user|>\n{prompt_text}\n<|assistant|>\n{response_text}{eos_token}"

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=1,  # Adjust if OOM
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            remove_unused_columns=False,  # Important if dataset has extra columns
            report_to="none",  # "tensorboard" or "wandb" if desired
            optim="adamw_torch",
            fp16=torch.cuda.is_available(),
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        )

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=sft_train_dataset,  # Use the list of dicts
            formatting_func=formatting_func,
            max_seq_length=1024,  # Max length of formatted prompt + response
            packing=True,  # Pack multiple short sequences
        )

        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("Starting model fine-tuning...")
        trainer.train()

        print(f"Saving fine-tuned model to {output_dir}")
        trainer.save_model(output_dir)  # Saves LoRA adapters
        self.tokenizer.save_pretrained(output_dir)

        # Perform evaluation if eval_data is available
        if eval_data_dicts:
            self._evaluate_model_and_print_metrics(eval_data_dicts, output_dir)
        else:
            print("Skipping evaluation as no evaluation data was prepared.")

        # To save the full model (merged with adapters) if needed:
        # merged_model_path = os.path.join(output_dir, "merged_model")
        # self.model.save_pretrained_merged(merged_model_path, self.tokenizer, save_method = "merged_16bit")
        # print(f"Full merged model saved to {merged_model_path}")

        return output_dir

    def load_finetuned_model(self, model_path: str):
        """Load a fine-tuned model (LoRA adapters)"""
        print(f"Loading fine-tuned LoRA adapters from {model_path}")
        # When loading a LoRA model, first load the base model, then apply adapters.
        # However, Unsloth's from_pretrained can often handle this if the path contains adapter_config.json.
        # For safety, let's assume model_path is where adapters were saved, and base model is self.model_name

        # Re-initialize base model first
        # self.model, self.tokenizer = FastLanguageModel.from_pretrained(
        #     model_name=self.model_name, # Or load from a config if base model changed
        #     max_seq_length=2048,
        #     dtype=None,
        #     load_in_4bit=True,
        # )
        # Then load adapters
        # self.model = FastLanguageModel.get_peft_model(self.model, peft_model_id=model_path)
        # print("Fine-tuned model with LoRA adapters loaded successfully.")

        # Simpler: Unsloth's from_pretrained can load a PEFT model directly if saved correctly
        # This assumes model_path is a directory containing the adapter files AND tokenizer files.
        # If you saved only adapters, you need to load base model first then apply adapters.
        # The SFTTrainer.save_model saves adapters. Tokenizer is saved separately.
        # So, we need to load the base model and then apply the adapters.

        # If `model_path` is the output_dir from finetuning:
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,  # This should point to the directory with adapter_config.json etc.
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        print(f"Fine-tuned model loaded from {model_path} successfully.")

    async def fetch_user_posts(self, user_handle: str, limit: int = 100) -> List[str]:
        """Fetch user posts from Bluesky"""
        if self.client is None:
            raise ValueError("Not logged in to Bluesky. Call login_bluesky first.")

        try:
            user_info = await self.client.get_profile(actor=user_handle)
            user_did = user_info.did

            posts = []
            cursor = None
            while len(posts) < limit:
                response = await self.client.get_author_feed(
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
        """Classify a user based on their posts using the fine-tuned model and heuristics."""
        if not posts:
            return {
                "primary_classification": "Unknown",
                "secondary_classification": "Unknown",
                "weeb_score": 0,
                "furry_score": 0,
                "normie_score": 1,
            }

        if self.model is None or self.tokenizer is None:
            raise ValueError(
                "Model not loaded. Call setup_model or load_finetuned_model first."
            )

        # Ensure model is on the correct device
        self.model.to(self.device)

        # Heuristic scores from term databases (overall tendency)
        combined_text = " ".join(posts).lower()
        heuristic_weeb_score = self._calculate_category_score(
            combined_text, self.weeb_terms
        )
        heuristic_furry_score = self._calculate_category_score(
            combined_text, self.furry_terms
        )

        # Heuristic overall classification (used as fallback or reference)
        h_primary, h_secondary = self._determine_classification_labels(
            heuristic_weeb_score, heuristic_furry_score
        )

        model_predicted_combined_labels = []
        posts_for_model = posts[: min(10, len(posts))]  # Use a subset

        for post_text in tqdm(
            posts_for_model,
            desc="Classifying posts with model",
            leave=False,
            mininterval=10.0,
        ):
            if not post_text.strip():
                continue

            prompt = f"Analyze this Bluesky post for weeb and furry traits: {post_text}"
            formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>"

            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.tokenizer.model_max_length - 70,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=70,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=False,
                    temperature=0.1,
                )

            input_ids_length = inputs.input_ids.shape[1]
            generated_tokens = outputs[0][input_ids_length:]
            model_response_part = self.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            ).strip()

            predicted_combined = self._parse_combined_label_from_text(
                model_response_part, source_type="model output for user classification"
            )
            if (
                "Unknown" not in predicted_combined
            ):  # Only consider valid model predictions
                model_predicted_combined_labels.append(predicted_combined)

        final_primary_classification = h_primary
        final_secondary_classification = h_secondary

        if model_predicted_combined_labels:
            # Aggregate model predictions (most common combined label)
            most_common_combined = max(
                set(model_predicted_combined_labels),
                key=model_predicted_combined_labels.count,
            )
            # Parse the aggregated primary and secondary from the combined string
            if "-" in most_common_combined:
                final_primary_classification, final_secondary_classification = (
                    most_common_combined.split("-", 1)
                )
            else:
                # Should not happen if _parse_combined_label_from_text works correctly and returns "X-Y"
                final_primary_classification = most_common_combined
                final_secondary_classification = "None"  # Fallback if parsing fails

        normie_score_val = max(
            0, 1.0 - max(heuristic_weeb_score, heuristic_furry_score)
        )

        return {
            "primary_classification": final_primary_classification,
            "secondary_classification": final_secondary_classification,
            "weeb_score": round(heuristic_weeb_score, 3),
            "furry_score": round(heuristic_furry_score, 3),
            "normie_score": round(normie_score_val, 3),
            "model_combined_labels_debug": model_predicted_combined_labels,
        }


class BlueskyUserDataset:
    @staticmethod
    def preprocess_and_save(input_csv: str, output_csv: str):
        if not os.path.exists(input_csv):
            raise FileNotFoundError(f"Input file {input_csv} not found")
        print(f"Loading data from {input_csv}...")
        df = pd.read_csv(input_csv)
        df = df[df["type"] == "post"]
        df = df.dropna(subset=["text"])
        df["text"] = df["text"].astype(str)  # Ensure text is string
        df["text"] = df["text"].apply(lambda x: "" if x == "0------------0" else x)
        df = df[df["text"].str.strip() != ""]  # Remove posts with only whitespace
        print(f"Saving processed data to {output_csv}...")
        df.to_csv(output_csv, index=False)
        print(f"Saved {len(df)} processed posts")


def main():
    parser = argparse.ArgumentParser(
        description="Bluesky User Classifier with Dual Labels & Metrics"
    )
    subparsers = parser.add_subparsers(
        dest="command", help="Command to run", required=True
    )

    preprocess_parser = subparsers.add_parser(
        "preprocess", help="Preprocess Bluesky data"
    )
    preprocess_parser.add_argument("--input", required=True, help="Input CSV file")
    preprocess_parser.add_argument("--output", required=True, help="Output CSV file")

    finetune_parser = subparsers.add_parser(
        "finetune", help="Fine-tune the model and evaluate"
    )
    finetune_parser.add_argument(
        "--data", required=True, help="Processed Bluesky data CSV"
    )
    finetune_parser.add_argument(
        "--output_dir", default="finetuned_model", help="Output directory"
    )
    finetune_parser.add_argument(
        "--epochs", type=int, default=3, help="Training epochs"
    )
    finetune_parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size (default: 4)"
    )
    finetune_parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    finetune_parser.add_argument(
        "--eval_split",
        type=float,
        default=0.2,
        help="Proportion of data for evaluation (0 to 1)",
    )

    classify_parser = subparsers.add_parser("classify", help="Classify a Bluesky user")
    classify_parser.add_argument(
        "--model", required=True, help="Path to fine-tuned model directory"
    )
    classify_parser.add_argument(
        "--username", required=True, help="Bluesky username to classify"
    )
    classify_parser.add_argument(
        "--bluesky_user", required=True, help="Your Bluesky login username"
    )
    classify_parser.add_argument(
        "--bluesky_pass", required=True, help="Your Bluesky login password"
    )

    args = parser.parse_args()

    if args.command == "preprocess":
        BlueskyUserDataset.preprocess_and_save(args.input, args.output)

    elif args.command == "finetune":
        # Pass batch_size from args to the classifier instance
        classifier = BlueskyClassifier(batch_size=args.batch_size)
        classifier.finetune_model(
            bluesky_data_csv=args.data,
            output_dir=args.output_dir,
            epochs=args.epochs,
            learning_rate=args.learning_rate,  # Pass learning_rate from args
            eval_split_size=args.eval_split,
        )

    elif args.command == "classify":
        import asyncio  # Import asyncio here as it's only needed for this command

        async def run_classification():
            classifier = BlueskyClassifier()
            classifier.load_finetuned_model(args.model)  # This loads the adapters

            success = await classifier.login_bluesky(
                args.bluesky_user, args.bluesky_pass
            )
            if not success:
                print("Failed to login to Bluesky")
                return

            posts = await classifier.fetch_user_posts(args.username)
            if not posts:
                print(
                    f"No posts found or error fetching posts for user {args.username}"
                )
                return

            result = classifier.classify_user(posts)

            print("\n" + "=" * 50)
            print(f"Classification Results for @{args.username}")
            print("=" * 50)
            print(f"Primary Classification: {result['primary_classification']}")
            print(f"Secondary Classification: {result['secondary_classification']}")
            print(f"  Heuristic Weeb Score: {result['weeb_score']:.3f}")
            print(f"  Heuristic Furry Score: {result['furry_score']:.3f}")
            print(f"  Heuristic Normie Score: {result['normie_score']:.3f}")
            if "model_combined_labels_debug" in result:
                print(
                    f"  Model Post Classifications (sample): {result['model_combined_labels_debug']}"
                )
            print("=" * 50 + "\n")

        asyncio.run(run_classification())
    else:
        parser.print_help()


if __name__ == "__main__":
    # Ensure your CSV files (weeb_terms.csv, furry_terms.csv) are in the same directory
    # or provide correct paths in _load_term_database.
    # Example usage:
    # 1. Preprocess: python your_script_name.py preprocess --input raw_bluesky_data.csv --output processed_data.csv
    # 2. Fine-tune: python your_script_name.py finetune --data processed_data.csv --output_dir my_finetuned_classifier --epochs 1 --batch_size 2 --learning_rate 1e-5 --eval_split 0.1
    # 3. Classify: python your_script_name.py classify --model my_finetuned_classifier --username targetuser.bsky.social --bluesky_user yourlogin.bsky.social --bluesky_pass yourpassword
    main()
