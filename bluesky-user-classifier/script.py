#!/usr/bin/env python3
import os
import sys
import random
import argparse
import pandas as pd
import numpy as np
import math
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import torch
from datasets import Dataset
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import seaborn as sns

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

    if not text or (
        isinstance(text, str) and not text.strip()
    ):  # Skip empty or whitespace-only posts
        return None

    # Ensure text is a string before lowercasing
    if not isinstance(text, str):
        try:
            text = str(text)
        except (TypeError, ValueError):
            # If conversion to string fails, skip this entry
            return None
        if not text.strip():  # Check again after conversion
            return None

    # Helper functions that were previously methods of BlueskyClassifier
    def calculate_category_score(text_content: str, terms_df: pd.DataFrame) -> float:
        if terms_df.empty or not isinstance(
            text_content, str
        ):  # Added type check for text_content
            return 0.0
        text_content = text_content.lower()  # Ensure text_content is string
        # Ensure 'term' column exists and contains strings
        if (
            "term" not in terms_df.columns
            or not terms_df["term"].apply(isinstance, args=(str,)).all()
        ):
            # print("Warning: 'term' column in terms_df is missing or contains non-string values.")
            return 0.0

        matched_terms = terms_df[
            terms_df["term"]
            .str.lower()
            .apply(lambda x: x in text_content if isinstance(x, str) else False)
        ]
        score = matched_terms["combined_score"].sum()
        max_potential_score = terms_df["combined_score"].sum()
        return score / max_potential_score if max_potential_score > 0 else 0.0

    def determine_classification_labels(
        w_score: float, f_score: float
    ) -> Tuple[str, str]:
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

    # Ensure text is a string for the prompt
    prompt_text_content = str(text) if not isinstance(text, str) else text
    prompt = (
        f"Analyze this Bluesky post for weeb and furry traits: {prompt_text_content}"
    )

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
    def __init__(self, model_name="unsloth/Qwen3-0.6B-unsloth-bnb-4bit", batch_size=8):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None
        self.tokenizer = None
        self.chunk_size = 100000  # Define a chunk size for reading CSV

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
            print(f"Error: Required term database {csv_file} not found.")
            sys.exit(1)  # Exit with error code 1

        df = pd.read_csv(csv_file)
        # Ensure 'term' and 'combined_score' columns exist
        if "term" not in df.columns or "combined_score" not in df.columns:
            print(
                f"Warning: {csv_file} is missing 'term' or 'combined_score' column. Scores for this category might be incorrect."
            )
            # Return what was loaded, or an empty df if critical columns are missing
            if "term" not in df.columns:
                df["term"] = pd.Series(dtype="str")
            if "combined_score" not in df.columns:
                df["combined_score"] = pd.Series(dtype="float")

        # Convert 'term' to string type to avoid issues with non-string data
        df["term"] = df["term"].astype(str)
        df["combined_score"] = pd.to_numeric(
            df["combined_score"], errors="coerce"
        ).fillna(0)

        df = df[["term", "combined_score"]].sort_values(
            by="combined_score", ascending=False
        )
        return df

    def setup_model(self):
        """Initialize the model and tokenizer"""
        print(f"Loading model: {self.model_name}")
        try:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=2048,
                dtype=None,  # None will use model's default dtype
                load_in_4bit=True,
            )

            # Set padding_side to 'left' for decoder-only models
            self.tokenizer.padding_side = "left"

            # Important for Qwen3: Ensure pad token is set if not already (Unsloth usually handles this)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                if (
                    self.model.config.pad_token_id is None
                ):  # Check if model config also needs update
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
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            print(
                "Please ensure the model name is correct, you have internet access, and Unsloth is installed correctly."
            )
            print(
                "If this is a GPU-related error, ensure CUDA is set up correctly and a compatible GPU is available."
            )
            # Exit or raise, as the classifier cannot function without a model
            raise RuntimeError(f"Failed to load model: {e}")

    def login_bluesky(self, username: str, password: str):
        """Log in to Bluesky"""
        try:
            self.client = Client()
            # Using async login, so this method should be async or run in an event loop
            # For simplicity in a synchronous class, we'll assume a synchronous login if available
            # or handle the async nature if the main script part calls it appropriately.
            # The original script uses asyncio.run() for the classify command.
            # If this login is called outside that, it might need adjustment.
            # For now, assuming it's called within an async context or the library handles it.
            self.client.login(
                username, password
            )  # This might need to be `await self.client.login(...)` if called from async
            print(f"Logged in to Bluesky as {username}")
            return True
        except Exception as e:
            print(f"Failed to log in to Bluesky: {e}")
            return False

    def _prepare_training_data(self, bluesky_data_csv: str) -> List[Dict[str, str]]:
        """Prepare training data from Bluesky posts CSV and term databases using multiprocessing"""
        print(f"Loading Bluesky data from {bluesky_data_csv} in chunks...")

        if not os.path.exists(bluesky_data_csv):
            raise FileNotFoundError(f"Bluesky data file {bluesky_data_csv} not found")

        all_training_data = []
        total_processed_posts = 0

        # Reduce number of processes and chunk size
        num_processes = min(4, max(1, cpu_count() - 1))  # Max 4 processes
        self.chunk_size = 50000  # Smaller chunks

        try:
            # Use tqdm to wrap the CSV reader (total unknown)
            for chunk_df in tqdm(
                pd.read_csv(
                    bluesky_data_csv,
                    chunksize=self.chunk_size,
                    usecols=["type", "text"],  # Only load needed columns
                ),
                desc="Reading CSV in chunks",
                mininterval=10.0,
            ):
                # Filter in place to reduce memory
                chunk_df = chunk_df[chunk_df["type"] == "post"]
                chunk_df = chunk_df.dropna(subset=["text"])
                chunk_df["text"] = chunk_df["text"].astype(str, errors="ignore")

                # Process in smaller batches with a progress bar
                batch_size = 5000
                for start_idx in tqdm(
                    range(0, len(chunk_df), batch_size),
                    desc="Processing batches",
                    leave=False,
                ):
                    batch_df = chunk_df.iloc[start_idx : start_idx + batch_size]

                    process_args = [
                        (text, self.weeb_terms, self.furry_terms)
                        for text in batch_df["text"]
                        if isinstance(text, str) and text.strip()
                    ]

                    if not process_args:
                        continue

                    # Create new pool for each batch
                    with Pool(processes=num_processes) as pool:
                        batch_results = list(
                            pool.imap(
                                process_post_for_training,
                                process_args,
                                chunksize=500,  # Process in smaller chunks
                            )
                        )

                    valid_results = [r for r in batch_results if r is not None]
                    all_training_data.extend(valid_results)
                    total_processed_posts += len(valid_results)

                    # Clear memory
                    del batch_results, valid_results

                # Clear memory after each chunk
                del chunk_df

        except Exception as e:
            print(f"Error processing {bluesky_data_csv}: {e}")
            return all_training_data

        print(f"Created {len(all_training_data)} training examples")
        return all_training_data

    def _parse_combined_label_from_text(
        self, text: str, source_type: str = "unknown source"
    ) -> str:
        """
        Parses Primary and Secondary classification labels from a given text.
        Returns a combined string "Primary-Secondary" or "Unknown-Unknown" if parsing fails.
        """
        primary_label = "Unknown"
        secondary_label = (
            "Unknown"  # Default to Unknown, which can be "None" if explicitly stated
        )

        try:
            lines = text.strip().split("\n")
            for line in lines:
                if "Primary Classification:" in line:
                    parsed_primary = line.split("Primary Classification:", 1)[1].strip()
                    if parsed_primary in self.primary_labels_set:
                        primary_label = parsed_primary
                    # else:
                    # print(f"Warning: Parsed primary label '{parsed_primary}' from {source_type} not in defined set: {self.primary_labels_set}. Text: '{text}'")

                elif "Secondary Classification:" in line:
                    parsed_secondary = line.split("Secondary Classification:", 1)[
                        1
                    ].strip()
                    if parsed_secondary in self.secondary_labels_set:
                        secondary_label = parsed_secondary
                    # else:
                    # print(f"Warning: Parsed secondary label '{parsed_secondary}' from {source_type} not in defined set: {self.secondary_labels_set}. Text: '{text}'")

            # If primary is still Unknown but secondary was found and isn't "None", it's an anomaly.
            # For now, we trust the parsed labels. If primary is "Unknown", it implies the model failed to classify.
            # If secondary is "Unknown" but primary is not, it means secondary wasn't found or wasn't "None".
            # We should ensure "None" is a valid secondary label.
            if primary_label != "Unknown" and secondary_label == "Unknown":
                # This case implies secondary was not found in the text, so it defaults to "None"
                # if the primary was successfully classified.
                # However, our default for secondary_label is "Unknown". If the text explicitly said "Secondary Classification: None",
                # it would be parsed as "None". If the line is missing, it remains "Unknown".
                # Let's refine: if primary is known, and secondary is still "Unknown" (meaning not explicitly parsed),
                # it's safer to assume "None" for secondary *if* primary is not "Normie".
                # Or, more simply, if the line for secondary classification is missing, we might infer "None".
                # For now, if "Secondary Classification:" line is absent, secondary_label remains "Unknown".
                # This is handled by the initial defaults.
                pass

        except Exception:
            # print(f"Error parsing labels from {source_type}: {e}. Text: '{text}'")
            return "Unknown-Unknown"  # Fallback on any parsing error

        # If primary is "Normie", secondary should always be "None" by definition of our heuristic.
        if primary_label == "Normie":
            secondary_label = "None"

        # If primary is "Unknown", it implies a failure, so secondary is also "Unknown".
        if primary_label == "Unknown":
            secondary_label = "Unknown"

        return f"{primary_label}-{secondary_label}"

    def _evaluate_model_and_print_metrics(
        self, eval_data: List[Dict[str, str]], output_dir: str
    ):
        """Evaluates the model on the evaluation data and prints metrics for combined labels."""
        if not self.model or not self.tokenizer:
            print("Model or tokenizer not available for evaluation. Skipping.")
            return

        MAX_EVAL_SAMPLES = 10000
        if len(eval_data) > MAX_EVAL_SAMPLES:
            eval_data = random.sample(eval_data, MAX_EVAL_SAMPLES)
            print(f"Evaluating on a random subset of {MAX_EVAL_SAMPLES} samples.")

        y_true_combined = []
        y_pred_combined = []

        # Ensure model and tokenizer are on the correct device
        self.model.to(self.device)

        batch_size = 16  # You can increase this if you have more VRAM
        num_batches = math.ceil(len(eval_data) / batch_size)

        for batch_idx in tqdm(
            range(num_batches), desc="Evaluating model (batched)", mininterval=10.0
        ):
            batch = eval_data[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            prompts = [f"<|user|>\n{item['prompt']}\n<|assistant|>" for item in batch]
            true_labels = [item["true_label_heuristic"] for item in batch]

            # Tokenize as a batch
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                truncation=True,
                max_length=getattr(self.tokenizer, "model_max_length", 2048) - 100,
                padding=True,
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

            for i in range(len(batch)):
                input_len = inputs.input_ids[i].shape[0]
                generated_tokens = outputs[i][input_len:]
                model_response_part = self.tokenizer.decode(
                    generated_tokens, skip_special_tokens=True
                ).strip()
                predicted_label = self._parse_combined_label_from_text(
                    model_response_part, source_type="evaluation output"
                )
                y_true_combined.append(true_labels[i])
                y_pred_combined.append(predicted_label)

        print("\n--- Classification Metrics (Combined Labels) ---")

        current_labels_in_data = sorted(list(set(y_true_combined + y_pred_combined)))

        # Use all defined labels for the report, plus any "Unknown-Unknown" if it appeared
        report_labels = sorted(
            list(set(self.defined_combined_labels + current_labels_in_data))
        )

        # Prepare metrics directory as a sibling of output_dir
        metrics_root = os.path.join(os.path.dirname(output_dir), "metrics")
        metrics_dir = os.path.join(metrics_root, os.path.basename(output_dir))
        if not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir)
            print(f"Created metrics directory: {metrics_dir}")

        # Save classification report to file
        report_str = classification_report(
            y_true_combined, y_pred_combined, labels=report_labels, zero_division=0
        )
        print(report_str)
        report_path = os.path.join(metrics_dir, "classification_report.txt")
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_str)
            print(f"Classification report saved to {report_path}")
        except Exception as e:
            print(f"Error saving classification report: {e}")

        print("\n--- Confusion Matrix (Combined Labels) ---")
        cm = confusion_matrix(y_true_combined, y_pred_combined, labels=report_labels)

        plt.figure(
            figsize=(
                max(12, len(report_labels) * 0.8),
                max(10, len(report_labels) * 0.6),
            )
        )
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=report_labels,
            yticklabels=report_labels,
            annot_kws={"size": 8},  # Adjust annotation font size
        )
        plt.xlabel("Predicted Label (Primary-Secondary)")
        plt.ylabel("True Label (Primary-Secondary)")
        plt.title("Confusion Matrix (Combined Labels)")
        plt.xticks(rotation=45, ha="right", fontsize=8)  # Adjust tick label font size
        plt.yticks(rotation=0, fontsize=8)  # Adjust tick label font size
        plt.tight_layout()

        cm_path = os.path.join(metrics_dir, "confusion_matrix_combined.png")
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
            try:
                self.setup_model()
            except RuntimeError as e:  # Catch error from setup_model
                print(f"Failed to setup model during fine-tuning: {e}")
                return None  # Cannot proceed with fine-tuning

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
                        "Warning: Stratification not possible. Splitting without stratification."
                    )
                if not enough_total_samples_for_split:
                    print(
                        "Warning: Not enough total samples for a split. Using all for training or adjusting."
                    )
                if (
                    enough_total_samples_for_split
                ):  # Still try non-stratified if enough samples
                    train_data_dicts, eval_data_dicts = train_test_split(
                        all_prepared_data,
                        test_size=eval_split_size,
                        random_state=RANDOM_SEED,
                    )
                else:  # Use all for training if not enough for split
                    print(
                        "Warning: Not enough samples for train/eval split. Using all data for training."
                    )
                    train_data_dicts = all_prepared_data
                    eval_data_dicts = []
        else:
            print(
                "Eval split size is 0 or not enough data. Using all data for training."
            )
            train_data_dicts = all_prepared_data
            eval_data_dicts = []

        print(
            f"Training with {len(train_data_dicts)} samples, evaluating with {len(eval_data_dicts)} samples."
        )
        if not train_data_dicts:
            print(
                "No training samples available after splitting. Aborting fine-tuning."
            )
            return None

        # The SFTTrainer expects data with 'prompt' and 'response' or a single text field.
        # Our train_data_dicts items have 'prompt', 'response', and 'true_label_heuristic'.
        # We need to ensure the formatting_func only uses 'prompt' and 'response'.
        sft_train_dataset = Dataset.from_list(
            [
                {"prompt": d["prompt"], "response": d["response"]}
                for d in train_data_dicts
            ]
        )

        # Create dataset formatter function
        # Capture tokenizer's EOS token for use in the formatting function
        eos_token = self.tokenizer.eos_token

        def formatting_func(example):
            # Standard ChatML format that Unsloth often uses for Qwen3 fine-tuning
            return [
                f"<|user|>\n{prompt}\n<|assistant|>\n{response}{eos_token}"
                for prompt, response in zip(example["prompt"], example["response"])
            ]

        sft_config = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=4,  # Adjust if OOM
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            logging_steps=max(1, len(sft_train_dataset) // (self.batch_size * 10))
            if sft_train_dataset
            else 10,  # Log ~10 times per epoch
            save_strategy="epoch",
            dataset_num_proc=1,  # Number of processes for dataset processing
            save_total_limit=2,
            remove_unused_columns=True,
            report_to="none",  # "tensorboard" or "wandb" if desired
            max_seq_length=384,  # Max length of formatted prompt + response
            packing=True,
            optim="adamw_torch",
            fp16=torch.cuda.is_available()
            and not hasattr(torch.cuda, "is_bf16_supported")
            and not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_available()
            and hasattr(torch.cuda, "is_bf16_supported")
            and torch.cuda.is_bf16_supported(),
            # Added for potential memory saving during training, if supported and useful:
            # gradient_checkpointing=True, # Can save memory but slows down training
        )

        trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            args=sft_config,
            train_dataset=sft_train_dataset,  # Use the list of dicts
            formatting_func=formatting_func,
        )

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("Starting model fine-tuning...")
        try:
            trainer.train()
            print(f"Saving fine-tuned model to {output_dir}")
            trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)
        except Exception as e:
            print(f"Error during training or saving model: {e}")
            if "CUDA out of memory" in str(e):
                print(
                    "CUDA out of memory during training. Try reducing batch_size, max_seq_length, or using gradient_accumulation_steps > 1."
                )
            return None  # Indicate failure

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
        # Set padding_side to 'left' for decoder-only models
        self.tokenizer.padding_side = "left"
        print(f"Fine-tuned model loaded from {model_path} successfully.")

    async def fetch_user_posts(self, user_handle: str, limit: int = 100) -> List[str]:
        """Fetch user posts from Bluesky"""
        if self.client is None:
            # Try to get credentials from environment variables if not logged in
            # This is a placeholder for a more robust credential management strategy
            bsky_user_env = os.getenv("BLUESKY_USERNAME")
            bsky_pass_env = os.getenv("BLUESKY_PASSWORD")
            if bsky_user_env and bsky_pass_env:
                print("Attempting to login to Bluesky using environment variables...")
                # Note: login_bluesky might need to be async if called from here
                # For simplicity, assuming it can be called, or this part is refactored.
                # This part is tricky because login_bluesky is synchronous but called from async.
                # Ideally, login should also be async or handled carefully.
                # Let's assume login_bluesky handles its own async if needed or is called from sync context before this.
                # For now, this is a conceptual addition.
                # await self.login_bluesky(bsky_user_env, bsky_pass_env) # This would require login_bluesky to be async
                print(
                    "Warning: Auto-login attempt from fetch_user_posts might not work as expected due to sync/async."
                )

            if self.client is None:  # Check again after potential auto-login attempt
                raise ValueError(
                    "Not logged in to Bluesky. Call login_bluesky first or set BLUESKY_USERNAME/BLUESKY_PASSWORD env vars."
                )

        try:
            user_info = await self.client.get_profile(actor=user_handle)
            user_did = user_info.did

            posts_texts = []  # Renamed to avoid conflict with 'posts' variable name if used elsewhere
            cursor = None
            fetched_count = 0

            # Limit the number of requests to avoid hitting rate limits too hard
            max_requests = (
                limit + 99
            ) // 100  # Calculate number of requests needed (100 posts per request)

            for _ in range(max_requests):
                if fetched_count >= limit:
                    break

                # Determine how many to fetch in this request
                fetch_limit_this_request = min(100, limit - fetched_count)
                if (
                    fetch_limit_this_request <= 0
                ):  # Should not happen if loop condition is correct
                    break

                response = await self.client.get_author_feed(
                    actor=user_did, limit=fetch_limit_this_request, cursor=cursor
                )
                if (
                    not response or not response.feed
                ):  # Check if response or response.feed is None
                    break

                for feed_item in response.feed:
                    if (
                        hasattr(feed_item, "post")
                        and feed_item.post  # Check if post exists
                        and hasattr(feed_item.post, "record")
                        and feed_item.post.record  # Check if record exists
                        and hasattr(feed_item.post.record, "text")
                    ):  # Check if text exists
                        posts_texts.append(feed_item.post.record.text)
                        fetched_count += 1
                        if fetched_count >= limit:
                            break

                cursor = response.cursor
                if cursor is None:
                    break
            print(f"Fetched {len(posts_texts)} posts for user {user_handle}")
            return posts_texts
        except Exception as e:
            print(f"Error fetching posts for {user_handle}: {e}")
            return []

    def _calculate_category_score(
        self, text_content: str, terms_df: pd.DataFrame
    ) -> float:
        """Helper to calculate score for a category based on text and terms. (Copied from process_post_for_training for internal use)"""
        if terms_df.empty or not isinstance(text_content, str):
            return 0.0
        text_content = text_content.lower()
        if (
            "term" not in terms_df.columns
            or not terms_df["term"].apply(isinstance, args=(str,)).all()
        ):
            return 0.0
        matched_terms = terms_df[
            terms_df["term"]
            .str.lower()
            .apply(lambda x: x in text_content if isinstance(x, str) else False)
        ]
        score = matched_terms["combined_score"].sum()
        max_potential_score = terms_df["combined_score"].sum()
        return score / max_potential_score if max_potential_score > 0 else 0.0

    def _determine_classification_labels(
        self, w_score: float, f_score: float
    ) -> Tuple[str, str]:
        """Helper to determine primary/secondary labels. (Copied from process_post_for_training for internal use)"""
        primary_label = "Normie"
        secondary_label = "None"
        if max(w_score, f_score) <= 0.4:
            return "Normie", "None"
        is_weeb_dominant = w_score >= f_score
        dominant_score, secondary_s = (
            (w_score, f_score) if is_weeb_dominant else (f_score, w_score)
        )
        dominant_type_strong, dominant_type_slight = (
            ("Weeb", "Slight Weeb") if is_weeb_dominant else ("Furry", "Slight Furry")
        )
        secondary_type_strong, secondary_type_slight = (
            ("Furry", "Slight Furry") if is_weeb_dominant else ("Weeb", "Slight Weeb")
        )

        if dominant_score > 0.7:
            primary_label = dominant_type_strong
            secondary_label = (
                secondary_type_strong
                if secondary_s > 0.7
                else (secondary_type_slight if secondary_s > 0.4 else "None")
            )
        elif dominant_score > 0.4:
            primary_label = dominant_type_slight
            secondary_label = (
                secondary_type_slight
                if secondary_s > 0.4 and secondary_s <= 0.7
                else "None"
            )
        else:  # Should be covered by first condition
            primary_label = "Normie"
            secondary_label = "None"
        return primary_label, secondary_label

    def classify_user(self, posts: List[str]) -> Dict[str, Any]:
        """Classify a user based on their posts using the fine-tuned model and heuristics."""
        if not posts:
            return {
                "primary_classification": "Unknown",
                "secondary_classification": "Unknown",
                "weeb_score": 0.0,  # Ensure float
                "furry_score": 0.0,  # Ensure float
                "normie_score": 1.0,  # Ensure float
                "model_combined_labels_debug": [],
            }

        if self.model is None or self.tokenizer is None:
            # Try to load a default model if not already loaded.
            # This is a fallback, ideally model should be explicitly loaded or fine-tuned.
            print(
                "Warning: Model not explicitly loaded for classify_user. Attempting to load default or fine-tuned if path known."
            )
            # This requires a path. If `self.model_name` points to a fine-tuned model dir, this might work.
            # Or, you might want to have a default_model_path attribute.
            try:
                self.load_finetuned_model(
                    self.model_name
                )  # Assuming model_name could be a path to fine-tuned
            except Exception as e:
                raise ValueError(
                    f"Model not loaded and failed to auto-load. Call setup_model or load_finetuned_model first. Error: {e}"
                )

        # Ensure model is on the correct device
        self.model.to(self.device)

        # Heuristic scores from term databases (overall tendency)
        # Ensure posts are strings and join them safely
        safe_posts = [str(p) if not isinstance(p, str) else p for p in posts]
        combined_text = " ".join(safe_posts).lower()

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
        # Use a smaller, manageable subset of posts for model classification to save time/resources
        # e.g., the 10 most recent or a random sample of 10, if posts list is long.
        # For now, using the first few as in original logic.
        posts_for_model = [p for p in posts if isinstance(p, str) and p.strip()][
            : min(10, len(posts))
        ]

        for post_text in tqdm(
            posts_for_model,
            desc="Classifying posts with model",
            leave=False,
            mininterval=10.0,
        ):
            # No need to check post_text.strip() again, already filtered in posts_for_model
            prompt = f"Analyze this Bluesky post for weeb and furry traits: {post_text}"
            formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>"

            max_input_length = getattr(self.tokenizer, "model_max_length", 2048) - 100

            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_length,  # Use calculated max_input_length
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
                "Unknown"
                not in predicted_combined  # Only consider valid model predictions
            ):
                model_predicted_combined_labels.append(predicted_combined)

        final_primary_classification = h_primary
        final_secondary_classification = h_secondary

        if model_predicted_combined_labels:
            # Aggregate model predictions (most common combined label)
            if model_predicted_combined_labels:  # Ensure list is not empty
                most_common_combined = max(
                    set(model_predicted_combined_labels),
                    key=model_predicted_combined_labels.count,
                )
                # Parse the aggregated primary and secondary from the combined string
                if "-" in most_common_combined:
                    parsed_labels = most_common_combined.split("-", 1)
                    if len(parsed_labels) == 2:
                        final_primary_classification, final_secondary_classification = (
                            parsed_labels
                        )
                    else:  # Should not happen with "X-Y" format
                        final_primary_classification = parsed_labels[0]
                        final_secondary_classification = "None"  # Fallback
                else:
                    final_primary_classification = most_common_combined
                    final_secondary_classification = "None"
            # else: primary/secondary remain heuristic-based if model had no valid predictions

        normie_score_val = max(
            0.0,
            1.0 - max(heuristic_weeb_score, heuristic_furry_score),  # Ensure float
        )

        return {
            "primary_classification": final_primary_classification,
            "secondary_classification": final_secondary_classification,
            "weeb_score": round(float(heuristic_weeb_score), 3),  # Ensure float
            "furry_score": round(float(heuristic_furry_score), 3),  # Ensure float
            "normie_score": round(normie_score_val, 3),
            "model_combined_labels_debug": model_predicted_combined_labels,
        }


class BlueskyUserDataset:
    @staticmethod
    def preprocess_and_save(input_csv: str, output_csv: str):
        if not os.path.exists(input_csv):
            raise FileNotFoundError(f"Input file {input_csv} not found")
        print(f"Loading data from {input_csv}...")
        try:
            df = pd.read_csv(input_csv)
        except pd.errors.EmptyDataError:
            print(f"Error: Input file {input_csv} is empty. Cannot preprocess.")
            return
        except Exception as e:
            print(f"Error reading {input_csv}: {e}")
            return

        df = df[df["type"] == "post"]
        df = df.dropna(subset=["text"])
        df["text"] = df["text"].astype(
            str, errors="ignore"
        )  # Convert to string, ignore errors for now
        # Remove known placeholder or clearly invalid text before stripping
        df["text"] = df["text"].apply(lambda x: "" if x == "0------------0" else x)
        # Filter out rows where text is not a string or is empty/whitespace after processing
        df = df[df["text"].apply(lambda x: isinstance(x, str) and x.strip() != "")]

        if df.empty:
            print(
                "No valid posts found after preprocessing. Output file will be empty or not created."
            )
            # Optionally, create an empty CSV with headers or just skip saving
            # pd.DataFrame(columns=df.columns).to_csv(output_csv, index=False)
            return

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
        "--batch_size", type=int, default=8, help="Batch size (default: 8)"
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
    finetune_parser.add_argument(
        "--model_name",
        default="unsloth/Qwen3-0.6B-unsloth-bnb-4bit",
        help="Base model name for fine-tuning",
    )
    finetune_parser.add_argument(
        "--chunk_size",
        type=int,
        default=100000,
        help="Chunk size for reading CSV data during training prep.",
    )

    classify_parser = subparsers.add_parser("classify", help="Classify a Bluesky user")
    classify_parser.add_argument(
        "--model",
        required=True,
        help="Path to fine-tuned model directory (or base model if not fine-tuned)",
    )
    classify_parser.add_argument(
        "--username", required=True, help="Bluesky username to classify"
    )
    # Optional login credentials, can also use environment variables
    classify_parser.add_argument(
        "--bluesky_user",
        default=os.getenv("BLUESKY_USERNAME"),
        help="Your Bluesky login username (or set BLUESKY_USERNAME env var)",
    )
    classify_parser.add_argument(
        "--bluesky_pass",
        default=os.getenv("BLUESKY_PASSWORD"),
        help="Your Bluesky login password (or set BLUESKY_PASSWORD env var)",
    )

    args = parser.parse_args()

    if args.command == "preprocess":
        BlueskyUserDataset.preprocess_and_save(args.input, args.output)

    elif args.command == "finetune":
        classifier = BlueskyClassifier(
            model_name=args.model_name, batch_size=args.batch_size
        )
        classifier.chunk_size = args.chunk_size  # Set chunk_size from args
        classifier.finetune_model(
            bluesky_data_csv=args.data,
            output_dir=args.output_dir,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            eval_split_size=args.eval_split,
        )

    elif args.command == "classify":
        import asyncio

        async def run_classification():
            # Use the --model argument as the model_name for the classifier initialization
            # This path could be a base model or a fine-tuned one.
            # load_finetuned_model will be called internally if it's a path to adapters.
            classifier = BlueskyClassifier(model_name=args.model)

            # Attempt to load the model. If model_name is a path to adapters,
            # from_pretrained in setup_model (or load_finetuned_model) should handle it.
            try:
                # If args.model is specifically a fine-tuned adapter path,
                # it's better to load it explicitly after base model setup.
                # However, Unsloth's from_pretrained is quite flexible.
                # Let's assume `model_name` in constructor handles it, or `load_finetuned_model` is called.
                # The current `classify_user` tries to load if model is None.
                # For clarity, we can call setup_model or load_finetuned_model here.
                if os.path.exists(os.path.join(args.model, "adapter_config.json")):
                    classifier.load_finetuned_model(args.model)
                else:  # Assume it's a base model name
                    classifier.setup_model()

            except RuntimeError as e:
                print(f"Failed to initialize or load model: {e}")
                return

            if not args.bluesky_user or not args.bluesky_pass:
                print(
                    "Bluesky username or password not provided. Please set BLUESKY_USERNAME and BLUESKY_PASSWORD environment variables or use --bluesky_user and --bluesky_pass arguments."
                )
                return

            # The login method in the class is currently synchronous.
            # To call it from an async function, it ideally should be async or run in an executor.
            # For now, let's assume the atproto client library might handle some level of sync/async internally for login,
            # or this needs refactoring if strict async is required for login.
            # The original script's login was synchronous.
            # For `await self.client.get_profile` etc., the client object itself must support async operations.

            # Let's make the login call synchronous for now, as the method is defined sync
            # This might block the event loop if it's a long operation.
            # Proper async handling would involve making login_bluesky async.
            # For now, we'll call it as is.

            # Create the client instance for login
            classifier.client = Client()
            try:
                # Perform login (assuming client.login can be called this way)
                # If client.login is async, this needs `await` and login_bluesky to be async
                await classifier.client.login(args.bluesky_user, args.bluesky_pass)
                print(f"Logged in to Bluesky as {args.bluesky_user}")
                # Assign the successfully logged-in client back to the classifier instance
                # This step is crucial if login_bluesky was not called or failed.
                # classifier.client = client_instance
            except Exception as e:
                print(f"Failed to log in to Bluesky: {e}")
                return

            posts = await classifier.fetch_user_posts(args.username)
            if not posts:
                print(
                    f"No posts found or error fetching posts for user {args.username}"
                )
                # Provide a default "Unknown" classification if no posts
                result = {
                    "primary_classification": "Unknown (No Posts)",
                    "secondary_classification": "None",
                    "weeb_score": 0.0,
                    "furry_score": 0.0,
                    "normie_score": 1.0,
                    "model_combined_labels_debug": [],
                }
            else:
                result = classifier.classify_user(posts)

            print("\n" + "=" * 50)
            print(f"Classification Results for @{args.username}")
            print("=" * 50)
            print(f"Primary Classification: {result['primary_classification']}")
            print(f"Secondary Classification: {result['secondary_classification']}")
            print(f"  Heuristic Weeb Score: {result['weeb_score']:.3f}")
            print(f"  Heuristic Furry Score: {result['furry_score']:.3f}")
            print(f"  Heuristic Normie Score: {result['normie_score']:.3f}")
            if (
                "model_combined_labels_debug" in result
                and result["model_combined_labels_debug"]
            ):
                print(
                    f"  Model Post Classifications (sample): {result['model_combined_labels_debug']}"
                )
            print("=" * 50 + "\n")

        asyncio.run(run_classification())
    else:
        parser.print_help()


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig

    main()
