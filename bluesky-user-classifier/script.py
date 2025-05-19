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
from datasets import Dataset  # type: ignore
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import seaborn as sns

# For Bluesky/ATProto
from atproto import Client  # type: ignore

# For metrics
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.metrics import classification_report, confusion_matrix  # type: ignore

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


def process_post_for_training(args) -> Dict[str, Any] | None:
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

        self.model_name = (
            model_name  # Can be a base model name or path to fine-tuned model
        )
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
                "Unknown-Unknown",  # Added for cases where parsing fails
            ]
        )
        # For parsing individual primary/secondary labels if needed elsewhere
        self.primary_labels_set = {
            "Weeb",
            "Furry",
            "Slight Weeb",
            "Slight Furry",
            "Normie",
            "Unknown",  # Added for robustness
        }
        self.secondary_labels_set = {
            "Weeb",
            "Furry",
            "Slight Weeb",
            "Slight Furry",
            "None",
            "Unknown",  # Added for robustness
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
            print(
                f"Warning: Term database {csv_file} not found. Scores for this category might be affected."
            )
            # Return an empty DataFrame with expected columns if file not found
            return pd.DataFrame(columns=["term", "combined_score"])

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

    def setup_model(self, model_name_or_path: str | None = None):
        """Initialize the base model and tokenizer, or load a full model from path."""
        # Use provided model_name_or_path if given, otherwise use instance's model_name
        load_path = model_name_or_path if model_name_or_path else self.model_name
        print(f"Loading model from: {load_path}")
        try:
            # Unsloth's from_pretrained can load base models or fine-tuned models (if PEFT config is present)
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=load_path,
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

            print(f"Model loaded successfully from {load_path}.")
            # If we loaded a base model using self.model_name and model_name_or_path was different (e.g. a PEFT path)
            # this means we might need to apply adapters separately if from_pretrained didn't pick them up.
            # However, for evaluation, from_pretrained on the PEFT adapter directory should work directly.
            # For fine-tuning, we set up PEFT adapters after this.

        except Exception as e:
            print(f"Error loading model {load_path}: {e}")
            print(
                "Please ensure the model name/path is correct, you have internet access, and Unsloth is installed correctly."
            )
            print(
                "If this is a GPU-related error, ensure CUDA is set up correctly and a compatible GPU is available."
            )
            # Exit or raise, as the classifier cannot function without a model
            raise RuntimeError(f"Failed to load model: {e}")

    def _setup_peft_adapters(self):
        """Sets up PEFT adapters for fine-tuning on the currently loaded model."""
        if not self.model:
            raise RuntimeError("Base model not loaded. Call setup_model() first.")
        try:
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
            print("PEFT adapters configured for fine-tuning.")
        except Exception as e:
            print(f"Error setting up PEFT adapters: {e}")
            raise RuntimeError(f"Failed to set up PEFT adapters: {e}")

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

    def _prepare_data_from_csv(self, data_csv_path: str) -> List[Dict[str, str]]:
        """Prepare data (prompts, responses, heuristic labels) from a CSV file using multiprocessing.
        This is used for preparing training, validation, or evaluation data.
        """
        print(f"Loading and preparing data from {data_csv_path} in chunks...")

        if not os.path.exists(data_csv_path):
            raise FileNotFoundError(f"Data file {data_csv_path} not found")

        all_prepared_data = []
        total_processed_posts = 0

        # Reduce number of processes and chunk size
        num_processes = min(4, max(1, cpu_count() - 1))  # Max 4 processes
        # self.chunk_size is an instance variable, can be set via args

        try:
            # Use tqdm to wrap the CSV reader (total unknown)
            for chunk_df in tqdm(
                pd.read_csv(
                    data_csv_path,
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
                batch_size_processing = 5000  # Internal batching for multiprocessing
                for start_idx in tqdm(
                    range(0, len(chunk_df), batch_size_processing),
                    desc="Processing batches",
                    leave=False,
                ):
                    batch_df = chunk_df.iloc[
                        start_idx : start_idx + batch_size_processing
                    ]

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
                                chunksize=max(
                                    1, len(process_args) // (num_processes * 2)
                                ),  # Dynamic chunksize for imap
                            )
                        )

                    valid_results = [r for r in batch_results if r is not None]
                    all_prepared_data.extend(valid_results)
                    total_processed_posts += len(valid_results)

                    # Clear memory
                    del batch_results, valid_results

                # Clear memory after each chunk
                del chunk_df

        except Exception as e:
            print(f"Error processing {data_csv_path}: {e}")
            return all_prepared_data  # Return what has been processed so far

        print(f"Created {len(all_prepared_data)} data examples from {data_csv_path}")
        return all_prepared_data

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
                # A more robust approach: if primary is known and secondary is "Unknown" (not explicitly "None" from text),
                # then set secondary to "None".
                secondary_label = "None"

        except Exception:  # pylint: disable=broad-except
            # print(f"Error parsing labels from {source_type}: {e}. Text: '{text}'")
            return "Unknown-Unknown"  # Fallback on any parsing error

        # If primary is "Normie", secondary should always be "None" by definition of our heuristic.
        if primary_label == "Normie":
            secondary_label = "None"

        # If primary is "Unknown", it implies a failure, so secondary is also "Unknown".
        if primary_label == "Unknown":
            secondary_label = "Unknown"  # Ensure consistency

        return f"{primary_label}-{secondary_label}"

    def _evaluate_model_and_print_metrics(
        self, eval_data: List[Dict[str, str]], metrics_output_dir: str
    ):
        """Evaluates the model on the evaluation data and prints metrics for combined labels."""
        if not self.model or not self.tokenizer:
            print("Model or tokenizer not available for evaluation. Skipping.")
            return

        MAX_EVAL_SAMPLES = 10000  # Limit evaluation samples for practical reasons
        if len(eval_data) > MAX_EVAL_SAMPLES:
            print(
                f"Warning: Evaluation data has {len(eval_data)} samples. Evaluating on a random subset of {MAX_EVAL_SAMPLES} samples."
            )
            eval_data_subset = random.sample(eval_data, MAX_EVAL_SAMPLES)
        else:
            eval_data_subset = eval_data

        if not eval_data_subset:
            print("No evaluation data to process. Skipping metrics calculation.")
            return

        y_true_combined = []
        y_pred_combined = []

        # Ensure model and tokenizer are on the correct device
        self.model.to(self.device)

        # Use self.batch_size for evaluation batching
        num_batches = math.ceil(len(eval_data_subset) / self.batch_size)

        for batch_idx in tqdm(
            range(num_batches), desc="Evaluating model (batched)", mininterval=10.0
        ):
            batch = eval_data_subset[
                batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size
            ]
            prompts = [f"<|user|>\n{item['prompt']}\n<|assistant|>" for item in batch]
            true_labels = [item["true_label_heuristic"] for item in batch]

            # Tokenize as a batch
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                truncation=True,
                max_length=getattr(self.tokenizer, "model_max_length", 2048)
                - 100,  # Leave space for generation
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=70,  # Max tokens for the response part
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=False,  # For consistent evaluation
                    temperature=0.1,  # Low temperature for less randomness
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
        # and any other labels that might have emerged unexpectedly
        report_labels = sorted(
            list(set(self.defined_combined_labels + current_labels_in_data))
        )

        # Ensure the metrics output directory exists
        if not os.path.exists(metrics_output_dir):
            os.makedirs(metrics_output_dir)
            print(f"Created metrics directory: {metrics_output_dir}")

        # Save classification report to file
        report_str = classification_report(
            y_true_combined,
            y_pred_combined,
            labels=report_labels,
            zero_division=0,
            target_names=report_labels,
        )
        print(report_str)
        report_path = os.path.join(metrics_output_dir, "classification_report.txt")
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_str)
            print(f"Classification report saved to {report_path}")
        except Exception as e:  # pylint: disable=broad-except
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

        cm_path = os.path.join(metrics_output_dir, "confusion_matrix_combined.png")
        try:
            plt.savefig(cm_path)
            print(f"Confusion matrix saved to {cm_path}")
        except Exception as e:  # pylint: disable=broad-except
            print(f"Error saving confusion matrix: {e}")
        plt.close()  # Close the plot to free memory

    def finetune_model(
        self,
        training_data_csv: str,
        output_dir: str = "finetuned_model",
        epochs: int = 3,
        learning_rate: float = 2e-4,
        eval_split_size: float = 0.20,
        skip_eval_after_train: bool = False,
    ):
        """Fine-tune the model using the prepared data and optionally evaluate it."""
        if self.model is None or self.tokenizer is None:
            try:
                # self.model_name here should be the base model for fine-tuning
                self.setup_model(self.model_name)
                self._setup_peft_adapters()  # Setup PEFT adapters after loading base model
            except RuntimeError as e:  # Catch error from setup_model
                print(f"Failed to setup model for fine-tuning: {e}")
                return None  # Cannot proceed with fine-tuning

        # Prepare all data from the training CSV
        all_prepared_data = self._prepare_data_from_csv(training_data_csv)
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
            else:  # Not enough samples for a split or stratification
                if not stratify_possible_sklearn:
                    print(
                        "Warning: Stratification not possible due to class imbalance or too few samples per class. Splitting without stratification."
                    )
                if not enough_total_samples_for_split:
                    print(
                        "Warning: Not enough total samples for a meaningful train/eval split. Adjusting."
                    )

                if enough_total_samples_for_split:
                    train_data_dicts, eval_data_dicts = train_test_split(
                        all_prepared_data,
                        test_size=eval_split_size,
                        random_state=RANDOM_SEED,  # No stratification
                    )
                else:
                    print(
                        "Warning: Not enough samples for train/eval split. Using all data for training and skipping in-training evaluation split."
                    )
                    train_data_dicts = all_prepared_data
                    eval_data_dicts = []  # No evaluation split
                    skip_eval_after_train = True  # Force skip if no eval data
        else:
            print(
                "Eval split size is 0 or not enough data. Using all data for training. In-training evaluation will be skipped."
            )
            train_data_dicts = all_prepared_data
            eval_data_dicts = []
            skip_eval_after_train = True  # Force skip if no eval data

        print(
            f"Training with {len(train_data_dicts)} samples, evaluating with {len(eval_data_dicts)} samples (if eval not skipped)."
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
            per_device_train_batch_size=self.batch_size,  # Use instance batch_size
            gradient_accumulation_steps=4,  # Adjust if OOM
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            logging_steps=max(1, len(sft_train_dataset) // (self.batch_size * 10))  # type: ignore
            if sft_train_dataset and self.batch_size > 0
            else 10,  # Log ~10 times per epoch
            save_strategy="epoch",
            dataset_num_proc=min(
                4, os.cpu_count() or 1
            ),  # Number of processes for dataset processing
            save_total_limit=2,
            remove_unused_columns=True,
            report_to="none",  # "tensorboard" or "wandb" if desired
            max_seq_length=384,  # Max length of formatted prompt + response
            packing=True,  # Packs multiple short examples into one sequence
            optim="adamw_torch",
            fp16=torch.cuda.is_available()
            and not (
                hasattr(torch.cuda, "is_bf16_supported")
                and torch.cuda.is_bf16_supported()
            ),  # type: ignore
            bf16=torch.cuda.is_available()
            and hasattr(torch.cuda, "is_bf16_supported")
            and torch.cuda.is_bf16_supported(),  # type: ignore
            # Added for potential memory saving during training, if supported and useful:
            # gradient_checkpointing=True, # Can save memory but slows down training
        )

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,  # Pass tokenizer explicitly
            args=sft_config,
            train_dataset=sft_train_dataset,
            formatting_func=formatting_func,
        )

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("Starting model fine-tuning...")
        try:
            trainer.train()
            print(f"Saving fine-tuned model (adapters) to {output_dir}")
            # SFTTrainer saves PEFT adapters by default.
            trainer.save_model(output_dir)
            # Save the tokenizer along with the adapters for easy reloading
            self.tokenizer.save_pretrained(output_dir)
        except Exception as e:  # pylint: disable=broad-except
            print(f"Error during training or saving model: {e}")
            if "CUDA out of memory" in str(e):
                print(
                    "CUDA out of memory during training. Try reducing batch_size, max_seq_length, or using gradient_accumulation_steps > 1."
                )
            return None  # Indicate failure

        if not skip_eval_after_train and eval_data_dicts:
            print("Evaluating model after fine-tuning...")
            # Metrics will be saved relative to the parent of output_dir
            metrics_parent_dir = (
                os.path.dirname(output_dir) if os.path.dirname(output_dir) else "."
            )
            eval_metrics_dir = os.path.join(
                metrics_parent_dir, "metrics", os.path.basename(output_dir)
            )
            self._evaluate_model_and_print_metrics(eval_data_dicts, eval_metrics_dir)
        elif skip_eval_after_train:
            print("Skipping evaluation after training as requested.")
        else:  # No eval_data_dicts
            print(
                "Skipping evaluation as no evaluation data was prepared/available from the split."
            )

        # To save the full model (merged with adapters) if needed:
        # print("Merging adapters and saving full model...")
        # merged_model_path = os.path.join(output_dir, "merged_model_16bit")
        # self.model.save_pretrained_merged(merged_model_path, self.tokenizer, save_method = "merged_16bit")
        # print(f"Full merged model saved to {merged_model_path}")
        # For 4bit merged model (if base was 4bit and you want to keep it that way, might need specific Unsloth methods)
        # merged_model_path_4bit = os.path.join(output_dir, "merged_model_4bit")
        # self.model.save_pretrained_merged(merged_model_path_4bit, self.tokenizer, save_method = "merged_4bit_forced") # Or other appropriate method
        # print(f"Full 4-bit merged model saved to {merged_model_path_4bit}")

        return output_dir

    def load_finetuned_model(self, model_path: str):
        """Load a fine-tuned model (PEFT/LoRA adapters or full model).
        Unsloth's from_pretrained can often handle loading PEFT models directly
        if the directory contains adapter_config.json.
        """
        print(f"Attempting to load fine-tuned model from {model_path}")
        try:
            # This will load the base model and apply adapters if adapter_config.json is present.
            # Or load a full fine-tuned model if it's saved that way.
            self.setup_model(model_name_or_path=model_path)
            # self.model_name is updated to the path of the loaded model
            self.model_name = model_path
        except RuntimeError as e:
            print(f"Failed to load fine-tuned model from {model_path}: {e}")
            raise  # Re-raise the exception to be handled by the caller

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
                # For now, we will rely on explicit login before calling classify
                print(
                    "Warning: Auto-login attempt from fetch_user_posts might not work as expected due to sync/async. Ensure client is logged in."
                )

            if self.client is None:  # Check again after potential auto-login attempt
                raise ValueError(
                    "Not logged in to Bluesky. Call login_bluesky first or set BLUESKY_USERNAME/BLUESKY_PASSWORD env vars and ensure login succeeded."
                )

        try:
            # Ensure client.get_profile and client.get_author_feed are awaited
            user_info = await self.client.get_profile(actor=user_handle)  # type: ignore
            user_did = user_info.did  # type: ignore

            posts_texts: List[
                str
            ] = []  # Renamed to avoid conflict with 'posts' variable name if used elsewhere
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

                response = await self.client.get_author_feed(  # type: ignore
                    actor=user_did, limit=fetch_limit_this_request, cursor=cursor
                )
                if (
                    not response or not response.feed  # type: ignore
                ):  # Check if response or response.feed is None
                    break

                for feed_item in response.feed:  # type: ignore
                    if (
                        hasattr(feed_item, "post")
                        and feed_item.post  # Check if post exists
                        and hasattr(feed_item.post, "record")
                        and feed_item.post.record  # Check if record exists
                        and hasattr(feed_item.post.record, "text")
                        and isinstance(
                            feed_item.post.record.text, str
                        )  # Ensure text is string
                    ):  # Check if text exists
                        posts_texts.append(feed_item.post.record.text)
                        fetched_count += 1
                        if fetched_count >= limit:
                            break

                if hasattr(response, "cursor"):
                    cursor = response.cursor  # type: ignore
                else:  # Should not happen with valid response
                    break

                if cursor is None:
                    break
            print(f"Fetched {len(posts_texts)} posts for user {user_handle}")
            return posts_texts
        except Exception as e:  # pylint: disable=broad-except
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
        """Classify a user based on their posts using the loaded model and heuristics."""
        if not posts:
            return {
                "primary_classification": "Unknown (No Posts)",
                "secondary_classification": "None",
                "weeb_score": 0.0,  # Ensure float
                "furry_score": 0.0,  # Ensure float
                "normie_score": 1.0,  # Ensure float
                "model_combined_labels_debug": [],
            }

        if self.model is None or self.tokenizer is None:
            # This should ideally not happen if setup_model or load_finetuned_model was called.
            # self.model_name should point to the model to be used.
            print(
                f"Warning: Model not explicitly loaded for classify_user. Attempting to load from self.model_name: {self.model_name}."
            )
            try:
                self.setup_model(
                    self.model_name
                )  # Try loading based on the instance's model_name
            except Exception as e:  # pylint: disable=broad-except
                raise ValueError(
                    f"Model not loaded and failed to auto-load from '{self.model_name}'. Call setup_model or load_finetuned_model first. Error: {e}"
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
            : min(
                10, len(posts)
            )  # Process at most 10 posts with the model
        ]

        prompts_for_model = []
        for post_text in posts_for_model:
            prompt = f"Analyze this Bluesky post for weeb and furry traits: {post_text}"
            formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>"
            prompts_for_model.append(formatted_prompt)

        if prompts_for_model:
            # Batch process posts with the model
            num_model_batches = math.ceil(len(prompts_for_model) / self.batch_size)
            for batch_idx in tqdm(
                range(num_model_batches),
                desc="Classifying posts with model (batched)",
                leave=False,
                mininterval=5.0,
            ):
                batch_prompts = prompts_for_model[
                    batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size
                ]

                max_input_length = (
                    getattr(self.tokenizer, "model_max_length", 2048) - 100
                )  # type: ignore

                inputs = self.tokenizer(  # type: ignore
                    batch_prompts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_input_length,
                    padding=True,
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model.generate(  # type: ignore
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=70,
                        pad_token_id=self.tokenizer.pad_token_id,  # type: ignore
                        eos_token_id=self.tokenizer.eos_token_id,  # type: ignore
                        do_sample=False,
                        temperature=0.1,
                    )

                for i in range(len(batch_prompts)):
                    input_ids_length = inputs.input_ids[i].shape[0]
                    generated_tokens = outputs[i][input_ids_length:]
                    model_response_part = self.tokenizer.decode(  # type: ignore
                        generated_tokens, skip_special_tokens=True
                    ).strip()

                    predicted_combined = self._parse_combined_label_from_text(
                        model_response_part,
                        source_type="model output for user classification",
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
                else:  # Should not happen if parsing is correct
                    final_primary_classification = most_common_combined
                    final_secondary_classification = "None"  # Fallback
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
            "model_combined_labels_debug": model_predicted_combined_labels,  # For seeing individual post model outputs
        }


class BlueskyUserDataset:
    @staticmethod
    def preprocess_and_save(input_csv: str, output_csv: str):
        if not os.path.exists(input_csv):
            raise FileNotFoundError(f"Input file {input_csv} not found")
        print(f"Loading data from {input_csv}...")
        try:
            # Try to read with error handling for bad lines
            df = pd.read_csv(input_csv, on_bad_lines="skip")
        except pd.errors.EmptyDataError:
            print(f"Error: Input file {input_csv} is empty. Cannot preprocess.")
            return
        except Exception as e:  # pylint: disable=broad-except
            print(f"Error reading {input_csv}: {e}")
            return

        if "type" not in df.columns or "text" not in df.columns:
            print(
                f"Error: Input CSV {input_csv} must contain 'type' and 'text' columns."
            )
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
        df[["text", "type"]].to_csv(
            output_csv, index=False
        )  # Save only relevant columns
        print(f"Saved {len(df)} processed posts")


def main():
    parser = argparse.ArgumentParser(
        description="Bluesky User Classifier with Fine-tuning, Evaluation, and Classification"
    )
    subparsers = parser.add_subparsers(
        dest="command", help="Command to run", required=True
    )

    # --- Preprocess Command ---
    preprocess_parser = subparsers.add_parser(
        "preprocess", help="Preprocess Bluesky data CSV (filters posts, cleans text)"
    )
    preprocess_parser.add_argument(
        "--input",
        required=True,
        help="Input CSV file (must contain 'type' and 'text' columns)",
    )
    preprocess_parser.add_argument(
        "--output", required=True, help="Output CSV file for processed posts"
    )

    # --- Finetune Command ---
    finetune_parser = subparsers.add_parser(
        "finetune", help="Fine-tune the language model and optionally evaluate"
    )
    finetune_parser.add_argument(
        "--data_csv",
        required=True,
        help="Path to the processed Bluesky data CSV for fine-tuning and validation split",
    )
    finetune_parser.add_argument(
        "--output_dir",
        default="finetuned_model",
        help="Directory to save the fine-tuned model (adapters) and tokenizer",
    )
    finetune_parser.add_argument(
        "--base_model_name",
        default="unsloth/Qwen3-0.6B-unsloth-bnb-4bit",
        help="Base model name from Hugging Face Hub for fine-tuning (e.g., 'unsloth/Qwen3-0.6B-unsloth-bnb-4bit')",
    )
    finetune_parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs (default: 3)"
    )
    finetune_parser.add_argument(
        "--batch_size", type=int, default=8, help="Training batch size (default: 8)"
    )
    finetune_parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,  # Corresponds to 0.0002
        help="Learning rate for fine-tuning (default: 2e-4)",
    )
    finetune_parser.add_argument(
        "--eval_split_size",
        type=float,
        default=0.2,
        help="Proportion of data from --data_csv to use for validation during fine-tuning (0 to 1, default: 0.2). If 0, no validation split is made from training data.",
    )
    finetune_parser.add_argument(
        "--skip_eval_after_train",
        action="store_true",  # Makes it a flag, default is False
        help="If set, skips the evaluation step after fine-tuning is complete.",
    )
    finetune_parser.add_argument(
        "--chunk_size_csv",  # Renamed for clarity
        type=int,
        default=50000,  # Reduced default
        help="Chunk size for reading the input CSV data during data preparation (default: 50000).",
    )

    # --- Evaluate Command ---
    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evaluate a pre-trained (fine-tuned or base) model"
    )
    evaluate_parser.add_argument(
        "--model_path",
        required=True,
        help="Path to the fine-tuned model directory (containing adapters and tokenizer) or a base model name from Hugging Face Hub.",
    )
    evaluate_parser.add_argument(
        "--eval_data_csv",
        required=True,
        help="Path to the CSV file containing data for evaluation (will be processed to create prompts and heuristic labels).",
    )
    evaluate_parser.add_argument(
        "--metrics_output_dir",
        default="evaluation_metrics",
        help="Directory to save evaluation metrics (classification report, confusion matrix).",
    )
    evaluate_parser.add_argument(
        "--batch_size", type=int, default=16, help="Evaluation batch size (default: 16)"
    )
    evaluate_parser.add_argument(
        "--chunk_size_csv",  # Renamed for clarity
        type=int,
        default=50000,  # Reduced default
        help="Chunk size for reading the evaluation CSV data during data preparation (default: 50000).",
    )

    # --- Classify Command ---
    classify_parser = subparsers.add_parser(
        "classify", help="Classify a Bluesky user based on their posts"
    )
    classify_parser.add_argument(
        "--model_path",  # Changed from --model to --model_path for consistency
        required=True,
        help="Path to the fine-tuned model directory (containing adapters and tokenizer) or a base model name from Hugging Face Hub to use for classification.",
    )
    classify_parser.add_argument(
        "--username",
        required=True,
        help="Bluesky username (handle) to classify (e.g., 'username.bsky.social')",
    )
    classify_parser.add_argument(
        "--bluesky_user",
        default=os.getenv("BLUESKY_USERNAME"),
        help="Your Bluesky login username (or set BLUESKY_USERNAME env var). Required for fetching posts.",
    )
    classify_parser.add_argument(
        "--bluesky_pass",
        default=os.getenv("BLUESKY_PASSWORD"),
        help="Your Bluesky login password (or set BLUESKY_PASSWORD env var). Required for fetching posts.",
    )
    classify_parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for classifying user posts with the model (default: 8)",
    )

    args = parser.parse_args()

    if args.command == "preprocess":
        BlueskyUserDataset.preprocess_and_save(args.input, args.output)

    elif args.command == "finetune":
        # For fine-tuning, model_name in BlueskyClassifier is the base model.
        classifier = BlueskyClassifier(
            model_name=args.base_model_name, batch_size=args.batch_size
        )
        classifier.chunk_size = args.chunk_size_csv
        classifier.finetune_model(
            training_data_csv=args.data_csv,
            output_dir=args.output_dir,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            eval_split_size=args.eval_split_size,
            skip_eval_after_train=args.skip_eval_after_train,
        )

    elif args.command == "evaluate":
        # For evaluation, model_name in BlueskyClassifier is the model to be evaluated.
        classifier = BlueskyClassifier(
            model_name=args.model_path,
            batch_size=args.batch_size,  # model_path is passed as model_name
        )
        classifier.chunk_size = args.chunk_size_csv
        try:
            # Load the specified model (could be fine-tuned adapters or a base model)
            classifier.load_finetuned_model(
                args.model_path
            )  # This calls setup_model internally

            print(f"Preparing evaluation data from: {args.eval_data_csv}")
            eval_prepared_data = classifier._prepare_data_from_csv(args.eval_data_csv)

            if eval_prepared_data:
                print(f"Starting evaluation using model: {args.model_path}")
                classifier._evaluate_model_and_print_metrics(
                    eval_prepared_data, args.metrics_output_dir
                )
            else:
                print(
                    f"No data prepared from {args.eval_data_csv}. Skipping evaluation."
                )

        except RuntimeError as e:
            print(f"Error during evaluation setup or execution: {e}")
        except FileNotFoundError as e:
            print(f"Error: {e}")

    elif args.command == "classify":
        import asyncio

        async def run_classification():
            # For classification, model_name in BlueskyClassifier is the model to be used.
            # It could be a base model name or a path to a fine-tuned model.
            classifier = BlueskyClassifier(
                model_name=args.model_path, batch_size=args.batch_size
            )

            try:
                # Load the model specified by --model_path.
                # load_finetuned_model handles paths to adapters or full models.
                # setup_model can handle base model names from Hub.
                # Unsloth's from_pretrained (called by setup_model/load_finetuned_model) is flexible.
                classifier.load_finetuned_model(args.model_path)

            except RuntimeError as e:
                print(
                    f"Failed to initialize or load model from '{args.model_path}': {e}"
                )
                return

            if not args.bluesky_user or not args.bluesky_pass:
                print(
                    "Bluesky login username or password not provided. Please set BLUESKY_USERNAME and BLUESKY_PASSWORD environment variables or use --bluesky_user and --bluesky_pass arguments."
                )
                return

            # Perform login using the atproto Client directly within the async function
            # This ensures async operations are handled correctly.
            # The classifier.client will be set upon successful login.
            temp_client = Client()
            try:
                print(f"Logging in to Bluesky as {args.bluesky_user}...")
                await temp_client.login(args.bluesky_user, args.bluesky_pass)
                classifier.client = temp_client  # Assign the logged-in client to the classifier instance
                print(f"Successfully logged in to Bluesky as {args.bluesky_user}")
            except Exception as e:  # pylint: disable=broad-except
                print(f"Failed to log in to Bluesky: {e}")
                return  # Exit if login fails

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
                result = classifier.classify_user(
                    posts
                )  # This method uses the loaded model

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
                    f"  Model Post Classifications (sample of up to 10 posts): {result['model_combined_labels_debug']}"
                )
            print("=" * 50 + "\n")

        asyncio.run(run_classification())
    else:
        parser.print_help()


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()  # For PyInstaller or cx_Freeze if used

    # These imports are potentially heavy and only needed if certain commands are run.
    # Unsloth and TRL are specific to fine-tuning and model loading with PEFT.
    try:
        from unsloth import FastLanguageModel  # type: ignore
    except ImportError:
        print(
            "Error: Unsloth library not found. Please install with 'pip install \"unsloth[cu1xx-ampere-torch210]\"' (adjust for your CUDA/torch version) or 'pip install unsloth'."
        )
        sys.exit(1)

    try:
        from trl import SFTTrainer, SFTConfig  # type: ignore
    except ImportError:
        print("Error: TRL library not found. Please install with 'pip install trl'.")
        sys.exit(1)

    main()
