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
import regex as re
import json

# For Bluesky/ATProto
from atproto import Client  # type: ignore

# For metrics
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
)  # type: ignore

# For classification
from dotenv import load_dotenv

load_dotenv()

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# --- Continuous Labeling Logic ---


def normalize_text(s: str) -> str:
    """
    Normalize text by removing URLs, emojis, mentions, and collapsing repeated characters.
    """
    # Remove URLs
    s = re.sub(r"https?://\S+|www\.\S+", "", s)

    # Remove @mentions
    s = re.sub(r"@\w+", "", s)

    # Remove emojis (comprehensive pattern)
    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f680-\U0001f6ff"  # transport & map
        "\U0001f1e0-\U0001f1ff"  # flags
        "\U00002600-\U000027bf"  # misc symbols
        "\U0001f900-\U0001f9ff"  # supplemental symbols
        "\U00002700-\U000027bf"  # dingbats
        "]+",
        flags=re.UNICODE,
    )
    s = emoji_pattern.sub("", s)

    # Remove numbers and non-letter characters (keep Unicode letters and accents)
    s = re.sub(r"[^\p{L}\s]", "", s, flags=re.UNICODE)

    # Collapse repeated characters: "soooo" → "soo"
    s = re.sub(r"(.)\1{2,}", r"\1\1", s)

    # Collapse whitespace and clean up
    s = re.sub(r"\s+", " ", s)

    return s.strip()


def calculate_category_score(text_content: str, terms_df: pd.DataFrame) -> float:
    """
    Calculate a normalized score for a category based on text and terms.
    Token-based: Score = sum of matched term scores / sum of all term scores in the category.
    This results in a score between 0 and 1.
    """
    if (
        terms_df.empty
        or not isinstance(text_content, str)
        or terms_df["combined_score"].sum() == 0
    ):
        return 0.0

    # Tokenize text (simple whitespace split, lowercased)
    tokens = set(normalize_text(text_content).lower().split())

    if "term" not in terms_df.columns or "combined_score" not in terms_df.columns:
        return 0.0
    if not terms_df["term"].apply(isinstance, args=(str,)).all():
        print(
            "Warning: Non-string values found in 'term' column of terms_df during score calculation."
        )

    matched_terms_score_sum = 0.0
    for term, score_val in zip(terms_df["term"], terms_df["combined_score"]):
        # Token-based match: check if the term (lowercased) is in the token set
        if str(term).lower() in tokens:
            matched_terms_score_sum += float(score_val)

    max_potential_score = terms_df["combined_score"].sum()

    return (
        matched_terms_score_sum / max_potential_score
        if max_potential_score > 0
        else 0.0
    )


def determine_continuous_labels_from_scores(
    w_score: float,
    f_score: float,
    min_threshold_weeb: float = 0.0031,
    min_threshold_furry: float = 0.0034,
    strong_threshold_weeb: float = 0.0047,
    strong_threshold_furry: float = 0.0051,
) -> Tuple[str, str, float, float]:
    """
    Determine primary and secondary classification labels based on continuous scores.
    Now returns labels with confidence scores instead of hard binning.
    Thresholds for weeb and furry are independent and configurable.
    """
    weeb_confidence = min(1.0, w_score * 20)
    furry_confidence = min(1.0, f_score * 20)

    # Determine primary classification based on which score is higher
    if w_score < min_threshold_weeb and f_score < min_threshold_furry:
        primary_label = "Normie"
        secondary_label = "None"
        primary_confidence = 1.0 - max(w_score, f_score) * 10
        secondary_confidence = 0.0
    elif w_score >= f_score:
        # Weeb is primary
        if w_score > strong_threshold_weeb:
            primary_label = "Weeb"
            primary_confidence = weeb_confidence
        else:
            primary_label = "Slight Weeb"
            primary_confidence = weeb_confidence * 0.7

        # Secondary based on furry score
        if f_score > min_threshold_furry:
            if f_score > strong_threshold_furry:
                secondary_label = "Furry"
                secondary_confidence = furry_confidence
            else:
                secondary_label = "Slight Furry"
                secondary_confidence = furry_confidence * 0.7
        else:
            secondary_label = "None"
            secondary_confidence = 0.0
    else:
        # Furry is primary
        if f_score > strong_threshold_furry:
            primary_label = "Furry"
            primary_confidence = furry_confidence
        else:
            primary_label = "Slight Furry"
            primary_confidence = furry_confidence * 0.7

        # Secondary based on weeb score
        if w_score > min_threshold_weeb:
            if w_score > strong_threshold_weeb:
                secondary_label = "Weeb"
                secondary_confidence = weeb_confidence
            else:
                secondary_label = "Slight Weeb"
                secondary_confidence = weeb_confidence * 0.7
        else:
            secondary_label = "None"
            secondary_confidence = 0.0

    primary_confidence = max(0.0, min(1.0, primary_confidence))
    secondary_confidence = max(0.0, min(1.0, secondary_confidence))

    if primary_label in ("Slight Weeb", "Slight Furry"):
        secondary_label = "None"
        secondary_confidence = 0.0

    return primary_label, secondary_label, primary_confidence, secondary_confidence


def process_post_for_training(args) -> Dict[str, Any] | None:
    """
    Process a single post for training data generation.
    Now includes continuous scores and confidence values in the training data.
    """
    (
        text,
        weeb_terms_df,
        furry_terms_df,
        min_threshold_weeb,
        min_threshold_furry,
        strong_threshold_weeb,
        strong_threshold_furry,
    ) = args

    if not text or (isinstance(text, str) and not text.strip()):
        return None
    if not isinstance(text, str):
        try:
            text = str(text)
        except (TypeError, ValueError):
            return None
        if not text.strip():
            return None

    weeb_score = calculate_category_score(text, weeb_terms_df)
    furry_score = calculate_category_score(text, furry_terms_df)

    primary_label, secondary_label, primary_conf, secondary_conf = (
        determine_continuous_labels_from_scores(
            weeb_score,
            furry_score,
            min_threshold_weeb,
            min_threshold_furry,
            strong_threshold_weeb,
            strong_threshold_furry,
        )
    )

    prompt_text_content = str(text)
    prompt = (
        "Analyze this Bluesky post using the provided scores. "
        "Respond with:\n"
        "Primary Classification: <label>\n"
        "Secondary Classification: <label>\n"
        "Valid labels: Normie, Weeb, Furry, Slight Weeb, Slight Furry, None.\n\n"
        f"Weeb Score: {weeb_score:.6f}\n"
        f"Furry Score: {furry_score:.6f}\n"
        f"Post Content: {prompt_text_content[:300]}"
    )

    response = (
        f"Primary Classification: {primary_label}\n"
        f"Secondary Classification: {secondary_label}\n"
    )

    return {
        "prompt": prompt,
        "response": response,
        "true_label_heuristic": f"{primary_label}-{secondary_label}",
        "weeb_score": weeb_score,
        "furry_score": furry_score,
        "primary_confidence": primary_conf,
        "secondary_confidence": secondary_conf,
        "combined_intensity": weeb_score + furry_score,
        "score_ratio": weeb_score / furry_score if furry_score > 0 else float("inf"),
    }


class BlueskyClassifier:
    def __init__(
        self,
        model_name="unsloth/Qwen3-0.6B-unsloth-bnb-4bit",
        batch_size=8,
        min_threshold_weeb: float = 0.0031,
        min_threshold_furry: float = 0.0034,
        strong_threshold_weeb: float = 0.0047,
        strong_threshold_furry: float = 0.0051,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None
        self.tokenizer = None
        self.chunk_size = 100000

        # Continuous classification parameters
        self.min_threshold_weeb = min_threshold_weeb
        self.min_threshold_furry = min_threshold_furry
        self.strong_threshold_weeb = strong_threshold_weeb
        self.strong_threshold_furry = strong_threshold_furry

        print("\n--- Using Continuous Score Classification ---")
        print(
            f"  Minimum Thresholds: weeb={self.min_threshold_weeb:.6f}, furry={self.min_threshold_furry:.6f}"
        )
        print(
            f"  Strong Thresholds: weeb={self.strong_threshold_weeb:.6f}, furry={self.strong_threshold_furry:.6f}"
        )
        print("  Classification: Based on continuous scores and learned boundaries")
        print("---------------------------------------------\n")

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
                "Slight Furry-None",
                "Unknown-Unknown",
            ]
        )
        self.primary_labels_set = {
            "Weeb",
            "Furry",
            "Slight Weeb",
            "Slight Furry",
            "Normie",
            "Unknown",
        }
        self.secondary_labels_set = {
            "Weeb",
            "Furry",
            "Slight Weeb",
            "Slight Furry",
            "None",
            "Unknown",
        }

        self.weeb_terms = self._load_term_database(
            "output/terms-analysis-bertopic/weeb_terms_bertopic.csv"
        )
        self.furry_terms = self._load_term_database(
            "output/terms-analysis-bertopic/furry_terms_bertopic.csv"
        )
        self.client = None

    def _load_term_database(self, csv_file: str) -> pd.DataFrame:
        if not os.path.exists(csv_file):
            print(
                f"Warning: Term database {csv_file} not found. Scores might be affected."
            )
            return pd.DataFrame(columns=["term", "combined_score"])
        try:
            df = pd.read_csv(csv_file)
            if "term" not in df.columns or "combined_score" not in df.columns:
                print(
                    f"Warning: {csv_file} is missing 'term' or 'combined_score' column."
                )
                df["term"] = df.get("term", pd.Series(dtype="str"))
                df["combined_score"] = df.get(
                    "combined_score", pd.Series(dtype="float")
                )
            df["term"] = df["term"].astype(str)
            df["combined_score"] = pd.to_numeric(
                df["combined_score"], errors="coerce"
            ).fillna(0)
            return df[["term", "combined_score"]].sort_values(
                by="combined_score", ascending=False
            )
        except Exception as e:
            print(f"Error loading term database {csv_file}: {e}")
            return pd.DataFrame(columns=["term", "combined_score"])

    def _calculate_category_score(
        self, text_content: str, terms_df: pd.DataFrame
    ) -> float:
        return calculate_category_score(text_content, terms_df)

    def setup_model(self, model_name_or_path: str | None = None):
        load_path = model_name_or_path if model_name_or_path else self.model_name
        print(f"Loading model from: {load_path}")
        try:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=load_path,
                max_seq_length=2048,
                dtype=torch.bfloat16,
                load_in_4bit=False,
                device_map="auto",
            )
            self.model.gradient_checkpointing_enable()

            # Force left padding more aggressively
            self.tokenizer.padding_side = "left"
            self.tokenizer.truncation_side = "left"

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.model.config.pad_token_id is None:
                self.model.config.pad_token_id = self.model.config.eos_token_id

            # Force it again after config changes
            self.tokenizer.padding_side = "left"

            torch.backends.cudnn.benchmark = True
            print(f"Model loaded successfully from {load_path}.")
        except Exception as e:
            print(f"Error loading model {load_path}: {e}")
            raise RuntimeError(f"Failed to load model: {e}")

    def _setup_peft_adapters(self):
        if not self.model:
            raise RuntimeError("Base model not loaded. Call setup_model() first.")
        try:
            self.model = FastLanguageModel.get_peft_model(
                model=self.model,
                r=32,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                use_gradient_checkpointing="unsloth",
            )
            print("PEFT adapters configured for fine-tuning.")
        except Exception as e:
            print(f"Error setting up PEFT adapters: {e}")
            raise RuntimeError(f"Failed to set up PEFT adapters: {e}")

    def _prepare_data_from_csv(self, data_csv_path: str) -> List[Dict[str, str]]:
        print(f"Preparing data from {data_csv_path}...")
        if not os.path.exists(data_csv_path):
            raise FileNotFoundError(f"Data file {data_csv_path} not found")

        all_prepared_data = []
        num_processes = min(max(1, cpu_count() // 2), 4)
        print(f"Processing posts using {num_processes} processes...")
        try:
            for chunk_df in tqdm(
                pd.read_csv(
                    data_csv_path,
                    chunksize=self.chunk_size,
                    usecols=["type", "text"],
                    on_bad_lines="warn",
                ),
                desc="Reading CSV for data processing",
                mininterval=10.0,
            ):
                chunk_df = chunk_df[chunk_df["type"] == "post"]
                chunk_df = chunk_df.dropna(subset=["text"])
                chunk_df["text"] = chunk_df["text"].astype(str, errors="ignore")
                process_args = [
                    (
                        text,
                        self.weeb_terms,
                        self.furry_terms,
                        self.min_threshold_weeb,
                        self.min_threshold_furry,
                        self.strong_threshold_weeb,
                        self.strong_threshold_furry,
                    )
                    for text in chunk_df["text"]
                    if isinstance(text, str) and text.strip()
                ]
                if not process_args:
                    continue
                with Pool(processes=num_processes) as pool:
                    imap_chunksize = max(1, len(process_args) // (num_processes * 4))
                    batch_results = list(
                        tqdm(
                            pool.imap(
                                process_post_for_training,
                                process_args,
                                chunksize=imap_chunksize,
                            ),
                            total=len(process_args),
                            desc="Processing chunk",
                            leave=False,
                            mininterval=5.0,
                        )
                    )
                all_prepared_data.extend([r for r in batch_results if r is not None])
                del chunk_df, process_args, batch_results
        except Exception as e:
            print(f"Error during main data processing of {data_csv_path}: {e}")
            return all_prepared_data
        print(
            f"Successfully created {len(all_prepared_data)} data examples from {data_csv_path}"
        )
        return all_prepared_data

    def _create_balanced_eval_set(self, all_prepared_data, test_size):
        """Create a truly balanced evaluation set."""
        from collections import defaultdict

        # Group by label
        label_groups = defaultdict(list)
        for item in all_prepared_data:
            label_groups[item["true_label_heuristic"]].append(item)

        # Remove empty groups
        label_groups = {k: v for k, v in label_groups.items() if v}

        if not label_groups:
            return []

        # Calculate samples per class for balanced distribution
        num_classes = len(label_groups)
        samples_per_class = test_size // num_classes
        remainder = test_size % num_classes

        eval_samples = []

        # Sample equally from each class
        for i, (label, items) in enumerate(sorted(label_groups.items())):
            current_class_size = samples_per_class + (1 if i < remainder else 0)
            if len(items) >= current_class_size:
                eval_samples.extend(random.sample(items, current_class_size))
            else:
                eval_samples.extend(items)
                needed = current_class_size - len(items)
                if needed > 0:
                    oversampled = random.choices(items, k=needed)
                    eval_samples.extend(oversampled)

        random.shuffle(eval_samples)
        return eval_samples[:test_size]

    def _create_balanced_training_set(
        self, train_data_dicts, max_samples_per_class=None
    ):
        """Create balanced training set using oversampling/downsampling."""
        from collections import defaultdict

        # Group by label
        label_groups = defaultdict(list)
        for item in train_data_dicts:
            label_groups[item["true_label_heuristic"]].append(item)

        # Remove empty groups
        label_groups = {k: v for k, v in label_groups.items() if v}

        if not label_groups:
            return train_data_dicts

        # Find target size (use largest class size or specified max)
        class_sizes = [len(items) for items in label_groups.values()]
        if max_samples_per_class is None:
            target_size = max(class_sizes)
        else:
            target_size = min(max_samples_per_class, max(class_sizes))

        print(f"Balancing {len(label_groups)} classes to {target_size} samples each")

        balanced_data = []

        for label, items in label_groups.items():
            original_size = len(items)
            if original_size >= target_size:
                sampled_items = random.sample(items, target_size)
            else:
                sampled_items = items.copy()
                needed = target_size - original_size
                oversampled = random.choices(items, k=needed)
                sampled_items.extend(oversampled)
            balanced_data.extend(sampled_items)
            print(f"  {label}: {original_size} -> {len(sampled_items)} samples")

        random.shuffle(balanced_data)
        print(f"Total balanced training samples: {len(balanced_data)}")
        return balanced_data

    def finetune_model(
        self,
        training_data_csv: str,
        output_dir: str = "output/user-classifier",
        epochs: int = 3,
        learning_rate: float = 5e-5,
        eval_split_size: float = 0.20,
        skip_eval_after_train: bool = False,
        max_training_samples: int = None,
        test_size: int = None,
    ):
        """Fine-tune the model on continuous scoring data."""
        if self.model is None or self.tokenizer is None:
            try:
                self.setup_model(self.model_name)
                self._setup_peft_adapters()
            except RuntimeError as e:
                print(f"Failed to setup base model for fine-tuning: {e}")
                return None

        all_prepared_data = self._prepare_data_from_csv(training_data_csv)
        if not all_prepared_data:
            print("No training data prepared. Skipping fine-tuning.")
            return None

        # --- Improved split and balancing logic ---
        if len(all_prepared_data) > 1:
            if test_size:
                # Create balanced eval set first
                eval_data_dicts = self._create_balanced_eval_set(
                    all_prepared_data, test_size
                )
                eval_ids = {id(item) for item in eval_data_dicts}
                train_data_dicts = [
                    item for item in all_prepared_data if id(item) not in eval_ids
                ]
            else:
                try:
                    labels_for_stratification = [
                        item["true_label_heuristic"] for item in all_prepared_data
                    ]
                    train_data_dicts, eval_data_dicts = train_test_split(
                        all_prepared_data,
                        test_size=eval_split_size,
                        random_state=RANDOM_SEED,
                        stratify=labels_for_stratification,
                    )
                except ValueError:
                    train_data_dicts, eval_data_dicts = train_test_split(
                        all_prepared_data,
                        test_size=eval_split_size,
                        random_state=RANDOM_SEED,
                    )
        else:
            train_data_dicts, eval_data_dicts = all_prepared_data, []

        print(
            f"Training with {len(train_data_dicts)} samples, evaluating with {len(eval_data_dicts)} samples."
        )

        if not train_data_dicts:
            print(
                "No training samples available after splitting. Aborting fine-tuning."
            )
            return None

        # --- Apply class balancing instead of intensity balancing ---
        if len(train_data_dicts) > 100000:
            max_per_class = 50000  # Limit to prevent memory issues with 6M dataset
            train_data_dicts = self._create_balanced_training_set(
                train_data_dicts, max_samples_per_class=max_per_class
            )
        else:
            train_data_dicts = self._create_balanced_training_set(train_data_dicts)

        # Optionally limit to max_training_samples if set
        if max_training_samples and len(train_data_dicts) > max_training_samples:
            from collections import defaultdict

            label_groups = defaultdict(list)
            for item in train_data_dicts:
                label_groups[item["true_label_heuristic"]].append(item)
            num_classes = len(label_groups)
            samples_per_class = max_training_samples // num_classes
            trimmed = []
            for items in label_groups.values():
                trimmed.extend(random.sample(items, samples_per_class))
            train_data_dicts = trimmed
            print(f"Trimmed balanced training set to {len(train_data_dicts)} samples.")

        sft_train_dataset = Dataset.from_list(
            [
                {"prompt": d["prompt"], "response": d["response"]}
                for d in train_data_dicts
            ]
        )
        eos_token = self.tokenizer.eos_token

        def formatting_func(example):
            return [
                f"<|user|>\n{prompt}\n<|assistant|>\n{response}{eos_token}"
                for prompt, response in zip(example["prompt"], example["response"])
            ]

        sft_config = SFTConfig(
            completion_only_loss=False,
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=self.batch_size,
            gradient_checkpointing=True,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            max_grad_norm=0.3,
            logging_steps=500,
            save_strategy="epoch",
            dataset_num_proc=1,
            dataloader_num_workers=4,
            save_total_limit=1,
            remove_unused_columns=True,
            report_to="none",
            max_seq_length=300,
            packing=False,
            optim="adamw_bnb_8bit",
            fp16=torch.cuda.is_available()
            and not (
                hasattr(torch.cuda, "is_bf16_supported")
                and torch.cuda.is_bf16_supported()
            ),
            bf16=torch.cuda.is_available()
            and hasattr(torch.cuda, "is_bf16_supported")
            and torch.cuda.is_bf16_supported(),
            dataloader_pin_memory=False,
            group_by_length=True,
            label_smoothing_factor=0.1,  # Helps with class imbalance
        )

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=sft_config,
            train_dataset=sft_train_dataset,
            formatting_func=formatting_func,
        )

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("Starting model fine-tuning with continuous scoring...")
        try:
            trainer.train()
            print(f"Saving fine-tuned model (adapters) to {output_dir}")
            trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)
        except Exception as e:
            print(f"Error during training or saving model: {e}")
            if "CUDA out of memory" in str(e):
                print(
                    "CUDA out of memory. Try reducing batch_size, max_seq_length, or increasing gradient_accumulation_steps."
                )
            return None

        if not skip_eval_after_train and eval_data_dicts:
            print("Evaluating model after fine-tuning...")
            eval_metrics_dir = os.path.join(
                "metrics",
                os.path.basename(output_dir),
            )
            self._evaluate_model_and_print_metrics(eval_data_dicts, eval_metrics_dir)
        elif skip_eval_after_train:
            print("Skipping evaluation after training as requested.")
        else:
            print(
                "Skipping evaluation as no evaluation data was prepared/available from the split."
            )
        return output_dir

    def _parse_combined_label_and_confidence_from_text(
        self, text: str, source_type: str = "unknown source"
    ) -> Tuple[str, float, float]:
        """Parse both classification labels and confidence scores from model output."""
        primary, secondary = "Unknown", "Unknown"
        primary_conf, secondary_conf = 0.0, 0.0

        for line in text.splitlines():
            line_lower = line.lower()
            if "primary classification:" in line_lower:
                lab = line.split(":", 1)[1].strip().title()
                if lab in self.primary_labels_set:
                    primary = lab
            elif "secondary classification:" in line_lower:
                lab = line.split(":", 1)[1].strip().title()
                if lab in self.secondary_labels_set:
                    secondary = lab
            elif "primary confidence:" in line_lower:
                try:
                    primary_conf = float(line.split(":", 1)[1].strip())
                except (ValueError, IndexError):
                    primary_conf = 0.0
            elif "secondary confidence:" in line_lower:
                try:
                    secondary_conf = float(line.split(":", 1)[1].strip())
                except (ValueError, IndexError):
                    secondary_conf = 0.0

        if primary != "Unknown" and secondary == "Unknown":
            secondary = "None"
        if primary == "Normie":
            secondary = "None"
        if primary == "Unknown":
            secondary = "Unknown"

        combined_label = f"{primary}-{secondary}"
        return combined_label, primary_conf, secondary_conf

    def _evaluate_model_and_print_metrics(
        self, eval_data: List[Dict[str, str]], metrics_output_dir: str
    ):
        if not self.model or not self.tokenizer:
            print("Model or tokenizer not available for evaluation. Skipping.")
            return

        MAX_EVAL_SAMPLES = 20000
        eval_data_subset = (
            random.sample(eval_data, MAX_EVAL_SAMPLES)
            if len(eval_data) > MAX_EVAL_SAMPLES
            else eval_data
        )
        if not eval_data_subset:
            print("No evaluation data to process. Skipping metrics calculation.")
            return

        y_true_combined, y_pred_combined = [], []
        y_true_primary_conf, y_pred_primary_conf = [], []
        y_true_secondary_conf, y_pred_secondary_conf = [], []
        y_true_weeb_scores, y_true_furry_scores = [], []

        self.model.to(self.device)
        num_batches = math.ceil(len(eval_data_subset) / self.batch_size)

        # Force left padding right before tokenization
        self.tokenizer.padding_side = "left"

        for batch_idx in tqdm(
            range(num_batches), desc="Evaluating model (batched)", mininterval=10.0
        ):
            batch = eval_data_subset[
                batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size
            ]
            prompts = [f"<|user|>\n{item['prompt']}\n<|assistant|>" for item in batch]
            true_labels = [item["true_label_heuristic"] for item in batch]
            true_primary_confs = [item.get("primary_confidence", 0.0) for item in batch]
            true_secondary_confs = [
                item.get("secondary_confidence", 0.0) for item in batch
            ]
            true_weeb_scores = [item.get("weeb_score", 0.0) for item in batch]
            true_furry_scores = [item.get("furry_score", 0.0) for item in batch]

            # Force left padding right before tokenization
            self.tokenizer.padding_side = "left"
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
                    max_new_tokens=50,  # Increased for confidence scores
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                )

            for i in range(len(batch)):
                input_len = inputs.input_ids[i].shape[0]
                generated_tokens = outputs[i][input_len:]
                model_response_part = self.tokenizer.decode(
                    generated_tokens, skip_special_tokens=True
                ).strip()

                # DEBUG: Print first few responses
                if batch_idx == 0 and i < 3:
                    print(f"\n=== DEBUG SAMPLE {i} ===")
                    print(f"Prompt: {batch[i]['prompt'][:200]}...")
                    print(f"Model response: '{model_response_part}'")
                    print(f"True label: {true_labels[i]}")

                predicted_label, pred_primary_conf, pred_secondary_conf = (
                    self._parse_combined_label_and_confidence_from_text(
                        model_response_part, "evaluation output"
                    )
                )

                if batch_idx == 0 and i < 3:
                    print(f"Parsed label: {predicted_label}")
                    print("=" * 30)

                y_true_combined.append(true_labels[i])
                y_pred_combined.append(predicted_label)
                y_true_primary_conf.append(true_primary_confs[i])
                y_pred_primary_conf.append(pred_primary_conf)
                y_true_secondary_conf.append(true_secondary_confs[i])
                y_pred_secondary_conf.append(pred_secondary_conf)
                y_true_weeb_scores.append(true_weeb_scores[i])
                y_true_furry_scores.append(true_furry_scores[i])

        # Classification metrics
        print("\n--- Classification Metrics (Combined Labels) ---")
        current_labels_in_data = sorted(list(set(y_true_combined + y_pred_combined)))
        report_labels = sorted(
            list(set(self.defined_combined_labels + current_labels_in_data))
        )

        if not os.path.exists(metrics_output_dir):
            os.makedirs(metrics_output_dir)
            print(f"Created metrics directory: {metrics_output_dir}")

        report_str = classification_report(
            y_true_combined,
            y_pred_combined,
            labels=report_labels,
            zero_division=0,
            target_names=report_labels,
        )
        print(report_str)

        # Regression metrics for confidence scores
        print("\n--- Confidence Score Regression Metrics ---")
        primary_conf_mse = mean_squared_error(y_true_primary_conf, y_pred_primary_conf)
        primary_conf_mae = mean_absolute_error(y_true_primary_conf, y_pred_primary_conf)
        secondary_conf_mse = mean_squared_error(
            y_true_secondary_conf, y_pred_secondary_conf
        )
        secondary_conf_mae = mean_absolute_error(
            y_true_secondary_conf, y_pred_secondary_conf
        )

        print(
            f"Primary Confidence - MSE: {primary_conf_mse:.4f}, MAE: {primary_conf_mae:.4f}"
        )
        print(
            f"Secondary Confidence - MSE: {secondary_conf_mse:.4f}, MAE: {secondary_conf_mae:.4f}"
        )

        # Save comprehensive report
        report_path = os.path.join(metrics_output_dir, "comprehensive_report.txt")
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("=== Classification Report ===\n")
                f.write(report_str)
                f.write("\n\n=== Confidence Score Metrics ===\n")
                f.write(
                    f"Primary Confidence - MSE: {primary_conf_mse:.4f}, MAE: {primary_conf_mae:.4f}\n"
                )
                f.write(
                    f"Secondary Confidence - MSE: {secondary_conf_mse:.4f}, MAE: {secondary_conf_mae:.4f}\n"
                )
            print(f"Comprehensive report saved to {report_path}")
        except Exception as e:
            print(f"Error saving comprehensive report: {e}")

        # Visualization: Confidence scores vs true scores
        plt.figure(figsize=(15, 5))

        # Primary confidence correlation
        plt.subplot(1, 3, 1)
        plt.scatter(y_true_primary_conf, y_pred_primary_conf, alpha=0.6)
        plt.plot([0, 1], [0, 1], "r--", lw=2)
        plt.xlabel("True Primary Confidence")
        plt.ylabel("Predicted Primary Confidence")
        plt.title("Primary Confidence Prediction")
        plt.grid(True, alpha=0.3)

        # Secondary confidence correlation
        plt.subplot(1, 3, 2)
        plt.scatter(y_true_secondary_conf, y_pred_secondary_conf, alpha=0.6)
        plt.plot([0, 1], [0, 1], "r--", lw=2)
        plt.xlabel("True Secondary Confidence")
        plt.ylabel("Predicted Secondary Confidence")
        plt.title("Secondary Confidence Prediction")
        plt.grid(True, alpha=0.3)

        # Score distribution
        plt.subplot(1, 3, 3)
        plt.scatter(y_true_weeb_scores, y_true_furry_scores, alpha=0.6, s=20)
        plt.xlabel("Weeb Score")
        plt.ylabel("Furry Score")
        plt.title("Score Distribution")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        confidence_plot_path = os.path.join(
            metrics_output_dir, "confidence_analysis.png"
        )
        try:
            plt.savefig(confidence_plot_path, dpi=150, bbox_inches="tight")
            print(f"Confidence analysis plot saved to {confidence_plot_path}")
        except Exception as e:
            print(f"Error saving confidence plot: {e}")
        plt.close()

        # Original confusion matrix
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
            annot_kws={"size": 8},
        )
        plt.xlabel("Predicted Label (Primary-Secondary)")
        plt.ylabel("True Label (Primary-Secondary)")
        plt.title("Confusion Matrix (Combined Labels)")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()

        cm_path = os.path.join(metrics_output_dir, "confusion_matrix_combined.png")
        try:
            plt.savefig(cm_path)
            print(f"Confusion matrix saved to {cm_path}")
        except Exception as e:
            print(f"Error saving confusion matrix: {e}")
        plt.close()

    def load_finetuned_model(self, model_path: str):
        print(f"Attempting to load fine-tuned model from {model_path}")
        try:
            self.setup_model(model_name_or_path=model_path)
            self.model_name = model_path
        except RuntimeError as e:
            print(f"Failed to load fine-tuned model from {model_path}: {e}")
            raise

    async def fetch_user_posts(self, user_handle: str, limit: int = 500) -> List[str]:
        if self.client is None:
            raise ValueError(
                "Bluesky client not authenticated. Ensure login is performed before fetching posts."
            )
        try:
            user_info = self.client.get_profile(actor=user_handle)
            user_did = user_info.did
            posts_texts: List[str] = []
            cursor = None
            fetched_count = 0
            max_requests = (limit + 99) // 100
            for _ in range(max_requests):
                if fetched_count >= limit:
                    break
                fetch_limit_this_request = min(100, limit - fetched_count)
                if fetch_limit_this_request <= 0:
                    break
                response = self.client.get_author_feed(
                    actor=user_did, limit=fetch_limit_this_request, cursor=cursor
                )
                if not response or not response.feed:
                    break
                for feed_item in response.feed:
                    if (
                        hasattr(feed_item, "post")
                        and feed_item.post
                        and hasattr(feed_item.post, "record")
                        and feed_item.post.record
                        and hasattr(feed_item.post.record, "text")
                        and isinstance(feed_item.post.record.text, str)
                    ):
                        posts_texts.append(feed_item.post.record.text)
                        fetched_count += 1
                        if fetched_count >= limit:
                            break
                if hasattr(response, "cursor"):
                    cursor = response.cursor
                else:
                    break
                if cursor is None:
                    break
            print(f"Fetched {len(posts_texts)} posts for user {user_handle}")
            return posts_texts
        except Exception as e:
            print(f"Error fetching posts for {user_handle}: {e}")
            return []

    def classify_user(self, posts: List[str]) -> Dict[str, Any]:
        """
        Continuous classification using all available posts with confidence scoring.
        """
        if not posts:
            return {
                "primary_classification": "Normie",
                "secondary_classification": "None",
                "primary_confidence": 1.0,
                "secondary_confidence": 0.0,
                "average_weeb_score": 0.0,
                "average_furry_score": 0.0,
                "score_distribution": {"weeb_scores": [], "furry_scores": []},
                "model_predictions": [],
            }

        if self.model is None or self.tokenizer is None:
            print(
                f"Warning: Model not explicitly loaded for classify_user. Attempting to load from self.model_name: {self.model_name}."
            )
            try:
                self.setup_model(self.model_name)
            except Exception as e:
                raise ValueError(
                    f"Model not loaded and failed to auto-load from '{self.model_name}'. Error: {e}"
                )

        self.model.to(self.device)
        safe_posts = [str(p) if not isinstance(p, str) else p for p in posts]
        posts_for_model = [p for p in safe_posts if p.strip()]

        # Calculate scores for all posts
        weeb_scores = []
        furry_scores = []
        model_predictions = []

        # Generate prompts with scores for ALL posts
        prompts_for_model = []
        for post_text in posts_for_model:
            post_weeb_score = self._calculate_category_score(post_text, self.weeb_terms)
            post_furry_score = self._calculate_category_score(
                post_text, self.furry_terms
            )
            weeb_scores.append(post_weeb_score)
            furry_scores.append(post_furry_score)

            prompt = (
                f"<|user|>\nAnalyze this Bluesky post using the provided scores. "
                f"Respond with:\n"
                f"Primary Classification: <label>\n"
                f"Secondary Classification: <label>\n"
                f"Valid labels: Normie, Weeb, Furry, Slight Weeb, Slight Furry, None.\n\n"
                f"Weeb Score: {post_weeb_score:.6f}\n"
                f"Furry Score: {post_furry_score:.6f}\n"
                f"Post Content: {post_text[:300]}\n<|assistant|>"
            )
            prompts_for_model.append(prompt)

        # Process with model
        if prompts_for_model:
            num_model_batches = math.ceil(len(prompts_for_model) / self.batch_size)
            for batch_idx in tqdm(
                range(num_model_batches),
                desc=f"Classifying {len(posts_for_model)} user posts with continuous scoring",
                leave=False,
                mininterval=5.0,
            ):
                batch_prompts = prompts_for_model[
                    batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size
                ]
                max_input_length = (
                    getattr(self.tokenizer, "model_max_length", 2048) - 100
                )
                # Force left padding right before tokenization
                self.tokenizer.padding_side = "left"
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_input_length,
                    padding=True,
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=50,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        top_k=None,
                    )

                for i in range(len(batch_prompts)):
                    input_ids_length = inputs.input_ids[i].shape[0]
                    model_response_part = self.tokenizer.decode(
                        outputs[i][input_ids_length:], skip_special_tokens=True
                    ).strip()

                    predicted_combined, pred_primary_conf, pred_secondary_conf = (
                        self._parse_combined_label_and_confidence_from_text(
                            model_response_part, "model output for user classification"
                        )
                    )

                    model_predictions.append(
                        {
                            "label": predicted_combined,
                            "primary_confidence": pred_primary_conf,
                            "secondary_confidence": pred_secondary_conf,
                        }
                    )

        # Aggregate results using continuous approach
        final_primary_classification = "Normie"
        final_secondary_classification = "None"

        if model_predictions:
            # Filter out "Unknown-Unknown" predictions
            valid_predictions = [
                pred for pred in model_predictions if pred["label"] != "Unknown-Unknown"
            ]

            if valid_predictions:
                # Weight predictions by their confidence scores
                label_confidence_pairs = [
                    (pred["label"], pred["primary_confidence"])
                    for pred in valid_predictions
                ]

                # Find the most confident prediction
                best_prediction = max(label_confidence_pairs, key=lambda x: x[1])
                most_common_combined = best_prediction[0]

                if "-" in most_common_combined:
                    parsed_labels = most_common_combined.split("-", 1)
                    if len(parsed_labels) == 2:
                        final_primary_classification, final_secondary_classification = (
                            parsed_labels
                        )
                    else:
                        final_primary_classification = parsed_labels[0]
                        final_secondary_classification = "None"
                else:
                    final_primary_classification = most_common_combined
                    final_secondary_classification = "None"

        return {
            "primary_classification": final_primary_classification,
            "secondary_classification": final_secondary_classification,
            "average_weeb_score": np.mean(weeb_scores) if weeb_scores else 0.0,
            "average_furry_score": np.mean(furry_scores) if furry_scores else 0.0,
            "score_distribution": {
                "weeb_scores": weeb_scores,
                "furry_scores": furry_scores,
            },
            "model_predictions": model_predictions,
            "posts_analyzed": len(posts_for_model),
        }


class BlueskyUserDataset:
    @staticmethod
    def preprocess_and_save(input_csv: str, output_csv: str):
        if not os.path.exists(input_csv):
            raise FileNotFoundError(f"Input file {input_csv} not found")
        print(f"Loading data from {input_csv}...")
        try:
            df = pd.read_csv(input_csv, on_bad_lines="skip")
        except pd.errors.EmptyDataError:
            print(f"Error: Input file {input_csv} is empty. Cannot preprocess.")
            return
        except Exception as e:
            print(f"Error reading {input_csv}: {e}")
            return

        if "type" not in df.columns or "text" not in df.columns:
            print(
                f"Error: Input CSV {input_csv} must contain 'type' and 'text' columns."
            )
            return

        df = df[df["type"] == "post"]
        df = df.dropna(subset=["text"])
        df["text"] = df["text"].astype(str, errors="ignore")
        df = df[df["text"].apply(lambda x: isinstance(x, str) and x.strip() != "")]
        if df.empty:
            print(
                "No valid posts found after preprocessing. Output file will be empty or not created."
            )
            return
        print(f"Saving {len(df)} processed posts to {output_csv}...")
        df[["text", "type"]].to_csv(output_csv, index=False)
        print("Processed data saved successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="Bluesky User Classifier with Continuous Scoring"
    )
    subparsers = parser.add_subparsers(
        dest="command", help="Command to run", required=True
    )

    preprocess_parser = subparsers.add_parser(
        "preprocess", help="Preprocess Bluesky data CSV"
    )
    preprocess_parser.add_argument("--input", required=True, help="Input CSV file")
    preprocess_parser.add_argument("--output", required=True, help="Output CSV file")

    # Finetune command
    finetune_parser = subparsers.add_parser(
        "finetune", help="Fine-tune the language model"
    )
    finetune_parser.add_argument(
        "--data_csv", required=True, help="Path to data CSV for fine-tuning"
    )
    finetune_parser.add_argument(
        "--output_dir",
        default="output/user-classifier",
        help="Directory to save fine-tuned model",
    )
    finetune_parser.add_argument(
        "--base_model_name",
        default="unsloth/Qwen3-0.6B-unsloth-bnb-4bit",
        help="Base model name",
    )
    finetune_parser.add_argument(
        "--epochs", type=int, default=3, help="Training epochs"
    )
    finetune_parser.add_argument(
        "--batch_size", type=int, default=8, help="Training batch size"
    )
    finetune_parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate"
    )
    finetune_parser.add_argument(
        "--eval_split_size", type=float, default=0.2, help="Validation split size"
    )
    finetune_parser.add_argument(
        "--skip_eval_after_train",
        action="store_true",
        help="Skip evaluation after training",
    )
    finetune_parser.add_argument(
        "--chunk_size", type=int, default=50000, help="Chunk size for CSV reading"
    )
    finetune_parser.add_argument(
        "--min_threshold_weeb",
        type=float,
        default=0.0031,
        help="Minimum threshold for weeb classification",
    )
    finetune_parser.add_argument(
        "--min_threshold_furry",
        type=float,
        default=0.0034,
        help="Minimum threshold for furry classification",
    )
    finetune_parser.add_argument(
        "--strong_threshold_weeb",
        type=float,
        default=0.0047,
        help="Strong threshold for weeb classification",
    )
    finetune_parser.add_argument(
        "--strong_threshold_furry",
        type=float,
        default=0.0051,
        help="Strong threshold for furry classification",
    )
    finetune_parser.add_argument(
        "--max_training_samples",
        type=int,
        default=400000,
        help="Maximum number of training samples to use",
    )
    finetune_parser.add_argument(
        "--test_size",
        type=int,
        default=20000,
        help="Fixed size for test set instead of using ratio",
    )

    # Evaluate command
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a model")
    evaluate_parser.add_argument(
        "--model_path", required=True, help="Path to model for evaluation"
    )
    evaluate_parser.add_argument(
        "--eval_data_csv", required=True, help="Path to evaluation data CSV"
    )
    evaluate_parser.add_argument(
        "--metrics_output_dir",
        default="evaluation_metrics",
        help="Directory for metrics",
    )
    evaluate_parser.add_argument(
        "--batch_size", type=int, default=16, help="Evaluation batch size"
    )
    evaluate_parser.add_argument(
        "--chunk_size", type=int, default=50000, help="Chunk size for CSV reading"
    )
    evaluate_parser.add_argument(
        "--min_threshold_weeb",
        type=float,
        default=0.0031,
        help="Minimum threshold for weeb classification",
    )
    evaluate_parser.add_argument(
        "--min_threshold_furry",
        type=float,
        default=0.0034,
        help="Minimum threshold for furry classification",
    )
    evaluate_parser.add_argument(
        "--strong_threshold_weeb",
        type=float,
        default=0.0047,
        help="Strong threshold for weeb classification",
    )
    evaluate_parser.add_argument(
        "--strong_threshold_furry",
        type=float,
        default=0.0051,
        help="Strong threshold for furry classification",
    )
    evaluate_parser.add_argument(
        "--test_size",
        type=int,
        default=None,
        help="Number of evaluation samples to use after balancing",
    )

    # Classify command
    classify_parser = subparsers.add_parser("classify", help="Classify a Bluesky user")
    classify_parser.add_argument(
        "--model_path", required=True, help="Path to model for classification"
    )
    classify_parser.add_argument(
        "--username", required=True, help="Bluesky username (handle)"
    )
    classify_parser.add_argument(
        "--bluesky_user",
        default=os.environ.get("BLUESKY_USERNAME"),
        help="Your Bluesky login username",
    )
    classify_parser.add_argument(
        "--bluesky_pass",
        default=os.environ.get("BLUESKY_PASSWORD"),
        help="Your Bluesky login password",
    )
    classify_parser.add_argument(
        "--batch_size", type=int, default=16, help="Model classification batch size"
    )
    classify_parser.add_argument(
        "--min_threshold_weeb",
        type=float,
        default=0.0031,
        help="Minimum threshold for weeb classification",
    )
    classify_parser.add_argument(
        "--min_threshold_furry",
        type=float,
        default=0.0034,
        help="Minimum threshold for furry classification",
    )
    classify_parser.add_argument(
        "--strong_threshold_weeb",
        type=float,
        default=0.0047,
        help="Strong threshold for weeb classification",
    )
    classify_parser.add_argument(
        "--strong_threshold_furry",
        type=float,
        default=0.0051,
        help="Strong threshold for furry classification",
    )

    args = parser.parse_args()

    if args.command == "preprocess":
        BlueskyUserDataset.preprocess_and_save(args.input, args.output)

    elif args.command in ["finetune", "evaluate", "classify"]:
        # Initialize classifier with continuous parameters
        classifier_kwargs = {
            "model_name": args.model_path
            if args.command != "finetune"
            else args.base_model_name,
            "batch_size": args.batch_size,
            "min_threshold_weeb": getattr(args, "min_threshold_weeb", 0.0031),
            "min_threshold_furry": getattr(args, "min_threshold_furry", 0.0034),
            "strong_threshold_weeb": getattr(args, "strong_threshold_weeb", 0.0047),
            "strong_threshold_furry": getattr(args, "strong_threshold_furry", 0.0051),
        }

        classifier = BlueskyClassifier(**classifier_kwargs)

        # Set chunk_size as an attribute if provided
        if hasattr(args, "chunk_size") and args.chunk_size is not None:
            classifier.chunk_size = args.chunk_size

        if args.command == "finetune":
            classifier.finetune_model(
                training_data_csv=args.data_csv,
                output_dir=args.output_dir,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                eval_split_size=args.eval_split_size,
                skip_eval_after_train=args.skip_eval_after_train,
                max_training_samples=args.max_training_samples,
                test_size=args.test_size,
            )
        elif args.command == "evaluate":
            try:
                classifier.load_finetuned_model(args.model_path)
                eval_prepared_data = classifier._prepare_data_from_csv(
                    args.eval_data_csv
                )
                # --- Class balancing for evaluation ---
                if eval_prepared_data:
                    # Limit to test_size if provided
                    if (
                        args.test_size is not None
                        and len(eval_prepared_data) > args.test_size
                    ):
                        eval_prepared_data = random.sample(
                            eval_prepared_data, k=args.test_size
                        )
                        print(
                            f"Randomly sampled {args.test_size} evaluation samples after balancing."
                        )

                if eval_prepared_data:
                    classifier._evaluate_model_and_print_metrics(
                        eval_prepared_data, args.metrics_output_dir
                    )
                else:
                    print(
                        f"No data prepared from {args.eval_data_csv}. Skipping evaluation."
                    )
            except (RuntimeError, FileNotFoundError) as e:
                print(f"Error during evaluation: {e}")

        elif args.command == "classify":
            import asyncio

            async def run_classification():
                try:
                    classifier.load_finetuned_model(args.model_path)
                except RuntimeError as e:
                    print(f"Failed to load model from '{args.model_path}': {e}")
                    return
                if not args.bluesky_user or not args.bluesky_pass:
                    print("Bluesky login credentials not provided.")
                    return
                temp_client = Client()
                try:
                    print(f"Logging in to Bluesky as {args.bluesky_user}...")
                    temp_client.login(args.bluesky_user, args.bluesky_pass)
                    classifier.client = temp_client
                    print("Successfully logged in to Bluesky.")
                except Exception as e:
                    print(f"Failed to log in to Bluesky: {e}")
                    return
                posts = await classifier.fetch_user_posts(args.username)
                result = classifier.classify_user(posts)

                # --- New section to add summary data to the result dict ---
                # Calculate prediction distribution
                if result.get("model_predictions"):
                    from collections import Counter

                    pred_counts = Counter(
                        [p["label"] for p in result["model_predictions"]]
                    )
                    distribution = []
                    total_preds = len(result["model_predictions"])
                    for pred, count in pred_counts.most_common():
                        avg_conf = np.mean(
                            [
                                p["primary_confidence"]
                                for p in result["model_predictions"]
                                if p["label"] == pred
                            ]
                        )
                        distribution.append(
                            {
                                "label": pred,
                                "count": count,
                                "percentage": round((count / total_preds) * 100, 2),
                                "average_confidence": round(avg_conf, 4),
                            }
                        )
                    result["prediction_distribution"] = distribution

                # Calculate score distribution summary
                weeb_scores = result["score_distribution"]["weeb_scores"]
                furry_scores = result["score_distribution"]["furry_scores"]
                if weeb_scores and furry_scores:
                    result["score_summary"] = {
                        "weeb": {
                            "min": round(min(weeb_scores), 6),
                            "max": round(max(weeb_scores), 6),
                            "mean": round(np.mean(weeb_scores), 6),
                            "std_dev": round(np.std(weeb_scores), 6),
                        },
                        "furry": {
                            "min": round(min(furry_scores), 6),
                            "max": round(max(furry_scores), 6),
                            "mean": round(np.mean(furry_scores), 6),
                            "std_dev": round(np.std(furry_scores), 6),
                        },
                    }
                # We don't need the raw list of scores in the final JSON
                # as it can be very large. The summary is more useful.
                del result["score_distribution"]
                del result["model_predictions"]  # Also potentially very large

                # --- Print the final result as a JSON object ---
                print(json.dumps(result, indent=2))

            asyncio.run(run_classification())
    else:
        parser.print_help()


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print(
            "Error: Unsloth library not found. Please install, e.g., 'pip install \"unsloth[cu1xx-ampere-torch210]\"' or 'pip install unsloth'."
        )
        sys.exit(1)
    try:
        from trl import SFTTrainer, SFTConfig
    except ImportError:
        print("Error: TRL library not found. Please install with 'pip install trl'.")
        sys.exit(1)

    main()
