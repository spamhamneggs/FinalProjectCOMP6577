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
import re

# For Bluesky/ATProto
from atproto import Client  # type: ignore

# For metrics
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.metrics import classification_report, confusion_matrix  # type: ignore

# For classification
from dotenv import load_dotenv

load_dotenv()

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# --- Heuristic Labeling Logic ---


def normalize_text(s: str) -> str:
    """
    Normalize text by collapsing repeated characters and whitespace.
    """
    s = re.sub(r"(.)\1{2,}", r"\1\1", s)  # Collapse repeated chars: "soooo" → "soo"
    s = re.sub(r"\s+", " ", s)  # Collapse whitespace
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


def determine_labels_from_scores_and_cutoffs(
    w_score: float,
    f_score: float,
    weeb_normie_cutoff: float,
    weeb_strong_cutoff: float,
    furry_normie_cutoff: float,
    furry_strong_cutoff: float,
) -> Tuple[str, str]:
    """
    Determine primary and secondary classification labels based on weeb and furry scores,
    using fixed absolute cutoff scores.
    """
    # Step 1: Determine the strength category for Weeb score
    weeb_strength_category = "Normie"
    if w_score > weeb_strong_cutoff:
        weeb_strength_category = "Strong"
    elif w_score > weeb_normie_cutoff:
        weeb_strength_category = "Slight"

    # Step 2: Determine the strength category for Furry score
    furry_strength_category = "Normie"
    if f_score > furry_strong_cutoff:
        furry_strength_category = "Strong"
    elif f_score > furry_normie_cutoff:
        furry_strength_category = "Slight"

    # Step 3: Combine these independent strength assessments
    primary_label = "Normie"
    secondary_label = "None"

    if weeb_strength_category == "Normie" and furry_strength_category == "Normie":
        primary_label = "Normie"
        secondary_label = "None"
    elif weeb_strength_category != "Normie" and furry_strength_category == "Normie":
        primary_label = "Weeb" if weeb_strength_category == "Strong" else "Slight Weeb"
        secondary_label = "None"
    elif furry_strength_category != "Normie" and weeb_strength_category == "Normie":
        primary_label = (
            "Furry" if furry_strength_category == "Strong" else "Slight Furry"
        )
        secondary_label = "None"
    else:  # Both have signal (neither is "Normie" strength for its category)
        if w_score >= f_score:  # Weeb is dominant or scores are equal
            primary_label = (
                "Weeb" if weeb_strength_category == "Strong" else "Slight Weeb"
            )
            if furry_strength_category == "Strong":
                secondary_label = "Furry"
            elif furry_strength_category == "Slight":
                secondary_label = "Slight Furry"
            else:
                secondary_label = "None"
        else:  # Furry is dominant
            primary_label = (
                "Furry" if furry_strength_category == "Strong" else "Slight Furry"
            )
            if weeb_strength_category == "Strong":
                secondary_label = "Weeb"
            elif weeb_strength_category == "Slight":
                secondary_label = "Slight Weeb"
            else:
                secondary_label = "None"

        primary_base_type = (
            "Weeb"
            if "Weeb" in primary_label
            else ("Furry" if "Furry" in primary_label else "Normie")
        )
        secondary_base_type = (
            "Weeb"
            if "Weeb" in secondary_label
            else ("Furry" if "Furry" in secondary_label else "None")
        )
        if (
            primary_base_type != "Normie"
            and primary_base_type == secondary_base_type
            and secondary_base_type != "None"
        ):
            secondary_label = "None"

    # If primary is a "Slight" classification, force secondary to "None"
    if primary_label.startswith("Slight "):
        secondary_label = "None"

    return primary_label, secondary_label


def process_post_for_training(args) -> Dict[str, Any] | None:
    """
    Process a single post for training data generation.
    Uses fixed absolute cutoffs and includes cardinal scores in the prompt.
    """
    (
        text,
        weeb_terms_df,
        furry_terms_df,
        weeb_normie_cutoff,
        weeb_strong_cutoff,
        furry_normie_cutoff,
        furry_strong_cutoff,
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

    primary_label, secondary_label = determine_labels_from_scores_and_cutoffs(
        weeb_score,
        furry_score,
        weeb_normie_cutoff,
        weeb_strong_cutoff,
        furry_normie_cutoff,
        furry_strong_cutoff,
    )

    prompt_text_content = str(text)
    # Modified prompt to include cardinal scores as input
    prompt = (
        "Analyze this Bluesky post using the provided scores and content. "
        "Respond ONLY with:\n"
        "Primary Classification: <label>\n"
        "Secondary Classification: <label>\n"
        "Valid labels: Normie, Weeb, Furry, Slight Weeb, Slight Furry, None.\n\n"
        f"Weeb Score: {weeb_score:.6f}\n"
        f"Furry Score: {furry_score:.6f}\n"
        f"Post Content: {prompt_text_content}"
    )
    response = (
        f"Primary Classification: {primary_label}\n"
        f"Secondary Classification: {secondary_label}"
    )

    return {
        "prompt": prompt,
        "response": response,
        "true_label_heuristic": f"{primary_label}-{secondary_label}",
        "weeb_score": weeb_score,
        "furry_score": furry_score,
    }


class BlueskyClassifier:
    def __init__(
        self,
        model_name="unsloth/Qwen3-0.6B-unsloth-bnb-4bit",
        batch_size=8,
        # Fixed absolute score cutoffs
        weeb_normie_cutoff=0.01,
        weeb_strong_cutoff=0.05,
        furry_normie_cutoff=0.015,
        furry_strong_cutoff=0.06,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None
        self.tokenizer = None
        self.chunk_size = 100000

        # Fixed absolute cutoff scores
        self.weeb_normie_cutoff_score_ = weeb_normie_cutoff
        self.weeb_strong_cutoff_score_ = weeb_strong_cutoff
        self.furry_normie_cutoff_score_ = furry_normie_cutoff
        self.furry_strong_cutoff_score_ = furry_strong_cutoff

        print(f"\n--- Using Fixed Absolute Score Cutoffs ---")
        print(f"  Weeb Normie Cutoff: {self.weeb_normie_cutoff_score_:.6f}")
        print(f"  Weeb Strong Cutoff: {self.weeb_strong_cutoff_score_:.6f}")
        print(f"  Furry Normie Cutoff: {self.furry_normie_cutoff_score_:.6f}")
        print(f"  Furry Strong Cutoff: {self.furry_strong_cutoff_score_:.6f}")
        print("-------------------------------------------\n")

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
                dtype=None,
                load_in_4bit=True,
            )

            # Force left padding more aggressively
            self.tokenizer.padding_side = "left"
            self.tokenizer.truncation_side = "left"  # Add this too

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
                r=16,
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
                        self.weeb_normie_cutoff_score_,
                        self.weeb_strong_cutoff_score_,
                        self.furry_normie_cutoff_score_,
                        self.furry_strong_cutoff_score_,
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

    def _parse_combined_label_from_text(
        self, text: str, source_type: str = "unknown source"
    ) -> str:
        primary, secondary = "Unknown", "Unknown"
        for line in text.splitlines():
            line_lower = line.lower()
            if "primary classification:" in line_lower:
                lab = line.split(":", 1)[1].strip().title()
                if lab in self.primary_labels_set:
                    primary = lab
            if "secondary classification:" in line_lower:
                lab = line.split(":", 1)[1].strip().title()
                if lab in self.secondary_labels_set:
                    secondary = lab
        if primary != "Unknown" and secondary == "Unknown":
            secondary = "None"
        if primary == "Normie":
            secondary = "None"
        if primary == "Unknown":
            secondary = "Unknown"
        return f"{primary}-{secondary}"

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
                    max_new_tokens=35,
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
                predicted_label = self._parse_combined_label_from_text(
                    model_response_part, "evaluation output"
                )
                y_true_combined.append(true_labels[i])
                y_pred_combined.append(predicted_label)

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
        report_path = os.path.join(metrics_output_dir, "classification_report.txt")
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

    def finetune_model(
        self,
        training_data_csv: str,
        output_dir: str = "finetuned_model",
        epochs: int = 3,
        learning_rate: float = 1e-5,
        eval_split_size: float = 0.20,
        skip_eval_after_train: bool = False,
        max_training_samples: int = None,
        test_size: int = None,
    ):
        """
        Fine-tune the model on custom data.

        Args:
            max_training_samples (int, optional): Maximum number of training samples to use
            test_size (int, optional): Fixed size for test set instead of using ratio
        """
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

        if (
            max_training_samples
            and test_size
            and len(all_prepared_data) >= (max_training_samples + test_size)
        ):
            total_samples = max_training_samples + test_size
            print(
                f"Sampling exactly {max_training_samples} train and {test_size} eval examples from {len(all_prepared_data)} total examples"
            )
            sampled_data = random.sample(all_prepared_data, total_samples)
            # Optionally shuffle for randomness
            random.shuffle(sampled_data)
            train_data_dicts = sampled_data[:max_training_samples]
            eval_data_dicts = sampled_data[max_training_samples:]
            skip_eval_after_train = False
        else:
            labels_for_stratification = [
                item["true_label_heuristic"] for item in all_prepared_data
            ]

            if len(all_prepared_data) > 1:
                if test_size:
                    train_data_dicts, eval_data_dicts = train_test_split(
                        all_prepared_data,
                        test_size=test_size,
                        random_state=RANDOM_SEED,
                        stratify=labels_for_stratification,
                    )
                else:
                    train_data_dicts, eval_data_dicts = train_test_split(
                        all_prepared_data,
                        test_size=eval_split_size,
                        random_state=RANDOM_SEED,
                        stratify=labels_for_stratification,
                    )
            else:
                # Fallback to proportional splitting
                labels_for_stratification = [
                    item["true_label_heuristic"] for item in all_prepared_data
                ]

                if len(all_prepared_data) > 1 and eval_split_size > 0:
                    try:
                        train_data_dicts, eval_data_dicts = train_test_split(
                            all_prepared_data,
                            test_size=eval_split_size,
                            random_state=RANDOM_SEED,
                            stratify=labels_for_stratification,
                        )
                    except ValueError:
                        # Stratification failed, split without it
                        train_data_dicts, eval_data_dicts = train_test_split(
                            all_prepared_data,
                            test_size=eval_split_size,
                            random_state=RANDOM_SEED,
                        )
                else:
                    train_data_dicts = all_prepared_data
                    eval_data_dicts = []
                    skip_eval_after_train = True

            print(
                f"Training with {len(train_data_dicts)} samples, evaluating with {len(eval_data_dicts)} samples."
            )

            if not train_data_dicts:
                print(
                    "No training samples available after splitting. Aborting fine-tuning."
                )
                return None

        # Oversample using sample weights (WeightedRandomSampler logic, but in-memory)
        label_counts = pd.Series(
            [ex["true_label_heuristic"] for ex in train_data_dicts]
        ).value_counts()
        label_to_weight = {label: 1.0 / count for label, count in label_counts.items()}
        sample_weights = [
            label_to_weight[ex["true_label_heuristic"]] for ex in train_data_dicts
        ]
        n_samples = len(train_data_dicts)
        indices = random.choices(range(n_samples), weights=sample_weights, k=n_samples)
        oversampled_train_data = [train_data_dicts[i] for i in indices]

        sft_train_dataset = Dataset.from_list(
            [
                {"prompt": d["prompt"], "response": d["response"]}
                for d in oversampled_train_data
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
            gradient_accumulation_steps=2,
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
            max_seq_length=256,
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
            dataloader_pin_memory=False,  # add this
            group_by_length=True,  # add this for efficiency
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
        print("Starting model fine-tuning...")
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
            metrics_parent_dir = (
                os.path.dirname(output_dir) if os.path.dirname(output_dir) else "."
            )
            eval_metrics_dir = os.path.join(
                metrics_parent_dir,
                "metrics_after_finetune",
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

    def load_finetuned_model(self, model_path: str):
        print(f"Attempting to load fine-tuned model from {model_path}")
        try:
            self.setup_model(model_name_or_path=model_path)
            self.model_name = model_path
        except RuntimeError as e:
            print(f"Failed to load fine-tuned model from {model_path}: {e}")
            raise

    async def fetch_user_posts(self, user_handle: str, limit: int = 1000) -> List[str]:
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
        LLM-only classification using all available posts.
        Defaults to "Normie-None" if final classification is "Unknown-Unknown".
        """
        if not posts:
            return {
                "primary_classification": "Normie",
                "secondary_classification": "None",
                "model_combined_labels_debug": [],
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

        # Use ALL posts (no limit of 10)
        posts_for_model = [p for p in safe_posts if p.strip()]

        model_predicted_combined_labels = []

        # Generate prompts with scores for ALL posts
        prompts_for_model = []
        for post_text in posts_for_model:
            # Calculate individual post scores for the model
            post_weeb_score = self._calculate_category_score(post_text, self.weeb_terms)
            post_furry_score = self._calculate_category_score(
                post_text, self.furry_terms
            )

            prompt = (
                f"<|user|>\nAnalyze this Bluesky post using the provided scores and content. "
                f"Respond ONLY with:\n"
                f"Primary Classification: <label>\n"
                f"Secondary Classification: <label>\n"
                f"Valid labels: Normie, Weeb, Furry, Slight Weeb, Slight Furry, None.\n\n"
                f"Weeb Score: {post_weeb_score:.6f}\n"
                f"Furry Score: {post_furry_score:.6f}\n"
                f"Post Content: {post_text}\n<|assistant|>"
            )
            prompts_for_model.append(prompt)

        if prompts_for_model:
            num_model_batches = math.ceil(len(prompts_for_model) / self.batch_size)
            for batch_idx in tqdm(
                range(num_model_batches),
                desc=f"Classifying {len(posts_for_model)} user posts with LLM",
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
                        max_new_tokens=35,
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
                    predicted_combined = self._parse_combined_label_from_text(
                        model_response_part, "model output for user classification"
                    )
                    model_predicted_combined_labels.append(predicted_combined)

        # Determine final classification from LLM predictions only
        final_primary_classification = "Normie"
        final_secondary_classification = "None"

        if model_predicted_combined_labels:
            # Filter out "Unknown-Unknown" predictions for determining the final label
            valid_predictions = [
                label
                for label in model_predicted_combined_labels
                if label != "Unknown-Unknown"
            ]

            if valid_predictions:
                # Use the most common valid prediction
                most_common_combined = max(
                    set(valid_predictions),
                    key=valid_predictions.count,
                )
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
            # If all predictions were "Unknown-Unknown", keep default "Normie-None"

        return {
            "primary_classification": final_primary_classification,
            "secondary_classification": final_secondary_classification,
            "model_combined_labels_debug": model_predicted_combined_labels,
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
        description="Bluesky User Classifier with Fixed Cutoffs and LLM-only Classification"
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
        default="finetuned_model",
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
        "--learning_rate", type=float, default=1e-5, help="Learning rate"
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
        "--chunk_size_csv", type=int, default=50000, help="Chunk size for CSV reading"
    )
    # Fixed cutoff arguments
    finetune_parser.add_argument(
        "--weeb-normie-cutoff",
        type=float,
        required=True,
        help="Fixed absolute cutoff for Weeb 'Normie' classification (default: 0.01)",
    )
    finetune_parser.add_argument(
        "--weeb-strong-cutoff",
        type=float,
        required=True,
        help="Fixed absolute cutoff for Weeb 'Strong' classification (default: 0.05)",
    )
    finetune_parser.add_argument(
        "--furry-normie-cutoff",
        type=float,
        required=True,
        help="Fixed absolute cutoff for Furry 'Normie' classification (default: 0.015)",
    )
    finetune_parser.add_argument(
        "--furry-strong-cutoff",
        type=float,
        required=True,
        help="Fixed absolute cutoff for Furry 'Strong' classification (default: 0.06)",
    )

    finetune_parser.add_argument(
        "--max_training_samples",
        type=int,
        default=400000,
        help="Maximum number of training samples to use (default: use 400,000)",
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
        "--chunk_size_csv", type=int, default=50000, help="Chunk size for CSV reading"
    )
    # Fixed cutoff arguments for evaluation
    evaluate_parser.add_argument(
        "--weeb-normie-cutoff",
        type=float,
        default=0.01,
        help="Fixed absolute cutoff for Weeb 'Normie' classification (default: 0.01)",
    )
    evaluate_parser.add_argument(
        "--weeb-strong-cutoff",
        type=float,
        default=0.05,
        help="Fixed absolute cutoff for Weeb 'Strong' classification (default: 0.05)",
    )
    evaluate_parser.add_argument(
        "--furry-normie-cutoff",
        type=float,
        default=0.015,
        help="Fixed absolute cutoff for Furry 'Normie' classification (default: 0.015)",
    )
    evaluate_parser.add_argument(
        "--furry-strong-cutoff",
        type=float,
        default=0.06,
        help="Fixed absolute cutoff for Furry 'Strong' classification (default: 0.06)",
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
        "--batch_size", type=int, default=8, help="Model classification batch size"
    )

    args = parser.parse_args()

    # Validate cutoffs: strong >= normie for each category
    if args.command in ["finetune", "evaluate"]:
        if args.weeb_strong_cutoff < args.weeb_normie_cutoff:
            parser.error(
                "--weeb-strong-cutoff cannot be less than --weeb-normie-cutoff"
            )
        if args.furry_strong_cutoff < args.furry_normie_cutoff:
            parser.error(
                "--furry-strong-cutoff cannot be less than --furry-normie-cutoff"
            )

    if args.command == "preprocess":
        BlueskyUserDataset.preprocess_and_save(args.input, args.output)

    elif args.command in ["finetune", "evaluate", "classify"]:
        # Initialize classifier with model_name and batch_size
        classifier_kwargs = {
            "model_name": args.model_path
            if args.command != "finetune"
            else args.base_model_name,
            "batch_size": args.batch_size,
        }

        # Add cutoff arguments for finetune and evaluate commands
        if args.command in ["finetune", "evaluate"]:
            classifier_kwargs.update(
                {
                    "weeb_normie_cutoff": args.weeb_normie_cutoff,
                    "weeb_strong_cutoff": args.weeb_strong_cutoff,
                    "furry_normie_cutoff": args.furry_normie_cutoff,
                    "furry_strong_cutoff": args.furry_strong_cutoff,
                }
            )

        classifier = BlueskyClassifier(**classifier_kwargs)

        if hasattr(args, "chunk_size_csv") and args.chunk_size_csv is not None:
            classifier.chunk_size = args.chunk_size_csv

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
                if not posts:
                    result = {
                        "primary_classification": "Normie",
                        "secondary_classification": "None",
                        "model_combined_labels_debug": [],
                    }
                else:
                    result = classifier.classify_user(posts)
                print(
                    "\n"
                    + "=" * 50
                    + f"\nLLM-Only Classification Results for @{args.username}\n"
                    + "=" * 50
                )
                print(f"Primary Classification: {result['primary_classification']}")
                print(f"Secondary Classification: {result['secondary_classification']}")
                if result.get("model_combined_labels_debug"):
                    print(
                        f"Model Post Classifications: {len(result['model_combined_labels_debug'])} posts analyzed"
                    )
                    # Show distribution of predictions
                    from collections import Counter

                    pred_counts = Counter(result["model_combined_labels_debug"])
                    print("Prediction distribution:")
                    for pred, count in pred_counts.most_common():
                        percentage = (
                            count / len(result["model_combined_labels_debug"])
                        ) * 100
                        print(f"  {pred}: {count} ({percentage:.1f}%)")
                print("=" * 50 + "\n")

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
