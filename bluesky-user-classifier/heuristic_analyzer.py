#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
from tqdm import tqdm
import re
import json

# --- Configuration ---
# Default paths for term databases, can be overridden by command-line arguments
DEFAULT_WEEB_TERMS_PATH = "output/terms-analysis-bertopic/weeb_terms_bertopic.csv"
DEFAULT_FURRY_TERMS_PATH = "output/terms-analysis-bertopic/furry_terms_bertopic.csv"
DEFAULT_OUTPUT_DIR = "heuristic_analysis_results_category_specific_percentile"


# --- Reusable Heuristic Logic ---
def normalize_text(s: str) -> str:
    s = re.sub(r"(.)\1{2,}", r"\1\1", s)  # "soooo" → "soo"
    s = re.sub(r"\s+", " ", s)  # Collapse whitespace
    return s.strip()


def load_term_database(csv_file: str) -> pd.DataFrame:
    """
    Load and process a terms CSV file.
    Expects 'term' and 'combined_score' columns.
    """
    if not os.path.exists(csv_file):
        print(f"Warning: Term database {csv_file} not found. Scores might be affected.")
        return pd.DataFrame(columns=["term", "combined_score"])
    try:
        df = pd.read_csv(csv_file)
        if "term" not in df.columns or "combined_score" not in df.columns:
            print(f"Warning: {csv_file} is missing 'term' or 'combined_score' column.")
            df["term"] = df.get("term", pd.Series(dtype="str"))
            df["combined_score"] = df.get("combined_score", pd.Series(dtype="float"))

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


def determine_classification_labels_category_specific(
    w_score: float,
    f_score: float,
    weeb_normie_cutoff: float,
    weeb_strong_cutoff: float,
    furry_normie_cutoff: float,
    furry_strong_cutoff: float,
) -> Tuple[str, str]:
    """
    Determine primary and secondary classification labels based on weeb and furry scores,
    using category-specific cutoff scores derived from their respective percentiles.
    Prevents "Slight Weeb-Furry" or "Slight Furry-Weeb" as a combined label.
    """
    # Step 1: Determine the strength category for Weeb score
    weeb_strength_category = "Normie"  # Default if below normie cutoff
    if w_score > weeb_strong_cutoff:
        weeb_strength_category = "Strong"
    elif w_score > weeb_normie_cutoff:  # And <= weeb_strong_cutoff
        weeb_strength_category = "Slight"

    # Step 2: Determine the strength category for Furry score
    furry_strength_category = "Normie"  # Default if below normie cutoff
    if f_score > furry_strong_cutoff:
        furry_strength_category = "Strong"
    elif f_score > furry_normie_cutoff:  # And <= furry_strong_cutoff
        furry_strength_category = "Slight"

    # Step 3: Combine these independent strength assessments
    primary_label = "Normie"
    secondary_label = "None"

    # Case A: Both categories are "Normie" strength
    if weeb_strength_category == "Normie" and furry_strength_category == "Normie":
        primary_label = "Normie"
        secondary_label = "None"

    # Case B: Weeb has signal, Furry is "Normie" strength
    elif weeb_strength_category != "Normie" and furry_strength_category == "Normie":
        if weeb_strength_category == "Strong":
            primary_label = "Weeb"
        else:  # Slight
            primary_label = "Slight Weeb"
        secondary_label = "None"

    # Case C: Furry has signal, Weeb is "Normie" strength
    elif furry_strength_category != "Normie" and weeb_strength_category == "Normie":
        if furry_strength_category == "Strong":
            primary_label = "Furry"
        else:  # Slight
            primary_label = "Slight Furry"
        secondary_label = "None"

    # Case D: Both Weeb and Furry have signal (neither is "Normie" strength)
    else:
        # Determine overall dominance by comparing raw scores
        if w_score >= f_score:  # Weeb is dominant or scores are equal
            # Assign Weeb as primary based on its strength
            if weeb_strength_category == "Strong":
                primary_label = "Weeb"
            else:
                primary_label = "Slight Weeb"  # Must be "Slight" as it's not "Normie"

            # Assign Furry as secondary based on its strength
            if furry_strength_category == "Strong":
                secondary_label = "Furry"
            elif furry_strength_category == "Slight":
                secondary_label = "Slight Furry"
            else:
                secondary_label = (
                    "None"  # Should not happen if furry_strength_category != "Normie"
                )

        else:  # Furry is dominant
            # Assign Furry as primary based on its strength
            if furry_strength_category == "Strong":
                primary_label = "Furry"
            else:
                primary_label = "Slight Furry"  # Must be "Slight"

            # Assign Weeb as secondary based on its strength
            if weeb_strength_category == "Strong":
                secondary_label = "Weeb"
            elif weeb_strength_category == "Slight":
                secondary_label = "Slight Weeb"
            else:
                secondary_label = "None"  # Should not happen

        # Final check: Prevent self-secondary if base types are the same
        # (e.g. "Weeb" primary and "Slight Weeb" secondary is not desired)
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

        # Prevent "Slight Weeb-Furry" or "Slight Furry-Weeb" as a combined label
        # If primary is "Slight Weeb" or "Slight Furry", secondary must be "None"
        if primary_label in ("Slight Weeb", "Slight Furry"):
            secondary_label = "None"

    return primary_label, secondary_label


# --- Main Analysis Function ---
def analyze_heuristics(
    data_csv_path: str,
    weeb_terms_df: pd.DataFrame,
    furry_terms_df: pd.DataFrame,
    output_dir_base: str,
    weeb_normie_p: int,  # Weeb normie percentile
    weeb_strong_p: int,  # Weeb strong percentile
    furry_normie_p: int,  # Furry normie percentile
    furry_strong_p: int,  # Furry strong percentile
    force_recalculate_scores: bool = False,
):
    output_subdir_name = (
        f"w_norm_p{weeb_normie_p}_str_p{weeb_strong_p}_"
        f"f_norm_p{furry_normie_p}_str_p{furry_strong_p}"
    )
    current_output_dir = os.path.join(output_dir_base, output_subdir_name)
    if not os.path.exists(current_output_dir):
        os.makedirs(current_output_dir)
        print(f"Created output subdirectory: {current_output_dir}")

    scores_csv_path = os.path.join(output_dir_base, "calculated_scores_for_data.csv")
    results_df = None

    if not force_recalculate_scores and os.path.exists(scores_csv_path):
        print(f"Loading pre-calculated scores from {scores_csv_path}...")
        try:
            results_df = pd.read_csv(scores_csv_path)
            if not {"text_snippet", "weeb_score", "furry_score"}.issubset(
                results_df.columns
            ):
                print(
                    "Pre-calculated scores file is missing required columns. Recalculating."
                )
                results_df = None
        except Exception as e:
            print(f"Error loading pre-calculated scores: {e}. Recalculating.")
            results_df = None

    if results_df is None:
        print(f"Calculating scores from input data: {data_csv_path}...")
        print("This may take a long time for large datasets.")
        try:
            df_text_input = pd.read_csv(
                data_csv_path, usecols=["text"], on_bad_lines="skip"
            )
            df_text_input.dropna(subset=["text"], inplace=True)
            df_text_input["text"] = df_text_input["text"].astype(str)
        except Exception as e:
            print(f"Error loading or processing input data CSV {data_csv_path}: {e}")
            return

        if df_text_input.empty:
            print("No text data found in the input CSV. Aborting analysis.")
            return

        print(f"Calculating weeb/furry scores for {len(df_text_input)} posts...")
        scores_data = []
        for _, row in tqdm(
            df_text_input.iterrows(),
            total=len(df_text_input),
            desc="Calculating scores",
        ):
            text = row["text"]
            if not text.strip():
                continue
            w_score = calculate_category_score(text, weeb_terms_df)
            f_score = calculate_category_score(text, furry_terms_df)
            scores_data.append(
                {
                    "text_snippet": text[:100] + "..." if len(text) > 100 else text,
                    "weeb_score": w_score,
                    "furry_score": f_score,
                }
            )

        if not scores_data:
            print("No scores were generated. Check input data or term lists.")
            return
        results_df = pd.DataFrame(scores_data)
        print(f"Saving calculated scores to {scores_csv_path}...")
        results_df.to_csv(scores_csv_path, index=False)
        print("Scores saved.")

    # Determine Category-Specific Thresholds from Percentiles
    print("\nDetermining category-specific thresholds based on score percentiles...")

    # For Weeb scores
    actual_weeb_normie_cutoff = 0.0
    actual_weeb_strong_cutoff = 0.0
    if not results_df["weeb_score"].empty:
        actual_weeb_normie_cutoff = np.percentile(
            results_df["weeb_score"], weeb_normie_p
        )
        actual_weeb_strong_cutoff = np.percentile(
            results_df["weeb_score"], weeb_strong_p
        )
        if actual_weeb_strong_cutoff < actual_weeb_normie_cutoff:
            print(
                f"Warning: Weeb strong percentile ({weeb_strong_p}th) score "
                f"({actual_weeb_strong_cutoff:.6f}) is less than normie ({weeb_normie_p}th) score "
                f"({actual_weeb_normie_cutoff:.6f}). Adjusting weeb_strong_cutoff to be at least weeb_normie_cutoff."
            )
            actual_weeb_strong_cutoff = actual_weeb_normie_cutoff
        print(
            f"  Weeb Normie Cutoff (at {weeb_normie_p}th percentile of weeb_scores): {actual_weeb_normie_cutoff:.6f}"
        )
        print(
            f"  Weeb Strong Cutoff (at {weeb_strong_p}th percentile of weeb_scores): {actual_weeb_strong_cutoff:.6f}"
        )
    else:
        print(
            "Warning: No weeb_scores to calculate percentiles from. Using 0.0 for weeb cutoffs."
        )

    # For Furry scores
    actual_furry_normie_cutoff = 0.0
    actual_furry_strong_cutoff = 0.0
    if not results_df["furry_score"].empty:
        actual_furry_normie_cutoff = np.percentile(
            results_df["furry_score"], furry_normie_p
        )
        actual_furry_strong_cutoff = np.percentile(
            results_df["furry_score"], furry_strong_p
        )
        if actual_furry_strong_cutoff < actual_furry_normie_cutoff:
            print(
                f"Warning: Furry strong percentile ({furry_strong_p}th) score "
                f"({actual_furry_strong_cutoff:.6f}) is less than normie ({furry_normie_p}th) score "
                f"({actual_furry_normie_cutoff:.6f}). Adjusting furry_strong_cutoff to be at least furry_normie_cutoff."
            )
            actual_furry_strong_cutoff = actual_furry_normie_cutoff
        print(
            f"  Furry Normie Cutoff (at {furry_normie_p}th percentile of furry_scores): {actual_furry_normie_cutoff:.6f}"
        )
        print(
            f"  Furry Strong Cutoff (at {furry_strong_p}th percentile of furry_scores): {actual_furry_strong_cutoff:.6f}"
        )
    else:
        print(
            "Warning: No furry_scores to calculate percentiles from. Using 0.0 for furry cutoffs."
        )

    # Save thresholds to JSON file
    thresholds = {
        "weeb_normie_cutoff": float(actual_weeb_normie_cutoff),
        "weeb_strong_cutoff": float(actual_weeb_strong_cutoff),
        "furry_normie_cutoff": float(actual_furry_normie_cutoff),
        "furry_strong_cutoff": float(actual_furry_strong_cutoff),
        "weeb_normie_percentile": weeb_normie_p,
        "weeb_strong_percentile": weeb_strong_p,
        "furry_normie_percentile": furry_normie_p,
        "furry_strong_percentile": furry_strong_p,
    }
    thresholds_json_path = os.path.join(current_output_dir, "category_thresholds.json")
    with open(thresholds_json_path, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)
    print(f"Saved category-specific thresholds to {thresholds_json_path}")

    # Apply Classification Labels using Category-Specific Dynamic Thresholds
    print(
        "\nApplying classification labels using category-specific dynamic thresholds..."
    )
    labels_list = []
    for _, row in tqdm(
        results_df.iterrows(),
        total=len(results_df),
        desc="Applying category-specific labels",
    ):
        primary_label, secondary_label = (
            determine_classification_labels_category_specific(
                row["weeb_score"],
                row["furry_score"],
                weeb_normie_cutoff=actual_weeb_normie_cutoff,
                weeb_strong_cutoff=actual_weeb_strong_cutoff,
                furry_normie_cutoff=actual_furry_normie_cutoff,
                furry_strong_cutoff=actual_furry_strong_cutoff,
            )
        )
        labels_list.append(
            {
                "primary_label": primary_label,
                "secondary_label": secondary_label,
                "combined_label": f"{primary_label}-{secondary_label}",
            }
        )

    labels_df = pd.DataFrame(labels_list)
    final_results_df = pd.concat(
        [results_df.reset_index(drop=True), labels_df.reset_index(drop=True)], axis=1
    )

    # Output Generation
    print(f"\nSaving analysis results to {current_output_dir}...")

    label_distribution = (
        final_results_df["combined_label"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )
    print("\n--- Heuristic Label Distribution (%) ---")
    print(label_distribution)
    label_distribution.to_csv(
        os.path.join(current_output_dir, "label_distribution.csv")
    )

    score_summary = results_df[["weeb_score", "furry_score"]].describe()
    print("\n--- Score Summary Statistics ---")
    print(score_summary)
    score_summary.to_csv(os.path.join(current_output_dir, "score_summary_stats.csv"))

    plt.figure(figsize=(12, 5))
    plt.suptitle(
        f"Score Distributions (Percentiles: WN{weeb_normie_p},WS{weeb_strong_p}, FN{furry_normie_p},FS{furry_strong_p})",
        fontsize=12,
    )
    plt.subplot(1, 2, 1)
    sns.histplot(results_df["weeb_score"], kde=True, bins=30)
    plt.title("Weeb Score Distribution")
    plt.xlabel("Normalized Weeb Score")
    plt.ylabel("Frequency")
    plt.axvline(
        actual_weeb_normie_cutoff,
        color="orange",
        linestyle="--",
        label=f"Normie Cutoff ({actual_weeb_normie_cutoff:.4f}) @ P{weeb_normie_p}",
    )
    plt.axvline(
        actual_weeb_strong_cutoff,
        color="red",
        linestyle="--",
        label=f"Strong Cutoff ({actual_weeb_strong_cutoff:.4f}) @ P{weeb_strong_p}",
    )
    plt.xlim(right=0.02)
    plt.legend(fontsize="small")

    plt.subplot(1, 2, 2)
    sns.histplot(results_df["furry_score"], kde=True, bins=30)
    plt.title("Furry Score Distribution")
    plt.xlabel("Normalized Furry Score")
    plt.ylabel("Frequency")
    plt.axvline(
        actual_furry_normie_cutoff,
        color="orange",
        linestyle="--",
        label=f"Normie Cutoff ({actual_furry_normie_cutoff:.4f}) @ P{furry_normie_p}",
    )
    plt.axvline(
        actual_furry_strong_cutoff,
        color="red",
        linestyle="--",
        label=f"Strong Cutoff ({actual_furry_strong_cutoff:.4f}) @ P{furry_strong_p}",
    )
    plt.xlim(right=0.02)
    plt.legend(fontsize="small")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    histograms_path = os.path.join(
        current_output_dir, "score_histograms_with_cutoffs.png"
    )
    plt.savefig(histograms_path)
    print(f"Saved score histograms with cutoffs to {histograms_path}")
    plt.close()

    detailed_data_path = os.path.join(
        current_output_dir, "heuristic_labelled_data_details.csv"
    )
    final_results_df.to_csv(detailed_data_path, index=False)
    print(f"Saved detailed heuristic labelled data to {detailed_data_path}")

    print("\nAnalysis complete for this percentile setting.")


# --- Command-Line Interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze heuristic labeling of text data using category-specific percentile-based thresholds."
    )
    parser.add_argument(
        "--data_csv",
        required=True,
        help="Path to the input CSV file with text data (must have a 'text' column).",
    )
    parser.add_argument(
        "--weeb_terms_csv",
        default=DEFAULT_WEEB_TERMS_PATH,
        help=f"Path to the weeb terms CSV file (default: {DEFAULT_WEEB_TERMS_PATH}).",
    )
    parser.add_argument(
        "--furry_terms_csv",
        default=DEFAULT_FURRY_TERMS_PATH,
        help=f"Path to the furry terms CSV file (default: {DEFAULT_FURRY_TERMS_PATH}).",
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Base directory to save analysis results (default: {DEFAULT_OUTPUT_DIR}). Subdirectories will be created per percentile setting.",
    )
    # Arguments for category-specific percentiles
    parser.add_argument(
        "--weeb-normie-percentile",
        type=int,
        default=70,
        choices=range(0, 101),
        metavar="[0-100]",
        help="Percentile for Weeb 'Normie' cutoff (default: 70).",
    )
    parser.add_argument(
        "--weeb-strong-percentile",
        type=int,
        default=90,
        choices=range(0, 101),
        metavar="[0-100]",
        help="Percentile for Weeb 'Strong' cutoff (default: 90).",
    )
    parser.add_argument(
        "--furry-normie-percentile",
        type=int,
        default=70,
        choices=range(0, 101),
        metavar="[0-100]",
        help="Percentile for Furry 'Normie' cutoff (default: 70).",
    )
    parser.add_argument(
        "--furry-strong-percentile",
        type=int,
        default=90,
        choices=range(0, 101),
        metavar="[0-100]",
        help="Percentile for Furry 'Strong' cutoff (default: 90).",
    )
    parser.add_argument(
        "--force_recalculate_scores",
        action="store_true",
        help="Force recalculation of weeb/furry scores even if a cached scores file exists.",
    )

    args = parser.parse_args()

    # Validate that strong percentile is not less than normie percentile for each category
    if args.weeb_strong_percentile < args.weeb_normie_percentile:
        print(
            f"Error: --weeb-strong-percentile ({args.weeb_strong_percentile}) cannot be less than --weeb-normie-percentile ({args.weeb_normie_percentile})."
        )
        exit(1)
    if args.furry_strong_percentile < args.furry_normie_percentile:
        print(
            f"Error: --furry-strong-percentile ({args.furry_strong_percentile}) cannot be less than --furry-normie-percentile ({args.furry_normie_percentile})."
        )
        exit(1)

    weeb_terms = load_term_database(args.weeb_terms_csv)
    furry_terms = load_term_database(args.furry_terms_csv)

    if (weeb_terms.empty or furry_terms.empty) and not (
        args.force_recalculate_scores
        or os.path.exists(
            os.path.join(args.output_dir, "calculated_scores_for_data.csv")
        )
    ):
        print(
            "Warning: One or both term databases are empty. If not forcing score recalculation and no cached scores exist, analysis might be based on incomplete term data."
        )
        if weeb_terms.empty and furry_terms.empty:
            print(
                "Error: Both term databases are empty, and cannot proceed without terms if scores need to be calculated. Exiting."
            )
            exit(1)

    analyze_heuristics(
        data_csv_path=args.data_csv,
        weeb_terms_df=weeb_terms,
        furry_terms_df=furry_terms,
        output_dir_base=args.output_dir,
        weeb_normie_p=args.weeb_normie_percentile,
        weeb_strong_p=args.weeb_strong_percentile,
        furry_normie_p=args.furry_normie_percentile,
        furry_strong_p=args.furry_strong_percentile,
        force_recalculate_scores=args.force_recalculate_scores,
    )
