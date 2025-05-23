# Bluesky User Classifier

This tool classifies Bluesky users with a **primary** and an optional **secondary** category (e.g., Primary: Weeb, Secondary: Slight Furry; or Primary: Normie, Secondary: None) based on their post history. It uses a fine-tuned language model (default: `unsloth/Qwen3-0.6B-unsloth-bnb-4bit`) to analyze post content and identify patterns associated with different user types. It also provides evaluation metrics after fine-tuning.

The core heuristic labeling now uses a more adaptive system: it calculates normalized scores (0-1 range) for "weeb" and "furry" categories based on term lists. Then, it determines category-specific cutoff scores based on user-defined percentiles of these score distributions (e.g., the 70th percentile of all weeb scores might define the "normie" cutoff for weeb-related content). These dynamic, category-specific cutoffs are then used to assign primary and secondary labels.

## Features

- Fine-tunes a language model (default: `unsloth/Qwen3-0.6B-unsloth-bnb-4bit`) with 4-bit quantization for efficient classification.  
- Uses a heuristic scoring system based on weighted term databases. Scores are normalized (0-1 range).  
- Heuristic labeling for training/evaluation data uses **category-specific percentile-based cutoffs** to determine "Normie", "Slight", and "Strong" signal strengths for each category (weeb, furry) independently before combining them into primary/secondary labels.  
- Fetches user posts directly from Bluesky using the ATProto API.  
- Provides primary and secondary classifications, along with heuristic scores for Weeb and Furry tendencies.  
- Generates classification metrics (classification report and confusion matrix for combined primary-secondary labels) after fine-tuning.

## Usage

The tool provides four main commands.

### 1. Preprocess Bluesky Data

Before training, you need to preprocess your Bluesky data (if you have it in a CSV format from another source):

```bash  
python script.py preprocess --input raw_bluesky_data.csv --output processed_data.csv
```

- --input: Path to the raw CSV file containing Bluesky posts.  
- --output: Path where the processed CSV file will be saved.

### **2. Fine-tune the Model**

Train the model using the preprocessed data. The heuristic labels for training will be generated using category-specific percentile cutoffs derived from the --data_csv.

```bash
python script.py finetune   
    --data_csv processed_data.csv   
    --output_dir finetuned_model   
    --base_model_name "unsloth/Qwen3-0.6B-unsloth-bnb-4bit"   
    --epochs 3   
    --weeb-normie-percentile 70   
    --weeb-strong-percentile 90   
    --furry-normie-percentile 70   
    --furry-strong-percentile 90
```

**Key Options:**

- --data_csv: Path to the processed Bluesky data CSV for fine-tuning.  
- --output_dir: Directory to save the fine-tuned model (default: finetuned_model).  
- --base_model_name: Base model for fine-tuning (default: unsloth/Qwen3-0.6B-unsloth-bnb-4bit).  
- --epochs: Number of training epochs (default: 3).  
- --batch_size: Training batch size (default: 8).  
- --learning_rate: Learning rate (default: 2e-4).  
- --eval_split_size: Proportion of data for validation (default: 0.2).  
- --skip_eval_after_train: If set, skips evaluation after training.  
- --chunk_size_csv: Chunk size for reading CSVs (default: 50000).  
- **Percentile Arguments for Heuristic Labeling:**  
  - --weeb-normie-percentile: Percentile of weeb scores to define the "Normie" cutoff for weeb signals (default: 70).  
  - --weeb-strong-percentile: Percentile of weeb scores to define the "Strong" cutoff for weeb signals (default: 90).  
  - --furry-normie-percentile: Percentile of furry scores to define the "Normie" cutoff for furry signals (default: 70).  
  - --furry-strong-percentile: Percentile of furry scores to define the "Strong" cutoff for furry signals (default: 90).

After fine-tuning, if evaluation is not skipped, a classification report and confusion matrix will be generated.

### **3. Evaluate a Model**

Evaluate a pre-trained (fine-tuned or base) model. Heuristic labels for the evaluation data will be generated using category-specific percentile cutoffs derived from the `--eval_data_csv`.

```bash
python script.py evaluate
    --model_path finetuned_model
    --eval_data_csv processed_eval_data.csv
    --metrics_output_dir evaluation_results
    --weeb-normie-percentile 70
    --weeb-strong-percentile 90
    --furry-normie-percentile 70
    --furry-strong-percentile 90
```

**Key Options:**

- --model_path: Path to the fine-tuned model or a base model name.  
- --eval_data_csv: Path to the CSV for evaluation.  
- --metrics_output_dir: Directory for metrics (default: evaluation_metrics).  
- --batch_size: Evaluation batch size (default: 16).  
- **Percentile Arguments:** Same as for finetune, applied to the --eval_data_csv.

### **4. Classify a User**

Classify a Bluesky user using a trained model. The heuristic fallback also uses the specified percentile cutoffs.

```bash
python script.py classify
    --model_path finetuned_model
    --username target_user.bsky.social
    --bluesky_user your_login.bsky.social
    --bluesky_pass your_app_password
    --weeb-normie-percentile 70
    --weeb-strong-percentile 90
    --furry-normie-percentile 70
    --furry-strong-percentile 90
```

**Key Options:**

- --model_path: Path to the fine-tuned model or a base model name.  
- --username: Bluesky handle of the user to classify.  
- --bluesky_user, --bluesky_pass: Your Bluesky login credentials.  
- --batch_size: Batch size for model inference on posts (default: 8).  
- **Percentile Arguments:** Same as for finetune. These define how the heuristic scores (calculated for the target user's posts) are interpreted for the heuristic part of the classification. For consistency, these should ideally match the percentiles used during fine-tuning of the model specified in --model_path.

## **How It Works**

1. **Term Databases**: The system uses weighted term databases (weeb_terms.csv, furry_terms.csv).  
2. **Score Calculation**: For each post, a weeb_score and a furry_score are calculated. This score is normalized (0-1 range) by dividing the sum of matched term scores by the total possible score from all terms in that category's list.  
3. **Percentile Cutoff Determination (for finetune and evaluate commands):**  
   - When preparing data (e.g., from --data_csv or --eval_data_csv), the script first performs a pre-pass over all posts in that specific CSV.  
   - It calculates weeb_score and furry_score for every post in the dataset.  
   - Based on the distributions of these scores *within that dataset*, it determines four actual cutoff score values:  
     - weeb_normie_cutoff: The score at the Nth percentile (e.g., 70th, from --weeb-normie-percentile) of all weeb scores.  
     - weeb_strong_cutoff: The score at the Mth percentile (e.g., 90th, from --weeb-strong-percentile) of all weeb scores.  
     - furry_normie_cutoff: The score at the Xth percentile (e.g., 70th, from --furry-normie-percentile) of all furry scores.  
     - furry_strong_cutoff: The score at the Yth percentile (e.g., 90th, from --furry-strong-percentile) of all furry scores.  
   - These four dynamically calculated cutoff scores are then used for heuristic labeling of each post in that dataset.  
4. **Heuristic Labeling (for Training Data & Evaluation Ground Truth)**:  
   - For each post, its weeb_score is compared against weeb_normie_cutoff and weeb_strong_cutoff to determine a "weeb strength" (Normie, Slight, or Strong).  
   - Similarly, its furry_score is compared against furry_normie_cutoff and furry_strong_cutoff to determine a "furry strength."  
   - These two independent strength assessments are then combined to assign a final primary and secondary label (e.g., "Weeb-Slight Furry", "Normie-None") to the post. This combined label becomes the true_label_heuristic.  
5. **Fine-tuning**: The language model is fine-tuned on a dataset of Bluesky posts formatted as prompts and the heuristically generated responses (which include the primary and secondary classifications derived from the percentile-based system).  
6. **Evaluation (Post-Finetuning)**: The fine-tuned model's performance is assessed against the true_label_heuristic generated using the same percentile-based system on the evaluation dataset.  
7. **Classification Process (for classify command)**:  
   - Fetches recent posts for the target user.  
   - Calculates overall heuristic weeb_score and furry_score for the user based on their combined posts.  
   - The percentile values provided to the classify command are used to set the four cutoff scores (weeb_normie, weeb_strong, furry_normie, furry_strong) for interpreting these heuristic scores.  
   - Uses the fine-tuned model to classify a sample of individual posts.  
   - Aggregates model predictions to determine final primary/secondary classifications. If the model is unclear, it may fall back to a classification based on the overall heuristic scores interpreted with the provided percentile cutoffs.  
8. **Score Interpretation & Heuristic Labeling Details**:  
   - The weeb_score and furry_score you see in the output are the normalized scores (0-1 range).  
   - The heuristic labeling logic:  
     1. Determines "weeb strength" (Normie, Slight, Strong) by comparing w_score to weeb_normie_cutoff and weeb_strong_cutoff.  
     2. Determines "furry strength" (Normie, Slight, Strong) by comparing f_score to furry_normie_cutoff and furry_strong_cutoff.  
     3. If both strengths are "Normie", result is "Normie-None".  
     4. If one has signal (Slight/Strong) and the other is "Normie", the one with signal becomes primary (e.g., "Weeb-None", "Slight Furry-None").  
     5. If both have signal, the one with the higher raw score (w_score vs f_score) becomes primary, and the other secondary, reflecting their respective strengths (e.g., "Weeb-Slight Furry", "Slight Furry-Strong Weeb"). Self-secondary labels (e.g., "Weeb-Slight Weeb") are resolved to "Primary-None".

## **Example Output (Classification)**

```txt
==================================================  
Classification Results for @example.bsky.social  
==================================================  

Primary Classification: Weeb  
Secondary Classification: Slight Furry  
  Heuristic Weeb Score: 0.018
  Heuristic Furry Score: 0.009
  Heuristic Normie Score: 0.982
  Model Post Classifications (sample): ['Weeb-Slight Furry', 'Normie-None', 'Weeb-None']  
==================================================
```

*(Note: Example scores are illustrative; actual normalized scores from the heuristic are typically very low, e.g., 0.001-0.025 range).

## **Notes**

- The accuracy of classification depends heavily on the quality of term databases, the chosen percentile cutoffs, and the training data.  
- Evaluation metrics show how well the model learned to mimic the sophisticated heuristic labeling, not necessarily absolute correctness against human judgment.  
- Requires a Bluesky account (username and app password).
