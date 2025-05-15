# Bluesky User Classifier

This tool classifies Bluesky users with a **primary** and an optional **secondary** category (e.g., Primary: Weeb, Secondary: Slight Furry; or Primary: Normie, Secondary: None) based on their post history. It uses a fine-tuned language model (Gemma 3 1B-IT) to analyze post content and identify patterns associated with different user types. It also provides evaluation metrics after fine-tuning.

## Features

- Fine-tunes a language model (`unsloth/gemma-3-1b-it-unsloth-bnb-4bit`) with 4-bit quantization for efficient classification.  
- Uses a heuristic scoring system based on weighted term databases for initial data labeling and fallback classification.  
- Fetches user posts directly from Bluesky using the ATProto API.  
- Provides primary and secondary classifications, along with heuristic scores for Weeb, Furry, and Normie tendencies.  
- **Generates classification metrics (classification report and confusion matrix for combined primary-secondary labels) after fine-tuning.**

## Usage

The tool provides three main commands. Replace `script.py` with the actual name of your Python script.

### 1. Preprocess Bluesky Data

Before training, you need to preprocess your Bluesky data (if you have it in a CSV format from another source):

```bash  
python script.py preprocess --input raw_bluesky_data.csv --output processed_data.csv
```

- --input: Path to the raw CSV file containing Bluesky posts.  
- --output: Path where the processed CSV file will be saved.

### **2. Fine-tune the Model**

Train the model using the preprocessed data:

```bash
python script.py finetune --data processed_data.csv --output_dir finetuned_model --epochs 3
```

Additional options:

- --output_dir: Directory to save the fine-tuned model and evaluation results (default: finetuned_model).  
- --epochs: Number of training epochs (default: 3).  
- --batch_size: Training batch size (default: 4).  
- --learning_rate: Learning rate (default: 2e-4).  
- --eval_split: Proportion of data to use for evaluation (default: 0.2). Set to 0 to disable evaluation and train on all data.

After fine-tuning, a classification report for combined primary-secondary labels will be printed to the console, and a confusion matrix plot (confusion_matrix_combined.png) will be saved in the specified --output_dir.

### **3. Classify a User**

Once the model is trained (or if you're using a pre-trained model specified in the script), you can classify any Bluesky user:

```bash
python script.py classify --model finetuned_model --username target_user.bsky.social --bluesky_user your_login_username.bsky.social --bluesky_pass your_login_password
```

- --model: Path to the directory containing the fine-tuned model adapters (e.g., finetuned_model).  
- --username: The Bluesky handle of the user you want to classify.  
- --bluesky_user: Your Bluesky username for logging in.  
- --bluesky_pass: Your Bluesky app password for logging in.

## **How It Works**

1. **Term Databases**: The system uses weighted term databases (weeb_terms.csv, furry_terms.csv) for initial heuristic scoring. These scores are used to generate labels for fine-tuning data and as a fallback in classification.  
2. **Data Preparation**: Raw post text is processed. For fine-tuning, posts are assigned heuristic **primary and secondary labels** (e.g., "Weeb-Slight Furry", "Normie-None") based on term scores using a defined logic. The response format for training includes these primary and secondary classifications.  
3. **Fine-tuning**: The language model is fine-tuned on a dataset of Bluesky posts formatted as prompts and the heuristically generated responses (which include primary and secondary classifications).  
4. **Evaluation (Post-Finetuning)**: If an evaluation split is used, the fine-tuned model's performance is assessed on unseen data. A classification report and confusion matrix are generated for the **combined primary-secondary labels**, based on the heuristic labels as ground truth.  
5. **Classification Process**:  
   - Fetches recent posts for the target user from Bluesky.  
   - Calculates overall heuristic scores (Weeb, Furry) based on the user's combined posts using term databases.  
   - Uses the fine-tuned model to classify a sample of individual posts, expecting it to predict primary and secondary classifications.  
   - Aggregates model predictions (e.g., by taking the most common predicted **primary-secondary label pair** for the sampled posts). These aggregated predictions determine the final primary and secondary classifications.  
   - If the model provides no clear classification, it may fall back to a classification based on the overall heuristic scores.  
6. **Score Interpretation & Heuristic Labeling**:  
   - The Weeb Score, Furry Score, and Normie Score provided in the output are based on the heuristic term-matching system applied to the user's combined posts.  
   - The heuristic labeling logic (used for training data generation and determining fallback classifications) assigns primary and secondary labels as follows:  
     - If max(weeb_score, furry_score) <= 0.4, then Primary: Normie, Secondary: None.  
     - Otherwise, the dominant score determines the primary label (e.g., >0.7 for "Weeb"/"Furry", >0.4 for "Slight Weeb"/"Slight Furry").  
     - The secondary score determines the secondary label (e.g., >0.7 for "Weeb"/"Furry", >0.4 for "Slight Weeb"/"Slight Furry"), if applicable, distinct from the primary, and not "Normie". If no significant secondary interest is detected, it's "None".

## **Example Output (Classification)**

```txt
==================================================  
Classification Results for @example.bsky.social  
==================================================  

Primary Classification: Weeb  
Secondary Classification: Slight Furry  
  Heuristic Weeb Score: 0.780  
  Heuristic Furry Score: 0.450  
  Heuristic Normie Score: 0.220  
  Model Post Classifications (sample): ['Weeb-Slight Furry', 'Normie-None', 'Weeb-None']  
==================================================
```

## **Notes**

- The accuracy of classification depends heavily on the quality and comprehensiveness of the term databases and the diversity and size of the training data.  
- The "ground truth" for evaluation metrics is derived from the same heuristic system used to label the training data. This means the evaluation shows how well the model learned to mimic the heuristic for assigning combined primary-secondary labels, not necessarily its absolute correctness against human judgment.  
- You need a valid Bluesky account (username and an app password) to fetch user posts.  
- The model is fine-tuned using LoRA (Low-Rank Adaptation), meaning only adapter weights are saved by default. The `load_finetuned_model`
