#!/usr/bin/env python3
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from atproto import Client
import argparse
from tqdm import tqdm
from unsloth import FastLanguageModel
from peft import LoraConfig
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define constants
MODEL_NAME = "unsloth/gemma-3-1b-it-unsloth-bnb-4bit"  # Unsloth-optimized 4-bit quantized Gemma 3B
THRESHOLD_WEEB = 0.4  # Threshold for weeb classification
THRESHOLD_FURRY = 0.4  # Threshold for furry classification
MAX_LENGTH = 512
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5
OUTPUT_DIR = "output/bluesky-user-classifier"
METRICS_DIR = "metrics/bluesky-user-classifier"

class BlueskyDataset(Dataset):
    def __init__(self, texts, weeb_scores, furry_scores, tokenizer, max_length=512):
        self.texts = texts
        self.weeb_scores = weeb_scores
        self.furry_scores = furry_scores
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create labels (0: normie, 1: weeb, 2: furry)
        self.labels = []
        for w_score, f_score in zip(weeb_scores, furry_scores):
            if w_score >= THRESHOLD_WEEB and w_score > f_score:
                self.labels.append(1)  # Weeb
            elif f_score >= THRESHOLD_FURRY and f_score > w_score:
                self.labels.append(2)  # Furry
            else:
                self.labels.append(0)  # Normie

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        weeb_score = self.weeb_scores[idx]
        furry_score = self.furry_scores[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "weeb_score": torch.tensor(weeb_score, dtype=torch.float),
            "furry_score": torch.tensor(furry_score, dtype=torch.float),
            "label": torch.tensor(label, dtype=torch.long)
        }

class BlueskyUserClassifier:
    def __init__(self, model_path=None):
        """Initialize the classifier with either a pre-trained model or a fresh one."""
        print("Initializing Bluesky User Classifier...")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load or initialize model
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            self.model = FastLanguageModel.from_pretrained(
                model_path,
                device_map="auto",
                # Use 4-bit quantized model configuration
                use_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4"
            )
        else:
            print("Initializing new model for training")
            self.model = None  # Will be created during train()
            
        # Initialize Bluesky client
        self.client = Client()
        self.is_logged_in = False
            
    def login_to_bluesky(self, username, password):
        """Log in to Bluesky with the provided credentials."""
        try:
            self.client.login(username, password)
            self.is_logged_in = True
            print(f"Logged in as {username}")
            return True
        except Exception as e:
            print(f"Failed to log in: {e}")
            return False
            
    def train(self, csv_path):
        """Train the model using the provided CSV data."""
        print(f"Loading training data from {csv_path}")
        try:
            df = pd.read_csv(csv_path)
            print(f"Loaded {len(df)} training examples")
            
            # Check required columns
            required_cols = ['text', 'weeb_score', 'furry_score']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in CSV")
                    
            # Split data
            texts = df['text'].tolist()
            weeb_scores = df['weeb_score'].astype(float).tolist()
            furry_scores = df['furry_score'].astype(float).tolist()
            
            # Split into train and validation
            train_texts, val_texts, train_weeb, val_weeb, train_furry, val_furry = train_test_split(
                texts, weeb_scores, furry_scores, test_size=0.2, random_state=42
            )
            
            # Initialize model if not already done
            if self.model is None:
                print("Creating model with Unsloth FastLanguageModel")
                self.model = FastLanguageModel.from_pretrained(
                    MODEL_NAME,
                    max_seq_length=MAX_LENGTH,
                    device_map="auto",
                    # Use 4-bit quantized model configuration
                    use_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4"
                )
                
                # Add LoRA adapters for efficient fine-tuning
                # Use Unsloth's recommended LoRA config for 4-bit quantized models
                lora_config = LoraConfig(
                    r=8,
                    lora_alpha=16,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    lora_dropout=0.05,
                    bias="none",
                    task_type="SEQ_CLS"
                )
                
                # Add a classification head for three classes: normie, weeb, furry
                self.model = FastLanguageModel.get_peft_model(
                    self.model,
                    lora_config,
                    train_task="seq_cls",
                    num_labels=3
                )
                
            # Create datasets
            train_dataset = BlueskyDataset(train_texts, train_weeb, train_furry, self.tokenizer, MAX_LENGTH)
            val_dataset = BlueskyDataset(val_texts, val_weeb, val_furry, self.tokenizer, MAX_LENGTH)
            
            # Create dataloaders
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
            
            # Train model
            print(f"Training model for {EPOCHS} epochs...")
            self.model.train()
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=LEARNING_RATE)
            criterion = torch.nn.CrossEntropyLoss()
            
            best_val_loss = float('inf')
            train_losses = []
            val_losses = []
            
            for epoch in range(EPOCHS):
                # Training loop
                self.model.train()
                epoch_loss = 0
                pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
                
                for batch in pbar:
                    optimizer.zero_grad()
                    
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    
                    logits = outputs.logits
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    pbar.set_postfix({'loss': loss.item()})
                
                avg_train_loss = epoch_loss / len(train_loader)
                train_losses.append(avg_train_loss)
                
                # Validation loop
                self.model.eval()
                val_loss = 0
                all_preds = []
                all_labels = []
                
                with torch.no_grad():
                    for batch in tqdm(val_loader, desc="Validation"):
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['label'].to(device)
                        
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                        
                        logits = outputs.logits
                        loss = criterion(logits, labels)
                        val_loss += loss.item()
                        
                        preds = torch.argmax(outputs.logits, dim=1)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                
                avg_val_loss = val_loss / len(val_loader)
                val_losses.append(avg_val_loss)
                
                print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
                
                # Print classification report
                print(classification_report(
                    all_labels, 
                    all_preds, 
                    target_names=['Normie', 'Weeb', 'Furry']
                ))
                
                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    print(f"New best model! Saving to {OUTPUT_DIR}")
                    self.model.save_pretrained(OUTPUT_DIR)
                    self.tokenizer.save_pretrained(OUTPUT_DIR)
            
            # Plot training history
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label='Training Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(os.path.join(METRICS_DIR, 'training_history.png'))
            
            # Plot confusion matrix
            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Normie', 'Weeb', 'Furry'],
                        yticklabels=['Normie', 'Weeb', 'Furry'])
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.savefig(os.path.join(METRICS_DIR, 'confusion_matrix.png'))
            
            print(f"Training complete! Model saved to {OUTPUT_DIR}")
            print(f"Model metrics saved to {METRICS_DIR}")
            
        except Exception as e:
            print(f"Error during training: {e}")
            raise
    
    def get_user_posts(self, username, limit=20):
        """Fetch recent posts from a Bluesky user."""
        if not self.is_logged_in:
            raise ValueError("Not logged in to Bluesky. Call login_to_bluesky() first")
            
        try:
            # Get user DID from handle
            profile = self.client.app.bsky.actor.getProfile({'actor': username})
            user_did = profile.did
            
            # Fetch user's posts
            posts = self.client.app.bsky.feed.getAuthorFeed({'actor': user_did, 'limit': limit})
            
            # Extract post text
            post_texts = []
            for feed_view in posts.feed:
                post = feed_view.post
                if hasattr(post.record, 'text'):
                    post_texts.append(post.record.text)
            
            return " ".join(post_texts)
        
        except Exception as e:
            print(f"Error fetching user posts: {e}")
            return ""
    
    def classify_user(self, username=None, text=None):
        """
        Classify a user as Weeb, Furry, or Normie.
        Either provide a username to fetch posts, or directly provide text to classify.
        """
        if self.model is None:
            raise ValueError("Model not initialized. Train or load a model first")
            
        try:
            # Get text to classify
            if username and not text:
                print(f"Fetching posts for user @{username}...")
                text = self.get_user_posts(username)
                if not text:
                    return {
                        "error": f"Could not fetch posts for user @{username}",
                        "classification": "Unknown",
                        "weeb_score": 0.0,
                        "furry_score": 0.0
                    }
            elif not text:
                raise ValueError("Either username or text must be provided")
                
            # Tokenize input
            inputs = self.tokenizer(
                text,
                max_length=MAX_LENGTH,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(device)
            
            # Get model prediction
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Get classification probabilities
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
            predicted_class = torch.argmax(probs).item()
            
            # Map to class names
            class_names = ["Normie", "Weeb", "Furry"]
            classification = class_names[predicted_class]
            
            # Return classification with confidence scores
            result = {
                "classification": classification,
                "normie_score": float(probs[0]),
                "weeb_score": float(probs[1]),
                "furry_score": float(probs[2]),
                "text_analyzed": text[:100] + "..." if len(text) > 100 else text
            }
            
            return result
            
        except Exception as e:
            print(f"Error during classification: {e}")
            return {
                "error": str(e),
                "classification": "Error",
                "weeb_score": 0.0,
                "furry_score": 0.0
            }

def main():
    """Main function to run the classifier from the command line."""
    parser = argparse.ArgumentParser(description="Bluesky User Classifier")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the classifier")
    train_parser.add_argument("--data", required=True, help="Path to CSV training data")
    
    # Classify command
    classify_parser = subparsers.add_parser("classify", help="Classify a Bluesky user")
    classify_parser.add_argument("--model", default=OUTPUT_DIR, help="Path to trained model")
    classify_parser.add_argument("--username", help="Bluesky username to classify")
    classify_parser.add_argument("--text", help="Text to classify directly")
    classify_parser.add_argument("--bluesky-user", help="Your Bluesky username")
    classify_parser.add_argument("--bluesky-pass", help="Your Bluesky password")
    
    args = parser.parse_args()
    
    if args.command == "train":
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(METRICS_DIR, exist_ok=True)
        classifier = BlueskyUserClassifier()
        classifier.train(args.data)
    
    elif args.command == "classify":
        if not args.username and not args.text:
            print("Error: Either --username or --text must be provided")
            return
            
        classifier = BlueskyUserClassifier(args.model)
        
        if args.username:
            if not args.bluesky_user or not args.bluesky_pass:
                print("Error: Bluesky credentials required to fetch user posts")
                return
                
            # Login to Bluesky
            if not classifier.login_to_bluesky(args.bluesky_user, args.bluesky_pass):
                return
                
        # Classify user
        result = classifier.classify_user(args.username, args.text)
        
        # Print result
        print("\n" + "="*50)
        print(f"Classification: {result['classification']}")
        print(f"Normie Score: {result['normie_score']:.4f}")
        print(f"Weeb Score: {result['weeb_score']:.4f}")
        print(f"Furry Score: {result['furry_score']:.4f}")
        print("="*50 + "\n")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()