"""Direct training script extracted from notebook"""
import json
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, set_seed,
)
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split

# Configuration
NAME = "Lovina Bhatia"
EMAIL = "lovinabhatia06@gmail.com"
LABELS = ["ANALYSIS", "HOT_TAKE", "DISCUSSION"]
BASE_MODEL = "distilbert-base-uncased"
SEED = 42
EPOCHS = 3
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
MAX_LENGTH = 128

print(f"Loading data...")
df = pd.read_csv("labels.csv")
print(f"Loaded {len(df)} examples")

# Check balance
print("\nLabel distribution:")
for label in LABELS:
    count = (df['label'] == label).sum()
    print(f"  {label}: {count} ({count/len(df)*100:.1f}%)")

# Split dataset
train_val_df, test_df = train_test_split(df, test_size=0.15, random_state=SEED, stratify=df['label'])
train_df, val_df = train_test_split(train_val_df, test_size=0.176, random_state=SEED, stratify=train_val_df['label'])  # 0.176 of 0.85 ≈ 0.15 overall

print(f"\nSplit sizes:")
print(f"  Train: {len(train_df)}")
print(f"  Val: {len(val_df)}")
print(f"  Test: {len(test_df)}")

# Convert labels to IDs
def to_ids(df, labels):
    lookup = {label: i for i, label in enumerate(labels)}
    out = df.copy()
    out["labels"] = out["label"].astype(str).str.strip().map(lookup)
    return out[["text", "labels"]]

# Tokenizer
print(f"\nLoading model: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=len(LABELS),
    id2label={i: l for i, l in enumerate(LABELS)},
    label2id={l: i for i, l in enumerate(LABELS)},
)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

train_ds = Dataset.from_pandas(to_ids(train_df, LABELS)).map(tokenize, batched=True)
val_ds = Dataset.from_pandas(to_ids(val_df, LABELS)).map(tokenize, batched=True)
test_ds = Dataset.from_pandas(to_ids(test_df, LABELS)).map(tokenize, batched=True)

def compute_metrics(eval_pred):
    logits, gold = eval_pred
    predicted = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(gold, predicted),
        "f1_macro": f1_score(gold, predicted, average="macro", zero_division=0),
    }

# Training arguments
args = TrainingArguments(
    output_dir="./takemeter_out",
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=10,
    seed=SEED,
    report_to="none",
    disable_tqdm=False,
)

print(f"\nTraining for {EPOCHS} epochs...")
set_seed(SEED)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)
trainer.train()

print("\n✓ Training complete")

# Test evaluation
print("\nEvaluating on test set...")
test_results = trainer.predict(test_ds)
test_logits = test_results.predictions
test_preds = np.argmax(test_logits, axis=-1)
test_gold = test_results.label_ids

# Compute metrics
accuracy = accuracy_score(test_gold, test_preds)
precision, recall, f1_scores, _ = precision_recall_fscore_support(
    test_gold, test_preds, average=None, zero_division=0
)
f1_macro = f1_score(test_gold, test_preds, average="macro", zero_division=0)
conf_matrix = confusion_matrix(test_gold, test_preds)

print(f"\nTest Results:")
print(f"  Accuracy: {accuracy:.3f}")
print(f"  Macro F1: {f1_macro:.3f}")
print(f"\nPer-label F1:")
for i, label in enumerate(LABELS):
    print(f"  {label}: {f1_scores[i]:.3f}")

# Save results.json
device_type = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
results = {
    "name": NAME,
    "email": EMAIL,
    "model": BASE_MODEL,
    "seed": SEED,
    "device": device_type,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "max_length": MAX_LENGTH,
    "train_size": len(train_df),
    "val_size": len(val_df),
    "test_size": len(test_df),
    "accuracy": float(accuracy),
    "f1_macro": float(f1_macro),
    "per_label": {
        LABELS[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1_scores[i]),
        }
        for i in range(len(LABELS))
    },
    "confusion_matrix": conf_matrix.tolist(),
}

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\n✓ Saved results.json")

# Save test_split.csv
test_df_with_preds = test_df.copy()
test_df_with_preds['predicted'] = [LABELS[p] for p in test_preds]
test_df_with_preds['correct'] = test_df_with_preds['label'] == test_df_with_preds['predicted']
test_df_with_preds.to_csv("test_split.csv", index=False)
print("✓ Saved test_split.csv")

print(f"\n{'='*60}")
print("TRAINING COMPLETE")
print(f"{'='*60}")
print(f"Device: {device_type}")
print(f"Accuracy: {accuracy:.3f}")
print(f"Macro F1: {f1_macro:.3f}")
