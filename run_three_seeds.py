"""Three-seed training and confusion matrix for Unit 6 Milestone 2"""
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
LABELS = ["ANALYSIS", "HOT_TAKE", "DISCUSSION"]
BASE_MODEL = "distilbert-base-uncased"
SEEDS = [42, 7, 2024]
RUN_LABEL = "before"  # Change to "after" for Milestone 5
EPOCHS = 3
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
MAX_LENGTH = 128

print(f"Loading data...")
df = pd.read_csv("labels.csv")

def to_ids(df, labels):
    lookup = {label: i for i, label in enumerate(labels)}
    out = df.copy()
    out["labels"] = out["label"].astype(str).str.strip().map(lookup)
    return out[["text", "labels"]]

def tokenize_fn(tokenizer):
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)
    return tokenize

def compute_metrics(eval_pred):
    logits, gold = eval_pred
    predicted = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(gold, predicted),
        "f1_macro": f1_score(gold, predicted, average="macro", zero_division=0),
    }

all_results = []
device_type = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

for i, seed in enumerate(SEEDS, 1):
    print(f"\n{'='*60}\nSeed {seed}  ({i} of {len(SEEDS)})\n{'='*60}")

    # Split dataset with this seed
    set_seed(seed)
    train_val_df, test_df = train_test_split(df, test_size=0.15, random_state=seed, stratify=df['label'])
    train_df, val_df = train_test_split(train_val_df, test_size=0.176, random_state=seed, stratify=train_val_df['label'])

    print(f"Split: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABELS),
        id2label={i: l for i, l in enumerate(LABELS)},
        label2id={l: i for i, l in enumerate(LABELS)},
    )

    # Prepare datasets
    tokenize = tokenize_fn(tokenizer)
    train_ds = Dataset.from_pandas(to_ids(train_df, LABELS)).map(tokenize, batched=True)
    val_ds = Dataset.from_pandas(to_ids(val_df, LABELS)).map(tokenize, batched=True)
    test_ds = Dataset.from_pandas(to_ids(test_df, LABELS)).map(tokenize, batched=True)

    # Training
    args = TrainingArguments(
        output_dir=f"./takemeter_out_seed{seed}",
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        seed=seed,
        report_to="none",
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    print(f"\nTraining...")
    trainer.train()

    # Test evaluation
    print(f"\nEvaluating on test set...")
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

    print(f"\n✓ Seed {seed} Results:")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  Macro F1: {f1_macro:.3f}")
    for j, label in enumerate(LABELS):
        print(f"  {label} F1: {f1_scores[j]:.3f}")

    # Store results
    result = {
        "seed": seed,
        "device": device_type,
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "per_label": {
            LABELS[j]: {
                "precision": float(precision[j]),
                "recall": float(recall[j]),
                "f1": float(f1_scores[j]),
            }
            for j in range(len(LABELS))
        },
        "confusion_matrix": conf_matrix.tolist(),
    }
    all_results.append(result)

# Save three-seeds results
output_file = f"results_three_seeds_{RUN_LABEL}.json"
with open(output_file, "w") as f:
    json.dump({
        "run_label": RUN_LABEL,
        "seeds": SEEDS,
        "device": device_type,
        "base_model": BASE_MODEL,
        "results": all_results
    }, f, indent=2)

print(f"\n{'='*60}")
print("THREE-SEED TRAINING COMPLETE")
print(f"{'='*60}")
print(f"\nResults saved to: {output_file}")
print(f"\nSummary across seeds:")
for i, (seed, result) in enumerate(zip(SEEDS, all_results)):
    print(f"Seed {seed}: Acc {result['accuracy']:.3f}, F1 {result['f1_macro']:.3f}")

# Print confusion matrix for first seed (for README)
print(f"\n\nConfusion Matrix (Seed {SEEDS[0]}) - Markdown table for README:")
print(f"\n| true \\ predicted | {' | '.join(LABELS)} |")
print(f"|---|{'|'.join(['---']*len(LABELS))}|")
conf = all_results[0]['confusion_matrix']
for i, true_label in enumerate(LABELS):
    cells = " | ".join(str(conf[i][j]) for j in range(len(LABELS)))
    print(f"| **{true_label}** | {cells} |")

# Find biggest off-diagonal
print(f"\nBiggest confusion:")
max_conf = 0
max_true = max_pred = None
for i, true_label in enumerate(LABELS):
    for j, pred_label in enumerate(LABELS):
        if i != j and conf[i][j] > max_conf:
            max_conf = conf[i][j]
            max_true = true_label
            max_pred = pred_label

if max_conf > 0:
    print(f"{max_conf} {max_true} posts were predicted as {max_pred}")
else:
    print("Perfect confusion matrix - no off-diagonal errors!")
