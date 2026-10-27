#!/usr/bin/env python3
"""Train an isolated 12-class ResumeMatch role classifier. Does not touch the 24-class model."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, Trainer, TrainingArguments
)
from datasets import Dataset

ROOT = Path(__file__).resolve().parent
TRAIN_FILE = ROOT / "train.csv"
VAL_FILE = ROOT / "validation.csv"
TEST_FILE = ROOT / "test.csv"
OUT_DIR = ROOT / "role_distilbert_v1"
RESULTS_DIR = ROOT / "results"
MODEL_NAME = "distilbert/distilbert-base-uncased"
MAX_LENGTH = 256
SEED = 42

for p in [TRAIN_FILE, VAL_FILE, TEST_FILE]:
    if not p.exists():
        raise SystemExit(f"Missing {p}. Keep this script beside train.csv, validation.csv and test.csv.")

train_df = pd.read_csv(TRAIN_FILE).fillna("")
val_df = pd.read_csv(VAL_FILE).fillna("")
test_df = pd.read_csv(TEST_FILE).fillna("")
labels = sorted(train_df["label"].astype(str).unique())
label2id = {name: i for i, name in enumerate(labels)}
id2label = {i: name for name, i in label2id.items()}

for name, frame in [("validation", val_df), ("test", test_df)]:
    unseen = set(frame["label"].astype(str)) - set(labels)
    if unseen:
        raise SystemExit(f"{name} has labels absent from train: {sorted(unseen)}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def to_dataset(frame):
    frame = frame[["text", "label"]].copy()
    frame["text"] = frame["text"].astype(str)
    frame["label"] = frame["label"].astype(str).map(label2id)
    ds = Dataset.from_pandas(frame, preserve_index=False)
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)
    return ds.map(tokenize, batched=True, remove_columns=["text"])

train_ds = to_dataset(train_df)
val_ds = to_dataset(val_df)
test_ds = to_dataset(test_df)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(labels), label2id=label2id, id2label=id2label
)
collator = DataCollatorWithPadding(tokenizer=tokenizer)

def metrics(eval_pred):
    logits, y_true = eval_pred
    y_pred = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
args_kwargs = dict(
    output_dir=str(OUT_DIR / "checkpoints"),
    num_train_epochs=3,
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    logging_steps=50,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    seed=SEED,
    report_to="none",
)
# Transformers changed this argument name across versions.
try:
    training_args = TrainingArguments(eval_strategy="epoch", **args_kwargs)
except TypeError:
    training_args = TrainingArguments(evaluation_strategy="epoch", **args_kwargs)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=metrics,
)
print(f"Device available: MPS={torch.backends.mps.is_available()}, CUDA={torch.cuda.is_available()}")
print(f"Classes: {len(labels)}; train={len(train_df)}, validation={len(val_df)}, test={len(test_df)}")
trainer.train()
print("Validation metrics:", trainer.evaluate())

pred = trainer.predict(test_ds)
y_true = pred.label_ids
y_pred = np.argmax(pred.predictions, axis=-1)
report = classification_report(
    y_true, y_pred, labels=list(range(len(labels))),
    target_names=labels, output_dict=True, zero_division=0
)
cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
trainer.save_model(str(OUT_DIR / "model"))
tokenizer.save_pretrained(str(OUT_DIR / "model"))
(OUT_DIR / "label_mapping.json").write_text(json.dumps(
    {"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}},
    indent=2
))
summary = {
    "model_name": MODEL_NAME,
    "max_length": MAX_LENGTH,
    "seed": SEED,
    "split_sizes": {"train": len(train_df), "validation": len(val_df), "test": len(test_df)},
    "test_metrics": {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    },
    "classification_report": report,
    "confusion_matrix": cm.tolist(),
    "labels": labels,
    "label_quality_note": "Candidate labels were title-derived and not fully human-verified."
}
(RESULTS_DIR / "test_evaluation.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary["test_metrics"], indent=2))
print(f"Saved model to {OUT_DIR / 'model'}")
print(f"Saved test metrics to {RESULTS_DIR / 'test_evaluation.json'}")
