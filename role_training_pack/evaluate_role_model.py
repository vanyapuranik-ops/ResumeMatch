from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "role_distilbert_v1" / "model"
TRAIN_CSV = ROOT / "train.csv"
TEST_CSV = ROOT / "test.csv"
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not MODEL_DIR.exists():
    raise FileNotFoundError(f"Saved model not found: {MODEL_DIR}")
if not TRAIN_CSV.exists() or not TEST_CSV.exists():
    raise FileNotFoundError("train.csv or test.csv is missing from this folder.")

train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)

for frame, name in [(train, "train"), (test, "test")]:
    if "text" not in frame or "label" not in frame:
        raise ValueError(f"{name}.csv must contain 'text' and 'label' columns.")
    frame["text"] = frame["text"].fillna("").astype(str)

# Map the numeric/string training labels to readable job-role names.
# Prefer candidate_role because it preserves the readable role names.
role_by_label = {}
if "candidate_role" in train.columns:
    pairs = train[["label", "candidate_role"]].dropna().drop_duplicates()
    for _, row in pairs.iterrows():
        role_by_label[str(row["label"])] = str(row["candidate_role"])

config = None
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
config = model.config

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
model.to(device)
model.eval()

num_labels = int(config.num_labels)

# Match readable names to the model's output IDs.
id2label = {}
configured = getattr(config, "id2label", {}) or {}
for key, value in configured.items():
    try:
        idx = int(key)
        if value and not str(value).startswith("LABEL_"):
            id2label[idx] = str(value)
    except (TypeError, ValueError):
        pass

# Use label IDs from the model config if meaningful; otherwise infer the
# mapping from the training CSV's label and candidate_role columns.
if len(id2label) != num_labels:
    id2label = {}
    if "candidate_role" in train.columns:
        for _, row in train[["label", "candidate_role"]].dropna().drop_duplicates().iterrows():
            try:
                idx = int(row["label"])
                id2label[idx] = str(row["candidate_role"])
            except (TypeError, ValueError):
                continue

if len(id2label) != num_labels:
    unique_labels = sorted(train["label"].unique(), key=lambda x: str(x))
    if len(unique_labels) == num_labels:
        id2label = {i: str(label) for i, label in enumerate(unique_labels)}

if len(id2label) != num_labels:
    raise ValueError(
        "Could not safely map model output IDs to role names. "
        f"Model has {num_labels} labels; config={configured}; "
        f"training labels={train['label'].unique().tolist()}. "
        "Inspect train_role_model.py to confirm label encoding."
    )

def encode_label(value):
    # Training labels are commonly integers in this dataset.
    try:
        return int(value)
    except (TypeError, ValueError):
        reverse = {name: idx for idx, name in id2label.items()}
        if str(value) in reverse:
            return reverse[str(value)]
        raise ValueError(f"Cannot map true label {value!r} to a model label ID.")

def predict_texts(texts, batch_size=16):
    predictions = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)
        predictions.extend(probabilities.cpu().numpy())
    return np.asarray(predictions)

print(f"Device: {device}")
print(f"Test resumes: {len(test)}")
print(f"Number of classes: {num_labels}")

probs = predict_texts(test["text"].tolist())
pred_ids = probs.argmax(axis=1)
true_ids = np.array([encode_label(x) for x in test["label"]])

# Restrict the report to the model's actual output IDs.
label_ids = list(range(num_labels))
target_names = [id2label[i] for i in label_ids]

report = classification_report(
    true_ids,
    pred_ids,
    labels=label_ids,
    target_names=target_names,
    output_dict=True,
    zero_division=0,
)
pd.DataFrame(report).transpose().to_csv(
    OUT_DIR / "classification_report.csv", index=True
)

cm = confusion_matrix(true_ids, pred_ids, labels=label_ids)
pd.DataFrame(cm, index=target_names, columns=target_names).to_csv(
    OUT_DIR / "confusion_matrix.csv", index=True
)

predicted_names = [id2label[int(i)] for i in pred_ids]
true_names = [id2label.get(int(i), str(i)) for i in true_ids]

results = test.copy()
results["true_role"] = true_names
results["predicted_role"] = predicted_names
results["correct"] = true_ids == pred_ids
results["confidence"] = probs.max(axis=1).round(4)
results.to_csv(OUT_DIR / "test_predictions.csv", index=False)

summary = {
    "device": str(device),
    "test_rows": int(len(test)),
    "num_classes": int(num_labels),
    "accuracy": float(accuracy_score(true_ids, pred_ids)),
    "macro_f1": float(f1_score(true_ids, pred_ids, average="macro", zero_division=0)),
    "weighted_f1": float(f1_score(true_ids, pred_ids, average="weighted", zero_division=0)),
    "id_to_role": {str(k): v for k, v in id2label.items()},
}
with open(OUT_DIR / "evaluation_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# Optional image output for reports/presentations.
try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(10, num_labels), max(8, num_labels * 0.75)))
    image = ax.imshow(cm, interpolation="nearest")
    ax.set_title("ResumeMatch — Test Confusion Matrix")
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(num_labels),
        yticks=np.arange(num_labels),
        xticklabels=target_names,
        yticklabels=target_names,
        xlabel="Predicted role",
        ylabel="Actual role",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
except ImportError:
    print("matplotlib not installed; confusion_matrix.png was skipped.")

print("\nEvaluation complete.")
print(json.dumps(summary, indent=2))
print(f"\nFiles saved in: {OUT_DIR}")
print("- classification_report.csv")
print("- confusion_matrix.csv")
print("- test_predictions.csv")
print("- evaluation_summary.json")
print("- confusion_matrix.png (if matplotlib is installed)")
