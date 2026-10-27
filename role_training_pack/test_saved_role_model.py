from pathlib import Path
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path.home() / "ResumeMatch"
MODEL_DIR = ROOT / "role_training_pack" / "role_distilbert_v1" / "model"
TRAIN_CSV = ROOT / "role_training_pack" / "train.csv"

print("\n=== 1. MODEL PATH CHECK ===")
print("Model path:", MODEL_DIR)
assert MODEL_DIR.is_dir(), f"Model folder not found: {MODEL_DIR}"
assert (MODEL_DIR / "config.json").exists(), "config.json is missing"
print("PASS: Saved model folder exists.")

print("\n=== 2. LOAD MODEL AND TOKENIZER ===")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(
    str(MODEL_DIR), local_files_only=True
)
model.eval()

num_labels = model.config.num_labels
print("Model classes:", num_labels)
print("Model type:", model.config.model_type)
assert num_labels == 12, (
    f"Expected the new 12-role model, but found {num_labels} classes."
)
print("PASS: The saved 12-class model loaded successfully.")

# Prefer the actual role names stored in the model configuration.
id2label = model.config.id2label
labels = [str(id2label.get(i, f"LABEL_{i}")) for i in range(num_labels)]

# A sample resume for a machine-learning/data-focused candidate.
sample_resume = """
Machine Learning and Data Science graduate with experience in Python,
SQL, pandas, NumPy, scikit-learn, data cleaning, exploratory data analysis,
feature engineering, model training, classification, regression and model
evaluation. Built academic projects involving predictive modelling,
visualisation and machine learning. Familiar with Jupyter Notebook,
Git and communicating analytical findings.
"""

print("\n=== 3. SAMPLE RESUME PREDICTION ===")
inputs = tokenizer(
    sample_resume,
    return_tensors="pt",
    truncation=True,
    max_length=min(512, getattr(tokenizer, "model_max_length", 512))
)

with torch.no_grad():
    logits = model(**inputs).logits
    probabilities = torch.softmax(logits, dim=-1)[0].cpu().numpy()

assert len(probabilities) == 12, "Prediction does not contain 12 class scores."
assert np.isfinite(probabilities).all(), "Prediction contains invalid scores."
assert np.isclose(probabilities.sum(), 1.0, atol=1e-4), (
    "Prediction probabilities do not sum to approximately 1."
)

top_indices = np.argsort(probabilities)[::-1][:3]

for rank, idx in enumerate(top_indices, start=1):
    print(f"{rank}. {labels[int(idx)]}: {probabilities[int(idx)] * 100:.2f}%")

print("\n=== 4. FINAL CHECK ===")
print("PASS: Model loaded, sample prediction ran, and scores are valid.")
print("Important: scores are model confidence estimates, not guaranteed probabilities of job suitability.")
