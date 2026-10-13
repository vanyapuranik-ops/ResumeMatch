"""Predict a job category from a resume using a saved ResumeMatch DistilBERT model."""

import argparse
import json
import re
import sys
from pathlib import Path

import pdfplumber
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "models" / "distilbert_B_seed42_artifacts1"
DEFAULT_MANIFEST = ROOT / "results" / "transformer" / "manifest_B_seed42_artifacts1.json"
MAX_LENGTH = 512
HEAD_CHARS = 100
PLACEHOLDER = "categorymask"


def norm(value):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(value).lower())).strip()


def mask_category_names(text, category_names):
    """Match Experiment B: mask category names in the first 100 characters."""
    names = sorted({norm(c) for c in category_names if norm(c)}, key=len, reverse=True)
    pattern = re.compile(
        r"(?<![a-z0-9])(?:" + "|".join(re.escape(n) for n in names) + r")(?![a-z0-9])"
    )
    head = norm(text[:HEAD_CHARS])
    if pattern.search(head):
        return pattern.sub(PLACEHOLDER, head) + " " + text[HEAD_CHARS:]
    return text


def extract_pdf_text(path):
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    text = "\n".join(pages).strip()
    if not text:
        raise ValueError(
            "No selectable text was found. This PDF may be scanned; OCR is not implemented yet."
        )
    return text


def predict(resume_text, model, tokenizer, label2id, device):
    categories = list(label2id.keys())
    prepared = mask_category_names(resume_text, categories)

    inputs = tokenizer(
        prepared,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    model.eval()
    with torch.inference_mode():
        logits = model(**inputs).logits[0]
        probabilities = torch.softmax(logits, dim=-1)
        scores, indices = torch.topk(probabilities, k=min(3, len(categories)))

    results = []
    for score, index in zip(scores.tolist(), indices.tolist()):
        category_id = int(index)
        category = next(
            (name for name, idx in label2id.items() if idx == category_id),
            model.config.id2label.get(category_id, str(category_id)),
        )
        results.append({"category": category, "score": round(float(score), 4)})

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Predict likely job categories from a resume PDF or text file."
    )
    parser.add_argument("file", help="Path to a PDF or plain-text resume")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    args = parser.parse_args()

    file_path = Path(args.file).expanduser()
    model_dir = Path(args.model_dir).expanduser()
    manifest_path = Path(args.manifest).expanduser()

    if not file_path.is_file():
        sys.exit("ERROR: Resume file not found: " + str(file_path))
    if not model_dir.is_dir():
        sys.exit("ERROR: Saved model directory not found: " + str(model_dir))
    if not manifest_path.is_file():
        sys.exit("ERROR: Manifest not found: " + str(manifest_path))

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    label2id = manifest.get("label2id")
    if not isinstance(label2id, dict) or len(label2id) != 24:
        sys.exit("ERROR: Manifest does not contain the expected 24-category label mapping.")

    if file_path.suffix.lower() == ".pdf":
        resume_text = extract_pdf_text(file_path)
    else:
        resume_text = file_path.read_text(encoding="utf-8").strip()

    if len(resume_text) < 30:
        sys.exit("ERROR: Resume text is empty or too short to classify reliably.")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), local_files_only=True
    ).to(device)

    results = predict(resume_text, model, tokenizer, label2id, device)

    print("\nResumeMatch prediction")
    print("----------------------")
    print("Primary predicted category:", results[0]["category"])
    print("\nTop category scores (not calibrated probabilities):")
    for item in results:
        print("  {}: {:.2%}".format(item["category"], item["score"]))
    print("\nDevice:", device)
    print("Input tokens are truncated to", MAX_LENGTH, "tokens.")
    print("Scores are model outputs, not guarantees of candidate suitability.")


if __name__ == "__main__":
    main()
