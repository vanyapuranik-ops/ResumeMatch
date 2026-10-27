"""Predict a job category from a resume using a saved ResumeMatch DistilBERT model."""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "models" / "distilbert_B_seed42_artifacts1"
DEFAULT_MANIFEST = ROOT / "results" / "transformer" / "manifest_B_seed42_artifacts1.json"
MAX_LENGTH = 512
HEAD_CHARS = 100
PLACEHOLDER = "categorymask"

# Input checks (P1). 50 words is the refusal floor; 113 is the shortest resume in the training data;
# the 250-word and 0.70 values come from validation analysis and are provisional, as is the letter share.
MIN_WORDS_REFUSE = 50
BELOW_TRAINING_MIN_WORDS = 113
SHORT_WARN_WORDS = 250
LOW_CONFIDENCE = 0.70
MIN_LETTER_SHARE = 0.60


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
    import pdfplumber  # imported here so plain-text resumes do not need pdfplumber

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


def assess_input(text):
    """Pre-prediction checks. Reads the text only; it never changes what the model sees."""
    stripped = text.strip()
    words = len(stripped.split())
    non_space = [c for c in stripped if not c.isspace()]
    letter_share = sum(c.isalpha() for c in non_space) / len(non_space) if non_space else 0.0
    out = {"words": words, "letter_share": letter_share, "refuse": None, "warnings": []}
    if words == 0:
        out["refuse"] = "The resume text is empty. Please provide a resume with its experience, skills and education."
    elif words < MIN_WORDS_REFUSE:
        out["refuse"] = (
            "Only {} words were found. A recommendation needs at least {} words of resume content "
            "(resumes in the training data have several hundred), so no category scores were generated. "
            "Please provide a more complete resume.".format(words, MIN_WORDS_REFUSE))
    else:
        if words < BELOW_TRAINING_MIN_WORDS:
            out["warnings"].append(
                "This resume has {} words, fewer than any resume the model was trained on (the shortest had {}). "
                "The recommendation may be less reliable.".format(words, BELOW_TRAINING_MIN_WORDS))
        elif words < SHORT_WARN_WORDS:
            out["warnings"].append(
                "This resume has {} words, shorter than typical training resumes (median about 757). In a validation "
                "test, accuracy was lower on shortened resumes, so the recommendation may be less reliable.".format(words))
        if letter_share < MIN_LETTER_SHARE:
            out["warnings"].append(
                "Fewer than {:.0%} of the characters are letters, so the text may not be normal resume prose "
                "(for example a PDF extraction problem).".format(MIN_LETTER_SHARE))
    return out


def confidence_notes(results):
    """Notes about how to read the scores. Scores themselves are never altered.
    Reliability figures are observed rates on the held-out seed-42 test set (497 resumes)."""
    lines = []
    top = results[0]["score"]
    if top <= LOW_CONFIDENCE:
        lines.append(
            "LOW CONFIDENCE: the top score ({:.2%}) is at or below {:.0%}. In the validation analysis, predictions "
            "at or below this level were correct about 37% of the time, against about 94% above it. Treat the "
            "category as a weak suggestion.".format(top, LOW_CONFIDENCE))
        lines.append(
            "In held-out testing, predictions at or below {:.0%} were correct 47.5% of the time (118 resumes), and "
            "the correct category was among the top 3 for 88.7% of all test resumes. Treat the three listed "
            "categories as a shortlist, not a single answer.".format(LOW_CONFIDENCE))
        lines.append(
            "The {:.0%} threshold is provisional: it was chosen using validation data, and the held-out test "
            "figures come from one model and one split.".format(LOW_CONFIDENCE))
    else:
        lines.append(
            "In held-out testing, predictions scoring above {:.0%} were correct 92.9% of the time (379 of 497 "
            "resumes). That is an observed rate for resumes like the test set, not a guarantee for this "
            "resume.".format(LOW_CONFIDENCE))
    lines.append("Scores are softmax model outputs, not calibrated probabilities; they do not give the chance "
                 "that a category is correct.")
    return lines


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
        try:
            resume_text = extract_pdf_text(file_path)
        except ImportError:
            pdf_error = "PDF support needs pdfplumber, which is not installed. Run: pip install pdfplumber"
        except ValueError as exc:
            pdf_error = str(exc)
        except Exception as exc:  # corrupt, encrypted or otherwise unreadable PDF
            pdf_error = ("The PDF could not be read ({}). It may be corrupted, password-protected "
                         "or not a real PDF.".format(type(exc).__name__))
        else:
            pdf_error = None
        if pdf_error:
            print("\nResumeMatch could not make a recommendation.")
            print(pdf_error)
            sys.exit(2)
    else:
        resume_text = file_path.read_text(encoding="utf-8").strip()

    assessment = assess_input(resume_text)
    if assessment["refuse"]:
        print("\nResumeMatch could not make a recommendation.")
        print(assessment["refuse"])
        sys.exit(2)

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
    print("\nInput checks")
    print("------------")
    print("Words in resume:", assessment["words"])
    total_tokens = len(tokenizer(mask_category_names(resume_text, list(label2id.keys())),
                                 verbose=False)["input_ids"])
    used_tokens = min(total_tokens, MAX_LENGTH)
    unread = "" if total_tokens <= MAX_LENGTH else " (the rest of the resume was not read)"
    print("Tokens read by the model: {} of {}{}".format(used_tokens, total_tokens, unread))
    for warning in assessment["warnings"]:
        print("WARNING:", warning)
    for line in confidence_notes(results):
        print(line)
    print("\nDevice:", device)
    print("Scores are model outputs, not guarantees of candidate suitability.")


if __name__ == "__main__":
    main()
