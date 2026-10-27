"""Combined 24-category prediction and specific-role recommendation."""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "distilbert_B_seed42_longer10"
DEFAULT_MANIFEST = ROOT / "results/transformer/manifest_B_seed42_longer10.json"
DEFAULT_CATALOG = ROOT / "data/role_catalog/job_roles.json"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import predict as category_predictor
from role_recommender import recommend


def main():
    parser = argparse.ArgumentParser(
        description="Predict a broad category and rank specific roles for a resume."
    )
    parser.add_argument("file", help="Path to a PDF or plain-text resume")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    file_path = Path(args.file).expanduser()
    model_dir = Path(args.model_dir).expanduser()
    manifest_path = Path(args.manifest).expanduser()
    catalog_path = Path(args.catalog).expanduser()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    for path, description in [
        (file_path, "Resume file"),
        (manifest_path, "Model manifest"),
        (catalog_path, "Role catalogue"),
    ]:
        if not path.is_file():
            sys.exit(f"ERROR: {description} not found: {path}")
    if not model_dir.is_dir():
        sys.exit(f"ERROR: Model directory not found: {model_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    label2id = manifest.get("label2id")
    if not isinstance(label2id, dict) or len(label2id) != 24:
        sys.exit("ERROR: Expected a manifest with 24 categories.")

    if file_path.suffix.lower() == ".pdf":
        try:
            resume_text = category_predictor.extract_pdf_text(file_path)
        except Exception as exc:
            sys.exit(f"ERROR: Could not extract resume text: {exc}")
    else:
        try:
            resume_text = file_path.read_text(encoding="utf-8").strip()
        except Exception as exc:
            sys.exit(f"ERROR: Could not read resume text: {exc}")

    assessment = category_predictor.assess_input(resume_text)
    if assessment["refuse"]:
        sys.exit("Resume check failed: " + assessment["refuse"])

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), local_files_only=True
    ).to(device)

    categories = category_predictor.predict(
        resume_text, model, tokenizer, label2id, device
    )
    predicted_categories = [item["category"] for item in categories]

    roles = recommend(
        resume_text,
        catalog_path=catalog_path,
        top_k=args.top_k,
        predicted_categories=predicted_categories,
    )

    print("\nResumeMatch — Integrated Recommendations")
    print("=" * 46)
    print("Primary predicted category:", categories[0]["category"])
    print("\nTop broad-category model outputs (not calibrated probabilities):")
    for item in categories:
        print(f"  {item['category']}: {item['score']:.2%}")

    print(f"\nTop {len(roles)} specific roles")
    print("-" * 46)
    print("Fit scores are heuristic keyword matches, NOT hiring probabilities.")

    for rank, role in enumerate(roles, 1):
        print(f"\n{rank}. {role['title']} — {role['fit_score']:.1f}/100")
        print("   Related categories:", ", ".join(role["categories"]))
        if role["category_supported"] is not None:
            print("   Category overlap with model shortlist:",
                  "Yes" if role["category_supported"] else "No")
        print("   Required evidence found:",
              ", ".join(role["matched_required"]) or "None identified")
        print("   Required evidence not found:",
              ", ".join(role["missing_required_evidence"]) or "None")
        print("   Preferred evidence found:",
              ", ".join(role["matched_preferred"]) or "None identified")
        print("   Project evidence found:",
              ", ".join(role["matched_project_evidence"]) or "None identified")

    print("\nInput checks")
    print("Words in resume:", assessment["words"])
    for warning in assessment["warnings"]:
        print("WARNING:", warning)
    print("\nDevice:", device)
    print("The broad category is model-predicted; specific roles are ranked by catalogue evidence.")


if __name__ == "__main__":
    main()
