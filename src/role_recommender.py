"""Explainable resume-to-role matching using a JSON role catalogue.

This is a transparent keyword-based baseline, not a trained role classifier.
Scores represent evidence matches, not probability of getting hired.
"""

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "data/role_catalog/job_roles.json"

ALIASES = {
    "python": ["python", "python3"],
    "javascript": ["javascript", "js", "ecmascript"],
    "machine learning": ["machine learning", "scikit-learn", "sklearn"],
    "data analysis": ["data analysis", "data analytics", "analyzed data", "analysed data"],
    "statistics": ["statistics", "statistical analysis", "statistical modelling"],
    "apis": ["api", "apis", "rest api", "restful api", "endpoint"],
    "databases": ["database", "databases", "mysql", "postgresql", "mongodb", "sqlite"],
    "sql": ["sql", "mysql", "postgresql", "sqlite", "sql server"],
    "power bi": ["power bi", "powerbi"],
    "ui design": ["ui design", "user interface design", "interface design"],
    "ux design": ["ux design", "user experience design", "usability"],
    "digital marketing": ["digital marketing", "online marketing"],
    "financial analysis": ["financial analysis", "financial modelling", "financial modeling"],
    "requirements analysis": ["requirements analysis", "business requirements"],
    "model evaluation": ["model evaluation", "model validation", "evaluation metrics"],
    "prototyping": ["prototyping", "prototype", "prototypes"],
    "content strategy": ["content strategy", "content planning"],
    "campaign analysis": ["campaign analysis", "marketing analytics"],
}

def normalize(text):
    return re.sub(r"[^a-z0-9+#.]+", " ", text.lower()).strip()

def contains_phrase(text, phrase):
    text_norm = normalize(text)
    phrase_norm = normalize(phrase)
    if not phrase_norm:
        return False
    return bool(re.search(
        r"(?<![a-z0-9])" + re.escape(phrase_norm) + r"(?![a-z0-9])",
        text_norm
    ))

def evidence_for(text, skill):
    candidates = [skill] + ALIASES.get(skill.lower(), [])
    return next((term for term in candidates if contains_phrase(text, term)), None)

def match_role(text, role, predicted_categories=None):
    required = role.get("required_skills", [])
    preferred = role.get("preferred_skills", [])
    projects = role.get("project_evidence", [])

    required_hits = [(s, evidence_for(text, s)) for s in required]
    preferred_hits = [(s, evidence_for(text, s)) for s in preferred]
    project_hits = [(s, evidence_for(text, s)) for s in projects]

    def rate(items):
        return sum(hit is not None for _, hit in items) / len(items) if items else 0.0

    required_rate = rate(required_hits)
    preferred_rate = rate(preferred_hits)
    project_rate = rate(project_hits)

    # Transparent heuristic weights; these are not statistically trained.
    score = 100 * (
        0.60 * required_rate +
        0.25 * preferred_rate +
        0.15 * project_rate
    )

    categories = role.get("categories", [])
    category_match = bool(predicted_categories and
                          set(categories).intersection(predicted_categories))

    return {
        "role_id": role["role_id"],
        "title": role["title"],
        "categories": categories,
        "fit_score": round(score, 1),
        "category_supported": category_match if predicted_categories else None,
        "matched_required": [s for s, hit in required_hits if hit],
        "missing_required_evidence": [s for s, hit in required_hits if not hit],
        "matched_preferred": [s for s, hit in preferred_hits if hit],
        "matched_project_evidence": [s for s, hit in project_hits if hit],
        "interpretation": (
            "Strong keyword evidence" if score >= 70 else
            "Some keyword evidence" if score >= 40 else
            "Limited keyword evidence"
        ),
    }

def recommend(text, catalog_path=DEFAULT_CATALOG, top_k=3,
              predicted_categories=None):
    data = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    roles = data.get("roles", [])
    if not roles:
        raise ValueError("The role catalogue contains no roles.")

    results = [match_role(text, role, predicted_categories) for role in roles]
    results.sort(key=lambda r: (r["fit_score"], len(r["matched_required"])), reverse=True)
    return results[:top_k]

def main():
    parser = argparse.ArgumentParser(description="Rank specific roles against resume text.")
    parser.add_argument("resume", help="Path to a plain-text resume")
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    resume_path = Path(args.resume).expanduser()
    if not resume_path.is_file():
        parser.error(f"Resume file not found: {resume_path}")

    text = resume_path.read_text(encoding="utf-8").strip()
    if not text:
        parser.error("Resume text is empty.")

    results = recommend(text, args.catalog, args.top_k)

    print("\nResumeMatch — Specific Job-Role Recommendations")
    print("=" * 48)
    print("These are heuristic evidence-match scores, not hiring probabilities.")

    for rank, item in enumerate(results, 1):
        print(f"\n{rank}. {item['title']} — Fit score: {item['fit_score']:.1f}/100")
        print("   Categories:", ", ".join(item["categories"]))
        print("   Assessment:", item["interpretation"])
        print("   Required evidence found:",
              ", ".join(item["matched_required"]) or "None identified")
        print("   Required evidence not found:",
              ", ".join(item["missing_required_evidence"]) or "None")
        print("   Preferred skills found:",
              ", ".join(item["matched_preferred"]) or "None identified")
        print("   Relevant project terms found:",
              ", ".join(item["matched_project_evidence"]) or "None identified")

if __name__ == "__main__":
    main()
