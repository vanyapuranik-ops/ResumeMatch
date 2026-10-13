"""Preprocess data/Resume.csv -> data/processed/resumes_clean.csv (raw CSV is never modified).

Duplicate strategy: no rows are deleted. Rows with identical normalized text share a
`group_id`. Future splits must use group-aware splitting (e.g. StratifiedGroupKFold or
GroupShuffleSplit with groups=group_id) so duplicates never cross train/test boundaries.
Only exact duplicates after normalization are detected; near-duplicates are not.
Output reports contain aggregate counts only, never resume text.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "Resume.csv"
OUT_DIR = ROOT / "data" / "processed"
CLEAN_PATH = OUT_DIR / "resumes_clean.csv"
SUMMARY_PATH = OUT_DIR / "preprocess_summary.json"
REQUIRED_COLS = ["ID", "Resume_str", "Resume_html", "Category"]


def normalize_label(value):
    """Strip and collapse whitespace; keep original casing. Returns None if missing/blank."""
    if pd.isna(value):
        return None
    label = re.sub(r"\s+", " ", str(value)).strip()
    return label or None


def normalize_text(value):
    """Lowercase, drop non-alphanumerics, collapse whitespace (used only for duplicate detection)."""
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(value).lower())).strip()


def text_hash(norm_text):
    return hashlib.sha1(norm_text.encode("utf-8")).hexdigest()


def main():
    if not RAW_PATH.exists():
        sys.exit(f"Dataset not found: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        sys.exit(f"Missing required columns: {missing}")

    summary = {"raw_rows": int(len(df))}

    # Normalize labels and text
    df["Category"] = df["Category"].map(normalize_label)
    df["_norm"] = df["Resume_str"].map(normalize_text)

    # Exclude unusable rows (record counts)
    missing_label = df["Category"].isna()
    empty_text = df["_norm"] == ""
    summary["excluded_missing_label"] = int(missing_label.sum())
    summary["excluded_empty_text"] = int(empty_text.sum())
    summary["excluded_total"] = int((missing_label | empty_text).sum())
    clean = df[~(missing_label | empty_text)].copy()

    # Duplicate / conflict detection on normalized text
    clean["text_hash"] = clean["_norm"].map(text_hash)
    grp = clean.groupby("text_hash")["Category"].agg(["size", "nunique"])
    dup = grp[grp["size"] > 1]
    conflicts = dup[dup["nunique"] > 1]
    summary["duplicate_text_groups"] = int(len(dup))
    summary["rows_in_duplicate_groups"] = int(dup["size"].sum())
    summary["duplicate_group_sizes"] = sorted(int(s) for s in dup["size"])
    summary["conflicting_label_groups"] = int(len(conflicts))

    # group_id for group-aware splitting (stable integer per unique text)
    clean["group_id"] = clean["text_hash"].factorize()[0]

    # Cleaned output: drop Resume_html and helper columns. ID kept for traceability ONLY, not a feature.
    out = clean[["ID", "Resume_str", "Category", "group_id"]].reset_index(drop=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(CLEAN_PATH, index=False)

    summary["clean_rows"] = int(len(out))
    summary["n_categories"] = int(out["Category"].nunique())
    summary["unique_groups"] = int(out["group_id"].nunique())
    summary["category_counts"] = {k: int(v) for k, v in out["Category"].value_counts().items()}
    summary["split_strategy"] = "group-aware split on group_id; no rows deleted"
    summary["clean_csv_note"] = "Contains resume text; keep under Git-ignored data/."

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    printable = {k: v for k, v in summary.items() if k != "category_counts"}
    print(json.dumps(printable, indent=2))
    print(f"Saved: {CLEAN_PATH.relative_to(ROOT)}, {SUMMARY_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
