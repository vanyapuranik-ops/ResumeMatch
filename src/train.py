"""Training pipeline for ResumeMatch (Experiments A and B, three classifiers).

Default mode is PREPARATION ONLY: load data, make a group-aware split, validate it, save
group assignments, build (but do not fit) pipelines. Models are fitted only with --run-training.

Dependencies: pandas, numpy, scikit-learn (>=1.0, for StratifiedGroupKFold).

Split: StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed), first fold is the
test set (~20%). Whole groups (group_id) stay on one side, so duplicates cannot cross the split.
Stratification is therefore approximate; duplicate groups are tiny, so the effect is small.
The script checks that every class appears in both train and test and warns otherwise.

Experiment A: raw Resume_str.
Experiment B: whole-word occurrences of any of the category names in the first 100
characters are replaced by a placeholder. The transform is label-agnostic, has no fitted
parameters, and is applied identically at training and inference.

Only Resume_str is used as a feature. ID, Resume_html and group_id are never features.
TF-IDF is fitted inside each Pipeline, i.e. on training data only.
Outputs are aggregate-only (no resume text, no feature names).
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = ROOT / "data" / "processed" / "resumes_clean.csv"
RESULTS_DIR = ROOT / "results"
SPLIT_DIR = RESULTS_DIR / "splits"
REQUIRED_COLS = ["ID", "Resume_str", "Category", "group_id"]
HEAD_CHARS = 100
PLACEHOLDER = "categorymask"
N_SPLITS = 5  # first fold = ~20% test


def norm(value):
    """Lowercase, replace punctuation with spaces, collapse whitespace (same as EDA)."""
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(value).lower())).strip()


def identity_text(texts):
    """Experiment A: raw text (only None/NaN safety)."""
    return [("" if pd.isna(t) else str(t)) for t in texts]


def mask_category_names(texts, category_names, head_chars=HEAD_CHARS, placeholder=PLACEHOLDER):
    """Experiment B: mask any category name (whole word/phrase) in the first head_chars chars.

    Rows with no match are returned unchanged, so A and B differ only where masking applies.
    """
    names = sorted({norm(c) for c in category_names if norm(c)}, key=len, reverse=True)
    pattern = re.compile(r"(?<![a-z0-9])(?:" + "|".join(re.escape(n) for n in names) + r")(?![a-z0-9])")
    out = []
    for t in texts:
        t = "" if pd.isna(t) else str(t)
        head = norm(t[:head_chars])
        if pattern.search(head):
            out.append(pattern.sub(placeholder, head) + " " + t[head_chars:])
        else:
            out.append(t)
    return out


def make_tfidf():
    return TfidfVectorizer(lowercase=True, stop_words="english", sublinear_tf=True,
                           ngram_range=(1, 2), min_df=2)


def make_classifiers(seed):
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced",
                                                 random_state=seed),
        "LinearSVM": LinearSVC(class_weight="balanced", random_state=seed),
        "MultinomialNB": MultinomialNB(),
    }


def make_pipeline(experiment, clf, category_names):
    if experiment == "A":
        prep = FunctionTransformer(identity_text, validate=False)
    else:
        prep = FunctionTransformer(mask_category_names, validate=False,
                                   kw_args={"category_names": list(category_names)})
    return Pipeline([("prep", prep), ("tfidf", make_tfidf()), ("clf", clf)])


def make_split(df, seed):
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    train_idx, test_idx = next(sgkf.split(df, df["Category"], groups=df["group_id"]))
    return np.sort(train_idx), np.sort(test_idx)


def validate_split(df, train_idx, test_idx):
    g_train = set(df.iloc[train_idx]["group_id"])
    g_test = set(df.iloc[test_idx]["group_id"])
    overlap = g_train & g_test
    if overlap:
        sys.exit(f"ERROR: {len(overlap)} group_id values appear in both train and test")
    if len(set(train_idx) & set(test_idx)) or len(train_idx) + len(test_idx) != len(df):
        sys.exit("ERROR: train/test indices overlap or do not cover all rows")
    classes = set(df["Category"])
    missing_train = classes - set(df.iloc[train_idx]["Category"])
    missing_test = classes - set(df.iloc[test_idx]["Category"])
    if missing_train or missing_test:
        print(f"WARNING: classes missing from train: {sorted(missing_train)}, "
              f"from test: {sorted(missing_test)}")
    return len(overlap)


def save_split(df, train_idx, test_idx, seed):
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    split = np.full(len(df), "train", dtype=object)
    split[test_idx] = "test"
    assign = pd.DataFrame({"row_index": np.arange(len(df)), "ID": df["ID"],
                           "group_id": df["group_id"], "split": split})
    path = SPLIT_DIR / f"split_seed{seed}.csv"
    assign.to_csv(path, index=False)
    per_cat = pd.DataFrame({
        "train": df.iloc[train_idx]["Category"].value_counts(),
        "test": df.iloc[test_idx]["Category"].value_counts(),
    }).fillna(0).astype(int)
    summary = {
        "seed": seed, "n_splits_for_holdout": N_SPLITS,
        "train_rows": int(len(train_idx)), "test_rows": int(len(test_idx)),
        "train_groups": int(df.iloc[train_idx]["group_id"].nunique()),
        "test_groups": int(df.iloc[test_idx]["group_id"].nunique()),
        "per_category_counts": per_cat.to_dict(orient="index"),
    }
    (SPLIT_DIR / f"split_seed{seed}_summary.json").write_text(json.dumps(summary, indent=2),
                                                              encoding="utf-8")
    return path, per_cat


def self_test_mask():
    """Synthetic-string sanity checks for transform B (no real resume data)."""
    cats = ["INFORMATION-TECHNOLOGY", "HR", "ACCOUNTANT"]
    out = mask_category_names(
        ["Information Technology Manager with HR duties " + "x" * 80 + " accountant later",
         "Chrome extension developer", "Shrink wrap"], cats)
    assert PLACEHOLDER in out[0].split(" ")[0:3] or out[0].startswith(PLACEHOLDER)
    assert out[0].count(PLACEHOLDER) == 2 and out[0].endswith("accountant later")
    assert out[1] == "Chrome extension developer" and out[2] == "Shrink wrap"  # no partial-word masking


def evaluate(y_true, y_pred, labels):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                  average="macro", zero_division=0)
    _, _, wf1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                   average="weighted", zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "macro_precision": p,
            "macro_recall": r, "macro_f1": f1, "weighted_f1": wf1}


def run_training(df, train_idx, test_idx, category_names, seed):
    X = df["Resume_str"]
    y = df["Category"]
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    labels = sorted(y.unique())
    rows, per_class_rows = [], []
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for exp in ("A", "B"):
        for name, clf in make_classifiers(seed).items():
            pipe = make_pipeline(exp, clf, category_names)
            pipe.fit(X_tr, y_tr)  # TF-IDF is fitted on training data only
            pred = pipe.predict(X_te)
            m = evaluate(y_te, pred, labels)
            rows.append({"experiment": exp, "model": name, **m})
            p, r, f, s = precision_recall_fscore_support(y_te, pred, labels=labels,
                                                         average=None, zero_division=0)
            for i, lab in enumerate(labels):
                per_class_rows.append({"experiment": exp, "model": name, "category": lab,
                                       "precision": p[i], "recall": r[i], "f1": f[i],
                                       "support": int(s[i])})
            cm = confusion_matrix(y_te, pred, labels=labels)
            pd.DataFrame(cm, index=labels, columns=labels).to_csv(
                RESULTS_DIR / f"confusion_matrix_{exp}_{name}.csv")
            print(f"[{exp}] {name}: acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f} "
                  f"weighted_f1={m['weighted_f1']:.4f}")
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(RESULTS_DIR / "metrics_per_class.csv", index=False)
    print("Saved metrics and confusion matrices to results/")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-training", action="store_true",
                        help="Fit and evaluate models. Without this flag nothing is fitted.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if not CLEAN_PATH.exists():
        sys.exit(f"Cleaned dataset not found: {CLEAN_PATH} (run src/preprocess.py first)")
    df = pd.read_csv(CLEAN_PATH)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        sys.exit(f"Missing required columns: {missing}")
    df = df.dropna(subset=["Resume_str", "Category"]).reset_index(drop=True)
    category_names = sorted(df["Category"].unique())

    train_idx, test_idx = make_split(df, args.seed)
    overlap = validate_split(df, train_idx, test_idx)
    split_path, per_cat = save_split(df, train_idx, test_idx, args.seed)
    self_test_mask()

    # Aggregate check of transform B on the full data: share of rows where the head is masked
    masked = mask_category_names(df["Resume_str"], category_names)
    changed = sum(1 for a, b in zip(df["Resume_str"].astype(str), masked) if a != b)

    # Build (do not fit) every pipeline to verify configuration
    n_pipes = sum(1 for exp in ("A", "B") for name, clf in make_classifiers(args.seed).items()
                  if make_pipeline(exp, clf, category_names) is not None)

    print(f"Rows: {len(df)} | Categories: {len(category_names)} | Seed: {args.seed}")
    print(f"Train rows: {len(train_idx)} | Test rows: {len(test_idx)} | "
          f"Group overlap: {overlap}")
    print(f"Smallest per-class test count: {int(per_cat['test'].min())} | "
          f"smallest per-class train count: {int(per_cat['train'].min())}")
    print(f"Transform B changes the first {HEAD_CHARS} chars in {changed} rows "
          f"({100 * changed / len(df):.1f}%)")
    print(f"Pipelines constructed (not fitted): {n_pipes}")
    print(f"Saved split assignments: {split_path.relative_to(ROOT)}")

    if args.run_training:
        run_training(df, train_idx, test_idx, category_names, args.seed)
    else:
        print("Preparation only: no models fitted. Use --run-training to train.")


if __name__ == "__main__":
    main()
