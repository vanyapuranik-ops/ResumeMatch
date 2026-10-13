"""Aggregate-only EDA and target-leakage diagnostics for data/processed/resumes_clean.csv.

Outputs: results/eda_summary.md, results/category_distribution.png,
results/resume_length_distribution.png. No individual resume text is printed or saved.
Read-only on the dataset; no models are trained.
"""
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = ROOT / "data" / "processed" / "resumes_clean.csv"
RESULTS_DIR = ROOT / "results"
SUMMARY_PATH = RESULTS_DIR / "eda_summary.md"
CAT_PLOT = RESULTS_DIR / "category_distribution.png"
LEN_PLOT = RESULTS_DIR / "resume_length_distribution.png"
REQUIRED_COLS = ["ID", "Resume_str", "Category", "group_id"]
HEAD_CHARS = 100


def norm(value):
    """Lowercase, replace punctuation with spaces, collapse whitespace."""
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(value).lower())).strip()


def has_phrase(norm_text, norm_phrase):
    """Whole-word/phrase match on normalized text."""
    return bool(norm_phrase) and (" " + norm_phrase + " ") in (" " + norm_text + " ")


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def main():
    if not CLEAN_PATH.exists():
        sys.exit("Cleaned dataset not found: " + str(CLEAN_PATH) + " (run src/preprocess.py first)")
    df = pd.read_csv(CLEAN_PATH)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        sys.exit("Missing required columns: " + str(missing))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    n = len(df)
    df["word_count"] = df["Resume_str"].fillna("").map(lambda t: len(str(t).split()))

    # Class distribution
    counts = df["Category"].value_counts()
    imbalance = counts.max() / counts.min()
    cat_rows = [(c, int(v), "{:.1f}%".format(100 * v / n)) for c, v in counts.items()]

    # Word-count statistics
    wc = df["word_count"]
    stat_rows = [
        ("min", int(wc.min())),
        ("p25", "{:.0f}".format(wc.quantile(0.25))),
        ("median", "{:.0f}".format(wc.median())),
        ("mean", "{:.1f}".format(wc.mean())),
        ("p75", "{:.0f}".format(wc.quantile(0.75))),
        ("p95", "{:.0f}".format(wc.quantile(0.95))),
        ("max", int(wc.max())),
        ("std", "{:.1f}".format(wc.std())),
    ]
    median_by_cat = df.groupby("Category")["word_count"].median()

    # Duplicate-group statistics
    gsize = df.groupby("group_id").size()
    dup_groups = gsize[gsize > 1]
    purity = df[df["group_id"].isin(dup_groups.index)].groupby("group_id")["Category"].nunique()
    n_conflict = int((purity > 1).sum())

    # Category-name matching diagnostic
    df["_norm_full"] = df["Resume_str"].map(norm)
    df["_norm_head"] = df["Resume_str"].fillna("").map(lambda t: norm(str(t)[:HEAD_CHARS]))
    df["_cat_norm"] = df["Category"].map(norm)
    df["match_head"] = [has_phrase(t, c) for t, c in zip(df["_norm_head"], df["_cat_norm"])]
    df["match_any"] = [has_phrase(t, c) for t, c in zip(df["_norm_full"], df["_cat_norm"])]
    lk = df.groupby("Category").agg(
        n=("Category", "size"), head=("match_head", "sum"), anywhere=("match_any", "sum")
    )
    lk["head_pct"] = 100 * lk["head"] / lk["n"]
    lk["any_pct"] = 100 * lk["anywhere"] / lk["n"]
    lk = lk.sort_values("any_pct", ascending=False)
    leak_rows = [
        (c, int(r["n"]), int(r["head"]), "{:.1f}%".format(r["head_pct"]),
         int(r["anywhere"]), "{:.1f}%".format(r["any_pct"]))
        for c, r in lk.iterrows()
    ]
    overall_head = 100 * df["match_head"].mean()
    overall_any = 100 * df["match_any"].mean()

    # Plot 1: category distribution
    asc = counts.sort_values()
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(asc.index, asc.values)
    for i, v in enumerate(asc.values):
        ax.text(v + 0.5, i, str(v), va="center", fontsize=8)
    ax.set_xlabel("Number of resumes")
    ax.set_title("Category distribution")
    fig.tight_layout()
    fig.savefig(CAT_PLOT, dpi=150)
    plt.close(fig)

    # Plot 2: resume length distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(wc, bins=50)
    ax.axvline(wc.median(), linestyle="--", label="median = {:.0f}".format(wc.median()))
    ax.set_xlabel("Words per resume")
    ax.set_ylabel("Number of resumes")
    ax.set_title("Resume length distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(LEN_PLOT, dpi=150)
    plt.close(fig)

    # Aggregate-only Markdown report
    md = [
        "# ResumeMatch EDA Summary",
        "",
        "Aggregate statistics only. No individual resume text is included.",
        "",
        "## Overview",
        "- Clean rows: " + str(n),
        "- Categories: " + str(counts.size),
        "- Unique text groups: " + str(df["group_id"].nunique()),
        "",
        "## Class distribution",
        "- Smallest class: {} ({}); largest: {} ({})".format(
            counts.idxmin(), counts.min(), counts.idxmax(), counts.max()),
        "- Imbalance ratio (max/min): {:.2f}".format(imbalance),
        "",
        md_table(["Category", "Count", "Share"], cat_rows),
        "",
        "![Category distribution](category_distribution.png)",
        "",
        "## Resume length (words, whitespace-split)",
        md_table(["Statistic", "Value"], stat_rows),
        "",
        "Median words by category: lowest {} ({:.0f}), highest {} ({:.0f}).".format(
            median_by_cat.idxmin(), median_by_cat.min(),
            median_by_cat.idxmax(), median_by_cat.max()),
        "",
        "![Resume length distribution](resume_length_distribution.png)",
        "",
        "## Duplicate groups (normalized exact text)",
        "- Groups with more than one row: " + str(len(dup_groups)),
        "- Rows in those groups: " + str(int(dup_groups.sum())),
        "- Group sizes: " + str(sorted(int(s) for s in dup_groups)),
        "- Groups with conflicting labels: " + str(n_conflict),
        "- Strategy: no rows deleted; use group_id for group-aware train/test splits.",
        "- Only exact duplicates after normalization are detected; near-duplicates are not.",
        "",
        "## Category-name matching (DIAGNOSTIC ONLY)",
        "This is a diagnostic, not proof of leakage. A category name appearing in a resume may be "
        "a legitimate skill or domain term, or it may be a job-title echo that makes classification "
        "artificially easy. Matching is case-insensitive, punctuation-normalized, whole-word/phrase "
        "(for example INFORMATION-TECHNOLOGY matches 'information technology'). Short names such as "
        "HR or BPO may match unrelated words or acronyms.",
        "",
        "- Overall, own-category name in first {} chars: {:.1f}% of resumes".format(HEAD_CHARS, overall_head),
        "- Overall, own-category name anywhere: {:.1f}% of resumes".format(overall_any),
        "",
        md_table(
            ["Category", "n", "First {} chars".format(HEAD_CHARS), "%", "Anywhere", "%"],
            leak_rows,
        ),
        "",
    ]
    SUMMARY_PATH.write_text("\n".join(md), encoding="utf-8")

    print("Rows: {} | Categories: {} | Imbalance ratio: {:.2f}".format(n, counts.size, imbalance))
    print("Median words: {:.0f} | Duplicate groups: {}".format(wc.median(), len(dup_groups)))
    print("Own-category name match: first {} chars {:.1f}%, anywhere {:.1f}%".format(
        HEAD_CHARS, overall_head, overall_any))
    print("Saved:", ", ".join(str(p.relative_to(ROOT)) for p in (SUMMARY_PATH, CAT_PLOT, LEN_PLOT)))


if __name__ == "__main__":
    main()
