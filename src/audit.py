import hashlib
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config

EXPECTED = {"ID", "Resume_str", "Resume_html", "Category"}
DATA_FILE = config.DATA_DIR / "Resume.csv"


def stop(message):
    print(f"\n[AUDIT] {message}")
    sys.exit(1)


def main():
    if not DATA_FILE.is_file():
        stop(
            "Resume.csv not found.\n"
            "Download the Kaggle dataset, extract the ZIP, and place "
            "Resume.csv inside ~/ResumeMatch/data/"
        )

    before = hashlib.sha256(DATA_FILE.read_bytes()).hexdigest()

    try:
        df = pd.read_csv(DATA_FILE)
    except Exception as exc:
        stop(f"Could not read CSV: {type(exc).__name__}")

    missing_columns = EXPECTED - set(df.columns)
    if missing_columns:
        stop(f"Unexpected dataset schema. Missing columns: {sorted(missing_columns)}")

    text = df["Resume_str"].fillna("").astype(str).str.strip()
    labels = df["Category"]
    usable = text.ne("") & labels.notna()

    clean_text = text[usable].str.lower().str.replace(
        r"\s+", " ", regex=True
    )
    clean_labels = labels[usable].astype(str).str.strip()

    data = pd.DataFrame({
        "text": clean_text,
        "label": clean_labels
    })

    class_counts = clean_labels.value_counts()
    duplicate_groups = data.groupby("text")["label"].agg(
        count="size", labels="nunique"
    )

    duplicate_groups = duplicate_groups[duplicate_groups["count"] > 1]
    conflicting = duplicate_groups[duplicate_groups["labels"] > 1]

    words = text[usable].str.split().str.len()
    category_leakage = []

    for category in class_counts.index:
        pattern = (
            r"\b" +
            r"[\s\-/&]+".join(
                re.escape(token)
                for token in re.split(r"[-_\s]+", category.lower())
                if token
            ) +
            r"\b"
        )
        matching = clean_text[clean_labels == category]
        rate = (
            matching.str[:100].str.contains(pattern, regex=True).mean() * 100
        )
        category_leakage.append((category, int(class_counts[category]), round(rate, 2)))

    report = [
        "# ResumeMatch Dataset Audit",
        "",
        "Aggregate statistics only. No resume contents are included.",
        "",
        "## Dataset overview",
        f"- Rows: {len(df)}",
        f"- Columns: {', '.join(df.columns)}",
        f"- Job categories: {labels.nunique()}",
        f"- Missing resume texts: {int(df['Resume_str'].isna().sum())}",
        f"- Empty resume texts: {int(text.eq('').sum())}",
        f"- Missing category labels: {int(labels.isna().sum())}",
        f"- Usable labelled resumes: {int(usable.sum())}",
        "",
        "## Resume lengths",
        f"- Median words: {words.median() if len(words) else 0:.1f}",
        f"- Mean words: {words.mean() if len(words) else 0:.1f}",
        f"- Resumes under 50 words: {int((words < 50).sum())}",
        "",
        "## Category distribution",
        "| Category | Count |",
        "|---|---:|",
    ]

    report.extend(
        f"| {category} | {count} |"
        for category, count in class_counts.items()
    )

    report += [
        "",
        "## Duplicate checks",
        f"- Exact duplicate rows: {int(df.duplicated().sum())}",
        f"- Duplicate resume texts after normalisation: {int(data['text'].duplicated().sum())}",
        f"- Duplicate text groups: {len(duplicate_groups)}",
        f"- Groups with conflicting labels: {len(conflicting)}",
        f"- Rows in conflicting-label groups: {int(conflicting['count'].sum())}",
        "",
        "## Potential title-line leakage",
        "The following measures exact category-name matches in the first 100 characters.",
        "This is a warning signal, not proof of leakage.",
        "",
        "| Category | Count | Match rate (%) |",
        "|---|---:|---:|",
    ]

    report.extend(
        f"| {category} | {count} | {rate} |"
        for category, count, rate in category_leakage
    )

    report += [
        "",
        "## Notes",
        "- Resume_str is the proposed model input.",
        "- Category is the target label.",
        "- ID and Resume_html must not be model input features.",
        "- Duplicate handling and leakage checks must be reviewed before training.",
        "- No model has been trained by this audit.",
    ]

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    (config.RESULTS_DIR / "eda_summary.md").write_text(
        "\n".join(report), encoding="utf-8"
    )

    class_counts.sort_values().plot(kind="barh", figsize=(10, 8))
    plt.title("Resume Counts by Job Category")
    plt.xlabel("Number of resumes")
    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / "class_distribution.png", dpi=150)
    plt.close()

    after = hashlib.sha256(DATA_FILE.read_bytes()).hexdigest()
    if before != after:
        stop("Dataset file changed during audit.")

    print("Dataset audit completed.")
    print(f"Rows: {len(df)}")
    print(f"Categories: {labels.nunique()}")
    print(f"Usable resumes: {int(usable.sum())}")
    print(f"Duplicate text groups: {len(duplicate_groups)}")
    print(f"Conflicting-label groups: {len(conflicting)}")
    print(f"Report: {config.RESULTS_DIR / 'eda_summary.md'}")
    print(f"Chart: {config.RESULTS_DIR / 'class_distribution.png'}")


if __name__ == "__main__":
    main()
