import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/resumes_clean.csv"
SPLIT = ROOT / "results/splits/split_seed42.csv"
PRED = ROOT / "results/transformer/predictions_B_seed42_artifacts1.csv"
MANIFEST = ROOT / "results/transformer/manifest_B_seed42_artifacts1.json"
MODEL_DIR = ROOT / "models/distilbert_B_seed42_artifacts1"
OUT_DIR = ROOT / "results/calibration"
OUT = OUT_DIR / "test_confidence_B_seed42_artifacts1.csv"

MAX_LENGTH = 512
BATCH_SIZE = 8
HEAD_CHARS = 100
PLACEHOLDER = "categorymask"


def norm(value):
    import re
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(value).lower())).strip()


def mask_category_names(text, category_names):
    import re
    names = sorted({norm(c) for c in category_names if norm(c)}, key=len, reverse=True)
    pattern = re.compile(
        r"(?<![a-z0-9])(?:" + "|".join(re.escape(n) for n in names) + r")(?![a-z0-9])"
    )
    head = norm(str(text)[:HEAD_CHARS])
    if pattern.search(head):
        return pattern.sub(PLACEHOLDER, head) + " " + str(text)[HEAD_CHARS:]
    return str(text)


def main():
    for p in (DATA, SPLIT, PRED, MANIFEST, MODEL_DIR):
        if not p.exists():
            sys.exit(f"ERROR: Required path missing: {p}")

    df = pd.read_csv(DATA).dropna(subset=["Resume_str", "Category"]).reset_index(drop=True)
    split = pd.read_csv(SPLIT)
    saved = pd.read_csv(PRED)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    label2id = manifest["label2id"]
    classes = [name for name, idx in sorted(label2id.items(), key=lambda kv: kv[1])]

    if len(df) != len(split) or not np.array_equal(split["row_index"], np.arange(len(df))):
        sys.exit("ERROR: Dataset and split row order mismatch.")
    if not np.array_equal(split["ID"], df["ID"]) or not np.array_equal(split["group_id"], df["group_id"]):
        sys.exit("ERROR: Dataset IDs/groups do not match saved split.")

    test_rows = split.index[split["split"].eq("test")].to_numpy()
    saved_test = saved[saved["split"].eq("test")].copy()
    saved_test["row_index"] = saved_test["row_index"].astype(int)
    if set(saved_test["row_index"]) != set(test_rows):
        sys.exit("ERROR: Saved predictions do not match the test split.")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Device:", device)
    print("Loading saved model and tokenizer; no training will occur...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR), local_files_only=True
    ).to(device)
    model.eval()

    test_texts = [
        mask_category_names(df.iloc[i]["Resume_str"], classes)
        for i in test_rows
    ]
    actual_labels = df.iloc[test_rows]["Category"].astype(str).to_numpy()
    all_probs = []

    print(f"Running inference on {len(test_rows)} test resumes in batches of {BATCH_SIZE}...")
    with torch.inference_mode():
        for start in range(0, len(test_texts), BATCH_SIZE):
            batch_texts = test_texts[start:start + BATCH_SIZE]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
            done = min(start + BATCH_SIZE, len(test_texts))
            if done % 80 < BATCH_SIZE or done == len(test_texts):
                print(f"Processed {done}/{len(test_texts)}")

    probs = np.concatenate(all_probs, axis=0)
    if probs.shape != (len(test_rows), len(classes)):
        sys.exit(f"ERROR: Unexpected probability matrix shape: {probs.shape}")

    pred_ids = probs.argmax(axis=1)
    pred_labels = np.array(classes, dtype=object)[pred_ids]

    expected = saved_test.set_index("row_index").loc[test_rows]
    expected_labels = expected["pred_label"].astype(str).to_numpy()
    mismatches = np.flatnonzero(pred_labels != expected_labels)
    if len(mismatches):
        print(f"PARITY CHECK FAILED: {len(mismatches)} predictions differ.")
        for j in mismatches[:10]:
            print(
                f"row_index={test_rows[j]} fresh={pred_labels[j]} "
                f"saved={expected_labels[j]}"
            )
        sys.exit("No output file saved. Investigate parity before evaluating confidence.")

    print("PASS: Fresh predictions match all 497 saved test labels.")

    out = pd.DataFrame({
        "row_index": test_rows,
        "true_category": actual_labels,
        "predicted_category": pred_labels,
        "top_score": probs.max(axis=1),
        "second_score": np.sort(probs, axis=1)[:, -2],
        "top_two_gap": np.sort(probs, axis=1)[:, -1] - np.sort(probs, axis=1)[:, -2],
        "correct": pred_labels == actual_labels,
    })
    for i, category in enumerate(classes):
        col = "score_" + norm(category).replace(" ", "_")
        out[col] = probs[:, i]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        sys.exit(f"ERROR: Output already exists; refusing to overwrite: {OUT}")

    out.to_csv(OUT, index=False)
    print("Saved:", OUT.relative_to(ROOT))
    print("Rows:", len(out), "| class-score columns:", len(classes))
    print("Accuracy from fresh inference:", float(np.mean(pred_labels == actual_labels)))
    print("Expected manifest accuracy:", manifest["test_metrics"]["accuracy"])
    print("Mean top score:", float(out["top_score"].mean()))
    print("Share with top score <= 0.70:", float((out["top_score"] <= 0.70).mean()))
    print("Done. Existing model and prediction files were not changed.")


if __name__ == "__main__":
    main()
