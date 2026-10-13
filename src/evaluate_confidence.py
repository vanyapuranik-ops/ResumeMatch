import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/resumes_clean.csv"
SPLIT = ROOT / "results/splits/split_seed42.csv"
MANIFEST = ROOT / "results/transformer/manifest_B_seed42_artifacts1.json"
MODEL_DIR = ROOT / "models/distilbert_B_seed42_artifacts1"
OUT = ROOT / "results/transformer/confidence_B_seed42_artifacts1.csv"
SEED = 42
MAX_LENGTH = 512
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
    head = norm(text[:HEAD_CHARS])
    if pattern.search(head):
        return pattern.sub(PLACEHOLDER, head) + " " + text[HEAD_CHARS:]
    return text


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    label2id = manifest["label2id"]
    classes = [name for name, idx in sorted(label2id.items(), key=lambda item: item[1])]

    data = pd.read_csv(DATA)
    split = pd.read_csv(SPLIT)
    train_idx = np.where(split["split"].to_numpy() == "train")[0]
    subset = data.iloc[train_idx]
    splitter = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=SEED)
    _, val_rel = next(splitter.split(subset, subset["Category"], groups=subset["group_id"]))
    val_idx = np.sort(train_idx[val_rel])
    val = data.iloc[val_idx].copy()

    saved = pd.read_csv(
        ROOT / "results/transformer/predictions_B_seed42_artifacts1.csv"
    )
    saved_val = saved[saved["split"] == "val"].sort_values("row_index")
    if len(val) != len(saved_val) or not np.array_equal(
        val_idx, saved_val["row_index"].to_numpy(dtype=int)
    ):
        raise SystemExit(
            "ERROR: Recreated validation split does not match saved predictions. "
            "No output was written."
        )

    if list(pd.unique(data["Category"])) != classes:
        if set(data["Category"]) != set(classes):
            raise SystemExit("ERROR: Dataset categories differ from the saved model mapping.")

    texts = [
        mask_category_names(text, classes)
        for text in val["Resume_str"].fillna("").astype(str).tolist()
    ]
    y = np.array([label2id[c] for c in val["Category"]], dtype=int)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR), local_files_only=True
    ).to(device)
    model.eval()

    all_logits = []
    with torch.inference_mode():
        for start in range(0, len(texts), 16):
            batch = tokenizer(
                texts[start:start + 16],
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
            )
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            all_logits.extend(logits.cpu().numpy())

    logits = np.asarray(all_logits)
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    pred = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    sorted_probs = np.sort(probs, axis=1)
    gap = sorted_probs[:, -1] - sorted_probs[:, -2]
    onehot = np.eye(len(classes))[y]
    brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))

    # Equal-width-bin expected calibration error; diagnostic, not definitive.
    ece = 0.0
    for lower in np.linspace(0, 0.9, 10):
        upper = lower + 0.1
        mask = (confidence >= lower) & (
            confidence < upper if upper < 1.0 else confidence <= upper
        )
        if mask.any():
            ece += mask.mean() * abs(
                (pred[mask] == y[mask]).mean() - confidence[mask].mean()
            )

    frame = pd.DataFrame({
        "row_index": val_idx,
        "true_category": val["Category"].to_numpy(),
        "predicted_category": [classes[i] for i in pred],
        "top_score": confidence,
        "second_score": sorted_probs[:, -2],
        "top_two_gap": gap,
        "correct": pred == y,
    })
    for i, category in enumerate(classes):
        frame["score_" + category.lower().replace("-", "_")] = probs[:, i]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)

    print("Validation resumes:", len(val))
    print("Accuracy:", round(accuracy_score(y, pred), 4))
    print("Macro F1:", round(f1_score(y, pred, labels=np.arange(len(classes)),
                                       average="macro", zero_division=0), 4))
    print("Multiclass log loss:", round(log_loss(y, probs, labels=np.arange(len(classes))), 4))
    print("Multiclass Brier score:", round(brier, 4))
    print("Expected calibration error (10 equal-width bins):", round(float(ece), 4))
    print("Mean top score:", round(float(confidence.mean()), 4))
    print("Median top-two score gap:", round(float(np.median(gap)), 4))
    print("Saved detailed validation scores to:", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
