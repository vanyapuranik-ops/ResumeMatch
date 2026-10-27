import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.metrics import log_loss
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train import mask_category_names
from train_transformer import load_data, load_split, carve_validation

ROOT = Path.cwd()
TAG = "B_seed42_longer10"
MODEL = ROOT / "models" / f"distilbert_{TAG}"
RESULTS = ROOT / "results" / "transformer"
OUTDIR = ROOT / "results" / "calibration"
OUT = OUTDIR / f"temperature_{TAG}.json"

def main():
    if OUT.exists():
        sys.exit(f"Refusing to overwrite {OUT}")

    manifest = json.loads((RESULTS / f"manifest_{TAG}.json").read_text())
    classes = [
        label for label, idx in
        sorted(manifest["label2id"].items(), key=lambda x: x[1])
    ]
    df = load_data()
    train_idx, _ = load_split(df, 42)
    _, val_idx = carve_validation(df, train_idx, 42)

    saved = pd.read_csv(RESULTS / f"predictions_{TAG}.csv")
    val_saved = saved[saved.split == "val"].sort_values("row_index")
    if not np.array_equal(val_saved.row_index.to_numpy(int), val_idx):
        sys.exit("Validation row mismatch; no output written.")

    texts = mask_category_names(
        df.iloc[val_idx]["Resume_str"].astype(str).tolist(), classes
    )
    labels = np.array([manifest["label2id"][x]
                       for x in df.iloc[val_idx]["Category"].astype(str)])

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL), local_files_only=True
    ).to(device)
    model.eval()

    chunks = []
    with torch.inference_mode():
        for start in range(0, len(texts), 8):
            batch = tokenizer(
                texts[start:start + 8], truncation=True, max_length=512,
                padding=True, return_tensors="pt"
            )
            batch = {k: v.to(device) for k, v in batch.items()}
            chunks.append(model(**batch).logits.float().cpu().numpy())

    logits = np.concatenate(chunks)
    if logits.shape != (len(val_idx), len(classes)) or not np.isfinite(logits).all():
        sys.exit("Invalid logits; no output written.")

    pred = np.array(classes)[logits.argmax(axis=1)]
    if not np.array_equal(pred, val_saved.pred_label.to_numpy()):
        sys.exit("Checkpoint predictions differ from saved validation predictions.")

    def nll(log_t):
        p = softmax(logits / np.exp(log_t), axis=1)
        return log_loss(labels, p, labels=np.arange(len(classes)))

    raw_nll = nll(0.0)
    result = minimize_scalar(nll, bounds=(-3, 3), method="bounded")
    temperature = float(np.exp(result.x))
    calibrated = softmax(logits / temperature, axis=1)

    def brier(p):
        onehot = np.eye(len(classes))[labels]
        return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))

    raw = softmax(logits, axis=1)
    def ece(p, bins=10):
        conf = p.max(axis=1)
        correct = (p.argmax(axis=1) == labels).astype(float)
        total = 0.0
        for lo in np.linspace(0, 1, bins + 1)[:-1]:
            hi = lo + 0.1
            mask = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
            if mask.any():
                total += mask.mean() * abs(conf[mask].mean() - correct[mask].mean())
        return float(total)

    report = {
        "model_tag": TAG,
        "split": "validation_only",
        "validation_rows": int(len(labels)),
        "temperature": temperature,
        "raw_log_loss": float(raw_nll),
        "calibrated_log_loss": float(nll(result.x)),
        "raw_brier": brier(raw),
        "calibrated_brier": brier(calibrated),
        "raw_ece_10_bins": ece(raw),
        "calibrated_ece_10_bins": ece(calibrated),
        "raw_mean_top_score": float(raw.max(axis=1).mean()),
        "calibrated_mean_top_score": float(calibrated.max(axis=1).mean()),
        "validation_accuracy_unchanged": float((raw.argmax(axis=1) == labels).mean()),
    }

    OUTDIR.mkdir(parents=True, exist_ok=True)
    with OUT.open("x") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nSaved: {OUT.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
