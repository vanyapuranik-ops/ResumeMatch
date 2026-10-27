"""Validation-only: recompute unrounded logits from the saved Experiment B checkpoint.

Inference only. Never reads test rows or test labels. Never writes model files. Refuses to overwrite.
Output: results/calibration/val_logits_B_seed42_artifacts1.csv (row_index + 24 logit columns; no text).
Parity gates: validation rows equal the saved ones, argmax equals the saved prediction on every row,
and softmax(logits) matches the stored probabilities within 1e-4.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train import CLEAN_PATH, ROOT, mask_category_names
from train_transformer import carve_validation, load_data, load_split

TAG = "B_seed42_artifacts1"
MODEL_DIR = ROOT / "models" / ("distilbert_" + TAG)
R = ROOT / "results" / "transformer"
OUT_DIR = ROOT / "results" / "calibration"
OUT = OUT_DIR / ("val_logits_" + TAG + ".csv")
SEED, MAX_LENGTH, BATCH = 42, 512, 8


def main():
    if OUT.exists():
        sys.exit("ERROR: refusing to overwrite " + str(OUT))
    if subprocess.run(["pgrep", "-f", "src/train_transformer.py"], capture_output=True).returncode == 0:
        sys.exit("STOP: a training process is running")
    man = json.loads((R / ("manifest_" + TAG + ".json")).read_text())
    label2id = man["label2id"]
    classes = [k for k, _ in sorted(label2id.items(), key=lambda kv: kv[1])]
    K = len(classes)
    df = load_data()
    train_idx, _test_idx = load_split(df, SEED)
    _tr, va_idx = carve_validation(df, train_idx, SEED)
    saved = pd.read_csv(R / ("predictions_" + TAG + ".csv"))
    sv = saved[saved["split"] == "val"].sort_values("row_index")
    conf = pd.read_csv(R / ("confidence_" + TAG + ".csv")).sort_values("row_index")
    cols = ["score_" + k.lower().replace("-", "_") for k in classes]
    if not (np.array_equal(sv["row_index"].to_numpy(int), va_idx) and np.array_equal(conf["row_index"].to_numpy(int), va_idx)):
        sys.exit("ERROR: rebuilt validation rows differ from saved files. Nothing written.")
    texts = mask_category_names(df.iloc[va_idx]["Resume_str"].astype(str).tolist(), classes)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR), local_files_only=True).to(device)
    model.eval()
    if [model.config.id2label[i] for i in range(K)] != classes:
        sys.exit("ERROR: checkpoint labels differ from the manifest")
    logits = np.zeros((len(texts), K), dtype=np.float64)
    with torch.inference_mode():
        for s in range(0, len(texts), BATCH):
            b = tok(texts[s:s + BATCH], return_tensors="pt", truncation=True, max_length=MAX_LENGTH, padding=True)
            b = {k: v.to(device) for k, v in b.items()}
            logits[s:s + BATCH] = model(**b).logits.float().cpu().numpy()
    if logits.shape != (len(va_idx), K) or not np.isfinite(logits).all():
        sys.exit("ERROR: unexpected or non-finite logits")

    z = logits - logits.max(1, keepdims=True)
    P = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    mism = int((np.array(classes)[logits.argmax(1)] != sv["pred_label"].to_numpy()).sum())
    pdiff = float(np.abs(P - conf[cols].to_numpy(float)).max())
    print("validation rows {} | argmax mismatches vs saved predictions {} | max |softmax - stored probability| {:.2e}".format(
        len(va_idx), mism, pdiff))
    if mism or pdiff > 1e-4:
        sys.exit("ERROR: parity gate failed. Nothing written.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(logits, columns=["logit_" + k.lower().replace("-", "_") for k in classes])
    frame.insert(0, "row_index", va_idx)
    frame.to_csv(OUT, index=False, float_format="%.9g")
    back = pd.read_csv(OUT)
    if not np.allclose(back.iloc[:, 1:].to_numpy(), logits, atol=1e-5) or not np.array_equal(back["row_index"].to_numpy(), va_idx):
        sys.exit("ERROR: read-back verification failed (file left in place; investigate)")
    print("Saved", OUT.relative_to(ROOT), "(read-back verified)")


if __name__ == "__main__":
    main()
