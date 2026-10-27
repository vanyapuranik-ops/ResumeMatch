"""Validation-only stress test: what happens when resumes are cut to their first N words?

Read-only on the saved Experiment B model; never saves weights; never reads test rows; never
overwrites outputs. Outputs hold no resume text and no true labels. The cut keeps the header, so this
is a proxy for short inputs, not a model of naturally short resumes. Experiment B masking is applied
after the cut (as predict.py would for a short pasted resume).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from evaluate_chunking import (CONFIDENCE, MANIFEST, MAX_LENGTH, MODEL_DIR, OUT_DIR, SAVED_PRED,
                               SPLIT, TAG, forward_logits, rebuild_validation, sha256_file, softmax)
from train import CLEAN_PATH, ROOT, evaluate, mask_category_names


def cut(text, n):
    """First n whitespace-separated words; n == 0 returns the text unchanged (full resume)."""
    return text if n == 0 else " ".join(text.split()[:n])


def run_all(model, device, tokenizer, texts, batch_size):
    ids = tokenizer(texts, truncation=True, max_length=MAX_LENGTH)["input_ids"]
    logits = forward_logits(model, ids, device, tokenizer.pad_token_id, batch_size)
    return logits, float(np.mean([len(i) for i in ids]))


def main():
    ap = argparse.ArgumentParser(description="Validation-only short-resume stress test")
    ap.add_argument("--limits", default="90,150,250", help="comma-separated word limits")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    assert cut("a b c d", 2) == "a b" and cut("x  y", 0) == "x  y" and cut("a b", 5) == "a b"
    try:
        limits = sorted({int(x) for x in args.limits.split(",")})
    except ValueError:
        sys.exit("ERROR: --limits must be comma-separated integers")
    if not limits or limits[0] < 10 or args.batch_size < 1:
        sys.exit("ERROR: each limit must be an integer >= 10 and batch-size >= 1")

    out_json = OUT_DIR / ("stress_short_" + TAG + "_metrics.json")
    out_rows = OUT_DIR / ("stress_short_" + TAG + "_rows.csv")
    clash = [str(p.relative_to(ROOT)) for p in (out_json, out_rows) if p.exists()]
    if clash:
        sys.exit("ERROR: refusing to overwrite: " + ", ".join(clash))

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    label2id = manifest["label2id"]
    classes = [c for c, _ in sorted(label2id.items(), key=lambda kv: kv[1])]
    K = len(classes)
    if (sha256_file(SPLIT) != manifest["sha256"]["split_file"]
            or sha256_file(CLEAN_PATH) != manifest["sha256"]["cleaned_csv"]):
        sys.exit("ERROR: split or cleaned CSV hash differs from the manifest")
    data = pd.read_csv(CLEAN_PATH)
    if data[["Resume_str", "Category"]].isna().any().any():
        sys.exit("ERROR: cleaned CSV has missing values")
    val_idx = rebuild_validation(data)
    saved_val = pd.read_csv(SAVED_PRED).query("split == 'val'").sort_values("row_index")
    conf = pd.read_csv(CONFIDENCE).sort_values("row_index")
    if not (np.array_equal(val_idx, saved_val["row_index"].to_numpy(dtype=int))
            and np.array_equal(val_idx, conf["row_index"].to_numpy(dtype=int))):
        sys.exit("ERROR: rebuilt validation rows differ from saved files. Nothing written.")
    val = data.iloc[val_idx]
    raw = val["Resume_str"].astype(str).tolist()
    y = np.array([label2id[c] for c in val["Category"]], dtype=int)
    n_words = np.array([len(t.split()) for t in raw])
    n = len(raw)
    print("Validation resumes: {} | words per resume: min {}, median {:.0f}, max {}".format(
        n, n_words.min(), np.median(n_words), n_words.max()))

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR), local_files_only=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    score_cols = ["score_" + c.lower().replace("-", "_") for c in classes]
    labels = list(range(K))
    results, preds, rows = {}, {}, {"row_index": val_idx, "n_words": n_words}

    for lim in [0] + limits:
        key = "full" if lim == 0 else str(lim)
        texts = mask_category_names([cut(t, lim) for t in raw], classes)
        try:
            logits, mean_tokens = run_all(model, device, tokenizer, texts, args.batch_size)
        except RuntimeError:
            if device.type != "mps":
                raise
            print("MPS failed; retrying on CPU")
            device = torch.device("cpu")
            model.to(device)
            logits, mean_tokens = run_all(model, device, tokenizer, texts, args.batch_size)
        if logits.shape != (n, K) or not np.isfinite(logits).all():
            sys.exit("ERROR: unexpected or non-finite logits")
        probs = softmax(logits)
        pred, top = probs.argmax(axis=1), probs.max(axis=1)
        if lim == 0:
            disagree = int((np.array(classes)[pred] != saved_val["pred_label"].to_numpy()).sum())
            diff = float(np.abs(probs - conf[score_cols].to_numpy()).max())
            print("Parity (full text): {} of {} predictions differ from saved | max prob diff {:.2e}".format(
                disagree, n, diff))
            if disagree or diff > 1e-2:
                sys.exit("ERROR: full-text pass does not reproduce saved results. Nothing written.")
        m = {k: float(v) for k, v in evaluate(y, pred, labels).items()}
        results[key] = {**m, "words_limit": lim, "resumes_cut": int((n_words > lim).sum()) if lim else 0,
                        "mean_tokens": mean_tokens, "mean_top_score": float(top.mean()),
                        "median_top_score": float(np.median(top)),
                        "share_top_score_le_0.5": float((top <= 0.5).mean())}
        preds[key] = pred
        rows["pred_" + key] = [classes[i] for i in pred]
        rows["top_" + key] = top

    print("\n{:<6}{:>6}{:>8}{:>8}{:>9}{:>8}{:>9}{:>8}{:>9}{:>10}".format(
        "limit", "cut", "tokens", "acc", "macroF1", "wF1", "meanTop", "medTop", "top<=.5", "fix/break"))
    for key, r in results.items():
        pc = ""
        if key != "full":
            ok0, ok1 = preds["full"] == y, preds[key] == y
            pc = "{}/{}".format(int((~ok0 & ok1).sum()), int((ok0 & ~ok1).sum()))
        print("{:<6}{:>6}{:>8.0f}{:>8.4f}{:>9.4f}{:>8.4f}{:>9.3f}{:>8.3f}{:>9.3f}{:>10}".format(
            key, r["resumes_cut"], r["mean_tokens"], r["accuracy"], r["macro_f1"], r["weighted_f1"],
            r["mean_top_score"], r["median_top_score"], r["share_top_score_le_0.5"], pc))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_rows, index=False)
    report = {"run_tag": TAG, "split_used": "validation only (248 rows); test rows never read",
              "weights_modified": False, "device": device.type, "word_limits": limits,
              "results": results, "parity_full_text": {"predictions_differing": disagree, "max_prob_diff": diff},
              "script_sha256": sha256_file(Path(__file__)),
              "limitations": ["Cutting a resume keeps its header; this is a proxy for short inputs, not real short resumes.",
                              "Model was trained on resumes of 113+ words; validation shortest quartile starts near 169.",
                              "Validation only, single seed."]}
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nSaved:", out_json.relative_to(ROOT), "and", out_rows.relative_to(ROOT))


if __name__ == "__main__":
    main()
