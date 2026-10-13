"""Validation-only comparison: overlapping-chunk inference vs the first-512-token baseline.

Read-only: loads the saved Experiment B model, never saves weights, never overwrites outputs.
Uses ONLY the 248 validation rows, rebuilt exactly as in src/evaluate_confidence.py and verified
against results/transformer/predictions_B_seed42_artifacts1.csv. Test rows are never read.
Preprocessing is Experiment B masking (train.mask_category_names, identical to training), applied
to raw text before tokenization; the mask never uses a row's label.
Strategies (fixed in advance): baseline_first_512, chunked_mean_logits, chunked_mean_probs.
Outputs contain no resume text.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train import CLEAN_PATH, ROOT, SPLIT_DIR, evaluate, mask_category_names

TAG = "B_seed42_artifacts1"
MODEL_DIR = ROOT / "models" / ("distilbert_" + TAG)
OUT_DIR = ROOT / "results" / "transformer"
MANIFEST = OUT_DIR / ("manifest_" + TAG + ".json")
SAVED_PRED = OUT_DIR / ("predictions_" + TAG + ".csv")
CONFIDENCE = OUT_DIR / ("confidence_" + TAG + ".csv")
SPLIT = SPLIT_DIR / "split_seed42.csv"
SEED = 42
VAL_FOLDS = 8
MAX_LENGTH = 512
SHORT_CHUNK = 64
STRATEGIES = ["baseline_first_512", "chunked_mean_logits", "chunked_mean_probs"]


# ---------- pure helpers (covered by the synthetic self-test) ----------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def softmax(x):
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def check_mapping(mapping, n):
    """chunk -> resume map must be non-decreasing and cover every resume 0..n-1 at least once."""
    mapping = np.asarray(mapping)
    if mapping.ndim != 1 or len(mapping) == 0:
        raise ValueError("chunk mapping is empty or not 1-D")
    if np.any(mapping[1:] < mapping[:-1]):
        raise ValueError("chunk mapping is not non-decreasing")
    if mapping.min() != 0 or mapping.max() != n - 1 or len(np.unique(mapping)) != n:
        raise ValueError("chunk mapping does not cover every resume exactly 0..n-1")


def aggregate(logits, mapping, n):
    """Per-resume mean logits and mean softmax probabilities (float64)."""
    logits = np.asarray(logits, dtype=np.float64)
    counts = np.bincount(mapping, minlength=n)
    sum_l = np.zeros((n, logits.shape[1]))
    np.add.at(sum_l, mapping, logits)
    probs = softmax(logits)
    sum_p = np.zeros_like(sum_l)
    np.add.at(sum_p, mapping, probs)
    return sum_l / counts[:, None], sum_p / counts[:, None], counts


def fast_macro_f1(y, pred, k):
    """Macro F1 over all k classes, zero_division=0 (equals sklearn with labels=range(k))."""
    tp = np.bincount(y[y == pred], minlength=k).astype(float)
    denom = (np.bincount(y, minlength=k) + np.bincount(pred, minlength=k)).astype(float)
    return float(np.mean(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)))


def paired_counts(y, base, other):
    bc, oc = base == y, other == y
    return {"fixed_by_chunking": int((~bc & oc).sum()), "broken_by_chunking": int((bc & ~oc).sum()),
            "both_correct": int((bc & oc).sum()), "both_wrong": int((~bc & ~oc).sum()),
            "prediction_changed": int((base != other).sum())}


def sign_test_p(fixed, broken):
    n = fixed + broken
    if n == 0:
        return 1.0
    from scipy.stats import binomtest
    return float(binomtest(fixed, n, 0.5).pvalue)


def group_bootstrap(y, base, other, groups, k, n_boot, seed):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    members = [np.where(groups == g)[0] for g in uniq]
    d_acc, d_f1 = [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[i] for i in pick])
        d_acc.append(float((other[idx] == y[idx]).mean() - (base[idx] == y[idx]).mean()))
        d_f1.append(fast_macro_f1(y[idx], other[idx], k) - fast_macro_f1(y[idx], base[idx], k))
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {"n_boot": n_boot, "resampling_unit": "group_id",
            "accuracy_diff_ci95": ci(d_acc), "macro_f1_diff_ci95": ci(d_f1)}


def per_class_frame(strategy, y, pred, classes):
    p, r, f, s = precision_recall_fscore_support(
        y, pred, labels=list(range(len(classes))), average=None, zero_division=0)
    return pd.DataFrame({"strategy": strategy, "category": classes, "precision": p,
                         "recall": r, "f1": f, "support": s.astype(int)})


def raises(fn, *a):
    try:
        fn(*a)
    except ValueError:
        return True
    return False


def self_test():
    big = softmax(np.array([[1000.0, 0.0, -1000.0]]))
    assert np.isfinite(big).all() and abs(big.sum() - 1) < 1e-12 and big[0, 0] > 0.999
    # one resume, 3 chunks: mean logits and mean probs disagree by construction
    lg = np.array([[10.0, 0, 0], [0, 2.0, 0], [0, 2.0, 0]])
    ml, mp, c = aggregate(lg, np.array([0, 0, 0]), 1)
    assert c.tolist() == [3] and ml[0].argmax() == 0 and mp[0].argmax() == 1
    assert abs(mp.sum() - 1) < 1e-12
    # two resumes with different chunk counts keep chunks with the right resume
    lg2 = np.array([[2.0, 0, 0], [0, 4.0, 0], [0, 1.0, 0]])
    ml2, mp2, c2 = aggregate(lg2, np.array([0, 0, 1]), 2)
    assert c2.tolist() == [2, 1] and np.allclose(ml2[0], [1, 2, 0]) and np.allclose(ml2[1], [0, 1, 0])
    assert raises(check_mapping, [0, 0, 2], 3) and raises(check_mapping, [1, 0], 2)
    assert raises(check_mapping, [0, 0, 1], 3) and raises(check_mapping, [], 1)
    check_mapping([0, 0, 1, 2, 2], 3)
    rng = np.random.default_rng(0)
    y = rng.integers(0, 6, 80)
    pred = np.where(rng.random(80) < 0.6, y, rng.integers(0, 6, 80))
    assert abs(fast_macro_f1(y, pred, 8) - f1_score(
        y, pred, labels=list(range(8)), average="macro", zero_division=0)) < 1e-12
    other = np.where(rng.random(80) < 0.8, y, pred)
    groups = np.arange(80) // 2
    b1 = group_bootstrap(y, pred, other, groups, 8, 50, 1)
    assert b1 == group_bootstrap(y, pred, other, groups, 8, 50, 1)
    pc = paired_counts(y, pred, other)
    assert pc["fixed_by_chunking"] + pc["broken_by_chunking"] + pc["both_correct"] + pc["both_wrong"] == 80
    assert sign_test_p(0, 0) == 1.0 and 0 < sign_test_p(9, 1) < 0.05


# ---------- data / model helpers ----------
def rebuild_validation(data):
    split = pd.read_csv(SPLIT)
    if len(split) != len(data) or not np.array_equal(split["row_index"].to_numpy(), np.arange(len(data))):
        sys.exit("ERROR: saved split does not match the cleaned dataset (row count/order)")
    train_idx = np.where(split["split"].to_numpy() == "train")[0]
    sub = data.iloc[train_idx]
    sgkf = StratifiedGroupKFold(n_splits=VAL_FOLDS, shuffle=True, random_state=SEED)
    _, va_rel = next(sgkf.split(sub, sub["Category"], groups=sub["group_id"]))
    return np.sort(train_idx[va_rel])


def forward_logits(model, id_lists, device, pad_id, batch_size):
    out = np.zeros((len(id_lists), model.config.num_labels), dtype=np.float64)
    model.eval()
    with torch.inference_mode():
        for s in range(0, len(id_lists), batch_size):
            batch = id_lists[s:s + batch_size]
            longest = max(len(b) for b in batch)
            inp = torch.full((len(batch), longest), pad_id, dtype=torch.long)
            att = torch.zeros((len(batch), longest), dtype=torch.long)
            for i, ids in enumerate(batch):
                inp[i, :len(ids)] = torch.tensor(ids)
                att[i, :len(ids)] = 1
            logits = model(input_ids=inp.to(device), attention_mask=att.to(device)).logits
            out[s:s + len(batch)] = logits.float().cpu().numpy()
    return out


def run_inference(model, device, pad_id, base_ids, chunk_ids, batch_size):
    model.to(device)
    forward_logits(model, base_ids[:batch_size], device, pad_id, batch_size)  # warm-up, untimed
    t0 = time.perf_counter()
    base = forward_logits(model, base_ids, device, pad_id, batch_size)
    t_base = time.perf_counter() - t0
    t0 = time.perf_counter()
    chunk = forward_logits(model, chunk_ids, device, pad_id, batch_size)
    t_chunk = time.perf_counter() - t0
    return base, chunk, t_base, t_chunk


def to_json(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o).__name__)


def main():
    ap = argparse.ArgumentParser(description="Validation-only chunked-inference evaluation")
    ap.add_argument("--stride", type=int, default=128, help="token overlap between windows")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--self-test", action="store_true", help="run synthetic checks only and exit")
    args = ap.parse_args()

    self_test()
    print("Synthetic self-tests passed.")
    if args.self_test:
        return
    if not 0 <= args.stride < MAX_LENGTH - 2 or args.batch_size < 1 or args.n_boot < 1:
        sys.exit("ERROR: need 0 <= stride < 510, batch-size >= 1, n-boot >= 1")

    run_tag = TAG + "_stride" + str(args.stride)
    out_json = OUT_DIR / ("chunking_" + run_tag + "_metrics.json")
    out_pc = OUT_DIR / ("chunking_" + run_tag + "_per_class.csv")
    out_pred = OUT_DIR / ("chunking_" + run_tag + "_predictions.csv")
    clash = [str(p.relative_to(ROOT)) for p in (out_json, out_pc, out_pred) if p.exists()]
    if clash:
        sys.exit("ERROR: refusing to overwrite: " + ", ".join(clash))

    # --- manifest, hashes, label mapping ---
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    label2id = manifest["label2id"]
    classes = [c for c, _ in sorted(label2id.items(), key=lambda kv: kv[1])]
    K = len(classes)
    if sorted(label2id.values()) != list(range(K)) or K != 24:
        sys.exit("ERROR: manifest label2id is not a 0..23 mapping")
    if sha256_file(SPLIT) != manifest["sha256"]["split_file"]:
        sys.exit("ERROR: split file hash differs from the manifest")
    if sha256_file(CLEAN_PATH) != manifest["sha256"]["cleaned_csv"]:
        sys.exit("ERROR: cleaned CSV hash differs from the manifest")

    # --- validation set (validation rows only) ---
    data = pd.read_csv(CLEAN_PATH)
    if data[["Resume_str", "Category"]].isna().any().any():
        sys.exit("ERROR: cleaned CSV has missing values; row indices would not align with training")
    if not set(data["Category"]) <= set(classes):
        sys.exit("ERROR: dataset has categories outside the model mapping")
    val_idx = rebuild_validation(data)
    saved = pd.read_csv(SAVED_PRED)
    saved_val = saved[saved["split"] == "val"].sort_values("row_index")
    if not np.array_equal(val_idx, saved_val["row_index"].to_numpy(dtype=int)):
        sys.exit("ERROR: rebuilt validation rows differ from saved predictions. Nothing written.")
    conf = pd.read_csv(CONFIDENCE).sort_values("row_index")
    if not np.array_equal(val_idx, conf["row_index"].to_numpy(dtype=int)):
        sys.exit("ERROR: confidence file rows differ from the validation rows. Nothing written.")
    score_cols = ["score_" + c.lower().replace("-", "_") for c in classes]
    if any(c not in conf.columns for c in score_cols):
        sys.exit("ERROR: confidence file lacks the expected score columns")
    val = data.iloc[val_idx]
    y = np.array([label2id[c] for c in val["Category"]], dtype=int)
    groups = val["group_id"].to_numpy()
    n = len(val)
    print("Validation resumes:", n, "| unique groups:", len(np.unique(groups)), "| classes:", K)

    # --- masking (Experiment B, label-agnostic) and tokenization ---
    texts = mask_category_names(val["Resume_str"].tolist(), classes)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    if not tokenizer.is_fast:
        sys.exit("ERROR: a fast tokenizer is required for overflow/stride windows")
    t0 = time.perf_counter()
    base_ids = tokenizer(texts, truncation=True, max_length=MAX_LENGTH)["input_ids"]
    enc = tokenizer(texts, truncation=True, max_length=MAX_LENGTH, stride=args.stride,
                    return_overflowing_tokens=True)
    t_tok = time.perf_counter() - t0
    chunk_ids = enc["input_ids"]
    mapping = np.asarray(enc["overflow_to_sample_mapping"], dtype=int)
    try:
        check_mapping(mapping, n)
    except ValueError as e:
        sys.exit("ERROR: " + str(e))
    if len(chunk_ids) != len(mapping):
        sys.exit("ERROR: chunk and mapping lengths differ")
    lens = np.array([len(c) for c in chunk_ids])
    if lens.max() > MAX_LENGTH or lens.min() < 3:
        sys.exit("ERROR: chunk length outside 3..512")
    if any(c[0] != tokenizer.cls_token_id or c[-1] != tokenizer.sep_token_id for c in chunk_ids):
        sys.exit("ERROR: a chunk lacks [CLS] at the start or [SEP] at the end")
    first = np.r_[True, mapping[1:] != mapping[:-1]]
    first_chunks = [c for c, f in zip(chunk_ids, first) if f]
    if first_chunks != base_ids:
        sys.exit("ERROR: first chunk differs from the standard 512-token tokenization")
    counts_tok = np.bincount(mapping, minlength=n)
    print("Chunks: {} total | per resume min/median/mean/max = {}/{:.0f}/{:.2f}/{} | "
          "resumes with >1 chunk: {} | chunks under {} tokens: {}".format(
              len(chunk_ids), counts_tok.min(), np.median(counts_tok), counts_tok.mean(),
              counts_tok.max(), int((counts_tok > 1).sum()), SHORT_CHUNK, int((lens < SHORT_CHUNK).sum())))

    # --- model (read-only) ---
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR), local_files_only=True)
    if model.config.num_labels != K or [model.config.id2label[i] for i in range(K)] != classes:
        sys.exit("ERROR: model id2label does not match the manifest label2id")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    try:
        base_lg, chunk_lg, t_base, t_chunk = run_inference(
            model, device, tokenizer.pad_token_id, base_ids, chunk_ids, args.batch_size)
    except RuntimeError as e:
        if device.type != "mps":
            raise
        print("MPS failed ({}); retrying on CPU".format(type(e).__name__))
        device = torch.device("cpu")
        base_lg, chunk_lg, t_base, t_chunk = run_inference(
            model, device, tokenizer.pad_token_id, base_ids, chunk_ids, args.batch_size)
    if base_lg.shape != (n, K) or chunk_lg.shape != (len(chunk_ids), K):
        sys.exit("ERROR: unexpected logits shape")
    if not (np.isfinite(base_lg).all() and np.isfinite(chunk_lg).all()):
        sys.exit("ERROR: non-finite logits")

    # --- parity gates against saved evidence ---
    base_pred = base_lg.argmax(axis=1)
    disagree = int((np.array(classes)[base_pred] != saved_val["pred_label"].to_numpy()).sum())
    max_prob_diff = float(np.abs(softmax(base_lg) - conf[score_cols].to_numpy()).max())
    chunk0_logit_diff = float(np.abs(chunk_lg[first] - base_lg).max())
    print("Parity | baseline vs saved training-time predictions: {} of {} rows differ | "
          "max |prob - saved confidence prob| = {:.2e} | max |chunk0 logit - baseline logit| = {:.2e}".format(
              disagree, n, max_prob_diff, chunk0_logit_diff))
    if disagree > 0 or max_prob_diff > 1e-2:
        sys.exit("ERROR: baseline does not reproduce the saved predictions. Nothing written; "
                 "chunking results would not be comparable.")

    # --- aggregation and metrics ---
    mean_lg, mean_pr, counts = aggregate(chunk_lg, mapping, n)
    if not np.allclose(mean_pr.sum(axis=1), 1.0, atol=1e-9) or not np.array_equal(counts, counts_tok):
        sys.exit("ERROR: aggregation check failed")
    preds = {"baseline_first_512": base_pred, "chunked_mean_logits": mean_lg.argmax(axis=1),
             "chunked_mean_probs": mean_pr.argmax(axis=1)}
    labels = list(range(K))
    metrics = {s: {k: float(v) for k, v in evaluate(y, preds[s], labels).items()} for s in STRATEGIES}
    base_f1_manifest = float(manifest["best_val_macro_f1"])
    print("Baseline macro F1 {:.4f} (manifest best_val_macro_f1 {:.4f})".format(
        metrics["baseline_first_512"]["macro_f1"], base_f1_manifest))
    print("\n{:<22}{:>9}{:>9}{:>9}{:>9}{:>11}".format(
        "strategy", "acc", "macroP", "macroR", "macroF1", "weightedF1"))
    for s in STRATEGIES:
        m = metrics[s]
        print("{:<22}{:>9.4f}{:>9.4f}{:>9.4f}{:>9.4f}{:>11.4f}".format(
            s, m["accuracy"], m["macro_precision"], m["macro_recall"], m["macro_f1"], m["weighted_f1"]))

    comparisons = {}
    for s in STRATEGIES[1:]:
        pc = paired_counts(y, preds["baseline_first_512"], preds[s])
        pc["sign_test_p_two_sided"] = sign_test_p(pc["fixed_by_chunking"], pc["broken_by_chunking"])
        pc["bootstrap"] = group_bootstrap(y, preds["baseline_first_512"], preds[s], groups, K,
                                          args.n_boot, SEED)
        comparisons[s] = pc
        print("\n{} vs baseline: fixed {} | broken {} | sign-test p {:.3f} | "
              "acc diff 95% CI {:.4f}..{:.4f} | macro-F1 diff 95% CI {:.4f}..{:.4f}".format(
                  s, pc["fixed_by_chunking"], pc["broken_by_chunking"], pc["sign_test_p_two_sided"],
                  *pc["bootstrap"]["accuracy_diff_ci95"], *pc["bootstrap"]["macro_f1_diff_ci95"]))

    pcf = pd.concat([per_class_frame(s, y, preds[s], classes) for s in STRATEGIES], ignore_index=True)
    wide = pcf.pivot(index="category", columns="strategy", values="f1")
    support = pcf[pcf["strategy"] == STRATEGIES[0]].set_index("category")["support"]
    print("\nPer-class F1 (validation):")
    print("{:<28}{:>5}{:>10}{:>10}{:>10}".format("category", "n", "base", "mean_lg", "mean_pr"))
    for c in classes:
        print("{:<28}{:>5}{:>10.3f}{:>10.3f}{:>10.3f}".format(
            c, int(support[c]), wide.loc[c, STRATEGIES[0]], wide.loc[c, STRATEGIES[1]],
            wide.loc[c, STRATEGIES[2]]))
    print("\nForward-pass time: baseline {:.1f}s ({} sequences) | chunked {:.1f}s ({} chunks, "
          "{:.1f}x) | tokenization {:.1f}s | device {}".format(
              t_base, n, t_chunk, len(chunk_ids), t_chunk / max(t_base, 1e-9), t_tok, device.type))

    # --- save new outputs (no text, no true labels) ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pcf.to_csv(out_pc, index=False)
    pd.DataFrame({"row_index": val_idx, "n_chunks": counts,
                  "pred_baseline": [classes[i] for i in preds["baseline_first_512"]],
                  "pred_mean_logits": [classes[i] for i in preds["chunked_mean_logits"]],
                  "pred_mean_probs": [classes[i] for i in preds["chunked_mean_probs"]]}
                 ).to_csv(out_pred, index=False)
    report = {
        "run_tag": run_tag, "split_used": "validation only (248 rows); test rows never read",
        "model_dir": str(MODEL_DIR.relative_to(ROOT)), "weights_modified": False,
        "preprocessing": "Experiment B head masking via train.mask_category_names, applied before tokenization",
        "max_length": MAX_LENGTH, "stride": args.stride,
        "chunk_stats": {"total_chunks": int(len(chunk_ids)), "min": int(counts_tok.min()),
                        "median": float(np.median(counts_tok)), "mean": float(counts_tok.mean()),
                        "max": int(counts_tok.max()), "resumes_with_multiple_chunks": int((counts_tok > 1).sum()),
                        "chunks_under_%d_tokens" % SHORT_CHUNK: int((lens < SHORT_CHUNK).sum())},
        "metrics": metrics, "comparisons_vs_baseline": comparisons,
        "parity": {"baseline_rows_differing_from_saved_predictions": disagree,
                   "max_abs_prob_diff_vs_confidence_file": max_prob_diff,
                   "max_abs_chunk0_vs_baseline_logit_diff": chunk0_logit_diff,
                   "baseline_macro_f1_vs_manifest": [metrics["baseline_first_512"]["macro_f1"], base_f1_manifest]},
        "runtime_seconds": {"baseline_forward": t_base, "chunked_forward": t_chunk, "tokenization": t_tok,
                            "device": device.type},
        "environment": {"torch": torch.__version__, "transformers": transformers.__version__},
        "script_sha256": sha256_file(Path(__file__)),
        "aggregation": "mean of chunk logits; mean of chunk softmax probabilities (float64, stable softmax)",
        "limitations": [
            "Model was trained on first-512-token inputs only; chunking is inference-time only.",
            "Validation set (248) was also used to select this checkpoint's epoch; small classes have ~2 resumes.",
            "Equal weight per chunk, including short final chunks.",
            "Bootstrap and sign test describe this validation set only; no test data used.",
        ],
    }
    out_json.write_text(json.dumps(report, indent=2, default=to_json), encoding="utf-8")
    print("\nSaved:", ", ".join(str(p.relative_to(ROOT)) for p in (out_json, out_pc, out_pred)))


if __name__ == "__main__":
    main()
