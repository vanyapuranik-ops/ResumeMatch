"""P3 candidate: chunk-level fine-tuning of DistilBERT (Experiment B, seed 42, max length 512).

STAGE 1 = DRY RUN ONLY. This script currently loads NO model weights, imports NO torch, trains
NOTHING and WRITES NOTHING (stdout only). --benchmark / --run-training are deliberately not
implemented yet; they are added only after the plan is approved.

Window semantics (verified, see checks below): each resume is Experiment-B masked, tokenized, and cut
into windows of max_len tokens = [CLS] + (max_len-2) content tokens + [SEP]. `stride` is the number of
CONTENT tokens shared by consecutive windows (default 128), so the step is max_len-2-stride = 382 and a
resume with L content tokens has 1 window if L <= 510, else 1 + ceil((L-510)/382). Window 1 equals the
baseline's first-512 input exactly. A final window is never shorter than stride+3 tokens.
Windowing happens AFTER the resume-level split, so all windows of a resume stay in one split.

Reuses unchanged: train.mask_category_names, train_transformer.load_data/load_split/carve_validation
(same seed-42 split, same validation carve). Does not modify any existing file or artifact.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from train import CLEAN_PATH, ROOT, SPLIT_DIR, mask_category_names
from train_transformer import (MODEL_NAME, OUT_DIR as BASE_OUT_DIR, carve_validation, load_data,
                               load_split, resumematch_sha256)

SEED = 42
BASE_TAG = "B_seed42_artifacts1"
BASE_MODEL_DIR = ROOT / "models" / ("distilbert_" + BASE_TAG)
BASE_MANIFEST = BASE_OUT_DIR / ("manifest_" + BASE_TAG + ".json")
BASE_PRED = BASE_OUT_DIR / ("predictions_" + BASE_TAG + ".csv")
BASE_CHUNK_JSON = BASE_OUT_DIR / ("chunking_" + BASE_TAG + "_stride128_metrics.json")
SPLIT_FILE = SPLIT_DIR / ("split_seed" + str(SEED) + ".csv")
P3_DIR = ROOT / "results" / "p3_chunked"
LOG_DIR = ROOT / "logs"


# ---------- pure helpers (covered by --self-test) ----------
def check_mapping(mapping, n):
    """window -> resume map must be non-decreasing and cover every resume 0..n-1."""
    m = np.asarray(mapping)
    if m.ndim != 1 or len(m) == 0:
        raise ValueError("window mapping is empty or not 1-D")
    if np.any(m[1:] < m[:-1]):
        raise ValueError("window mapping is not non-decreasing")
    if m.min() != 0 or m.max() != n - 1 or len(np.unique(m)) != n:
        raise ValueError("window mapping does not cover every resume 0..n-1")


def window_positions(mapping):
    """0-based position of each window inside its own resume (mapping must be sorted)."""
    m = np.asarray(mapping)
    return np.arange(len(m)) - np.searchsorted(m, m, side="left")


def expected_windows(content_tokens, max_len, stride):
    c = max_len - 2
    return np.where(content_tokens <= c, 1, 1 + np.ceil((content_tokens - c) / (c - stride))).astype(int)


def mass_schemes(y, k, K):
    """Per-resume share of the training loss under four window-weighting schemes.

    y: class id per training resume, k: windows kept per resume (after cap), K: classes.
    PyTorch weighted CE normalises by the sum of weights in the batch, so only relative masses matter.
      S0 baseline (first window only):         m_i = N / (K n_c)
      S1 window-count class weights, no 1/k:   m_i = k_i * M / (K W_c)      (W_c = windows of class c)
      S2 resume-equal (class w / k per window): m_i = N / (K n_c)            (== S0 by construction)
      S3 naive (resume class w on every window): m_i = k_i * N / (K n_c)
    """
    y, k = np.asarray(y), np.asarray(k, dtype=float)
    N, M = len(y), k.sum()
    n_c = np.bincount(y, minlength=K).astype(float)
    W_c = np.bincount(y, weights=k, minlength=K)
    return {"S0 baseline (1st window)": N / (K * n_c[y]),
            "S1 window-count wts": k * M / (K * W_c[y]),
            "S2 resume-equal": N / (K * n_c[y]),
            "S3 naive resume wts": k * N / (K * n_c[y])}


def mass_stats(m, y, k, K, cap):
    m, y, k = np.asarray(m, dtype=float), np.asarray(y), np.asarray(k)
    cls = np.bincount(y, weights=m, minlength=K)
    rel = cls / (m.sum() / K)
    ess = 0.0
    for c in range(K):
        mc = m[y == c]
        if len(mc):
            ess += mc.sum() ** 2 / (mc ** 2).sum()
    return {"class_rel_min": float(rel.min()), "class_rel_max": float(rel.max()),
            "within_class_ess_frac": float(ess / len(m)), "share_from_k_eq_cap": float(m[k >= cap].sum() / m.sum())}


def raises(fn, *a):
    try:
        fn(*a)
    except ValueError:
        return True
    return False


def self_test():
    check_mapping([0, 0, 1, 2, 2], 3)
    assert raises(check_mapping, [0, 0, 2], 3) and raises(check_mapping, [1, 0], 2) and raises(check_mapping, [], 1)
    assert window_positions([0, 0, 0, 1, 2, 2]).tolist() == [0, 1, 2, 0, 0, 1]
    assert expected_windows(np.array([1, 510, 511, 892, 893]), 512, 128).tolist() == [1, 1, 2, 2, 3]
    # two classes: A has 2 resumes (k=1,1), B has 2 resumes (k=6,2); K=2
    y, k = np.array([0, 0, 1, 1]), np.array([1, 1, 6, 2])
    s = mass_schemes(y, k, 2)
    assert np.allclose(s["S0 baseline (1st window)"], s["S2 resume-equal"]) and np.allclose(s["S2 resume-equal"], 1.0)
    st = {n: mass_stats(m, y, k, 2, 6) for n, m in s.items()}
    for n in ("S0 baseline (1st window)", "S1 window-count wts", "S2 resume-equal"):
        assert abs(st[n]["class_rel_min"] - 1) < 1e-12 and abs(st[n]["class_rel_max"] - 1) < 1e-12  # class balance kept
    s3 = st["S3 naive resume wts"]
    assert abs(s3["class_rel_max"] / s3["class_rel_min"] - 4) < 1e-9   # class B has 4x the windows per resume of A
    assert abs(st["S2 resume-equal"]["within_class_ess_frac"] - 1) < 1e-12
    ess_b = (6 + 2) ** 2 / (36 + 4)
    assert abs(st["S1 window-count wts"]["within_class_ess_frac"] - (2 + ess_b) / 4) < 1e-12
    assert abs(st["S1 window-count wts"]["share_from_k_eq_cap"] - 0.375) < 1e-12  # 3.75 of total mass 10
    for _, m in s.items():
        assert np.all(m > 0)


# ---------- data helpers ----------
def load_tokenizer(tokenizer_dir):
    from transformers import AutoTokenizer
    if tokenizer_dir:
        return AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True), str(tokenizer_dir)
    try:
        return AutoTokenizer.from_pretrained(MODEL_NAME), MODEL_NAME
    except Exception:  # offline: the baseline saved an identical tokenizer copy (no weights are read)
        return (AutoTokenizer.from_pretrained(str(BASE_MODEL_DIR), local_files_only=True),
                str(BASE_MODEL_DIR.relative_to(ROOT)) + " (saved baseline tokenizer copy)")


def planned_outputs(tag, run_name):
    return {
        "model dir": ROOT / "models" / ("distilbert_" + tag),
        "manifest": P3_DIR / ("manifest_" + tag + ".json"),
        "metrics": P3_DIR / ("metrics_" + tag + ".json"),
        "per-class val": P3_DIR / ("metrics_per_class_val_" + tag + ".csv"),
        "per-class test": P3_DIR / ("metrics_per_class_test_" + tag + ".csv"),
        "confusion val": P3_DIR / ("confusion_matrix_val_" + tag + ".csv"),
        "confusion test": P3_DIR / ("confusion_matrix_test_" + tag + ".csv"),
        "predictions": P3_DIR / ("predictions_" + tag + ".csv"),
        "comparison vs baseline": P3_DIR / ("comparison_vs_baseline_" + tag + ".json"),
        "log": LOG_DIR / ("p3_" + run_name + ".log"),
    }


def is_inside(path, folder):
    return path == folder or folder in path.parents


def main():
    ap = argparse.ArgumentParser(description="P3 chunk-level fine-tuning, Stage 1: dry run only")
    ap.add_argument("--stride", type=int, default=128, help="content tokens shared by consecutive windows")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--max-windows", type=int, default=6, help="cap on windows per TRAINING resume")
    ap.add_argument("--run-name", default="chunktrain1")
    ap.add_argument("--tokenizer-dir", default=None, help="local tokenizer dir (default: MODEL_NAME, then baseline copy)")
    ap.add_argument("--baseline-epoch-min", type=float, default=None,
                    help="measured baseline minutes per epoch (optional; enables an absolute estimate)")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--run-training", action="store_true")
    args = ap.parse_args()

    self_test()
    print("Synthetic self-tests passed.")
    if args.self_test:
        return
    if args.benchmark or args.run_training:
        sys.exit("STAGE 1: --benchmark and --run-training are intentionally NOT implemented. Dry run only.")
    if not args.run_name or not args.run_name.isascii() or not args.run_name.replace("-", "").replace("_", "").isalnum():
        sys.exit("ERROR: run-name may contain only letters, digits, dash and underscore")
    if not 0 <= args.stride < args.max_len - 2 or args.max_windows < 1:
        sys.exit("ERROR: need 0 <= stride < max_len-2 and max-windows >= 1")
    cap, ML, ST = args.max_windows, args.max_len, args.stride
    tag = "B_seed" + str(SEED) + "_" + args.run_name

    # --- same data, split and validation carve as the baseline (read-only) ---
    df = load_data()
    classes = sorted(df["Category"].unique())
    K = len(classes)
    y = df["Category"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    train_all, test_idx = load_split(df, SEED)
    tr_idx, va_idx = carve_validation(df, train_all, SEED)
    if not BASE_MANIFEST.is_file():
        sys.exit("ERROR: baseline manifest not found: " + str(BASE_MANIFEST))
    man = json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))
    if resumematch_sha256(SPLIT_FILE) != man["sha256"]["split_file"] or resumematch_sha256(CLEAN_PATH) != man["sha256"]["cleaned_csv"]:
        sys.exit("ERROR: split or cleaned CSV hash differs from the baseline manifest")
    if [c for c, _ in sorted(man["label2id"].items(), key=lambda kv: kv[1])] != classes:
        sys.exit("ERROR: dataset categories differ from the baseline label mapping")
    if man["rows"] != {"train": len(tr_idx), "val": len(va_idx), "test": len(test_idx)}:
        sys.exit("ERROR: split sizes differ from the baseline manifest rows")
    print("Data/split verified against baseline manifest (hashes, label map, rows): train {} | val {} | test {} | K {}".format(
        len(tr_idx), len(va_idx), len(test_idx), K))

    # --- Experiment B masking (unchanged), then windowing of ALL resumes ---
    texts = mask_category_names(df["Resume_str"], classes)
    tok, tok_src = load_tokenizer(args.tokenizer_dir)
    if not tok.is_fast:
        sys.exit("ERROR: a fast tokenizer is required for overflow/stride windows")
    print("Tokenizer:", tok_src)
    n = len(texts)
    base_ids = tok(texts, truncation=True, max_length=ML)["input_ids"]
    enc = tok(texts, truncation=True, max_length=ML, stride=ST, return_overflowing_tokens=True)
    win = enc["input_ids"]
    mapping = np.asarray(enc["overflow_to_sample_mapping"], dtype=int)
    try:
        check_mapping(mapping, n)
    except ValueError as e:
        sys.exit("ERROR: " + str(e))
    lens = np.array([len(w) for w in win])
    k_all = np.bincount(mapping, minlength=n)
    pos = window_positions(mapping)
    # invariants on the REAL data
    if lens.max() > ML or any(w[0] != tok.cls_token_id or w[-1] != tok.sep_token_id for w in win):
        sys.exit("ERROR: a window exceeds max_len or lacks [CLS]/[SEP]")
    if [win[i] for i in np.flatnonzero(pos == 0)] != base_ids:
        sys.exit("ERROR: window 1 differs from the baseline first-512 tokenization")
    total_content = np.bincount(mapping, weights=lens - 2, minlength=n)  # includes overlap
    nxt = np.flatnonzero(pos > 0)
    for i in nxt:  # exact overlap: head of window j+1 == tail of window j (content tokens)
        prev, cur = win[i - 1], win[i]
        if ST and (len(prev) != ML or prev[-1 - ST:-1] != cur[1:1 + ST]):
            sys.exit("ERROR: window overlap is not exactly stride={} content tokens".format(ST))
    # unique content length L = windows' content minus duplicated overlap
    L = total_content - ST * (k_all - 1)
    if not np.array_equal(k_all, expected_windows(L, ML, ST)):
        sys.exit("ERROR: window counts disagree with the closed-form 1 + ceil((L-510)/step)")
    short_last = int((lens[np.r_[pos[1:] == 0, True] & (pos > 0)] < ST + 3).sum())
    print("Window semantics verified on real data: window 1 == first-512 input | consecutive windows share exactly {} "
          "content tokens (step {}) | counts match 1+ceil((L-{})/{}) | final windows under stride+3 tokens: {}".format(
              ST, ML - 2 - ST, ML - 2, ML - 2 - ST, short_last))

    # --- tables ---
    splits = {"train": tr_idx, "val": va_idx, "test": test_idx}
    print("\nWindows by split (uncapped; val/test are always scored on all windows)")
    print("{:<7}{:>8}{:>9}{:>7}{:>8}{:>5}{:>10}{:>11}".format("split", "resumes", "windows", "mean", "median", "max", ">1 win", "k>cap(%d)" % cap))
    for name, idx in splits.items():
        kk = k_all[idx]
        print("{:<7}{:>8}{:>9}{:>7.2f}{:>8.0f}{:>5}{:>10}{:>11}".format(
            name, len(idx), int(kk.sum()), kk.mean(), np.median(kk), kk.max(), int((kk > 1).sum()), int((kk > cap).sum())))

    ktr_raw = k_all[tr_idx]
    ktr = np.minimum(ktr_raw, cap)
    ytr = y[tr_idx]
    N, M0, M = len(tr_idx), int(ktr_raw.sum()), int(ktr.sum())
    print("\nTrain windows per resume (uncapped histogram):",
          " ".join("{}:{}".format(j, int((ktr_raw == j).sum())) for j in range(1, cap + 1)), "| >{}: {}".format(cap, int((ktr_raw > cap).sum())))
    print("Cap {} effect (train only): windows {} -> {} (drops {}, {:.1f}%) | resumes affected {} ({:.1f}%) | "
          "max windows dropped from one resume {}".format(
              cap, M0, M, M0 - M, 100 * (M0 - M) / M0, int((ktr_raw > cap).sum()), 100 * (ktr_raw > cap).mean(),
              int((ktr_raw - ktr).max())))
    tr_win_mask = np.isin(mapping, tr_idx) & (pos < cap)
    tok_tr_cap = int(lens[tr_win_mask].sum())
    tok_base = int(sum(len(base_ids[i]) for i in tr_idx))
    print("Train tokens: capped windows {} vs baseline first-512 {} (ratio {:.2f}x)".format(tok_tr_cap, tok_base, tok_tr_cap / tok_base))

    print("\nBy category (train windows after cap)")
    print("{:<24}{:>6}{:>8}{:>8}{:>7}{:>6}{:>6}{:>6}{:>6}".format("category", "trRes", "trWin", "uncap", "mean_k", "vaRes", "vaWin", "teRes", "teWin"))
    for c, name in enumerate(classes):
        t = ytr == c
        v, e = y[va_idx] == c, y[test_idx] == c
        print("{:<24}{:>6}{:>8}{:>8}{:>7.2f}{:>6}{:>6}{:>6}{:>6}".format(
            name, int(t.sum()), int(ktr[t].sum()), int(ktr_raw[t].sum()), ktr[t].mean(),
            int(v.sum()), int(k_all[va_idx][v].sum()), int(e.sum()), int(k_all[test_idx][e].sum())))

    print("\nLoss-mass analysis (train, after cap). Class-mass share is relative to 1/K (1.00 = balanced as in baseline);")
    print("within-class ESS = fraction of resumes' worth of independent weight kept (1.00 = no resume dominates its class).")
    print("{:<26}{:>11}{:>11}{:>14}{:>13}".format("scheme", "class min", "class max", "in-class ESS", "mass k==cap"))
    for name, m in mass_schemes(ytr, ktr, K).items():
        s = mass_stats(m, ytr, ktr, K, cap)
        print("{:<26}{:>11.2f}{:>11.2f}{:>14.3f}{:>13.1%}".format(
            name, s["class_rel_min"], s["class_rel_max"], s["within_class_ess_frac"], s["share_from_k_eq_cap"]))

    # --- output paths / conflicts ---
    plan = planned_outputs(tag, args.run_name)
    protected = [BASE_MODEL_DIR, BASE_OUT_DIR, SPLIT_DIR, CLEAN_PATH.parent]
    print("\nPlanned outputs (tag {}):".format(tag))
    conflicts = 0
    for label, p in plan.items():
        inside = [str(q.relative_to(ROOT)) for q in protected if is_inside(p, q)]
        flag = "EXISTS - CONFLICT" if p.exists() else "free"
        if inside:
            flag = "INSIDE PROTECTED " + inside[0] + " - REFUSE"
        conflicts += flag != "free"
        print("  {:<24}{:<72}{}".format(label, str(p.relative_to(ROOT)), flag))
    print("Baseline artifacts (read-only, never written): " + ", ".join(
        "{} {}".format(p.relative_to(ROOT), "present" if p.exists() else "MISSING")
        for p in (BASE_MODEL_DIR, BASE_MANIFEST, BASE_PRED, SPLIT_FILE)))
    print("Output-path conflicts:", conflicts)

    # --- cost proxies ---
    val_win = int(k_all[va_idx].sum())
    print("\nCost proxies (hypotheses, not measurements of this run)")
    print("  baseline batches/epoch {} | candidate {} ({:.2f}x) | token ratio {:.2f}x".format(
        -(-N // 8), -(-M // 8), M / N, tok_tr_cap / tok_base))
    if BASE_CHUNK_JSON.is_file():
        rt = json.loads(BASE_CHUNK_JSON.read_text(encoding="utf-8"))["runtime_seconds"]
        per_win = rt["chunked_forward"] / 745
        print("  per-epoch validation scoring: {} windows x {:.3f} s/window (measured earlier) = about {:.0f} s".format(
            val_win, per_win, val_win * per_win))
    if args.baseline_epoch_min:
        est = args.baseline_epoch_min * tok_tr_cap / tok_base
        print("  from --baseline-epoch-min {:.1f}: about {:.1f} min/epoch training, up to 6 epochs = {:.0f} min (+ validation scoring)".format(
            args.baseline_epoch_min, est, 6 * est))
    print("\nDry run complete. Nothing was written, no weights were loaded, nothing was trained.")


if __name__ == "__main__":
    main()
