"""TF-IDF baselines (Experiments A and B) on the saved seed-42 split.

Fits on exactly the 1,738 rows DistilBERT trained on (train minus the group-aware validation
carve), so the comparison uses identical training data. Never rewrites the split file. Hyperparameters
are the defaults in train.py; nothing is tuned. Each pipeline is scored once on the saved test set.
Outputs go to results/baselines/ only, contain no resume text, and are never overwritten.
Default mode only checks alignment and runs a synthetic test; --run-training fits the models.
"""
import argparse
import hashlib
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from train import CLEAN_PATH, ROOT, SPLIT_DIR, evaluate, make_classifiers, make_pipeline
from train_transformer import carve_validation, load_data, load_split

SEED = 42
TAG = "tfidf_seed42"
OUT_DIR = ROOT / "results" / "baselines"
TR_DIR = ROOT / "results" / "transformer"
TR_MANIFEST = TR_DIR / "manifest_B_seed42_artifacts1.json"
TR_PRED = TR_DIR / "predictions_B_seed42_artifacts1.csv"
TR_A_METRICS = TR_DIR / "metrics_A_seed42.json"
MODELS = ["LogisticRegression", "LinearSVM", "MultinomialNB"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sign_test_p(only_a, only_b):
    n = only_a + only_b
    if n == 0:
        return 1.0
    from scipy.stats import binomtest
    return float(binomtest(only_a, n, 0.5).pvalue)


def self_test(classes_demo=("CHEF", "TEACHER", "BANKING")):
    """Fit all six pipelines on tiny synthetic data to confirm they run end to end."""
    texts, labels = [], []
    for c in classes_demo:
        for i in range(8):
            words = " ".join("{}word{}".format(c.lower(), j) for j in range(6))
            texts.append("{} resume {} common skills number{}".format(c, words, i))
            labels.append(c)
    X, y = pd.Series(texts), np.array(labels)
    for exp in ("A", "B"):
        for name, clf in make_classifiers(SEED).items():
            pipe = make_pipeline(exp, clf, list(classes_demo))
            pred = pipe.fit(X, y).predict(X)
            assert len(pred) == len(y) and set(pred) <= set(classes_demo)
    assert sign_test_p(0, 0) == 1.0 and 0 < sign_test_p(9, 1) < 0.05


def check_alignment(df, tr_idx, va_idx, te_idx, classes):
    manifest = json.loads(TR_MANIFEST.read_text(encoding="utf-8"))
    if sha256_file(SPLIT_DIR / "split_seed42.csv") != manifest["sha256"]["split_file"]:
        sys.exit("ERROR: split file hash differs from the transformer manifest")
    if sha256_file(CLEAN_PATH) != manifest["sha256"]["cleaned_csv"]:
        sys.exit("ERROR: cleaned CSV hash differs from the transformer manifest")
    sets = [set(df.iloc[i]["group_id"]) for i in (tr_idx, va_idx, te_idx)]
    if (sets[0] & sets[1]) or (sets[0] & sets[2]) or (sets[1] & sets[2]):
        sys.exit("ERROR: group overlap between train/validation/test")
    if len(set(tr_idx) | set(va_idx) | set(te_idx)) != len(df):
        sys.exit("ERROR: train/validation/test do not cover the dataset")
    pred = pd.read_csv(TR_PRED)
    for name, idx in (("val", va_idx), ("test", te_idx)):
        rows = pred[pred["split"] == name].sort_values("row_index")["row_index"].to_numpy(dtype=int)
        if not np.array_equal(rows, idx):
            sys.exit("ERROR: {} rows differ from the transformer's saved predictions".format(name))
    if manifest["rows"] != {"train": len(tr_idx), "val": len(va_idx), "test": len(te_idx)}:
        sys.exit("ERROR: split sizes differ from the transformer manifest")
    return manifest


def main():
    ap = argparse.ArgumentParser(description="TF-IDF baselines on the saved seed-42 split")
    ap.add_argument("--run-training", action="store_true", help="fit and evaluate the six pipelines")
    args = ap.parse_args()

    self_test()
    print("Synthetic self-test passed (6 pipelines fit on tiny synthetic data).")

    df = load_data()
    classes = sorted(df["Category"].unique())
    train_idx, te_idx = load_split(df, SEED)
    tr_idx, va_idx = carve_validation(df, train_idx, SEED)
    manifest = check_alignment(df, tr_idx, va_idx, te_idx, classes)
    print("Rows: {} | categories: {} | train {} | validation {} | test {} | group overlaps: 0".format(
        len(df), len(classes), len(tr_idx), len(va_idx), len(te_idx)))
    print("Alignment verified against the transformer manifest and saved predictions.")

    names = ["summary", "per_class", "predictions"]
    outputs = {"summary": OUT_DIR / ("summary_" + TAG + ".csv"),
               "per_class": OUT_DIR / ("per_class_" + TAG + ".csv"),
               "predictions": OUT_DIR / ("predictions_" + TAG + ".csv"),
               "json": OUT_DIR / ("metrics_" + TAG + ".json")}
    cm_paths = {(e, m): OUT_DIR / "confusion_matrix_{}_{}_{}.csv".format(e, m, TAG)
                for e in "AB" for m in MODELS}
    clash = [str(p.relative_to(ROOT)) for p in list(outputs.values()) + list(cm_paths.values()) if p.exists()]
    if clash:
        sys.exit("ERROR: refusing to overwrite: " + ", ".join(clash))
    if not args.run_training:
        print("Preparation only: no models fitted. Use --run-training to fit.")
        return

    X, y = df["Resume_str"], df["Category"].to_numpy()
    y_tr, y_va, y_te = y[tr_idx], y[va_idx], y[te_idx]
    rows, per_class, preds_va, preds_te, cms, report_models = [], [], {}, {}, {}, {}
    for exp in ("A", "B"):
        for name, clf in make_classifiers(SEED).items():
            pipe = make_pipeline(exp, clf, classes)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                t0 = time.perf_counter()
                pipe.fit(X.iloc[tr_idx], y_tr)  # TF-IDF fitted on these training rows only
                fit_s = time.perf_counter() - t0
            n_conv = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
            pv, pt = pipe.predict(X.iloc[va_idx]), pipe.predict(X.iloc[te_idx])
            mv, mt = evaluate(y_va, pv, classes), evaluate(y_te, pt, classes)
            rows.append({"experiment": exp, "model": name,
                         "val_accuracy": mv["accuracy"], "val_macro_f1": mv["macro_f1"],
                         "test_accuracy": mt["accuracy"], "test_macro_precision": mt["macro_precision"],
                         "test_macro_recall": mt["macro_recall"], "test_macro_f1": mt["macro_f1"],
                         "test_weighted_f1": mt["weighted_f1"], "fit_seconds": round(fit_s, 1),
                         "convergence_warnings": n_conv})
            for split, yy, pp in (("val", y_va, pv), ("test", y_te, pt)):
                p, r, f, s = precision_recall_fscore_support(yy, pp, labels=classes, average=None,
                                                             zero_division=0)
                per_class.append(pd.DataFrame({"experiment": exp, "model": name, "split": split,
                                               "category": classes, "precision": p, "recall": r,
                                               "f1": f, "support": s.astype(int)}))
            preds_va[(exp, name)], preds_te[(exp, name)] = pv, pt
            cms[(exp, name)] = pd.DataFrame(confusion_matrix(y_te, pt, labels=classes),
                                            index=classes, columns=classes)
            report_models["{}_{}".format(exp, name)] = {
                "val": {k: float(v) for k, v in mv.items()}, "test": {k: float(v) for k, v in mt.items()},
                "classifier_params": {k: str(v) for k, v in clf.get_params().items()},
                "n_tfidf_features": int(len(pipe.named_steps["tfidf"].vocabulary_)),
                "convergence_warnings": n_conv, "fit_seconds": fit_s}
            print("[{}] {:<19} fitted in {:5.1f}s | convergence warnings: {}".format(exp, name, fit_s, n_conv))

    summary = pd.DataFrame(rows)
    ref = {"DistilBERT B (artifacts1)": manifest["test_metrics"]}
    if TR_A_METRICS.exists():
        ref["DistilBERT A (seed 42)"] = json.loads(TR_A_METRICS.read_text(encoding="utf-8"))["test_metrics"]
    print("\nTEST set (497 rows, 24 labels, zero_division=0). TF-IDF trained on the same 1,738 rows.")
    print("{:<26}{:>9}{:>9}{:>9}{:>9}{:>11}{:>10}".format(
        "model", "acc", "macroP", "macroR", "macroF1", "weightedF1", "val_mF1"))
    for _, r in summary.iterrows():
        print("{:<26}{:>9.4f}{:>9.4f}{:>9.4f}{:>9.4f}{:>11.4f}{:>10.4f}".format(
            r.experiment + " " + r.model, r.test_accuracy, r.test_macro_precision, r.test_macro_recall,
            r.test_macro_f1, r.test_weighted_f1, r.val_macro_f1))
    for label, m in ref.items():
        print("{:<26}{:>9.4f}{:>9.4f}{:>9.4f}{:>9.4f}{:>11.4f}{:>10}".format(
            label, m["accuracy"], m["macro_precision"], m["macro_recall"], m["macro_f1"],
            m["weighted_f1"], "-"))

    tr_pred = pd.read_csv(TR_PRED)
    tr_pred = tr_pred[tr_pred["split"] == "test"].sort_values("row_index")["pred_label"].to_numpy()
    comparisons = {}
    print("\nPaired comparison, Experiment B, test set (TF-IDF vs saved DistilBERT B predictions):")
    for name in MODELS:
        tf_ok, tr_ok = preds_te[("B", name)] == y_te, tr_pred == y_te
        only_tf, only_tr = int((tf_ok & ~tr_ok).sum()), int((~tf_ok & tr_ok).sum())
        p = sign_test_p(only_tf, only_tr)
        comparisons["B_" + name] = {"tfidf_only_correct": only_tf, "distilbert_only_correct": only_tr,
                                    "both_correct": int((tf_ok & tr_ok).sum()),
                                    "both_wrong": int((~tf_ok & ~tr_ok).sum()), "sign_test_p": p}
        print("  {:<19} TF-IDF-only correct {:>3} | DistilBERT-only correct {:>3} | sign-test p {:.3f}".format(
            name, only_tf, only_tr, p))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(outputs["summary"], index=False)
    pd.concat(per_class, ignore_index=True).to_csv(outputs["per_class"], index=False)
    pred_df = pd.DataFrame({"row_index": np.concatenate([va_idx, te_idx]),
                            "split": ["val"] * len(va_idx) + ["test"] * len(te_idx)})
    for (exp, name) in preds_te:
        pred_df["pred_{}_{}".format(exp, name)] = np.concatenate([preds_va[(exp, name)], preds_te[(exp, name)]])
    pred_df.to_csv(outputs["predictions"], index=False)
    for key, cm in cms.items():
        cm.to_csv(cm_paths[key])
    report = {"run_tag": TAG, "training_rows": int(len(tr_idx)),
              "training_set": "same 1,738 rows as DistilBERT (saved train minus group-aware validation carve)",
              "tuning": "none; default hyperparameters from src/train.py", "test_scored_once_per_pipeline": True,
              "split_sha256": sha256_file(SPLIT_DIR / "split_seed42.csv"), "cleaned_csv_sha256": sha256_file(CLEAN_PATH),
              "versions": {"scikit_learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
              "models": report_models, "distilbert_reference_test": ref,
              "paired_comparison_B_vs_distilbert": comparisons,
              "limitations": ["Single seed; test classes have 4-23 resumes.",
                              "DistilBERT A has no saved row-level predictions, so only aggregate comparison is possible.",
                              "Default TF-IDF hyperparameters were not tuned; tuned baselines could differ."]}
    outputs["json"].write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nSaved to results/baselines/ (summary, per-class, predictions, 6 confusion matrices, metrics JSON).")


if __name__ == "__main__":
    main()
