"""Fine-tune distilbert-base-uncased for 24-class resume classification (primary experiment).

Modes (mutually exclusive): default = preparation only (no model load, no torch import);
--benchmark = time a few training steps (loads weights, no evaluation); --run-training = full run.

Split: uses the saved results/splits/split_seed{seed}.csv (same as baselines). Validation is carved
from TRAIN groups only (StratifiedGroupKFold, ~12.5%). The test set is used once, after the best
checkpoint is chosen by validation macro F1 with early stopping.
Macro precision/recall/F1 are computed over all classes with zero_division=0.
Experiment A = raw Resume_str; B = header masking (train.mask_category_names), label-agnostic.
Truncation keeps the first max_len tokens. Outputs are aggregate-only (no resume text).
Seeded but MPS is not bit-deterministic.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold

from train import (CLEAN_PATH, REQUIRED_COLS, RESULTS_DIR, ROOT, SPLIT_DIR, evaluate,
                   identity_text, mask_category_names)

MODEL_NAME = "distilbert-base-uncased"
OUT_DIR = RESULTS_DIR / "transformer"
VAL_FOLDS = 8


def load_data():
    if not CLEAN_PATH.exists():
        sys.exit("Cleaned dataset not found: " + str(CLEAN_PATH))
    df = pd.read_csv(CLEAN_PATH)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        sys.exit("Missing required columns: " + str(missing))
    return df.dropna(subset=["Resume_str", "Category"]).reset_index(drop=True)


def load_split(df, seed):
    path = SPLIT_DIR / ("split_seed" + str(seed) + ".csv")
    if not path.exists():
        sys.exit("Saved split not found: " + str(path) + " (run src/train.py first)")
    a = pd.read_csv(path)
    if len(a) != len(df) or not np.array_equal(a["row_index"].to_numpy(), np.arange(len(df))):
        sys.exit("Saved split does not match the cleaned dataset (row count/order)")
    if not np.array_equal(a["ID"].to_numpy(), df["ID"].to_numpy()):
        sys.exit("Saved split IDs do not match the cleaned dataset")
    if not np.array_equal(a["group_id"].to_numpy(), df["group_id"].to_numpy()):
        sys.exit("Saved split group_ids do not match the cleaned dataset")
    train_idx = np.where(a["split"].to_numpy() == "train")[0]
    test_idx = np.where(a["split"].to_numpy() == "test")[0]
    if set(df.iloc[train_idx]["group_id"]) & set(df.iloc[test_idx]["group_id"]):
        sys.exit("ERROR: group overlap between train and test")
    return train_idx, test_idx


def carve_validation(df, train_idx, seed):
    sub = df.iloc[train_idx]
    sgkf = StratifiedGroupKFold(n_splits=VAL_FOLDS, shuffle=True, random_state=seed)
    tr_rel, va_rel = next(sgkf.split(sub, sub["Category"], groups=sub["group_id"]))
    tr_idx, va_idx = train_idx[tr_rel], train_idx[va_rel]
    if set(df.iloc[tr_idx]["group_id"]) & set(df.iloc[va_idx]["group_id"]):
        sys.exit("ERROR: group overlap between train and validation")
    return np.sort(tr_idx), np.sort(va_idx)


def resumematch_run_tag(args):
    tag = args.experiment + "_seed" + str(args.seed)
    suffix = getattr(args, "run_name", "")
    return tag + ("_" + suffix if suffix else "")


def resumematch_sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resumematch_write_predictions(va_idx, va_pred, te_idx, te_pred, classes, path):
    frame = pd.DataFrame({
        "row_index": np.concatenate([va_idx, te_idx]).astype(int),
        "split": ["val"] * len(va_idx) + ["test"] * len(te_idx),
        "pred_id": np.concatenate([va_pred, te_pred]).astype(int),
    })
    if frame["row_index"].duplicated().any():
        raise ValueError("Duplicate row indices in prediction output")
    if (frame["pred_id"] < 0).any() or (frame["pred_id"] >= len(classes)).any():
        raise ValueError("Prediction ID out of range")
    frame["pred_label"] = [classes[i] for i in frame["pred_id"]]
    frame.to_csv(path, index=False)
    check = pd.read_csv(path)
    if not frame.equals(check):
        raise ValueError("Prediction CSV verification failed")
    return resumematch_sha256(path), len(frame)


def head_tail_ids(tok, texts, max_len, tail_tokens):
    """Head+tail truncation: texts longer than max_len - 2 content tokens keep their first
    (max_len - 2 - tail_tokens) and last tail_tokens tokens, wrapped in [CLS]/[SEP]. Shorter texts are
    unchanged and equal the standard truncation=True output. No labels are used."""
    budget = max_len - 2
    if not 0 < tail_tokens < budget:
        raise ValueError("tail_tokens must be between 1 and max_len - 3")
    full = tok(texts, add_special_tokens=False, verbose=False)["input_ids"]
    out = []
    for ids in full:
        if len(ids) > budget:
            ids = ids[:budget - tail_tokens] + ids[-tail_tokens:]
        out.append([tok.cls_token_id] + ids + [tok.sep_token_id])
    return out


def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run(args, texts, y, classes, tr_idx, va_idx, te_idx, benchmark):
    import torch
    import transformers
    from torch.utils.data import DataLoader
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)
    transformers.logging.set_verbosity_error()
    set_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    K = len(classes)
    all_labels = list(range(K))
    print("Device:", device.type)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    def encode(idx):
        ids = []
        for s in range(0, len(idx), 128):
            chunk = [texts[i] for i in idx[s:s + 128]]
            if args.truncation == "head":
                ids += tok(chunk, truncation=True, max_length=args.max_len)["input_ids"]
            else:
                ids += head_tail_ids(tok, chunk, args.max_len, args.tail_tokens)
        return ids

    pad_id = tok.pad_token_id

    def collate(batch):
        L = max(len(b[0]) for b in batch)
        inp = torch.full((len(batch), L), pad_id, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for i, (ids, _) in enumerate(batch):
            inp[i, :len(ids)] = torch.tensor(ids)
            att[i, :len(ids)] = 1
        return inp, att, torch.tensor([b[1] for b in batch], dtype=torch.long)

    def make_loader(ids, labels, bs, shuffle):
        gen = torch.Generator().manual_seed(args.seed)
        return DataLoader(list(zip(ids, labels)), batch_size=bs, shuffle=shuffle,
                          collate_fn=collate, generator=gen)

    def sync():
        if device.type == "mps":
            torch.mps.synchronize()

    y_tr, y_va, y_te = y[tr_idx], y[va_idx], y[te_idx]
    ids_tr = encode(tr_idx)
    trunc = float(np.mean([len(x) >= args.max_len for x in ids_tr]))
    print("Train resumes hitting the {}-token limit: {:.1f}%".format(args.max_len, 100 * trunc))
    train_loader = make_loader(ids_tr, y_tr.tolist(), args.batch_size, True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=K,
        id2label={i: c for i, c in enumerate(classes)},
        label2id={c: i for i, c in enumerate(classes)}).to(device)
    counts = np.bincount(y_tr, minlength=K)
    weights = torch.tensor(len(y_tr) / (K * np.maximum(counts, 1)), dtype=torch.float32).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if benchmark:
        model.train()
        times = []
        for b, (inp, att, yb) in enumerate(train_loader):
            if b >= args.bench_batches:
                break
            t0 = time.time()
            loss = loss_fn(model(input_ids=inp.to(device), attention_mask=att.to(device)).logits,
                           yb.to(device))
            loss.backward()
            opt.step()
            opt.zero_grad()
            sync()
            times.append(time.time() - t0)
        per = float(np.mean(times[2:])) if len(times) > 3 else float(np.mean(times))
        print("Timed {} batches (batch size {}): {:.2f} s/batch".format(
            len(times), args.batch_size, per))
        print("Extrapolated train time per epoch: {:.1f} min (estimate, not a result)".format(
            per * len(train_loader) / 60))
        print("Benchmark only: nothing saved, no evaluation performed.")
        return

    ids_va, ids_te = encode(va_idx), encode(te_idx)
    steps_per_epoch = math.ceil(len(train_loader) / args.accum)
    total_steps = steps_per_epoch * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)

    def predict(ids, labels):
        model.eval()
        out = []
        with torch.no_grad():
            for inp, att, _ in make_loader(ids, labels, 32, False):
                logits = model(input_ids=inp.to(device), attention_mask=att.to(device)).logits
                out.append(logits.argmax(-1).cpu().numpy())
        return np.concatenate(out)

    best_f1, best_epoch, best_state, bad, history = -1.0, 0, None, 0, []
    best_pv = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        opt.zero_grad()
        for b, (inp, att, yb) in enumerate(train_loader):
            loss = loss_fn(model(input_ids=inp.to(device), attention_mask=att.to(device)).logits,
                           yb.to(device))
            (loss / args.accum).backward()
            total += loss.item()
            n += 1
            if n % 20 == 0:
                print("  epoch {} batch {}/{}".format(epoch, n, len(train_loader)), flush=True)
            if (b + 1) % args.accum == 0 or (b + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
        pv = predict(ids_va, y_va.tolist())
        f1 = float(f1_score(y_va, pv, labels=all_labels, average="macro", zero_division=0))
        acc = float((pv == y_va).mean())
        history.append({"epoch": epoch, "train_loss": total / n,
                        "val_macro_f1": f1, "val_accuracy": acc})
        print("epoch {} | train_loss {:.4f} | val_macro_f1 {:.4f} | val_acc {:.4f}".format(
            epoch, total / n, f1, acc))
        if f1 > best_f1 + 1e-6:
            best_f1, best_epoch, bad = f1, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_pv = pv.copy()
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stopping at epoch", epoch)
                break

    model.load_state_dict(best_state)
    pt = predict(ids_te, y_te.tolist())  # test set used once, after checkpoint selection
    m = {k: float(v) for k, v in evaluate(y_te, pt, all_labels).items()}
    cm = confusion_matrix(y_te, pt, labels=all_labels)
    p, r, f, s = precision_recall_fscore_support(y_te, pt, labels=all_labels,
                                                 average=None, zero_division=0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = resumematch_run_tag(args)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(OUT_DIR / ("confusion_matrix_" + tag + ".csv"))
    pd.DataFrame({"category": classes, "precision": p, "recall": r, "f1": f,
                  "support": s.astype(int)}).to_csv(OUT_DIR / ("metrics_per_class_" + tag + ".csv"), index=False)
    report = {
        "experiment": args.experiment, "head_masking": args.experiment == "B", "model": MODEL_NAME,
        "seed": args.seed, "device": device.type, "torch": torch.__version__,
        "transformers": transformers.__version__,
        "hyperparameters": {"max_len": args.max_len, "batch_size": args.batch_size,
                            "grad_accum": args.accum, "lr": args.lr, "weight_decay": args.weight_decay,
                            "epochs_max": args.epochs, "patience": args.patience, "warmup": 0.1,
                            "class_weighted_loss": True},
        "rows": {"train": int(len(tr_idx)), "val": int(len(va_idx)), "test": int(len(te_idx))},
        "train_share_truncated": trunc, "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
        "classes_with_val_support": int(len(set(y_va.tolist()))),
        "classes_with_test_support": int(len(set(y_te.tolist()))),
        "history": history, "test_metrics": m,
    }
    if args.truncation != "head":
        report["hyperparameters"]["truncation"] = args.truncation
        report["hyperparameters"]["tail_tokens"] = args.tail_tokens
    (OUT_DIR / ("metrics_" + tag + ".json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
    prediction_info = None

    if args.save_predictions:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        prediction_path = OUT_DIR / ("predictions_" + tag + ".csv")
        pred_hash, pred_count = resumematch_write_predictions(
            va_idx, best_pv, te_idx, pt, classes, prediction_path
        )
        prediction_info = {
            "file": str(prediction_path.relative_to(ROOT)),
            "rows": pred_count,
            "sha256": pred_hash,
            "columns": ["row_index", "split", "pred_id", "pred_label"],
        }
        print("Saved predictions to", prediction_path.relative_to(ROOT))

    if args.save_artifacts:
        import platform
        import sklearn
        model_dir = ROOT / "models" / ("distilbert_" + tag)
        model_dir.mkdir(parents=True, exist_ok=False)
        model.save_pretrained(model_dir)
        tok.save_pretrained(model_dir)
        split_path = SPLIT_DIR / ("split_seed" + str(args.seed) + ".csv")
        manifest = {
            "schema_version": 1,
            "model_name": MODEL_NAME,
            "experiment": args.experiment,
            "seed": args.seed,
            "label2id": {c: i for i, c in enumerate(classes)},
            "hyperparameters": report["hyperparameters"],
            "command_line_args": vars(args),
            "rows": report["rows"],
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_f1,
            "test_metrics": m,
            "history": history,
            "device": device.type,
            "versions": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "sha256": {
                "split_file": resumematch_sha256(split_path),
                "cleaned_csv": resumematch_sha256(CLEAN_PATH),
                "train_transformer_py": resumematch_sha256(Path(__file__).resolve()),
                "train_py": resumematch_sha256(ROOT / "src" / "train.py"),
                "predictions_csv": prediction_info["sha256"] if prediction_info else None,
            },
            "predictions": prediction_info,
        }
        manifest_text = json.dumps(manifest, indent=2)
        (model_dir / "manifest.json").write_text(manifest_text, encoding="utf-8")
        (OUT_DIR / ("manifest_" + tag + ".json")).write_text(
            manifest_text, encoding="utf-8"
        )
        print("Saved model, tokenizer and manifest to", model_dir.relative_to(ROOT))

    if args.save_model:
        (ROOT / "models").mkdir(exist_ok=True)
        torch.save(best_state, ROOT / "models" / ("distilbert_" + tag + ".pt"))
    print("TEST (best checkpoint from epoch {}): ".format(best_epoch) +
          " | ".join("{}={:.4f}".format(k, v) for k, v in m.items()))
    print("Saved aggregate results to", OUT_DIR.relative_to(ROOT))


def main():
    p = argparse.ArgumentParser(description="DistilBERT fine-tuning for ResumeMatch")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--experiment", choices=["A", "B"], default="B",
                   help="A = raw text, B = header masking")
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--bench-batches", type=int, default=12)
    p.add_argument("--save-model", action="store_true")
    p.add_argument("--truncation", choices=["head", "head_tail"], default="head",
                   help="head = first max-len tokens (default); head_tail = first and last tokens")
    p.add_argument("--tail-tokens", type=int, default=128,
                   help="tokens kept from the end of a long resume when --truncation head_tail")
    p.add_argument("--save-predictions", action="store_true",
                   help="save row indices and predicted labels without resume text")
    p.add_argument("--save-artifacts", action="store_true",
                   help="save model, tokenizer and JSON manifest")
    p.add_argument("--run-name", default="",
                   help="unique suffix for new output files, e.g. rerun1")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--benchmark", action="store_true", help="time a few steps only")
    mode.add_argument("--run-training", action="store_true", help="full fine-tuning run")
    args = p.parse_args()
    if args.truncation == "head_tail" and not 0 < args.tail_tokens < args.max_len - 2:
        sys.exit("ERROR: --tail-tokens must be between 1 and max-len - 3")
    if args.run_name and (
        not args.run_name.isascii()
        or not args.run_name.replace("-", "").replace("_", "").isalnum()
    ):
        sys.exit("ERROR: run-name may contain only letters, digits, dash and underscore")
    if args.benchmark and (args.save_predictions or args.save_artifacts):
        sys.exit("ERROR: saving options require --run-training")
    if (args.save_predictions or args.save_artifacts) and not args.run_training:
        sys.exit("ERROR: saving options require --run-training")

    df = load_data()
    classes = sorted(df["Category"].unique())
    y = df["Category"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    train_idx, test_idx = load_split(df, args.seed)
    tr_idx, va_idx = carve_validation(df, train_idx, args.seed)
    miss_val = set(range(len(classes))) - set(y[va_idx].tolist())
    print("Rows: {} | Categories: {} | Seed: {}".format(len(df), len(classes), args.seed))
    print("Train: {} | Validation: {} | Test: {} | Group overlaps: 0".format(
        len(tr_idx), len(va_idx), len(test_idx)))
    print("Smallest validation class count: {} | classes missing from validation: {}".format(
        int(np.bincount(y[va_idx], minlength=len(classes)).min()), len(miss_val)))
    print("Experiment:", args.experiment)
    print("Truncation:", args.truncation + (" (tail {} tokens)".format(args.tail_tokens) if args.truncation == "head_tail" else ""))

    tag = resumematch_run_tag(args)
    if args.save_predictions or args.save_artifacts or args.save_model:
        targets = [
            OUT_DIR / ("metrics_" + tag + ".json"),
            OUT_DIR / ("confusion_matrix_" + tag + ".csv"),
            OUT_DIR / ("metrics_per_class_" + tag + ".csv"),
        ]
        if args.save_predictions:
            targets.append(OUT_DIR / ("predictions_" + tag + ".csv"))
        if args.save_artifacts:
            targets.extend([
                ROOT / "models" / ("distilbert_" + tag),
                OUT_DIR / ("manifest_" + tag + ".json"),
            ])
        if args.save_model:
            targets.append(ROOT / "models" / ("distilbert_" + tag + ".pt"))
        existing = [str(x) for x in targets if x.exists()]
        if existing:
            sys.exit(
                "ERROR: output already exists; choose a new --run-name: "
                + ", ".join(existing)
            )
        print("Saving preflight passed; existing results will not be overwritten.")

    if not (args.benchmark or args.run_training):
        print("Preparation only: no model loaded or trained. Use --benchmark or --run-training.")
        return
    if args.experiment == "B":
        texts = mask_category_names(df["Resume_str"], classes)
    else:
        texts = identity_text(df["Resume_str"])
    run(args, texts, y, classes, tr_idx, va_idx, test_idx, benchmark=args.benchmark)


if __name__ == "__main__":
    main()
