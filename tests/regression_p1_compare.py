"""Read-only regression test: original predict.py (pre-P1 backup) vs updated predict.py.

Loads the SAME saved checkpoint and tokenizer once, runs both versions' predict() on one sample
resume (each twice), and compares top-3 labels, returned scores and an independent probability
reference. Writes nothing; hashes all inputs before and after to prove they were not modified.
Never prints resume text. Exit code 0 = pass, 1 = fail.
"""
import argparse
import ast
import hashlib
import importlib.machinery
import importlib.util
import inspect
import os
import sys
import types
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ORIGINAL_SHA256 = "aab27c258171c0726dcea0e9a753360e5a764e3fac463d056882ed4cf90ed72a"
SCORE_TOL = 1e-4      # returned scores are rounded to 4 decimals, so 1e-4 is one rounding step
ROUND_TOL = 0.5e-4 + 2e-6   # max |rounded float32 score - double-precision reference|
SAME_FUNCS = ("norm", "mask_category_names", "predict")
SAME_CONSTS = ("MAX_LENGTH", "HEAD_CHARS", "PLACEHOLDER", "DEFAULT_MODEL_DIR", "DEFAULT_MANIFEST")
RESULTS = []


def fail(msg):
    print("FAIL: " + msg)
    sys.exit(1)


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print("  [{}] {}{}".format("ok" if ok else "FAIL", name, (" - " + detail) if detail else ""))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def snapshot(model_dir, extra):
    files = sorted(p for p in Path(model_dir).rglob("*") if p.is_file()) + [Path(e) for e in extra]
    return {str(p): sha256_file(p) for p in files}


def function_nodes(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def ensure_pdfplumber():
    try:
        import pdfplumber  # noqa: F401
    except ImportError:
        stub = types.ModuleType("pdfplumber")

        def unavailable(*a, **k):
            raise ImportError("pdfplumber is stubbed in this regression test")
        stub.open = unavailable
        sys.modules["pdfplumber"] = stub
        print("NOTE: pdfplumber is not installed. A stub is used ONLY so the original module can be imported; "
              "predict() does not use it (checked below).")


def load_module(path, name):
    path = Path(path)
    if not path.is_file():
        fail("source file not found: " + str(path))
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    try:
        loader.exec_module(module)
    except Exception as exc:
        fail("could not import {} ({}: {})".format(path.name, type(exc).__name__, exc))
    return module


def validate_output(out, label2id, who):
    n = min(3, len(label2id))
    if not isinstance(out, list) or len(out) != n:
        fail("{}: predict() must return a list of {} items, got {}".format(
            who, n, "a " + type(out).__name__ if not isinstance(out, list) else str(len(out)) + " items"))
    seen, prev = set(), None
    for item in out:
        if not isinstance(item, dict) or set(item) != {"category", "score"}:
            fail("{}: each item must be a dict with exactly 'category' and 'score'".format(who))
        cat, score = item["category"], item["score"]
        if cat not in label2id or cat in seen:
            fail("{}: category {!r} is unknown or repeated".format(who, cat))
        seen.add(cat)
        if not isinstance(score, float) or score != score or not 0.0 <= score <= 1.0:
            fail("{}: score {!r} is not a float in [0, 1]".format(who, score))
        if prev is not None and score > prev:
            fail("{}: scores are not in descending order".format(who))
        prev = score


def compare(a, b, label):
    same = [x["category"] for x in a] == [x["category"] for x in b]
    diff = max(abs(x["score"] - y["score"]) for x, y in zip(a, b))
    check(label + ": top-3 labels and order identical", same)
    check(label + ": returned scores agree", diff <= SCORE_TOL,
          "max |diff| = {:.6f}{}".format(diff, " (exact)" if diff == 0 else ""))
    return diff


def main():
    ap = argparse.ArgumentParser(description="Read-only P1 regression comparison")
    ap.add_argument("--original", default=str(ROOT / "src" / "predict.py.pre_p1.bak"))
    ap.add_argument("--updated", default=str(ROOT / "src" / "predict.py"))
    ap.add_argument("--sample", default=str(ROOT / "sample_resume.txt"))
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--device", choices=["auto", "cpu"], default="auto")
    args = ap.parse_args()
    original_path = Path(args.original).expanduser()
    updated_path = Path(args.updated).expanduser()
    sample_path = Path(args.sample).expanduser()
    if not sample_path.is_file():
        fail("sample resume not found: {} (pass --sample /path/to/sample_resume.txt)".format(sample_path))

    print("1) Static comparison of the two source files")
    print("   original sha256:", sha256_file(original_path))
    print("   updated  sha256:", sha256_file(updated_path))
    if sha256_file(original_path) != EXPECTED_ORIGINAL_SHA256:
        print("   WARNING: the original differs from the hash recorded before the P1 patch.")
    fo, fu = function_nodes(original_path), function_nodes(updated_path)
    for name in SAME_FUNCS:
        if name not in fo or name not in fu:
            fail("function {!r} missing in one of the files".format(name))
        check("{}() AST identical".format(name), ast.dump(fo[name]) == ast.dump(fu[name]))
    uses_pdf = any(isinstance(n, ast.Name) and n.id == "pdfplumber" for n in ast.walk(fo["predict"]))
    check("predict() does not reference pdfplumber", not uses_pdf)

    print("2) Importing both modules")
    ensure_pdfplumber()
    original = load_module(original_path, "predict_original")
    updated = load_module(updated_path, "predict_updated")
    for name in SAME_CONSTS:
        check("constant {} equal".format(name), str(getattr(original, name, None)) == str(getattr(updated, name, None)))
    check("predict() signature equal",
          str(inspect.signature(original.predict)) == str(inspect.signature(updated.predict)),
          str(inspect.signature(updated.predict)))

    model_dir = Path(args.model_dir or updated.DEFAULT_MODEL_DIR).expanduser()
    manifest_path = Path(args.manifest or updated.DEFAULT_MANIFEST).expanduser()
    if not model_dir.is_dir() or not manifest_path.is_file():
        fail("model directory or manifest not found: {} | {}".format(model_dir, manifest_path))
    before = snapshot(model_dir, [manifest_path, original_path, updated_path, sample_path])
    print("   hashed {} input files before running".format(len(before)))

    print("3) Manifest and checkpoint")
    import json
    try:
        import torch
        import transformers
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        fail("torch/transformers import failed: {}".format(exc))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    label2id = manifest.get("label2id")
    if not isinstance(label2id, dict) or len(label2id) != 24:
        fail("manifest label2id is not a dict of 24 categories")
    if sorted(label2id.values()) != list(range(24)):
        fail("manifest label ids are not exactly 0..23")
    device = torch.device("mps" if args.device == "auto" and torch.backends.mps.is_available() else "cpu")
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), local_files_only=True).to(device)
        model.eval()
    except Exception as exc:
        fail("checkpoint loading failed ({}: {})".format(type(exc).__name__, exc))
    config_map = {int(k): v for k, v in model.config.id2label.items()}
    manifest_map = {v: k for k, v in label2id.items()}
    check("checkpoint id2label matches manifest label2id", config_map == manifest_map,
          "" if config_map == manifest_map else "mismatching ids: {}".format(
              sorted(i for i in manifest_map if config_map.get(i) != manifest_map[i])[:10]))
    check("model has 24 labels", model.config.num_labels == 24)
    print("   device: {} | torch {} | transformers {}".format(device.type, torch.__version__, transformers.__version__))

    print("4) Sample input (read as the CLI does: UTF-8, stripped)")
    text = sample_path.read_text(encoding="utf-8").strip()
    print("   words: {} | characters: {} | sample sha256: {}".format(
        len(text.split()), len(text), sha256_file(sample_path)[:16] + "..."))
    if not text:
        fail("sample is empty after stripping")

    print("5) Running predict() (each version twice, same model/tokenizer objects)")

    def call(module, who):
        try:
            out = module.predict(text, model, tokenizer, label2id, device)
        except Exception as exc:
            fail("{} predict() raised {}: {}".format(who, type(exc).__name__, exc))
        validate_output(out, label2id, who)
        return out
    o1, u1, o2, u2 = call(original, "original#1"), call(updated, "updated#1"), call(original, "original#2"), call(updated, "updated#2")
    d_ou = compare(o1, u1, "original vs updated")
    compare(o1, o2, "original repeatability")
    compare(u1, u2, "updated repeatability")

    print("6) Independent reference probabilities")
    prepared = updated.mask_category_names(text, list(label2id.keys()))
    inputs = tokenizer(prepared, return_tensors="pt", truncation=True,
                       max_length=updated.MAX_LENGTH, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        logits = model(**inputs).logits[0].detach().float().cpu()
    if logits.shape != (24,) or not torch.isfinite(logits).all():
        fail("reference logits have unexpected shape or non-finite values")
    probs = torch.softmax(logits.double(), dim=-1)
    check("reference probabilities sum to 1", abs(float(probs.sum()) - 1.0) < 1e-9)
    vals, idx = torch.topk(probs, 3)
    ref_names = [manifest_map[int(i)] for i in idx]
    check("reference top-3 labels match returned labels", ref_names == [x["category"] for x in o1])
    ref_diff = max(abs(float(v) - x["score"]) for v, x in zip(vals, o1))
    check("returned scores equal rounded reference", ref_diff <= ROUND_TOL, "max |diff| = {:.2e}".format(ref_diff))
    print("   tokens fed to the model: {}".format(int(inputs["input_ids"].shape[1])))
    print("\n   rank  category                 original  updated   unrounded ref   |orig-upd|")
    for r, (a, b, v) in enumerate(zip(o1, u1, vals), 1):
        print("   {:<5} {:<24} {:<9.4f} {:<9.4f} {:<15.6f} {:.6f}".format(
            r, a["category"], a["score"], b["score"], float(v), abs(a["score"] - b["score"])))

    print("7) P1 additions (updated version only; informational, no resume text shown)")
    if not hasattr(updated, "assess_input") or not hasattr(updated, "confidence_notes"):
        check("updated file defines assess_input and confidence_notes", False)
    else:
        a = updated.assess_input(text)
        print("   words {} | letter share {:.3f} | refused: {}".format(
            a["words"], a["letter_share"], bool(a["refuse"])))
        for w in a["warnings"]:
            print("   WARNING:", w)
        if a["refuse"]:
            print("   The updated CLI would STOP for this sample and print no scores:", a["refuse"])
        for n in updated.confidence_notes(u1):
            print("   NOTE:", n)

    print("8) Inputs unchanged after the run")
    after = snapshot(model_dir, [manifest_path, original_path, updated_path, sample_path])
    changed = [p for p in before if before[p] != after.get(p)] + [p for p in after if p not in before]
    check("all {} hashed files unchanged".format(len(before)), not changed, ", ".join(changed[:5]))

    print()
    if all(RESULTS):
        print("RESULT: PASS ({} checks)".format(len(RESULTS)))
        print("Returned-score difference between versions: {:.6f}".format(d_ou))
        return 0
    print("RESULT: FAIL ({} of {} checks failed)".format(RESULTS.count(False), len(RESULTS)))
    return 1


if __name__ == "__main__":
    sys.exit(main())
