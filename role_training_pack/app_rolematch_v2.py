import io
import json
from pathlib import Path

import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = (
    ROOT / "data" / "role_v2_sources" / "v2_experiment_32class"
    / "trained_models" / "distilbert"
)
MAX_LENGTH = 384
MIN_WORDS = 50

st.set_page_config(page_title="ResumeMatch V2", page_icon="📄", layout="centered")
st.title("ResumeMatch V2")
st.subheader("AI-based job role recommendation")
st.caption("Experimental 32-role DistilBERT model")

@st.cache_resource
def load_model():
    with open(MODEL_DIR / "label_mapping.json", encoding="utf-8") as f:
        mapping = json.load(f)

    label2id = mapping["label2id"]
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR), local_files_only=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR), local_files_only=True
    )

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    model.to(device)
    model.eval()

    if len(label2id) != model.config.num_labels:
        raise ValueError("Label mapping and model output size do not match.")

    return tokenizer, model, device, label2id

try:
    tokenizer, model, device, label2id = load_model()
except Exception as exc:
    st.error(f"Could not load V2 model: {exc}")
    st.stop()

st.caption(f"Model loaded successfully · Device: {device}")

uploaded = st.file_uploader("Upload resume (PDF or TXT)", type=["pdf", "txt"])
pasted = st.text_area(
    "Or paste resume text",
    height=240,
    placeholder="Include skills, education, projects and work experience..."
)

resume_text = pasted

if uploaded is not None:
    try:
        if uploaded.name.lower().endswith(".txt"):
            resume_text = uploaded.getvalue().decode("utf-8", errors="replace")
        else:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(uploaded.getvalue())) as pdf:
                resume_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            if not resume_text.strip():
                st.warning(
                    "No selectable text found. Scanned PDFs need OCR; "
                    "you can paste the resume text instead."
                )
    except Exception as exc:
        st.error(f"Could not read uploaded file: {exc}")
        resume_text = ""

if st.button("Recommend job roles", type="primary"):
    text = (resume_text or "").strip()
    word_count = len(text.split())

    if word_count < MIN_WORDS:
        st.warning(
            f"Please provide a more complete resume. Found {word_count} words; "
            f"at least {MIN_WORDS} are required for this demo."
        )
    else:
        with st.spinner("Analyzing resume..."):
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH,
            )
            token_count = len(tokenizer(text, truncation=False)["input_ids"])
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.inference_mode():
                logits = model(**inputs).logits[0]
                probabilities = torch.softmax(logits, dim=-1)
                scores, indices = torch.topk(
                    probabilities, k=min(3, len(label2id))
                )

        st.markdown("### Recommended roles")
        for rank, (score, index) in enumerate(
            zip(scores.tolist(), indices.tolist()), start=1
        ):
            role = model.config.id2label.get(
                str(index), model.config.id2label.get(index)
            )
            if not role or role.startswith("LABEL_"):
                role = next(
                    name for name, idx in label2id.items() if idx == index
                )
            st.write(f"**{rank}. {role}** — {score:.2%} model score")

        st.caption(f"Resume words: {word_count} · Tokens before truncation: {token_count}")
        if token_count > MAX_LENGTH:
            st.warning(
                f"The model reads at most {MAX_LENGTH} tokens. "
                "Some resume content was omitted."
            )

        st.info(
            "This is an experimental recommendation, not a hiring decision. "
            "Scores are softmax outputs, not calibrated probabilities or "
            "guarantees of suitability. The reported test performance is "
            "from the project's dataset; real-world performance is not yet established."
        )
