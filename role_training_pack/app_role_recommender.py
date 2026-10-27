from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "role_distilbert_v1" / "model"
TRAIN_CSV = ROOT / "train.csv"

st.set_page_config(page_title="ResumeMatch", page_icon="📄", layout="centered")
st.title("ResumeMatch")
st.subheader("AI-based job role recommendation")
st.write("Upload a resume or paste its text to see the roles most closely predicted by the trained model.")

if not MODEL_DIR.exists():
    st.error(f"Saved model not found: {MODEL_DIR}")
    st.stop()

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    model.to(device)
    model.eval()

    config_map = getattr(model.config, "id2label", {}) or {}
    id2label = {}
    for key, value in config_map.items():
        try:
            idx = int(key)
            if value and not str(value).startswith("LABEL_"):
                id2label[idx] = str(value)
        except (TypeError, ValueError):
            pass

    if len(id2label) != model.config.num_labels:
        id2label = {}
        train = pd.read_csv(TRAIN_CSV)
        if "candidate_role" not in train.columns:
            raise ValueError("candidate_role column is missing from train.csv")
        pairs = train[["label", "candidate_role"]].dropna().drop_duplicates()
        for _, row in pairs.iterrows():
            try:
                id2label[int(row["label"])] = str(row["candidate_role"])
            except (TypeError, ValueError):
                continue

    if len(id2label) != model.config.num_labels:
        raise ValueError("Could not map all model output IDs to job roles. Check the training label encoding.")

    return tokenizer, model, device, id2label

try:
    tokenizer, model, device, id2label = load_model()
except Exception as exc:
    st.error(f"Could not load model or role labels: {exc}")
    st.stop()

st.caption(f"Model loaded • Device: {device}")

uploaded_file = st.file_uploader("Upload a resume (PDF or TXT)", type=["pdf", "txt"])
resume_text = st.text_area(
    "Or paste resume text here",
    height=220,
    placeholder="Paste the resume's skills, education, experience, and projects...",
)

if uploaded_file is not None:
    try:
        if uploaded_file.name.lower().endswith(".txt"):
            resume_text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        else:
            import pdfplumber
            import io
            with pdfplumber.open(io.BytesIO(uploaded_file.getvalue())) as pdf:
                resume_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        st.success(f"Read text from {uploaded_file.name}")
        if not resume_text.strip():
            st.warning("No selectable text was found. If this is a scanned PDF, use OCR or paste the resume text.")
    except Exception as exc:
        st.error(f"Could not read this file: {exc}")

if st.button("Recommend job roles", type="primary"):
    text = resume_text.strip()
    if len(text) < 30:
        st.warning("Please provide more resume text before requesting a recommendation.")
    else:
        with st.spinner("Analyzing resume..."):
            inputs = tokenizer(
                text,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
                probabilities = torch.softmax(logits, dim=-1)[0]

            top_k = min(3, len(probabilities))
            scores, indices = torch.topk(probabilities, k=top_k)

        st.markdown("### Recommended roles")
        for rank, (score, idx) in enumerate(zip(scores.tolist(), indices.tolist()), start=1):
            role = id2label[int(idx)]
            st.write(f"**{rank}. {role}** — {score * 100:.2f}% model confidence")
            st.progress(float(score))

        st.info(
            "These are model predictions, not a guarantee of job suitability. "
            "Confidence scores are not verified probabilities of getting hired. "
            "Review the resume and recommendations before making decisions."
        )
