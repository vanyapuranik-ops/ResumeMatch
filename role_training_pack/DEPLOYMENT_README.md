# ResumeMatch V2 Deployment Handoff

Run from the repository root:
python -m pip install -r requirements.txt
python -m streamlit run role_training_pack/app_rolematch_v2.py

## Model hosting
The app expects the model at data/role_v2_sources/v2_experiment_32class/trained_models/distilbert. This directory is excluded from GitHub. The deployment maintainer must host the model separately, update app_rolematch_v2.py to load the hosted model, and verify the model, tokenizer, and label_mapping.json are available.

## Checklist
- Install requirements.txt.
- Configure model hosting and access.
- Verify all 32 output labels.
- Test TXT, text-based PDF, scanned PDF, and short inputs.
- Keep private resumes, secrets, and model weights out of GitHub.

Recommendation scores are not calibrated probabilities. Existing held-out metrics are not independent real-world validation.
