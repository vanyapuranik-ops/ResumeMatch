# ResumeMatch V2 Deployment Guide

## Application
- Branch: rolematch-v2-deployment
- App: role_training_pack/app_rolematch_v2.py
- Dependencies: requirements.txt

## Run
python -m pip install -r requirements.txt
python -m streamlit run role_training_pack/app_rolematch_v2.py

## Model hosting requirement
The app currently loads the model from data/role_v2_sources/v2_experiment_32class/trained_models/distilbert. The data directory is excluded from GitHub. The deployment maintainer must host the trained model separately and update the app to load it from the hosted location before cloud deployment.

Do not commit model weights, private resumes, access tokens, virtual environments, or backups. Test PDF, TXT, scanned PDF, short-input handling, and all 32 output labels before publishing. Existing test metrics do not guarantee real-world performance.
