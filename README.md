# ResumeMatch: ML-Based Job Role Recommendation from Resumes

ResumeMatch is a machine learning and Natural Language Processing (NLP) project that predicts suitable job categories from resume text. The project compares traditional TF-IDF-based machine learning models with a fine-tuned DistilBERT transformer model.

## Objectives

- Automate resume classification into predefined job categories.
- Compare traditional machine learning models with transformer-based NLP.
- Investigate category-name leakage through a text-masking experiment.
- Evaluate model performance using multiple classification metrics.
- Explore methods for handling long resumes during inference.

## Dataset Overview

- **Total resumes:** 2,483
- **Job categories:** 24
- **Median resume length:** 757 words
- **Class imbalance ratio:** 5.45:1
- **Data preparation:** Text cleaning, duplicate identification, and group-aware data splitting.

The original dataset is not included in this repository.

## Models Implemented

### Traditional Machine Learning
- Logistic Regression
- Linear Support Vector Machine (Linear SVM)
- Multinomial Naive Bayes

### Transformer-Based Model
- DistilBERT fine-tuned for 24-class resume classification.

## Experimental Results

| Model | Experiment | Accuracy | Macro F1 |
|---|---|---:|---:|
| Logistic Regression | Original text | 60.36% | 54.38% |
| Linear SVM | Original text | 65.79% | 60.55% |
| Multinomial Naive Bayes | Original text | 51.51% | 42.66% |
| Logistic Regression | Header-masked text | 58.75% | 52.48% |
| Linear SVM | Header-masked text | 63.78% | 58.60% |
| Multinomial Naive Bayes | Header-masked text | 51.11% | 42.37% |
| DistilBERT | Original text | 83.90% | 77.74% |
| DistilBERT | Header-masked text | 82.09% | 77.28% |

*Results are from experiments using random seed 42. Further evaluation across multiple seeds and independent datasets is needed to assess generalisation.*

## Methodology

The project follows these main stages:

1. Resume text preprocessing and dataset auditing.
2. Exploratory data analysis and duplicate-group identification.
3. Group-aware training, validation, and test splitting.
4. TF-IDF feature extraction and traditional classifier training.
5. DistilBERT fine-tuning for resume classification.
6. Evaluation using accuracy, macro precision, macro recall, macro F1, and weighted F1.
7. Experiments involving header masking and long-resume chunking.

For detailed experimental settings and findings, see [`docs/methodology.md`](docs/methodology.md).

## Project Structure

```text
ResumeMatch/
├── docs/
│   └── methodology.md
├── src/
│   ├── audit.py
│   ├── config.py
│   ├── eda.py
│   ├── evaluate_chunking.py
│   ├── evaluate_confidence.py
│   ├── predict.py
│   ├── preprocess.py
│   ├── run_baselines.py
│   └── train_transformer.py
├── requirements.txt
├── sample_resume.txt
└── README.md
```

## Installation

Python 3.12 is recommended for the current project environment.

Clone the repository:

```bash
git clone https://github.com/vanyapuranik-ops/ResumeMatch.git
cd ResumeMatch
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running the Project

The `src/` directory contains scripts for preprocessing, auditing, exploratory analysis, training, prediction, confidence evaluation, and chunking experiments.

Before running data-dependent scripts, place the original dataset in the location expected by `src/config.py`. Check the configuration and script arguments before executing individual workflows.

## Limitations

- The dataset contains imbalanced job categories.
- Current reported results use a single random seed.
- The standard transformer input length limits how much resume text is processed at once.
- Performance on independent datasets has not yet been established.
- Predicted categories reflect patterns in the training dataset and are not guarantees of job suitability.

## Future Work

- Evaluate performance across multiple random seeds.
- Test generalisation on independent resume datasets.
- Improve long-document processing.
- Expand explainability and confidence analysis.
- Develop a user interface for resume upload and job-role recommendations.

## Disclaimer

ResumeMatch is an academic machine learning project. Its predictions should support career exploration and should not be treated as definitive hiring or career decisions.