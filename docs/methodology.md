# ResumeMatch: ML-Based Job Role Recommendation from Resumes

## 1. Introduction

ResumeMatch is a machine-learning-based job role recommendation system that predicts suitable job categories from resume text. The project compares traditional TF-IDF-based machine learning models with a transformer-based DistilBERT model to evaluate their effectiveness in resume classification.

## 2. Problem Statement

Recruiters and job seekers often need to identify appropriate job roles from large amounts of resume information. Manual classification can be time-consuming and inconsistent. ResumeMatch addresses this problem by automatically classifying resumes into predefined job categories using natural language processing and machine learning.

## 3. Objectives

The objectives of the project are:

- To preprocess resume text and prepare a labelled dataset.
- To build traditional machine-learning baselines using TF-IDF features.
- To train a transformer-based DistilBERT classifier.
- To compare models using accuracy, precision, recall, macro F1-score, and weighted F1-score.
- To investigate the influence of category-name leakage through a controlled text-masking experiment.
- To examine whether processing resumes in overlapping chunks improves classification performance.

## 4. Dataset Description

The project uses a labelled resume dataset containing resume text and corresponding job categories. After removing one empty resume, the processed dataset contains 2,483 resumes across 24 job categories.

The dataset includes resume identifiers, resume text, category labels, and group identifiers used to identify duplicate resume text. The median resume length is 757 words. The dataset has a class imbalance ratio of approximately 5.45:1, indicating that some categories contain substantially more examples than others.

Duplicate-text analysis identified two groups of duplicate resumes. Group-aware splitting was used to prevent resumes containing identical text from appearing in both training and test sets.

## 5. Data Preprocessing

The dataset was cleaned by removing the empty resume and preparing the resume text and category labels for modelling. Group identifiers were retained to support leakage-controlled dataset splitting.

Two text conditions were evaluated:

- **Experiment A — Original text:** The original cleaned resume text was used.
- **Experiment B — Header-masked text:** Occurrences of category names within the first 100 characters of the resume were masked using a label-agnostic transformation.

The masking experiment was designed to investigate whether models relied heavily on explicit job-category names near the beginning of resumes. The transformation was applied without using the individual resume's target label to determine which category name to mask.

## 6. TF-IDF Baseline Models

Traditional machine-learning models were trained using Term Frequency–Inverse Document Frequency (TF-IDF) features. The vectorizer used lowercase text, English stop-word removal, sublinear term-frequency scaling, unigram and bigram features, and a minimum document frequency of two.

Three classifiers were evaluated:

1. **Logistic Regression:** Trained with balanced class weights and a maximum of 1,000 iterations.
2. **Linear Support Vector Machine:** Implemented using LinearSVC with balanced class weights.
3. **Multinomial Naive Bayes:** Used as a probabilistic baseline for text classification.

All three models were evaluated under both Experiment A and Experiment B, resulting in six TF-IDF pipelines.

## 7. Transformer-Based Classification

DistilBERT was used as the transformer-based classification model. The pretrained transformer was fine-tuned for the 24-class resume classification task.

The transformer experiments used a maximum sequence length of 512 tokens, a batch size of eight, gradient accumulation of two steps, a learning rate of 3 × 10⁻⁵, weight decay of 0.01, and a maximum of six training epochs. Class-weighted loss was used to address class imbalance.

The checkpoint was selected using validation macro F1-score, with early stopping patience of two epochs and a warm-up proportion of 0.1. Training was performed on the Apple Silicon Metal Performance Shaders (MPS) device.

The transformer was evaluated under both original-text and header-masked conditions.

## 8. Dataset Splitting and Evaluation Metrics

A group-aware stratified split with random seed 42 was used. The initial split contained 1,986 training-partition rows and 497 test rows. A group-aware validation subset of 248 rows was selected from the training partition, leaving 1,738 rows for model fitting.

The final partitions were therefore:

- Training: 1,738 resumes
- Validation: 248 resumes
- Testing: 497 resumes

Group overlap between training and test partitions was zero. The validation set was used for checkpoint selection, while the test set was used for final evaluation.

The following metrics were reported:

- **Accuracy:** Proportion of predictions that are correct.
- **Macro Precision:** Precision calculated independently for each category and averaged equally across categories.
- **Macro Recall:** Recall calculated independently for each category and averaged equally across categories.
- **Macro F1-score:** Harmonic mean of precision and recall calculated per category and averaged equally.
- **Weighted F1-score:** F1-score averaged according to the number of examples in each category.

Macro F1-score was particularly important because the dataset contains imbalanced job categories.

## 9. Reproducibility and Implementation

The experiments were implemented in Python using the project's virtual environment. The training scripts, saved split assignments, metrics, predictions, model artifacts, and experiment manifests were retained in the project directory.

The experiments used random seed 42 and the same underlying train, validation, and test assignments to support comparisons between models and text conditions.

The saved outputs allow the reported results to be inspected without retraining the models.

## 10. TF-IDF Baseline Results

The following table presents the test-set performance of the three TF-IDF classifiers under both text conditions.

| Model | Experiment | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|---:|---:|
| Logistic Regression | A | 0.6036 | 0.5889 | 0.5627 | 0.5438 | 0.5775 |
| Linear SVM | A | 0.6579 | 0.6457 | 0.6164 | 0.6055 | 0.6391 |
| Multinomial Naive Bayes | A | 0.5151 | 0.4924 | 0.4667 | 0.4266 | 0.4684 |
| Logistic Regression | B | 0.5875 | 0.5579 | 0.5477 | 0.5248 | 0.5571 |
| Linear SVM | B | 0.6378 | 0.6213 | 0.5994 | 0.5860 | 0.6162 |
| Multinomial Naive Bayes | B | 0.5111 | 0.4883 | 0.4631 | 0.4237 | 0.4652 |

Among the TF-IDF baselines, Linear SVM achieved the highest accuracy and macro F1-score in both experiments. Masking category names in the first 100 characters reduced the performance of all three baseline classifiers, although the size of the reduction varied by model.

## 11. DistilBERT Results and Comparison

DistilBERT achieved the following test-set results:

| Metric | Experiment A | Experiment B |
|---|---:|---:|
| Accuracy | 0.8390 | 0.8209 |
| Macro Precision | 0.7903 | 0.7807 |
| Macro Recall | 0.7799 | 0.7727 |
| Macro F1 | 0.7774 | 0.7728 |
| Weighted F1 | 0.8306 | 0.8158 |
| Best epoch | 3 | 6 |

Experiment A achieved the higher overall scores. Experiment B also performed strongly despite masking category names near the beginning of resumes.

Under Experiment B, DistilBERT achieved an accuracy of 0.8209 and macro F1-score of 0.7728. The strongest TF-IDF baseline under the same condition was Linear SVM, with an accuracy of 0.6378 and macro F1-score of 0.5860. DistilBERT therefore improved accuracy by approximately 18.31 percentage points and macro F1-score by 0.1868 compared with this baseline.

These results show that the transformer-based model outperformed the traditional TF-IDF baselines on the evaluated test set. However, the experiments used a single random seed, so the results do not establish performance variability across different seeds.

## 12. Chunked Inference Experiment

Long resumes can exceed the transformer's maximum input length of 512 tokens. A chunking experiment was therefore conducted on the 248-row validation set to investigate whether processing longer resumes in overlapping windows would improve predictions.

The experiment used windows of 512 tokens with an overlap of 128 tokens. It generated 745 chunks across the validation resumes, with 229 resumes containing multiple chunks.

Three approaches were compared:

1. Standard inference using the first 512 tokens.
2. Averaging the predicted class probabilities across chunks.
3. Averaging the model logits across chunks.

The standard first-512-token approach achieved validation accuracy of 0.8226 and macro F1-score of 0.7605. Mean-probability aggregation achieved accuracy of 0.8024 and macro F1-score of 0.7450. Mean-logit aggregation achieved accuracy of 0.7460 and macro F1-score of 0.6951.

Mean-probability aggregation corrected 11 predictions but changed 16 previously correct predictions to incorrect. The observed difference was not statistically significant in the paired sign test (p = 0.442). Mean-logit aggregation performed worse, with a paired sign-test p-value of 0.005.

Chunked inference took approximately 47.3 seconds for 745 chunks, compared with 12.8 seconds for 248 standard sequences. Given the additional computational cost and the lack of demonstrated improvement, the first-512-token approach was retained for the reported evaluation.

These findings are limited to the validation set and the tested aggregation strategies. They do not demonstrate that all chunking methods are ineffective.

## 13. Limitations and Future Work

The project has several limitations. First, the dataset contains only 2,483 labelled resumes across 24 categories and exhibits class imbalance. Second, many resumes exceed the 512-token transformer input limit, meaning that the standard inference approach may not use all available resume information. Third, masking category names in the first 100 characters tests one potential source of leakage but does not eliminate every possible source of label-related information. Finally, the reported transformer results use one random seed.

Future work should include:

- Evaluating performance across multiple random seeds.
- Investigating alternative strategies for handling long resumes.
- Exploring class-specific performance and additional error analysis.
- Investigating probability calibration and prediction confidence.
- Testing the system on an independent resume dataset to evaluate generalisation.
- Developing an interface that accepts resume text and returns a predicted job category with an explanation of the recommendation.

## 14. Conclusion

ResumeMatch compares traditional TF-IDF-based classifiers with a fine-tuned DistilBERT model for resume classification across 24 job categories. Linear SVM was the strongest traditional baseline, while DistilBERT achieved the best overall performance in the reported experiments.

The category-name masking experiment examined the effect of explicit category terms near the beginning of resumes. The chunking experiment found no improvement from the tested aggregation methods on the validation set, while increasing inference time.

Overall, the results support the use of transformer-based text representations for this resume classification task. Further multi-seed evaluation, external testing, and improvements to long-document processing are needed before drawing broader conclusions about real-world performance.
