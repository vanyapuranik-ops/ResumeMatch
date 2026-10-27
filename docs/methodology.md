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

These results show that the transformer-based model outperformed the traditional TF-IDF baselines on the evaluated test set. However, the comparison with the TF-IDF baselines and the Experiment A results rest on a single random seed (42), so they do not establish performance variability across seeds. Variability for Experiment B alone is reported in the three-seed results below.

**Paired comparison with the TF-IDF baselines (Experiment B, test set).** Because the TF-IDF models were fitted on the same 1,738 training rows and scored on the same 497 test resumes, predictions can be compared row by row against the saved DistilBERT B predictions. DistilBERT was correct where the baseline was wrong on 106 resumes for Linear SVM, 129 for Logistic Regression and 165 for Multinomial Naive Bayes; the reverse occurred on 15, 13 and 11 resumes respectively (two-sided sign tests, all p < 1e-17). Of the test resumes, 74 were misclassified by both DistilBERT B and Linear SVM B. Experiment A could not be compared row by row because DistilBERT A did not save row-level predictions, so its comparison rests on the aggregate metrics above. These comparisons hold for the default-hyperparameter baselines on one seed; the TF-IDF models were not tuned, so they are lower bounds on what classical methods can achieve.

**Error analysis (Experiment B, test set; diagnostic only).** Of DistilBERT B's 89 test errors, 74 were also made by Linear SVM B; 15 were DistilBERT-only (106 resumes were wrong for Linear SVM only). This overlap is partly expected, since a stronger model's errors tend to be a subset of a weaker model's. Five categories (AUTOMOBILE, BPO, AGRICULTURE, ARTS, APPAREL) make up 64 of 497 test resumes (13%) but account for 35 of the 89 errors (39%) and 32 of the 74 shared errors, with test recall of 0.14, 0.25, 0.46, 0.48 and 0.58. The training pool (1,986 rows before the validation carve) holds 29, 18, 50, 82 and 78 rows for these categories against a per-category median of 92 and a maximum of 96. BPO, AUTOMOBILE and AGRICULTURE therefore have far fewer training examples than typical, so scarcity is a plausible but untested cause; ARTS and APPAREL have near-typical counts, so sample size does not explain them, and overlap with other categories is a hypothesis not yet tested. The most frequent confusion, FITNESS to SALES (4 resumes), occurs in both models; all other confusion pairs involve 1-2 resumes. Resume length does not separate right from wrong: median 738 words for correct predictions and 735 for errors (interquartile ranges 646-885 and 653-928). DistilBERT's largest per-category gains over Linear SVM were in CONSULTANT (20 vs 3 correct of 23), SALES (18 vs 8 of 23), ADVOCATE (19 vs 8 of 24) and FINANCE (23 vs 14 of 24); the SVM matched or exceeded it on AUTOMOBILE, INFORMATION-TECHNOLOGY and PUBLIC-RELATIONS. Caveats: single seed, test classes of 4-24 resumes make per-category rates unstable, and because this analysis used the test set, any change motivated by it must be evaluated on validation data, cross-validation or new seeds.

**Confidence and short-input analysis (validation set, 248 resumes; diagnostic only).** Scores are softmax outputs, not calibrated probabilities. The top score never exceeded 0.895 (median 0.817, mean 0.737). Resumes with a top score above 0.7 (196 of 248) were correct 94% of the time (185 of 196); those at or below 0.7 (52) were correct 37% of the time (19 of 52); the 18 resumes at or below 0.30 were correct 3 times. The gap between the top two scores gives a similar split: the lowest-gap quartile (gap up to 0.687, 62 resumes) was 43.5% accurate and held 35 of the 44 validation errors, while the other three quartiles were 93.5% to 96.8% accurate. Scores understated accuracy at the high end (mean score 0.851 against 96.4% accuracy above 0.8) and overstated it in the 0.6 to 0.7 bin (0.662 against 54.5%, 11 resumes); bins below 0.6 hold 6 to 10 resumes each and are not interpreted. The 0.7 cut point was read from this validation set and is not an independently validated threshold.

To test short inputs, each validation resume was cut to its first N words (header retained), masked as in Experiment B and scored with the saved model (`src/stress_short.py`); the full-text pass reproduced the saved predictions exactly.

| Words kept | Accuracy | Macro F1 | Weighted F1 | Mean top score | Top score <= 0.5 |
|---|---:|---:|---:|---:|---:|
| Full text | 0.8226 | 0.7605 | 0.8097 | 0.737 | 14.1% |
| 250 | 0.7782 | 0.7096 | 0.7629 | 0.703 | 19.8% |
| 150 | 0.6895 | 0.6515 | 0.6865 | 0.635 | 30.6% |
| 90 | 0.6129 | 0.5923 | 0.6225 | 0.561 | 39.9% |

Accuracy and confidence both fell as less text was provided (at 90 words, 60 predictions that were correct on full text became wrong and 8 went the other way). Caveats: cutting a resume keeps its header, so this is a proxy for short inputs, not a model of naturally short resumes; the dataset's shortest resume has 113 words and the shortest validation resume 169, so very short inputs lie outside the training distribution; validation set only, single seed, test set not used.

**Robustness across three group-aware splits (Experiment B, test sets; seeds 42, 43 and 44).** Experiment B was trained with identical settings on the group-aware splits for seeds 43 and 44 (each 1,738 training, 248 validation and 497 test resumes, zero group overlap), in addition to seed 42. Each run used the same hyperparameters, and each seed changes the split, the validation carve, the initialization and the batch order together, so this measures variation across random splits, not repeated training on one split.

| Test metric | Seed 42 | Seed 43 | Seed 44 | Mean | Min-max |
|---|---:|---:|---:|---:|---:|
| Accuracy | 0.8209 | 0.8229 | 0.8028 | 0.8156 | 0.8028-0.8229 |
| Macro precision | 0.7807 | 0.7976 | 0.7659 | 0.7814 | 0.7659-0.7976 |
| Macro recall | 0.7727 | 0.7830 | 0.7587 | 0.7715 | 0.7587-0.7830 |
| Macro F1 | 0.7728 | 0.7848 | 0.7554 | 0.7710 | 0.7554-0.7848 |
| Weighted F1 | 0.8158 | 0.8176 | 0.7980 | 0.8105 | 0.7980-0.8176 |
| Best validation macro F1 | 0.7605 | 0.7477 | 0.7852 | 0.7645 | 0.7477-0.7852 |

Sample standard deviations over the three runs were 0.011 for accuracy and 0.015 for macro F1; with three runs these are rough, and only the mean and range are reported. The saved predictions, metrics and manifests for seeds 43 and 44 were audited against their splits, configurations and hashes. Test sets of different seeds overlap partly (86 to 109 shared rows between pairs), so the three results are not fully independent.

**Findings.** Macro F1 ranged from 0.755 to 0.785 and accuracy from 0.803 to 0.823, so the single-seed values in the tables above lie inside this range. The seed with the highest validation macro F1 (44) had the lowest test macro F1, so the 248-resume validation set is a noisy guide. The best epoch was 6, the maximum allowed, in all three seeds, and seed 44's validation macro F1 was still rising (0.769, 0.778, 0.785 over epochs 4 to 6); whether a larger epoch limit would help was not tested. AUTOMOBILE, BPO, APPAREL and AGRICULTURE had test F1 below 0.70 in all three seeds (mean F1 0.22, 0.32, 0.52 and 0.60), while ARTS fell below 0.70 in two seeds but reached 0.81 on seed 43 (range 0.54 to 0.81). BPO and AUTOMOBILE have 4 to 8 test resumes per seed, so their individual F1 values are unreliable; no cause for any weak category has been established. HR, ACCOUNTANT, FINANCE, BUSINESS-DEVELOPMENT, DESIGNER and CONSTRUCTION had mean F1 of 0.90 or higher.

**Caveats.** The seed-42 run used a different `train_transformer.py` file hash than seeds 43 and 44 (the latter two ran identical code); the training operations reviewed were the same but exact code identity for seed 42 is not proven. MPS training is seeded but not bit-deterministic. Experiment A, the TF-IDF baselines and the chunked-inference comparison were not repeated on seeds 43 and 44.

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

The project has several limitations. First, the dataset contains only 2,483 labelled resumes across 24 categories and exhibits class imbalance. Second, many resumes exceed the 512-token transformer input limit, meaning that the standard inference approach may not use all available resume information. Third, masking category names in the first 100 characters tests one potential source of leakage but does not eliminate every possible source of label-related information. Finally, the Experiment B transformer results were repeated on three group-aware splits (seeds 42, 43 and 44), but the Experiment A, TF-IDF baseline and chunked-inference results use a single seed, and three seeds are too few to estimate confidence intervals.

Future work should include:

- Extending the multi-seed evaluation: Experiment B was run on seeds 42, 43 and 44 (see section 11); Experiment A, the TF-IDF baselines and the chunking experiment were not repeated on seeds 43 and 44, and three seeds are too few for confidence intervals.
- Investigating alternative strategies for handling long resumes.
- Exploring class-specific performance and additional error analysis.
- Investigating probability calibration and prediction confidence.
- Testing the system on an independent resume dataset to evaluate generalisation.
- Developing an interface that accepts resume text and returns a predicted job category with an explanation of the recommendation.

## 14. Conclusion

ResumeMatch compares traditional TF-IDF-based classifiers with a fine-tuned DistilBERT model for resume classification across 24 job categories. Linear SVM was the strongest traditional baseline, while DistilBERT achieved the best overall performance in the reported experiments.

The category-name masking experiment examined the effect of explicit category terms near the beginning of resumes. The chunking experiment found no improvement from the tested aggregation methods on the validation set, while increasing inference time.

Overall, the results support the use of transformer-based text representations for this resume classification task. Experiment B was repeated on three group-aware splits, with test macro F1 between 0.755 and 0.785 and accuracy between 0.803 and 0.823. Multi-seed evaluation of Experiment A, the TF-IDF baselines and chunked inference, external testing, and improvements to long-document processing are still needed before drawing broader conclusions about real-world performance.
