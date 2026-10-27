# ResumeMatch Final Training Pack

## Files
- `ResumeMatch_Final_Training_Dataset.csv`: full reviewed candidate dataset.
- `train.csv`: stratified training split (70%).
- `validation.csv`: stratified validation split (~15%).
- `test.csv`: stratified held-out test split (~15%).
- `dataset_manifest.json`: class counts, split sizes, exclusions and limitations.

## Training columns
- `text`: resume text
- `label`: 12-class candidate role
- `source_row_id`, `resume_group_id`, `job_title`: provenance
- remaining columns retain original category/label evidence and review status

## Review decisions
Three rows were excluded from training:
- source row 1559: mixed Big Data platform support/data science title
- source row 4772: excerpt appeared dominated by scraped/non-resume website text
- source row 6835: excerpt emphasized ERP techno-functional consulting rather than a clear software-development role

Two ambiguous records (4109, 5341) were retained provisionally rather than automatically relabelled. They are marked in `review_status`.

## Important limitation
The labels are candidate roles derived from job-title rules. This dataset has not been manually verified resume-by-resume and must not be described as fully validated ground truth. Use validation for model selection and test only for final evaluation. Report macro-F1, weighted-F1, per-class precision/recall and a confusion matrix. Avoid claiming a strong result based on accuracy alone.
