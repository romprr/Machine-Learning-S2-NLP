# Stack Overflow Tag Classification

This project tackles a 20-class text classification problem on Stack Overflow questions. Each row contains a question body and one target tag, and the goal is to predict the correct programming-language or framework tag from the post text.

The repository is organized as an end-to-end NLP pipeline: preprocessing, handcrafted feature extraction, vectorization, model benchmarking, hyperparameter tuning, ensembling, and evaluation. After adding `Model 3`, we benchmarked both 2-model and 3-model ensembles. The best overall system in the repo is still the 2-model `soft_calibrated_weighted` ensemble, which reaches `0.8553` accuracy and `0.8552` macro F1 on the held-out test set.

## Recommended Presentation Flow

If you use this README for slides, this order usually works better than a raw code walkthrough:

1. Problem and dataset
2. Data preprocessing
3. Handcrafted features
4. Pipeline overview
5. Vectorization methods
6. Model 1, Model 2, Model 3
7. Ensemble models
8. Evaluation and final takeaway

## Dataset and Experimental Protocol

- Dataset: `stack-overflow-data.csv`
- Total rows: `40,000`
- Number of labels: `20`
- Class balance: `2,000` posts per label
- Train/test split: `75% / 25%`
- Actual split used in the repo: `30,000` train and `10,000` test
- Per-class split: `1,500` train and `500` test
- Random seed: `42`

The split is fully balanced across all labels, which keeps the evaluation easy to interpret and aligns with the assignment requirement.

Main evaluation metrics:

- Accuracy
- Micro-precision, micro-recall, micro-F1
- Macro-precision, macro-recall, macro-F1
- Weighted precision, recall, and F1
- Confusion matrix

The saved split summary is available at `model_1/outputs/pipeline/model_1_splits/split_summary.json`.

## Data Preprocessing

Preprocessing is one of the most important parts of this project because Stack Overflow posts mix natural language, HTML, URLs, inline code, and multi-line code blocks.

The preprocessing pipeline does the following:

- Lowercases and normalizes text
- Decodes HTML entities and percent-encoded fragments
- Parses HTML structure with BeautifulSoup
- Replaces links with explicit markers such as `link` and `url`
- Preserves useful technical tokens such as `c++`, `c#`, `.net`, array types, and method-like tokens
- Cleans inline code and code blocks separately
- Keeps code content instead of deleting it, because code syntax is highly predictive for tags
- Removes English stop words while protecting important technical markers
- Preserves informative punctuation patterns used in programming text

This design matters because many labels are better identified from code syntax than from plain English. For example:

- `std::`, `cout`, `vector` strongly indicate `c++`
- `System.out.println` strongly indicates `java`
- `$_POST` and `echo` strongly indicate `php`
- `$scope` and `ng-model` strongly indicate `angularjs`

Summary statistics from the saved preprocessing run:

- Rows processed: `40,000`
- Unique tags: `20`
- Rows containing code blocks after preprocessing: `27,847`
- Average preprocessed length: `747.76` characters
- Average preprocessed size: `94.07` tokens

See:

- `model_1/preprocessing.py`
- `model_1/outputs/pipeline/preprocessing_summary.json`

## Handcrafted Features

Besides standard text vectorization, the repo also includes a handcrafted feature pipeline in `handcrafted_features/`. The motivation is simple: some tags are associated with highly distinctive lexical or syntactic cues, and these cues can be turned into explicit features.

The handcrafted pipeline builds two kinds of features:

- Text statistics features
- Regex and token-indicator features

Examples of text statistics features:

- Post length
- Number of words
- Number of code blocks
- Number of URLs
- Punctuation count and punctuation ratio

Examples of regex-based manual features:

- `feat_cpp_std` from `std::`
- `feat_java_println` from `System.out.println`
- `feat_php_echo` from `echo`
- `feat_ang_scope` from `$scope`
- `feat_c_printf` from `printf(`
- `feat_cs_writeline` from `Console.WriteLine`

Feature selection logic from the exploration report:

- Coverage threshold: at least `200` matched posts
- Precision threshold: at least `0.50`

Outcome:

- `25` handcrafted patterns were kept
- `79` candidate patterns were discarded
- The final handcrafted feature matrix contains `187` engineered features for `40,000` samples

This is a nice point to emphasize in a presentation: handcrafted features were not used blindly. They were explored, measured, filtered, and documented.

Useful references:

- `handcrafted_features/exploration_report.md`
- `handcrafted_features/features_metadata.json`
- `handcrafted_features/pipeline.py`
- `handcrafted_features/extractors_text.py`
- `handcrafted_features/extractors_code.py`

## Pipeline Overview

The overall workflow can be summarized as:

`raw csv -> preprocessing -> balanced split -> vectorization -> model training/tuning -> ensemble -> evaluation`

In this repo, that flow is implemented as:

1. `model_1/preprocessing.py` or `model_2/preprocessing.py`
2. `model_1/split_data.py` or `model_2/split_data.py`
3. `model_1/vectorization.py` or `model_2/vectorization.py`
4. `model_1/train_models.py` or `model_2/train_models.py`
5. `model_1/params_search.py` or `model_2/params_search.py`
6. `ensemble_models/run_ensemble.py`

There are effectively three experimental branches:

- `model_1`: word-level vectorizations and classifiers
- `model_2`: character-level TF-IDF with separate tuning
- `ensemble_models`: combinations of the best `model_1`, `model_2`, and `model_3` runs

One important presentation note: `Model 3` is not a separate preprocessing branch. It is a specific benchmark configuration inside the `model_1` branch: `one_hot + logistic_regression`.

## Vectorization Methods

The project benchmarks several sparse text representations:

| Method | Branch | Main idea | Notes |
| --- | --- | --- | --- |
| `one_hot` | `model_1` | Binary word/ngram presence | Strong simple baseline |
| `count` | `model_1` | Raw term counts | Keeps frequency information |
| `tfidf` | `model_1` | Word-level TF-IDF | Best word-level representation |
| `char_tfidf` | `model_2` | Character n-gram TF-IDF | Best standalone representation |

Word-level vectorization settings in `model_1`:

- Token pattern preserves technical terms such as `c++`, `c#`, `.net`, `foo.bar`, and `[]`
- `ngram_range=(1, 2)`
- `min_df=2`
- `max_df=0.95`
- `sublinear_tf=True` for TF-IDF

Character-level TF-IDF settings in `model_2`:

- `analyzer="char_wb"`
- `ngram_range=(3, 5)`
- `min_df=2`
- `max_df=0.95`
- `sublinear_tf=True`

Why character-level features help here:

- They capture punctuation-heavy programming patterns
- They remain robust to spelling variants and token boundary issues
- They encode syntax fragments such as `::`, `()`, `[]`, `ng-`, `$_`, `.js`, and HTML-like snippets

## Model 1: TF-IDF Word Level + SGD Classifier

`Model 1` is the main word-level branch.

Chosen configuration:

- Vectorization: `tfidf`
- Classifier: `sgd_classifier`
- Search objective: `f1_macro`
- Cross-validation for the saved tuned run: `5` folds

Why this model is a good candidate:

- Linear models scale well on high-dimensional sparse text
- SGD is efficient for large feature spaces
- The hinge-loss version behaves similarly to a linear SVM while remaining fast to train

Best tuned parameters from the saved run:

- `alpha=1e-06`
- `average=False`
- `class_weight="balanced"`
- `eta0=0.03`
- `learning_rate="adaptive"`
- `loss="hinge"`
- `penalty="l2"`

Saved performance:

- Accuracy: `0.8409`
- Micro-F1: `0.8409`
- Macro-F1: `0.8407`

Artifacts:

- Tuned summary: `model_1/outputs/pipeline/model_1_finetuning/aggregate_macro_f1_summary.csv`
- Chosen run summary: `model_1/outputs/pipeline/model_1_finetuning/tfidf_sgd_classifier/tfidf_sgd_classifier/tfidf/metrics_summary.csv`
- Confusion matrix: `model_1/outputs/pipeline/model_1_finetuning/tfidf_sgd_classifier/tfidf_sgd_classifier/tfidf/sgd_classifier/confusion_matrix.csv`

## Model 2: TF-IDF Character Level + Ridge Classifier

`Model 2` is the strongest standalone model in the repo.

Chosen configuration:

- Vectorization: `char_tfidf`
- Classifier: `ridge_classifier`
- Search objective: `f1_macro`
- Cross-validation for the saved tuned run: `3` folds

Why this model works well:

- Character n-grams capture code fragments, API spellings, and framework naming conventions
- Ridge classification is strong and stable for sparse linear classification
- It complements Model 1 because it focuses less on whole words and more on local patterns

Best tuned parameters from the saved run:

- `alpha=2.0`
- `class_weight=None`

Saved performance:

- Accuracy: `0.8525`
- Micro-F1: `0.8525`
- Macro-F1: `0.8519`

Artifacts:

- Summary: `model_2/outputs/pipeline/model_2_finetuning/char_tfidf_ridge_classifier_fast/char_tfidf/metrics_summary.csv`
- Confusion matrix: `model_2/outputs/pipeline/model_2_finetuning/char_tfidf_ridge_classifier_fast/char_tfidf/ridge_classifier/confusion_matrix.csv`

## Model 3: One-Hot + Logistic Regression

`Model 3` is best presented as a comparison model rather than a final system. It is simpler than Models 1 and 2, but still gives a strong baseline.

Configuration:

- Vectorization: `one_hot`
- Classifier: `logistic_regression`
- Branch location: benchmarked inside `model_1`

Why include it:

- It is easy to explain
- It shows that even a binary bag-of-ngrams representation is competitive
- It helps justify why TF-IDF and character features were worth exploring

Saved performance:

- Accuracy: `0.8191`
- Micro-F1: `0.8191`
- Macro-F1: `0.8195`

Artifacts:

- Summary file: `model_1/outputs/pipeline/model_1_models/metrics_summary_v1.csv`
- Confusion matrix: `model_1/outputs/pipeline/model_1_models/one_hot/logistic_regression/confusion_matrix.csv`

## Ensemble Models

We tested two ensemble settings:

- A 2-model ensemble using `Model 1 + Model 2`
- A 3-model ensemble using `Model 1 + Model 2 + Model 3`

### 2-Model Ensemble

Branches used:

- Model 1: `tfidf + sgd_classifier`
- Model 2: `char_tfidf + ridge_classifier`

Tested variants:

- `hard_tiebreak_model_1`
- `hard_tiebreak_model_2`
- `hard_weighted`
- `soft_score_equal`
- `soft_score_weighted`
- `soft_calibrated_equal`
- `soft_calibrated_weighted`

Best 2-model strategy:

- Variant: `soft_calibrated_weighted`
- Accuracy: `0.8553`
- Micro-F1: `0.8553`
- Macro-F1: `0.8552`

Why it works well:

- It combines complementary word-level and character-level evidence
- Both classifiers are calibrated with `CalibratedClassifierCV(method="sigmoid", cv=3)`
- Voting weights are proportional to each branch's macro F1

Artifacts:

- Run configuration: `ensemble_models/outputs/runs/model_1_model_2_ensemble/run_config.json`
- Summary: `ensemble_models/outputs/runs/model_1_model_2_ensemble/metrics_summary.csv`
- Best confusion matrix: `ensemble_models/outputs/runs/model_1_model_2_ensemble/soft_calibrated_weighted/confusion_matrix.csv`

### 3-Model Ensemble

Branches used:

- Model 1: `tfidf + sgd_classifier`
- Model 2: `char_tfidf + ridge_classifier`
- Model 3: `one_hot + logistic_regression`

Tested variants:

- `hard_majority`
- `hard_weighted`
- `soft_score_equal`
- `soft_score_weighted`
- `soft_calibrated_equal`
- `soft_calibrated_weighted`

Best 3-model strategy:

- Variant: `soft_calibrated_equal`
- Accuracy: `0.8523`
- Micro-F1: `0.8523`
- Macro-F1: `0.8522`

Important observation:

- Adding `Model 3` did not improve the best 2-model ensemble
- The 3-model ensemble is slightly below the 2-model best result
- This suggests `Model 3` adds diversity, but not enough new signal to outperform the stronger 2-model calibrated setup

Artifacts:

- Run configuration: `ensemble_models/outputs/runs/model_1_model_2_model_3_ensemble/run_config.json`
- Summary: `ensemble_models/outputs/runs/model_1_model_2_model_3_ensemble/metrics_summary.csv`
- Best confusion matrix: `ensemble_models/outputs/runs/model_1_model_2_model_3_ensemble/soft_calibrated_equal/confusion_matrix.csv`

### Final Selected Ensemble

After comparing all ensemble runs, the final selected system remains:

- `Model 1 + Model 2`
- Variant: `soft_calibrated_weighted`
- Accuracy: `0.8553`
- Macro-F1: `0.8552`

Compared with the best standalone model:

- Accuracy improves from `0.8525` to `0.8553`
- Macro-F1 improves from `0.8519` to `0.8552`

## Evaluation

The most useful summary table for a report or presentation is:

| System | Vectorization | Classifier / Strategy | Accuracy | Micro-F1 | Macro-F1 |
| --- | --- | --- | --- | --- | --- |
| Model 1 | `tfidf` | `sgd_classifier` | `0.8409` | `0.8409` | `0.8407` |
| Model 2 | `char_tfidf` | `ridge_classifier` | `0.8525` | `0.8525` | `0.8519` |
| Model 3 | `one_hot` | `logistic_regression` | `0.8191` | `0.8191` | `0.8195` |
| Best 3-model ensemble | `tfidf + char_tfidf + one_hot` | `soft_calibrated_equal` | `0.8523` | `0.8523` | `0.8522` |
| Final selected ensemble | `tfidf + char_tfidf` | `soft_calibrated_weighted` | `0.8553` | `0.8553` | `0.8552` |

Main takeaway:

- Character-level TF-IDF is the strongest standalone representation
- Word-level TF-IDF with a tuned linear classifier is also strong
- One-hot + logistic regression is a solid baseline, but weaker than Models 1 and 2
- The 3-model ensemble is competitive, but it does not beat the best 2-model ensemble
- The best overall performance comes from combining the tuned word-level and character-level models

For the assignment requirements:

- Accuracy is reported
- Micro average is reported
- Confusion matrices are saved for all key runs
- The experimental protocol is reproducible from saved artifacts

## How To Reproduce

Minimal end-to-end commands:

```bash
python -m handcrafted_features.pipeline

python -m model_1.preprocessing
python -m model_1.split_data
python -m model_1.vectorization --vectorizations one_hot count tfidf
python -m model_1.train_models --vectorizations one_hot count tfidf --no-tuning
python -m model_1.params_search --vectorization tfidf --models sgd_classifier --run-name tfidf_sgd_classifier

python -m model_2.preprocessing
python -m model_2.split_data
python -m model_2.vectorization --vectorizations char_tfidf
python -m model_2.params_search --vectorization char_tfidf --models ridge_classifier --run-name char_tfidf_ridge_classifier_fast

python -m ensemble_models.run_ensemble --run-name model_1_model_2_ensemble --variants soft_calibrated_weighted
python -m ensemble_models.run_ensemble --include-model-3 --run-name model_1_model_2_model_3_ensemble --variants soft_calibrated_equal soft_calibrated_weighted
```

## Suggested Report / Slide Conclusion

A concise conclusion for the final presentation could be:

> We compared word-level, character-level, handcrafted, and ensemble approaches for Stack Overflow tag prediction on a balanced 20-class dataset. Character-level TF-IDF with RidgeClassifier was the best standalone model, while the best overall system was a calibrated weighted ensemble of the tuned word-level SGD model and the tuned character-level Ridge model. Adding the one-hot logistic regression branch created a competitive 3-model ensemble, but it did not outperform the best 2-model setup.
