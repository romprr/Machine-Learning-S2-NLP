# How to Run Model 3 Pipeline

This guide explains how to run the standalone `model3` pipelines from raw data to trained models. It covers the primary hybrid model (One-Hot + Handcrafted) as well as the TF-IDF and Handcrafted-Only variants.

## 1. Prerequisites
Ensure you are in the project root directory and your virtual environment is activated. The data splits and preprocessing steps from the main project scripts (`scripts/`) must be available, as `model3` uses the central dataset output (`scripts/outputs/pipeline/splits/`).

## 2. Feature Discovery (Handcrafted Rules)

Before running the pipelines, you must extract the rule-based features. 
We have configured the discovery script to find features with **at least 200 coverage** and **60% precision (accuracy)**. 

To run the discovery and update the underlying rules:
```bash
python3 -m handcrafted_features.discover_features
```
*This generates `handcrafted_features/discovered_candidates.json` used by the pipelines below.*

## 3. Consolidated Execution (Pipelines)

You can run the entire pipeline (from loading splits to vectorization, training, and metrics output) using the consolidated scripts below. 

These scripts use the `FeaturePipeline` class to handle vectorization and feature extraction on the fly, saving all artifacts (models, scalers, vectorizers, `metrics.json`, and `confusion_matrix.csv`) to their respective output directories.

**To run the default Hybrid pipeline (One-Hot + Handcrafted):**
```bash
python3 -m model3.run_pipeline
```
*Outputs to: `model3/outputs/`*

**To run the TF-IDF Hybrid pipeline (TF-IDF + Rule-based Handcrafted):**
```bash
python3 -m model3.run_tfidf_pipeline
```
*Outputs to: `model3/outputs/tfidf/`*

**To run the Handcrafted-Only pipeline:**
```bash
python3 -m model3.run_handcrafted_pipeline
```
*Outputs to: `model3/outputs/handcrafted_only/`*

### Hyperparameter Tuning
You can append the `--tune` flag to any of the consolidated pipeline scripts above to run hyperparameter tuning before training:
```bash
python3 -m model3.run_pipeline --tune
```