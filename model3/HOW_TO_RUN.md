# How to Run Model 3 Pipeline

This guide explains how to run the standalone `model3` pipeline from raw data to a trained model.

## 1. Prerequisites
Ensure you are in the project root directory and your virtual environment is activated.

## 2. Step-by-Step Execution

Run the following commands in order:

### A. Preprocessing
Cleans the raw Stack Overflow posts.
```bash
python3 -m model3.preprocessing
```

### B. Data Splitting
Splits the preprocessed data into train and test sets.
```bash
python3 -m model3.split_data
```

### C. Hybrid Vectorization
Generates the combined feature set (Handcrafted Features + Text One-Hot encoding).
```bash
python3 -m model3.vectorization
```

### D. Hyperparameter Tuning (Optional)
Runs a `HalvingRandomSearchCV` to find the best parameters.
```bash
python3 -m model3.params_search
```

### E. Final Training & Verification
Trains the Logistic Regression model using optimized hyperparameters and prints final metrics.
```bash
python3 -m model3.final_verify
```

## 3. Outputs
All generated artifacts (CSVs, `.npz` matrices, models, and metrics) are saved in the `model3/outputs/` directory.
