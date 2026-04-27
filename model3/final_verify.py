import pandas as pd
from scipy.sparse import load_npz
from model3.train_models import build_model, evaluate_and_save_estimator
from model3.pipeline_paths import VECTORIZATION_OUTPUT_DIR, MODELS_OUTPUT_DIR

def verify():
    vec_dir = VECTORIZATION_OUTPUT_DIR / "hybrid"
    X_train = load_npz(vec_dir / "X_train.npz")
    X_test = load_npz(vec_dir / "X_test.npz")
    y_train = pd.read_csv(vec_dir / "y_train.csv")["tags"].astype(str)
    y_test = pd.read_csv(vec_dir / "y_test.csv")["tags"].astype(str)
    labels = sorted(y_train.unique())

    model = build_model("logistic_regression")
    print("Training model...")
    model.fit(X_train, y_train)
    
    output_dir = MODELS_OUTPUT_DIR / "hybrid" / "logistic_regression_verify"
    summary = evaluate_and_save_estimator(
        model_name="logistic_regression",
        estimator=model,
        model_output_dir=output_dir,
        x_train=X_train,
        y_train=y_train,
        x_test=X_test,
        y_test=y_test,
        labels=labels,
        vectorization_name="hybrid",
        tune_hyperparameters=False,
        cv_folds=0,
        scoring="f1_macro",
        best_cv_score=None,
        best_params=None,
        cv_results_df=None
    )
    print(f"Verification Success! Macro F1: {summary['macro_f1']:.4f}")

if __name__ == "__main__":
    verify()
