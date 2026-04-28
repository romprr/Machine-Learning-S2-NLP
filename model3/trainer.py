import joblib
import json
import numpy as np
from pathlib import Path
from time import perf_counter
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from .config import MODEL_PATH

def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series, labels: list[str]) -> dict[str, float]:
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "label_count": len(labels),
    }

class Model3Trainer:
    def __init__(self, params=None):
        self.params = params or {
            "solver": "saga",
            "max_iter": 200,
            "tol": 1e-2,
            "random_state": 42,
            "C": 1.0,
            "penalty": "l2"
        }
        self.model = LogisticRegression(**self.params)
        self.fit_time_seconds = 0

    def train(self, X_train, y_train):
        print(f"Training Logistic Regression with params: {self.params}")
        start_time = perf_counter()
        self.model.fit(X_train, y_train)
        self.fit_time_seconds = perf_counter() - start_time
        return self.model

    def evaluate(self, X_test, y_test, label_encoder=None, vectorization_name="hybrid", output_dir=None):
        start_time = perf_counter()
        y_pred = self.model.predict(X_test)
        predict_time_seconds = perf_counter() - start_time
        
        f1_macro = f1_score(y_test, y_pred, average="macro")
        print(f"Test Macro F1: {f1_macro:.4f}")
        
        labels = None
        if label_encoder:
            labels = [str(c) for c in label_encoder.classes_]
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=labels))
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if labels is None:
                labels = sorted(list(set(y_test)))

            metrics = evaluate_predictions(pd.Series(y_test), pd.Series(y_pred), labels)
            
            # Handle case where y_test/y_pred are integers and labels are strings
            conf_labels = labels
            if label_encoder and np.issubdtype(np.array(y_test).dtype, np.integer):
                conf_labels = range(len(labels))
            
            confusion = confusion_matrix(y_test, y_pred, labels=conf_labels)
            confusion_df = pd.DataFrame(confusion, index=labels, columns=labels)
            confusion_df.to_csv(output_dir / "confusion_matrix.csv")
            
            metrics_payload = {
                "vectorization": vectorization_name,
                "model": "logistic_regression",
                "component_models": None,
                "tune_hyperparameters": False,
                "cv_folds": None,
                "scoring": None,
                "best_cv_score": None,
                "best_params": self.params,
                "fit_time_seconds": round(self.fit_time_seconds, 4),
                "predict_time_seconds": round(predict_time_seconds, 4),
                **metrics,
            }
            (output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
            
        return f1_macro

    def save(self, path=MODEL_PATH):
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path=MODEL_PATH):
        instance = cls()
        instance.model = joblib.load(path)
        instance.params = instance.model.get_params()
        return instance
