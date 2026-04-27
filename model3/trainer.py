import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from .config import MODEL_PATH

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

    def train(self, X_train, y_train):
        print(f"Training Logistic Regression with params: {self.params}")
        self.model.fit(X_train, y_train)
        return self.model

    def evaluate(self, X_test, y_test, label_encoder=None):
        y_pred = self.model.predict(X_test)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        print(f"Test Macro F1: {f1_macro:.4f}")
        
        if label_encoder:
            target_names = [str(c) for c in label_encoder.classes_]
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=target_names))
            
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
