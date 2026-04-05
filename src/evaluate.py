

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

from src.config import (
    FEATURED_DATA_PATH,
    MODEL_PATH
)

def evaluate_model(
    data_path: str = None,
    model_path: str = None,
    test_size: float = 0.2,
    random_state: int = 42
):
    if data_path is None:
        data_path = FEATURED_DATA_PATH
    if model_path is None:
        model_path = MODEL_PATH

    df = pd.read_csv(data_path)

    target_col = "FTR"

    # Ensure we have a model
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    bundle = joblib.load(model_path)
    model = bundle["model"]
    label_encoder = bundle["label_encoder"]
    feature_cols = bundle["feature_cols"]

    # Filter rows that have valid target
    df.dropna(subset=[target_col], inplace=True)
    df = df.dropna(subset=feature_cols)

    X = df[feature_cols]
    y = df[target_col]
    y_enc = label_encoder.transform(y)  # encode "H", "D", "A" to numeric

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc
    )

    # Predict on test set
    y_pred_raw = model.predict(X_test)
    # Convert any string labels back into numeric codes
    try:
        y_pred = label_encoder.transform(y_pred_raw)
    except Exception:
        y_pred = y_pred_raw

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.3f}")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix (rows=actual, cols=pred):")
    print(cm)

def main():
    evaluate_model()

if __name__ == "__main__":
    main()
