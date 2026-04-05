

import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def main():
    # 1. Load best model
    model_path = "models/best_random_forest.pkl"
    bundle = joblib.load(model_path)
    model = bundle["model"]
    label_encoder = bundle["label_encoder"]
    feature_cols = bundle["feature_cols"]

    print(f"Loaded model from {model_path}: {model}")

    # 2. Load featured data
    featured_path = "data/processed/featured.csv"
    df = pd.read_csv(featured_path)
    print(f"Loaded data from {featured_path}. Shape: {df.shape}")

    df.dropna(subset=feature_cols + ["FTR"], inplace=True)

    X = df[feature_cols]
    y = df["FTR"]

    y_encoded = label_encoder.transform(y)

    # 3. Compute overall accuracy (just as a reference)
    y_pred = model.predict(X)
    acc = accuracy_score(y_encoded, y_pred)
    print(f"Overall Accuracy (All Features): {acc:.2%}")

    # 4. Feature Importances
    importances = model.feature_importances_
    feat_imp = sorted(
        zip(feature_cols, importances),
        key=lambda x: x[1],
        reverse=True
    )

    print("\nFeature Importances:")
    for f, imp in feat_imp:
        print(f"{f}: {imp:.4f}")

    # 5. Experiment: Drop the least important feature
    if len(feature_cols) > 1:
        least_important = feat_imp[-1][0]  # name of the least important feature
        print(f"\nDropping least important feature: {least_important}")

        reduced_features = [f for f in feature_cols if f != least_important]
        X_reduced = df[reduced_features]


        from sklearn.ensemble import RandomForestClassifier

        best_params = model.get_params()
        temp_rf = RandomForestClassifier(**best_params)
        temp_rf.fit(X_reduced, y_encoded)

        # Evaluate
        y_reduced_pred = temp_rf.predict(X_reduced)
        acc_reduced = accuracy_score(y_encoded, y_reduced_pred)
        print(f"Accuracy after dropping '{least_important}': {acc_reduced:.2%}")

    print("\nFeature importance analysis complete.")

if __name__ == "__main__":
    main()
