"""
tune_model.py

Automates hyperparameter tuning for a RandomForestClassifier
using GridSearchCV. Saves the best model to disk.

Usage:
    python tune_model.py

References:
    - Scikit-learn docs: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
"""

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def main():
    # 1. Load featured data
    featured_path = "data/processed/featured.csv"  # Adjust path if needed
    df = pd.read_csv(featured_path)
    print(f"Loaded data from {featured_path}. Shape: {df.shape}")

    # Ensure necessary columns
    if "FTR" not in df.columns:
        print("Error: 'FTR' column not found in featured data.")
        return

    # Example feature columns (adjust to match your pipeline)
    # If you have them stored in a config or model .pkl, you can load from there.
    feature_cols = [
        "home_avg_goals_scored_5", "away_avg_goals_scored_5",
        "home_avg_goals_conceded_5", "away_avg_goals_conceded_5",
        "home_win_rate_5", "away_win_rate_5",
        "home_team_elo", "away_team_elo", "elo_diff","h2h_avg_goals_for_5",
        "h2h_avg_goals_against_5",
        "h2h_win_rate_5"
        # ... add any others (day_of_week, is_weekend, etc.)
    ]

    # Drop rows if they are missing crucial features or target
    df.dropna(subset=feature_cols + ["FTR"], inplace=True)

    X = df[feature_cols]
    y = df["FTR"]

    # Encode target (H, D, A) into numeric labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # 2. Train/test split
    # (shuffle=False if your data is time-series and you want a time-based split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    # 3. Define the model and parameter grid
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "max_features": ["sqrt", "log2"]
    }

    # 4. Grid Search
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring="accuracy",
        cv=5,       # 5-fold cross-validation
        n_jobs=-1,  # Use all available CPU cores
        verbose=1
    )

    print("Starting Grid Search...")
    grid_search.fit(X_train, y_train)
    print("Grid Search complete.")

    # 5. Best model & evaluation
    best_model = grid_search.best_estimator_
    print(f"Best Params: {grid_search.best_params_}")
    print(f"Best CV Score: {grid_search.best_score_:.4f}")

    # Evaluate on test set
    y_pred_test = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    print(f"Test Accuracy: {test_acc:.2%}")

    print("Classification Report (Test Set):")
    print(classification_report(y_test, y_pred_test, target_names=label_encoder.classes_))

    # 6. Save best model
    bundle = {
        "model": best_model,
        "label_encoder": label_encoder,
        "feature_cols": feature_cols
    }
    joblib.dump(bundle, "models/best_random_forest.pkl")
    print("Best model saved to models/best_random_forest.pkl")

if __name__ == "__main__":
    main()
