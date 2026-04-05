

import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from src.config import (
    FEATURED_DATA_PATH,  # e.g., "data/processed/featured.csv"
    MODEL_PATH           # e.g., "models/random_forest_model.pkl"
)

def train_model(
    input_path: str = None,
    model_path: str = None,
    random_state: int = 42
):



    if input_path is None:
        input_path = FEATURED_DATA_PATH
    if model_path is None:
        model_path = MODEL_PATH

    print(f"Loading featured data from: {input_path}")
    df = pd.read_csv(input_path)


    # Target is the final result: "H", "D", or "A"
    target_col = "FTR"

    feature_cols = [
        "home_avg_goals_scored_5",
        "home_avg_goals_conceded_5",
        "away_avg_goals_scored_5",
        "away_avg_goals_conceded_5",
        "home_win_rate_5",
        "away_win_rate_5",
        "home_avg_shots_on_target_5",
        "home_avg_corners_5",
        "away_avg_shots_on_target_5",
        "away_avg_corners_5",
        "day_of_week",
        "is_weekend",
        "home_team_elo",
        "away_team_elo",
        "elo_diff"
    ]

    # Drop rows with missing features or target
    df.dropna(subset=feature_cols + [target_col], inplace=True)

    X = df[feature_cols]
    y = df[target_col]

    # Encode the target "H", "D", "A" into numeric labels (0, 1, 2 for example)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)


    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.1,
        random_state=random_state,
        stratify=y_encoded  # keep class distribution
    )

    print(f"Train set: {X_train.shape[0]} rows, Test set: {X_test.shape[0]} rows.")


    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=random_state
    )
    from src.config import RF_PARAMS
    model = RandomForestClassifier(**RF_PARAMS)

    print("Training Random Forest...")
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.3f}")

    # Detailed classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))


    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    joblib.dump({
        "model": model,
        "label_encoder": label_encoder,
        "feature_cols": feature_cols
    }, model_path)

    print(f"Model saved to: {model_path}")
    return model

def main():
    train_model()

if __name__ == "__main__":
    main()
