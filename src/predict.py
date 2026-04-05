
import os
import joblib
import pandas as pd

from src.config import (
    MODEL_PATH,
    UPCOMING_FIXTURES_PATH,
    PREDICTIONS_OUTPUT_PATH
)

def predict_upcoming_fixtures(
    fixtures_path: str = None,
    model_path: str = None,
    output_path: str = None
):
    if fixtures_path is None:
        fixtures_path = UPCOMING_FIXTURES_PATH
    if model_path is None:
        model_path = MODEL_PATH
    if output_path is None:
        output_path = PREDICTIONS_OUTPUT_PATH

    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model_bundle = joblib.load(model_path)
    model = model_bundle["model"]
    label_encoder = model_bundle["label_encoder"]
    feature_cols = model_bundle["feature_cols"]

    # Load upcoming fixtures
    if not os.path.exists(fixtures_path):
        raise FileNotFoundError(f"Fixtures file not found: {fixtures_path}")
    df_fixtures = pd.read_csv(fixtures_path)


    # Example placeholders:
    df_fixtures["day_of_week"] = pd.to_datetime(df_fixtures["Date"], dayfirst=True, errors="coerce").dt.weekday
    df_fixtures["is_weekend"] = df_fixtures["day_of_week"].isin([5, 6]).astype(int)

    # Elo, rolling stats, etc. would need to be fetched from a pipeline or logic
    df_fixtures["home_team_elo"] = 1500
    df_fixtures["away_team_elo"] = 1500
    df_fixtures["elo_diff"] = df_fixtures["home_team_elo"] - df_fixtures["away_team_elo"]

    df_fixtures["home_avg_goals_scored_5"] = 1.5
    df_fixtures["away_avg_goals_scored_5"] = 1.2

    for col in feature_cols:
        if col not in df_fixtures.columns:
            df_fixtures[col] = 0

    X_upcoming = df_fixtures[feature_cols]

    # Predict
    y_pred_enc = model.predict(X_upcoming)
    y_pred = label_encoder.inverse_transform(y_pred_enc)
    y_probs = model.predict_proba(X_upcoming)

    df_fixtures["predicted_label"] = y_pred

    classes_ = label_encoder.classes_
    for i, c in enumerate(classes_):
        df_fixtures[f"pred_prob_{c}"] = y_probs[:, i]

    df_fixtures.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

def main():
    predict_upcoming_fixtures()

if __name__ == "__main__":
    main()
