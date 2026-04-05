
import pytest
import os
import pandas as pd
import joblib

from src.predict import predict_upcoming_fixtures
from src.config import PREDICTIONS_OUTPUT_PATH

@pytest.fixture
def mock_upcoming_fixtures_csv(tmp_path):
    df = pd.DataFrame({
        "Date": ["2025-08-15", "2025-08-16"],
        "HomeTeam": ["TeamA", "TeamB"],
        "AwayTeam": ["TeamC", "TeamD"]
    })
    csv_path = tmp_path / "upcoming_fixtures.csv"
    df.to_csv(csv_path, index=False)
    return csv_path

@pytest.fixture
def mock_model_pkl(tmp_path):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    model = RandomForestClassifier()
    label_encoder = LabelEncoder()
    label_encoder.fit(["H", "D", "A"])
    bundle = {
        "model": model,
        "label_encoder": label_encoder,
        "feature_cols": [
            "day_of_week", "is_weekend", "home_team_elo",
            "away_team_elo", "elo_diff", "home_avg_goals_scored_5",
            "away_avg_goals_scored_5"  # etc.
        ]
    }
    model_path = tmp_path / "random_forest_model.pkl"
    joblib.dump(bundle, model_path)
    return model_path

def test_predict_upcoming_fixtures(mock_upcoming_fixtures_csv, mock_model_pkl, tmp_path):
    output_path = tmp_path / "upcoming_predictions.csv"

    predict_upcoming_fixtures(
        fixtures_path=str(mock_upcoming_fixtures_csv),
        model_path=str(mock_model_pkl),
        output_path=str(output_path)
    )

    # Check that predictions file is created
    assert output_path.exists(), "Predictions output file not created."

    df_pred = pd.read_csv(output_path)
    assert "predicted_label" in df_pred.columns, "Predictions file missing predicted_label column."
