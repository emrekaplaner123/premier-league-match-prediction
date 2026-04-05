
import pytest
import os
import pandas as pd
from unittest.mock import patch, MagicMock
import joblib

from src.train import train_model

@pytest.fixture
def mock_featured_csv(tmp_path):

    df = pd.DataFrame({
        "home_avg_goals_scored_5": [1.2, 2.0],
        "home_avg_goals_conceded_5": [1.0, 1.5],
        "away_avg_goals_scored_5": [1.1, 0.8],
        "away_avg_goals_conceded_5": [1.2, 1.0],
        "home_win_rate_5": [0.5, 0.6],
        "away_win_rate_5": [0.4, 0.3],
        "home_avg_shots_on_target_5": [4, 5],
        "home_avg_corners_5": [5, 6],
        "away_avg_shots_on_target_5": [3, 4],
        "away_avg_corners_5": [4, 3],
        "day_of_week": [2, 6],
        "is_weekend": [0, 1],
        "home_team_elo": [1500, 1510],
        "away_team_elo": [1500, 1480],
        "elo_diff": [0, 30],
        "FTR": ["H", "A"]
    })
    path = tmp_path / "featured.csv"
    df.to_csv(path, index=False)
    return path

def test_train_model(mock_featured_csv, tmp_path):
    model_path = tmp_path / "random_forest_model.pkl"
    train_model(
        input_path=str(mock_featured_csv),
        model_path=str(model_path),
        random_state=42
    )
    # Check that model file is created
    assert model_path.exists(), "Model pickle file not created."

    # Load it and check contents
    bundle = joblib.load(model_path)
    assert "model" in bundle, "No model found in saved file."
    assert "label_encoder" in bundle, "No label encoder found in saved file."
    assert "feature_cols" in bundle, "No feature columns list found in saved file."
