
import pytest
import pandas as pd
from src.feature_engineering import generate_features

@pytest.fixture
def mock_with_elo_csv(tmp_path):

    data = {
        "Date": ["2022-01-01", "2022-01-10", "2022-01-20", "2022-02-01"],
        "HomeTeam": ["TeamA", "TeamA", "TeamB", "TeamA"],
        "AwayTeam": ["TeamB", "TeamC", "TeamA", "TeamC"],
        "FTHG": [2, 3, 1, 0],
        "FTAG": [1, 0, 2, 2],
        "FTR": ["H", "H", "A", "A"],  # 'H' home win, 'A' away win
        "HomeEloPre": [1500, 1510, 1490, 1520],
        "AwayEloPre": [1500, 1480, 1510, 1485],
        "official_result": [True, True, True, True]  # all official for simplicity
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "with_elo.csv"
    df.to_csv(csv_path, index=False)
    return csv_path

def test_generate_features(mock_with_elo_csv, tmp_path):
    featured_path = tmp_path / "featured.csv"
    df_features = generate_features(
        input_path=str(mock_with_elo_csv),
        output_path=str(featured_path),
        rolling_window=2
    )

    assert featured_path.exists(), "Featured CSV not created."
    assert "home_avg_goals_scored_5" in df_features.columns, "Expected rolling columns not found."
    assert "home_team_elo" in df_features.columns, "Elo columns not renamed properly."


    second_match = df_features[(df_features["Date"] == "2022-01-10") & (df_features["HomeTeam"] == "TeamA")]
    assert not second_match.empty, "Test row not found."

    val = second_match.iloc[0]["home_avg_goals_scored_5"]

    assert pd.notnull(val), "Expected a numeric rolling average, got null."
