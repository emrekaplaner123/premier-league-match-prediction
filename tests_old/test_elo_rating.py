
import pytest
import pandas as pd
from src.elo_rating import generate_elo_ratings
from src.config import ELO_BASE, K_FACTOR, HOME_ADVANTAGE

@pytest.fixture
def mock_clean_data():

    data = {
        "Date": ["2022-01-01", "2022-01-05"],
        "HomeTeam": ["TeamA", "TeamA"],
        "AwayTeam": ["TeamB", "TeamB"],
        "FTHG": [2, 3],
        "FTAG": [1, 2],
        "FTR": ["H", "H"],
        "official_result": [True, True]
    }
    return pd.DataFrame(data)

def test_generate_elo_ratings(mock_clean_data):
    df_elo = generate_elo_ratings(mock_clean_data)

    assert "HomeEloPre" in df_elo.columns, "Elo column missing."
    assert "AwayEloPre" in df_elo.columns, "Elo column missing."

    first_row = df_elo.iloc[0]
    home_elo_pre = first_row["HomeEloPre"]
    away_elo_pre = first_row["AwayEloPre"]

    assert home_elo_pre == ELO_BASE, "Expected home pre-match Elo to be base."
    assert away_elo_pre == ELO_BASE, "Expected away pre-match Elo to be base."

    second_row = df_elo.iloc[1]
    second_home_elo_pre = second_row["HomeEloPre"]
    assert second_home_elo_pre > 1500, "Home Elo didn't increase after a home win."

