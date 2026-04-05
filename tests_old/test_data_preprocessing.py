
import os
import pytest
import pandas as pd

from src.data_preprocessing import preprocess_data

@pytest.fixture
def mock_combined_csv(tmp_path):

    df = pd.DataFrame({
        "Date": ["08/08/2015", "09/08/2015"],
        "HomeTeam": ["TeamA", "TeamB"],
        "AwayTeam": ["TeamC", "TeamD"],
        "FTHG": [1, None],    # one missing value
        "FTAG": [1, 2],
        "FTR": ["H", None],   # missing FTR in second row
    })
    csv_path = tmp_path / "combined.csv"
    df.to_csv(csv_path, index=False)
    return csv_path

def test_preprocess_data(mock_combined_csv, tmp_path):
    cleaned_path = tmp_path / "cleaned.csv"
    df_clean = preprocess_data(
        input_path=str(mock_combined_csv),
        output_path=str(cleaned_path)
    )
    assert cleaned_path.exists(), "Cleaned CSV not created."

    assert len(df_clean) == 1, "Expected 1 row after dropping missing FTR."

    assert pd.api.types.is_datetime64_any_dtype(df_clean["Date"]), "Date column not converted to datetime."
