
import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from src.data_collection import collect_data

@pytest.fixture
def mock_excel_files(tmp_path):

    d = tmp_path / "raw_data"
    d.mkdir()
    df_dummy = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    excel_path_1 = d / "PL-15-16.xlsx"
    excel_path_2 = d / "PL-16-17.xlsx"
    df_dummy.to_excel(excel_path_1, index=False)
    df_dummy.to_excel(excel_path_2, index=False)
    return d

def test_collect_data(mock_excel_files, tmp_path):

    output_csv = tmp_path / "processed" / "combined.csv"
    collect_data(input_dir=str(mock_excel_files), output_path=str(output_csv))

    assert output_csv.exists(), "Output CSV not created."

    df_out = pd.read_csv(output_csv)
    assert len(df_out) == 4, "Expected 4 rows of data, got something else."

