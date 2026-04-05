import os
import pandas as pd
from src.config import (
    COMBINED_DATA_PATH,  # e.g., "data/processed/combined.csv"
    CLEANED_DATA_PATH    # e.g., "data/processed/cleaned.csv"
)

def preprocess_data(
    input_path: str = None,
    output_path: str = None
) -> pd.DataFrame:

    if input_path is None:
        input_path = COMBINED_DATA_PATH
    if output_path is None:
        output_path = CLEANED_DATA_PATH

    # Make sure directories exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)

    if "Date" in df.columns:
        try:
            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        except Exception as e:
            print("Warning: could not parse dates:", e)

    critical_cols = ["HomeTeam", "AwayTeam", "FTR"]
    df.dropna(subset=critical_cols, inplace=True)

    possible_goal_cols = ["FTHG", "FTAG", "HTHG", "HTAG"]
    for col in possible_goal_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["FTHG", "FTAG"], how="any", inplace=True)

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df.sort_values(by="Date", inplace=True)

    print(f"Data shape after cleaning: {df.shape}")
    print(f"Saving cleaned data to: {output_path}")
    df.to_csv(output_path, index=False)

    return df

def main():
    preprocess_data()


if __name__ == "__main__":
    main()
