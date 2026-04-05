
import os
import pandas as pd
from src.config import (
    CLEANED_DATA_PATH,
    WITH_ELO_DATA_PATH,
    FEATURED_DATA_PATH
)
from src.elo_rating import generate_elo_ratings
from src.feature_engineering import generate_features

def update_after_match(new_match: dict, is_official=True):

    if not is_official:
        print("Match marked as test/hypothetical. Not updating.")
        return

    # 1. Load cleaned dataset
    df_cleaned = pd.read_csv(CLEANED_DATA_PATH, parse_dates=["Date"], dayfirst=True)

    # 2. Append the new match
    df_cleaned = df_cleaned.append(new_match, ignore_index=True)
    df_cleaned.sort_values("Date", inplace=True)

    # 3. Save updated cleaned dataset
    df_cleaned.to_csv(CLEANED_DATA_PATH, index=False)
    print("New match appended to cleaned data.")

    # 4. Re-run Elo
    df_with_elo = generate_elo_ratings(df_cleaned)
    df_with_elo.to_csv(WITH_ELO_DATA_PATH, index=False)
    print("Elo updated and saved.")

    # 5. Re-run feature engineering
    df_features = generate_features(
        input_path=WITH_ELO_DATA_PATH,
        output_path=FEATURED_DATA_PATH
    )
    print("Feature engineering complete. Updated data is ready.")

def main():

    is_official_input = input("Is this an official result? (y/n): ").lower().strip()
    is_official = True if is_official_input == "y" else False

    if not is_official:
        print("Skipping official update.")
        return

    date_str = input("Match Date (YYYY-MM-DD): ")
    home_team = input("Home Team: ")
    away_team = input("Away Team: ")
    fthg = int(input("Home goals (FTHG): "))
    ftag = int(input("Away goals (FTAG): "))

    if fthg > ftag:
        ftr = "H"
    elif fthg < ftag:
        ftr = "A"
    else:
        ftr = "D"

    new_match_row = {
        "Date": date_str,
        "HomeTeam": home_team,
        "AwayTeam": away_team,
        "FTHG": fthg,
        "FTAG": ftag,
        "FTR": ftr,
        "official_result": True
    }

    update_after_match(new_match_row, is_official=True)

if __name__ == "__main__":
    main()
