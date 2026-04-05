

import pandas as pd
from src.config import ELO_BASE, K_FACTOR, HOME_ADVANTAGE

def generate_elo_ratings(df: pd.DataFrame) -> pd.DataFrame:


    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()

    # Sort by date so Elo updates happen in chronological order
    df.sort_values(by="Date", inplace=True)

    if "official_result" in df.columns:
        df = df[df["official_result"] == True]
    # ------------------------------------------------

    # Dictionary to store current Elo rating for each team
    current_elo = {}

    # Lists to hold pre-match Elo for home/away teams
    home_elo_pre_list = []
    away_elo_pre_list = []

    for _, row in df.iterrows():
        home_team = row["HomeTeam"]
        away_team = row["AwayTeam"]
        match_result = row["FTR"]  # 'H', 'D', or 'A'

        # If teams haven't been assigned an Elo yet, initialize them
        if home_team not in current_elo:
            current_elo[home_team] = ELO_BASE
        if away_team not in current_elo:
            current_elo[away_team] = ELO_BASE

        home_elo = current_elo[home_team]
        away_elo = current_elo[away_team]

        # Record the pre-match Elo
        home_elo_pre_list.append(home_elo)
        away_elo_pre_list.append(away_elo)

        # Compute expected score for the home team
        rating_diff = (away_elo - home_elo) + HOME_ADVANTAGE
        expected_home = 1.0 / (1.0 + 10 ** (rating_diff / 400))
        expected_away = 1.0 - expected_home

        # Determine actual score
        if match_result == "H":
            score_home = 1.0
            score_away = 0.0
        elif match_result == "D":
            score_home = 0.5
            score_away = 0.5
        else:  # 'A'
            score_home = 0.0
            score_away = 1.0

        # Update Elo
        new_home_elo = home_elo + K_FACTOR * (score_home - expected_home)
        new_away_elo = away_elo + K_FACTOR * (score_away - expected_away)

        current_elo[home_team] = new_home_elo
        current_elo[away_team] = new_away_elo

    official_df = df.copy()
    official_df["HomeEloPre"] = home_elo_pre_list
    official_df["AwayEloPre"] = away_elo_pre_list


    merged_df = df.copy().join(
        official_df[["HomeEloPre", "AwayEloPre"]],
        how="left"
    )
    return official_df

def main():
    import os
    input_path = "data/processed/cleaned.csv"
    output_path = "data/processed/with_elo.csv"

    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        return

    df_input = pd.read_csv(input_path)
    df_with_elo = generate_elo_ratings(df_input)
    df_with_elo.to_csv(output_path, index=False)
    print(f"Elo ratings saved to: {output_path}")

if __name__ == "__main__":
    main()
