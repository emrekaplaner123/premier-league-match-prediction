
import os
import pandas as pd

WITH_ELO_DATA_PATH = "data/processed/with_elo.csv"
FEATURED_DATA_PATH = "data/processed/featured.csv"

def generate_features(
    input_path: str = None,
    output_path: str = None,
    rolling_window: int = 5
) -> pd.DataFrame:

    if input_path is None:
        input_path = WITH_ELO_DATA_PATH
    if output_path is None:
        output_path = FEATURED_DATA_PATH

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)


    # Make sure 'Date' is datetime for day-of-week computations
    if "Date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)

    # Sort by date so rolling logic is chronological
    df.sort_values(by="Date", inplace=True)

    def _create_rolling_home_features(grp: pd.DataFrame) -> pd.DataFrame:

        grp = grp.sort_values(by="Date")

        # Convert columns to numeric if needed (goals might already be numeric)
        grp["FTHG"] = pd.to_numeric(grp["FTHG"], errors="coerce")
        grp["FTAG"] = pd.to_numeric(grp["FTAG"], errors="coerce")

        # Rolling avg goals scored/conceded
        grp["home_avg_goals_scored_5"] = (
            grp["FTHG"].shift(1).rolling(rolling_window).mean()
        )
        grp["home_avg_goals_conceded_5"] = (
            grp["FTAG"].shift(1).rolling(rolling_window).mean()
        )

        # Rolling win rate: We treat (FTR == "H") as 1, else 0
        # shift(1) so we only consider past matches
        grp["home_win_rate_5"] = (
            grp["FTR"].shift(1).eq("H").rolling(rolling_window).mean()
        )

        # Shots on target, corners
        if "HST" in grp.columns:
            grp["HST"] = pd.to_numeric(grp["HST"], errors="coerce")
            grp["home_avg_shots_on_target_5"] = (
                grp["HST"].shift(1).rolling(rolling_window).mean()
            )
        if "HC" in grp.columns:
            grp["HC"] = pd.to_numeric(grp["HC"], errors="coerce")
            grp["home_avg_corners_5"] = (
                grp["HC"].shift(1).rolling(rolling_window).mean()
            )

        return grp

    def _create_rolling_away_features(grp: pd.DataFrame) -> pd.DataFrame:

        grp = grp.sort_values(by="Date")

        grp["FTHG"] = pd.to_numeric(grp["FTHG"], errors="coerce")
        grp["FTAG"] = pd.to_numeric(grp["FTAG"], errors="coerce")

        grp["away_avg_goals_scored_5"] = (
            grp["FTAG"].shift(1).rolling(rolling_window).mean()
        )
        grp["away_avg_goals_conceded_5"] = (
            grp["FTHG"].shift(1).rolling(rolling_window).mean()
        )

        # Rolling away win rate: (FTR == "A") => 1 else 0
        grp["away_win_rate_5"] = (
            grp["FTR"].shift(1).eq("A").rolling(rolling_window).mean()
        )

        if "AST" in grp.columns:
            grp["AST"] = pd.to_numeric(grp["AST"], errors="coerce")
            grp["away_avg_shots_on_target_5"] = (
                grp["AST"].shift(1).rolling(rolling_window).mean()
            )
        if "AC" in grp.columns:
            grp["AC"] = pd.to_numeric(grp["AC"], errors="coerce")
            grp["away_avg_corners_5"] = (
                grp["AC"].shift(1).rolling(rolling_window).mean()
            )

        return grp

    # Group by HomeTeam -> create home-based rolling features
    df = df.groupby("HomeTeam", group_keys=False).apply(_create_rolling_home_features)

    # Group by AwayTeam -> create away-based rolling features
    df = df.groupby("AwayTeam", group_keys=False).apply(_create_rolling_away_features)


    if "Date" in df.columns:
        df["day_of_week"] = df["Date"].dt.weekday
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)


    if "HomeEloPre" in df.columns:
        df["home_team_elo"] = df["HomeEloPre"]
    if "AwayEloPre" in df.columns:
        df["away_team_elo"] = df["AwayEloPre"]

    # Create a difference feature
    if "home_team_elo" in df.columns and "away_team_elo" in df.columns:
        df["elo_diff"] = df["home_team_elo"] - df["away_team_elo"]

    df["MatchPair"] = df["HomeTeam"] + "_vs._" + df["AwayTeam"]

    def _create_h2h_features(grp: pd.DataFrame) -> pd.DataFrame:

        grp = grp.sort_values(by="Date")
        grp["FTHG"] = pd.to_numeric(grp["FTHG"], errors="coerce")
        grp["FTAG"] = pd.to_numeric(grp["FTAG"], errors="coerce")

        # "H2H goals for" = FTHG if we keep orientation consistent
        grp["h2h_avg_goals_for_5"] = grp["FTHG"].shift(1).rolling(rolling_window).mean()
        # "H2H goals against" = FTAG
        grp["h2h_avg_goals_against_5"] = grp["FTAG"].shift(1).rolling(rolling_window).mean()

        # H2H win rate for the home side in these direct matchups
        # If FTR == 'H', that means the listed HomeTeam in "MatchPair" won
        grp["h2h_win_rate_5"] = grp["FTR"].shift(1).eq("H").rolling(rolling_window).mean()

        return grp

    df = df.groupby("MatchPair", group_keys=False).apply(_create_h2h_features)


    print(f"Feature engineering complete. Saving to {output_path}")
    df.to_csv(output_path, index=False)

    return df

def main():

    generate_features()

if __name__ == "__main__":
    main()

