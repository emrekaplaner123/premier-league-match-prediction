
import os
import joblib
import pandas as pd
from datetime import datetime

from src.config import (
    FEATURED_DATA_PATH,  # e.g., "data/processed/featured.csv"
    MODEL_PATH           # e.g., "models/random_forest_model.pkl"
)

ROLLING_WINDOW = 5
def compute_rolling_for_home_away(df: pd.DataFrame, team_name: str, match_date: pd.Timestamp, is_home=True):

    prefix = "home" if is_home else "away"

    # Filter the dataframe to matches involving this team in the correct role (home/away)
    if is_home:
        df_team = df[(df["HomeTeam"] == team_name) & (df["Date"] < match_date)].sort_values(by="Date")
        gf_col, ga_col = "FTHG", "FTAG"
        win_condition = "H"
        shots_col, corners_col = "HST", "HC"
    else:
        df_team = df[(df["AwayTeam"] == team_name) & (df["Date"] < match_date)].sort_values(by="Date")
        gf_col, ga_col = "FTAG", "FTHG"
        win_condition = "A"
        shots_col, corners_col = "AST", "AC"

    df_recent = df_team.tail(ROLLING_WINDOW)
    num_matches = len(df_recent)
    if num_matches == 0:
        return {
            f"{prefix}_avg_goals_scored_5": None,
            f"{prefix}_avg_goals_conceded_5": None,
            f"{prefix}_win_rate_5": None,
            f"{prefix}_avg_shots_on_target_5": None,
            f"{prefix}_avg_corners_5": None
        }

    # Compute stats
    avg_gf = df_recent[gf_col].mean()
    avg_ga = df_recent[ga_col].mean()
    win_rate = (df_recent["FTR"] == win_condition).sum() / num_matches
    avg_sot = df_recent[shots_col].mean() if shots_col in df_recent.columns else None
    avg_corners = df_recent[corners_col].mean() if corners_col in df_recent.columns else None

    return {
        f"{prefix}_avg_goals_scored_5": avg_gf,
        f"{prefix}_avg_goals_conceded_5": avg_ga,
        f"{prefix}_win_rate_5": win_rate,
        f"{prefix}_avg_shots_on_target_5": avg_sot,
        f"{prefix}_avg_corners_5": avg_corners
    }

def get_latest_elo(df: pd.DataFrame, team_name: str, match_date: pd.Timestamp, is_home=True):

    if is_home:
        df_team = df[(df["HomeTeam"] == team_name) & (df["Date"] < match_date)].sort_values(by="Date")
        elo_col = "HomeEloPre"
    else:
        df_team = df[(df["AwayTeam"] == team_name) & (df["Date"] < match_date)].sort_values(by="Date")
        elo_col = "AwayEloPre"

    if df_team.empty:
        return 1500  # default if no data
    return df_team.iloc[-1][elo_col]

def predict_single_match():
    # Prompt user
    date_str = input("Match date (DD/MM/YYYY): ")
    home_team = input("Home Team: ")
    away_team = input("Away Team: ")

    try:
        match_date = datetime.strptime(date_str, "%d/%m/%Y")
        match_date = pd.to_datetime(match_date)
    except ValueError:
        print("Invalid date format! Please use DD/MM/YYYY.")
        return

    if not os.path.exists(FEATURED_DATA_PATH):
        print(f"Historical data not found: {FEATURED_DATA_PATH}")
        return

    # Load the full dataset (featured.csv)
    df = pd.read_csv(FEATURED_DATA_PATH, parse_dates=["Date"], dayfirst=True)
    df.sort_values(by="Date", inplace=True)

    # Compute rolling features for home & away
    home_rolling = compute_rolling_for_home_away(df, home_team, match_date, is_home=True)
    away_rolling = compute_rolling_for_home_away(df, away_team, match_date, is_home=False)

    # day_of_week, is_weekend
    day_of_week = match_date.weekday()  # 0=Monday, 6=Sunday
    is_weekend = 1 if day_of_week in [5,6] else 0

    # Elo
    home_elo = get_latest_elo(df, home_team, match_date, is_home=True)
    away_elo = get_latest_elo(df, away_team, match_date, is_home=False)
    elo_diff = home_elo - away_elo

    # Build feature dict
    feature_dict = {
        # home rolling
        "home_avg_goals_scored_5": home_rolling["home_avg_goals_scored_5"],
        "home_avg_goals_conceded_5": home_rolling["home_avg_goals_conceded_5"],
        "home_win_rate_5": home_rolling["home_win_rate_5"],
        "home_avg_shots_on_target_5": home_rolling["home_avg_shots_on_target_5"],
        "home_avg_corners_5": home_rolling["home_avg_corners_5"],

        # away rolling
        "away_avg_goals_scored_5": away_rolling["away_avg_goals_scored_5"],
        "away_avg_goals_conceded_5": away_rolling["away_avg_goals_conceded_5"],
        "away_win_rate_5": away_rolling["away_win_rate_5"],
        "away_avg_shots_on_target_5": away_rolling["away_avg_shots_on_target_5"],
        "away_avg_corners_5": away_rolling["away_avg_corners_5"],

        # match context
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,

        # Elo
        "home_team_elo": home_elo,
        "away_team_elo": away_elo,
        "elo_diff": elo_diff
    }

    df_features = pd.DataFrame([feature_dict])

    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        return
    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle["model"]
    label_encoder = model_bundle["label_encoder"]
    feature_cols = model_bundle["feature_cols"]  # the list of columns model expects

    # Ensure columns exist in df_features
    for col in feature_cols:
        if col not in df_features.columns:
            df_features[col] = 0  # default

    df_features = df_features[feature_cols]

    # Make prediction
    y_pred_enc = model.predict(df_features)
    y_pred = label_encoder.inverse_transform(y_pred_enc)  # "H", "D", "A"


    # Predict probabilities
    y_probs = model.predict_proba(df_features)
    classes_ = label_encoder.classes_

    #implied odds
    y_odds = [1/i for i in y_probs]
    print(y_odds)

    print("\nPrediction:")
    print(f"{home_team} vs. {away_team} on {match_date.strftime('%d/%m/%Y')}")
    print(f"Predicted Outcome: {y_pred[0]}")
    print("Probabilities:")
    for i, c in enumerate(classes_):
        print(f"  {c}: {y_probs[0][i]:.3f}")
    print("Implied odds:")
    for i, c in enumerate(classes_):
        print((f" {c}: {y_odds[0][i]:.3f}"))

def main():

    predict_single_match()

if __name__ == "__main__":
    main()
