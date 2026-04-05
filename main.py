

import argparse
import sys

# Import your scripts
from src.data_collection import collect_data
from src.data_preprocessing import preprocess_data
from src.elo_rating import generate_elo_ratings
from src.feature_engineering import generate_features
from src.train import train_model
from src.predict import predict_upcoming_fixtures
from src.predict_single_match import predict_single_match
from src.update_after_match import update_after_match
from src.evaluate import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="ML Pipeline CLI for Premier League predictions.")
    parser.add_argument(
        "--action",
        type=str,
        required=True,
        help="Which pipeline action to run. Options: collect_data, preprocess, elo, feature_engineering, train, predict, predict_single, update_match, evaluate"
    )
    args = parser.parse_args()
    action = args.action.lower()

    if action == "collect_data":
        # Run data collection (merging Excel files from data/raw)
        collect_data()
    elif action == "preprocess":
        # Clean up and unify data
        preprocess_data()
    elif action == "elo":
        # Recompute Elo ratings
        import pandas as pd
        from src.config import CLEANED_DATA_PATH, WITH_ELO_DATA_PATH
        df_clean = pd.read_csv(CLEANED_DATA_PATH)
        df_elo = generate_elo_ratings(df_clean)
        df_elo.to_csv(WITH_ELO_DATA_PATH, index=False)
        print("Elo ratings updated.")
    elif action == "feature_engineering":
        # Build rolling stats, day_of_week, etc.
        generate_features()
    elif action == "train":
        # Train the random forest model
        train_model()
    elif action == "predict":
        # Predict upcoming fixtures from CSV
        predict_upcoming_fixtures()
    elif action == "predict_single":
        predict_single_match()
    elif action == "update_match":

        print("This action is usually interactive. Run `update_after_match.py` directly or build logic here.")

    elif action == "evaluate":
        evaluate_model()
    else:
        print(f"Unknown action: {action}")
        sys.exit(1)

if __name__ == "__main__":
    main()
