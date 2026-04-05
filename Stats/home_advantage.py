import pandas as pd


def calculate_home_advantage(df):

    home_advantage = {}

    # Group by HomeTeam and calculate average goals scored and conceded
    home_stats = df.groupby("HomeTeam").agg(
        avg_home_goals=("FTHG", "mean"),
        avg_away_goals_conceded=("FTAG", "mean"),
    )

    # Group by AwayTeam and calculate average goals scored and conceded
    away_stats = df.groupby("AwayTeam").agg(
        avg_away_goals=("FTAG", "mean"),
        avg_home_goals_conceded=("FTHG", "mean"),
    )

    # Merge stats to compute home advantage
    merged_stats = home_stats.merge(away_stats, left_index=True, right_index=True)
    merged_stats["home_advantage"] = (
            merged_stats["avg_home_goals"] - merged_stats["avg_away_goals"]
    )

    # Store in dictionary
    home_advantage = merged_stats["home_advantage"].to_dict()

    return home_advantage


if __name__ == "__main__":
    # Load your data (adjust the file path)
    file_path = "data/processed/featured.csv"  # Replace with your file path
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully.")

        # Ensure required columns exist
        required_columns = ["HomeTeam", "AwayTeam", "FTHG", "FTAG"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {required_columns}")

        # Calculate home advantage
        home_advantage = calculate_home_advantage(df)

        # Print results
        print("\nHome Advantage (Team-Specific):")
        for team, adv in home_advantage.items():
            print(f"{team}: {adv:.2f}")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Please check the path.")
    except Exception as e:
        print(f"An error occurred: {e}")