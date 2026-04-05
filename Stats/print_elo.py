
import pandas as pd

def get_latest_elo(file_path="data/processed/with_elo.csv"):

    try:
        df = pd.read_csv(file_path)

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
            df.sort_values(by="Date", inplace=True)
        else:
            raise KeyError("The 'Date' column is missing in the input file.")

        required_cols = {"HomeTeam", "AwayTeam", "HomeEloPre", "AwayEloPre"}
        if not required_cols.issubset(df.columns):
            raise KeyError(f"The input file must include the following columns: {required_cols}")

        latest_elo = {}

        for _, row in df.iterrows():
            home_team = row["HomeTeam"]
            home_elo = row["HomeEloPre"]
            latest_elo[home_team] = home_elo

            away_team = row["AwayTeam"]
            away_elo = row["AwayEloPre"]
            latest_elo[away_team] = away_elo

        return latest_elo

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Ensure the Elo data file exists.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def main():
    file_path = "data/processed/with_elo.csv"

    latest_elo = get_latest_elo(file_path)

    if latest_elo:
        print("\nLatest Elo Ratings for All Teams:")
        for team, elo in sorted(latest_elo.items(), key=lambda x: -x[1]):  # Sort by Elo descending
            print(f"{team}: {elo:.2f}")

if __name__ == "__main__":
    main()
