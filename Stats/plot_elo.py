import pandas as pd
import matplotlib.pyplot as plt


def get_team_elo_timeseries(df, team_name):

    home_matches = df[df["HomeTeam"] == team_name].copy()
    home_matches["Elo"] = home_matches["HomeEloPre"]

    away_matches = df[df["AwayTeam"] == team_name].copy()
    away_matches["Elo"] = away_matches["AwayEloPre"]

    team_df = pd.concat([home_matches, away_matches], ignore_index=True)

    team_df = team_df[["Date", "Elo"]].copy()
    team_df["Team"] = team_name

    team_df.sort_values(by="Date", inplace=True)
    return team_df


def main():
    max_teams = 20
    while True:
        try:
            n = int(input(f"How many teams do you want to plot? (1-{max_teams}): "))
            if 1 <= n <= max_teams:
                break
            else:
                print(f"Please enter a number between 1 and {max_teams}.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    team_names = []
    for i in range(n):
        tname = input(f"Enter Team #{i + 1}: ")
        team_names.append(tname.strip())

    df = pd.read_csv("data/processed/with_elo.csv")

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    else:
        print("No 'Date' column found. Exiting.")
        return

    plt.figure(figsize=(10, 6))
    for team in team_names:
        team_df = get_team_elo_timeseries(df, team)

        if team_df.empty:
            print(f"Warning: No matches found for '{team}'. Skipping plot.")
            continue

        plt.plot(team_df["Date"], team_df["Elo"], label=team)

    plt.title("Elo Rating Over Time")
    plt.xlabel("Date")
    plt.ylabel("Elo Rating")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
