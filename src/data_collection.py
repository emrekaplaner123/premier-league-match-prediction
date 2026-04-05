
import os
import pandas as pd


def collect_data(
        input_dir: str = "data/raw",
        output_path: str = "data/processed/combined.csv"
) -> None:

    # List to keep all dataframes
    all_dfs = []

    # Loop through the input directory
    for filename in os.listdir(input_dir):
        # Check if the file is an Excel file
        if filename.lower().endswith((".csv")):
            file_path = os.path.join(input_dir, filename)
            print(f"Reading file: {file_path}")

            # Read the Excel file into a DataFrame
            df = pd.read_csv(
                file_path,
                encoding="latin-1",
                on_bad_lines="warn",
                engine="python",
            )

            all_dfs.append(df)

    # Concatenate all the dataframes
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"Total rows combined: {combined_df.shape[0]}")

        # Save the combined dataframe to CSV
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined_df.to_csv(output_path, index=False)
        print(f"Combined CSV saved to: {output_path}")
    else:
        print(f"No Excel files found in {input_dir}.")


def main():
    collect_data()


if __name__ == "__main__":
    main()
