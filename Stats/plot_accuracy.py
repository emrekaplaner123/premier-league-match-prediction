
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def calculate_cumulative_accuracy(df):

    # Add a "Correct" column: 1 if prediction matches actual result, else 0
    df["Correct"] = (df["FTR"] == df["Predicted"]).astype(int)

    # Calculate cumulative accuracy
    df["CumulativeAccuracy"] = df["Correct"].expanding().mean()

    return df

def main():
    # Paths
    model_path = "models/random_forest_model.pkl"
    featured_data_path = "data/processed/featured.csv"

    # Load the trained model
    try:
        bundle = joblib.load(model_path)
        model = bundle["model"]
        feature_cols = bundle["feature_cols"]
        label_encoder = bundle["label_encoder"]
    except FileNotFoundError:
        print(f"Error: Model not found at {model_path}. Ensure you have trained and saved the model.")
        return

    # Load the featured data
    try:
        df = pd.read_csv(featured_data_path)
    except FileNotFoundError:
        print(f"Error: Featured data not found at {featured_data_path}. Ensure feature engineering is complete.")
        return

    # Ensure 'Date' is in datetime format and sort by Date
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df.sort_values(by="Date", inplace=True)
    else:
        print("Error: 'Date' column missing in featured data.")
        return

    # Check required columns
    required_cols = {"FTR"} | set(feature_cols)
    if not required_cols.issubset(df.columns):
        print(f"Error: Featured data must include columns: {required_cols}.")
        return

    # Prepare features and actual results
    X = df[feature_cols]
    y = df["FTR"]

    # Split into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=0.2, random_state=42, shuffle=False  # Ensure no time leakage
    )

    # Train the model on the training set
    model.fit(X_train, label_encoder.transform(y_train))

    # Predict outcomes for the test set
    y_pred_encoded = model.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    # Add predictions to the test DataFrame
    df_test["Predicted"] = y_pred

    # Calculate accuracy metrics
    overall_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Set Overall Accuracy: {overall_accuracy:.2%}")

    # Print confusion matrix and classification report
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=label_encoder.classes_))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, labels=label_encoder.classes_))

    # Calculate cumulative accuracy
    df_test = calculate_cumulative_accuracy(df_test)

    # Plot cumulative accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(df_test["Date"], df_test["CumulativeAccuracy"], label="Cumulative Accuracy", linewidth=2)
    plt.title("Prediction Accuracy Over Time (Test Set)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
