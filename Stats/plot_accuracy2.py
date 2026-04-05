
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from src.config import FEATURED_DATA_PATH, MODEL_PATH


def main():
    df = pd.read_csv(FEATURED_DATA_PATH, parse_dates=['Date'])

    # Load model bundle
    bundle = joblib.load(MODEL_PATH)
    model = bundle['model']
    feature_cols = bundle['feature_cols']

    # Filter to complete records and sort by date
    df = df.dropna(subset=['FTR'] + feature_cols)
    df.sort_values('Date', inplace=True)

    # Prepare features and true labels
    X = df[feature_cols]
    y_true = df['FTR']

    # Make predictions and flag correctness
    y_pred = model.predict(X)
    df['correct'] = (y_pred == y_true).astype(int)

    # 1) Compute and plot monthly accuracy
    df['month'] = df['Date'].dt.to_period('M')
    monthly_acc = df.groupby('month')['correct'].mean()
    times = monthly_acc.index.to_timestamp()

    plt.figure(figsize=(10, 5))
    plt.plot(times, monthly_acc.values, marker='o', linestyle='-')
    plt.title('Monthly Prediction Accuracy')
    plt.xlabel('Month')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 2) Compute and plot cumulative accuracy
    df['cumulative_accuracy'] = df['correct'].expanding().mean()

    plt.figure(figsize=(10, 5))
    plt.plot(df['Date'], df['cumulative_accuracy'], marker='', linestyle='-')
    plt.title('Cumulative Prediction Accuracy Over Time')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()