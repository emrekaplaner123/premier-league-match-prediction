
import argparse
import os
import joblib
import json
import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit

from src.config import FEATURED_PATH, MODEL_PATH, RANDOM_SEARCH_CV_SPLITS


def main(args):
    # Load featured data
    df = pd.read_csv(FEATURED_PATH)
    X = df.drop(columns=[args.target_col])
    y = df[args.target_col]

    # Load existing model bundle to get feature list and base model
    bundle = joblib.load(args.model_in)
    model = bundle['model']
    label_encoder = bundle['label_encoder']
    feature_cols = bundle['feature_cols']

    # Keep only numeric features
    X = X.select_dtypes(include=[np.number])
    X = X[feature_cols]

    # Time-series cross-validator for calibration
    tscv = TimeSeriesSplit(n_splits=RANDOM_SEARCH_CV_SPLITS)

    # Calibrate probabilities
    calibrator = CalibratedClassifierCV(estimator=model,
                                        method=args.method,
                                        cv=tscv)
    calibrator.fit(X, y)

    # Package calibrated model bundle
    calibrated_bundle = {
        "model": calibrator,
        "label_encoder": label_encoder,
        "feature_cols": feature_cols
    }

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    joblib.dump(calibrated_bundle, args.model_out)
    print(f"Calibrated model saved to: {args.model_out}")

    # Save calibration metadata
    metadata = {
        "method": args.method,
        "cv_splits": RANDOM_SEARCH_CV_SPLITS
    }
    with open(args.metadata_out, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Calibration metadata saved to: {args.metadata_out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calibrate RF probabilities")
    parser.add_argument('--method', choices=['sigmoid', 'isotonic'], default='isotonic',
                        help='Calibration method')
    parser.add_argument('--model-in', default=MODEL_PATH,
                        help='Path to the existing model bundle (.pkl)')
    parser.add_argument('--model-out', default='models/rf_calibrated.pkl',
                        help='Path to save the calibrated model bundle')
    parser.add_argument('--metadata-out', default='models/rf_calibration_metadata.json',
                        help='Path to save calibration metadata')
    parser.add_argument('--target-col', default='FTR', help='Target column name')
    args = parser.parse_args()
    main(args)
