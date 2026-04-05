
import argparse
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from joblib import dump

from src.config import (
    FEATURED_PATH,
    RANDOM_SEARCH_PARAMS_PATH,
    RANDOM_SEARCH_N_ITER,
    RANDOM_SEARCH_CV_SPLITS,
    RANDOM_SEARCH_RANDOM_STATE
)
from src.model_pipeline import build_pipeline


def main(args):
    # Load featured data
    df = pd.read_csv(FEATURED_PATH)
    X = df.drop(columns=[args.target_col])
    y = df[args.target_col]

    # Keep only numeric features for model fitting
    X = X.select_dtypes(include=[np.number])

    pipeline = build_pipeline()

    # Time-series cross-validator
    tscv = TimeSeriesSplit(n_splits=RANDOM_SEARCH_CV_SPLITS)

    # Define randomized search over specified parameter grid
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=args.param_grid,
        n_iter=RANDOM_SEARCH_N_ITER,
        cv=tscv,
        scoring=args.scoring,
        random_state=RANDOM_SEARCH_RANDOM_STATE,
        n_jobs=-1,
        verbose=2,
    )

    # Run search
    search.fit(X, y)

    # Persist best parameters
    best_params = search.best_params_
    with open(RANDOM_SEARCH_PARAMS_PATH, 'w') as f:
        json.dump(best_params, f, indent=2)

    if args.save_model:
        dump(search.best_estimator_, args.model_out)
        print(f"Saved best model pipeline to {args.model_out}")

    print("Best parameters:")
    print(json.dumps(best_params, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run RandomizedSearchCV for the match outcome pipeline"
    )
    parser.add_argument(
        '--target-col', default='FTR',
        help="Name of the target column in featured CSV"
    )
    parser.add_argument(
        '--scoring', default='accuracy',
        help="Scoring metric for RandomizedSearchCV"
    )
    parser.add_argument(
        '--save-model', action='store_true',
        help="Whether to save the best estimator to disk"
    )
    parser.add_argument(
        '--model-out', default='models/best_pipeline.pkl',
        help="File path to save the tuned pipeline"
    )
    parser.add_argument(
        '--param-grid', type=json.loads,
        default=json.dumps({
            'clf__n_estimators': [100, 200, 500],
            'clf__max_depth': [None, 10, 20],
            'clf__min_samples_leaf': [1, 2, 5],
            'clf__max_features': ['sqrt', 'log2', None],
        }),
        help="JSON string of parameter distributions to sample from"
    )
    args = parser.parse_args()
    main(args)
