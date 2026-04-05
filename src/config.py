

# -----------------------------
# Elo Rating Settings
# -----------------------------
ELO_BASE = 1500       # Starting Elo for teams not previously encountered
K_FACTOR = 50         # Determines how reactive Elo updates are
HOME_ADVANTAGE = 0   # Extra Elo points given to the home team in expectation calculations

# -----------------------------
# Data Paths
# -----------------------------
# After data_collection merges raw files, it writes to combined.csv (or similar).
COMBINED_DATA_PATH = "data/processed/combined.csv"
CLEANED_DATA_PATH = "data/processed/cleaned.csv"
WITH_ELO_DATA_PATH = "data/processed/with_elo.csv"
FEATURED_DATA_PATH = "data/processed/featured.csv"


# -----------------------------
# Model Paths
# -----------------------------
MODEL_PATH = "models/random_forest_model.pkl"  # Where the trained model is saved
# now using the calibrated RF for all predictions
MODEL_PATH = "models/rf_calibrated.pkl"

# -----------------------------
# Prediction Files
# -----------------------------
UPCOMING_FIXTURES_PATH = "data/raw/upcoming_fixtures.csv"  # For batch predictions
PREDICTIONS_OUTPUT_PATH = "data/predictions/upcoming_predictions.csv"  # Where predictions are saved

# -----------------------------
# Hyperparameter‐Search Settings
# -----------------------------
RANDOM_SEARCH_PARAMS_PATH = "tuning/best_params.json"
# How many parameter combos to try
RANDOM_SEARCH_N_ITER      = 50
# How many folds for TimeSeriesSplit during search
RANDOM_SEARCH_CV_SPLITS   = 5
# Random seed for reproducibility
RANDOM_SEARCH_RANDOM_STATE= 42

# Alias so hyperparameter_search.py can still import FEATURED_PATH
FEATURED_PATH = FEATURED_DATA_PATH

# ─── In src/config.py ────────────────────────────────

RF_PARAMS = {
  "n_estimators": 200,
  "min_samples_leaf": 2,
  "max_features": None,
  "max_depth": None,
  "random_state": RANDOM_SEARCH_RANDOM_STATE
}