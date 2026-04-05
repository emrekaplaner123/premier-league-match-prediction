from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from src.config import RANDOM_SEARCH_RANDOM_STATE

from src.config import RF_PARAMS

def build_pipeline():
    return Pipeline([
        ("clf", RandomForestClassifier(**RF_PARAMS))
    ])

def build_pipeline():

    return Pipeline([
        ("clf", RandomForestClassifier(
            random_state=RANDOM_SEARCH_RANDOM_STATE
        ))
    ])
