"""Capa de inferencia: carga el pipeline serializado y expone predict()."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .preprocessing import FEATURE_COLUMNS, email_to_features

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "modelo.pkl"


@lru_cache(maxsize=1)
def load_model(path: str | Path | None = None):
    return joblib.load(Path(path) if path else MODEL_PATH)


def predict_from_features(features: pd.DataFrame, path: str | Path | None = None) -> dict:
    model = load_model(path)
    features = features[FEATURE_COLUMNS]
    label = int(model.predict(features)[0])
    score = float(model.decision_function(features)[0])
    proba = 1.0 / (1.0 + np.exp(-score))
    return {
        "label": label,
        "label_name": "spam" if label == 1 else "ham",
        "decision_score": score,
        "spam_probability": proba,
    }


def predict_from_email(text: str, path: str | Path | None = None) -> dict:
    return predict_from_features(email_to_features(text), path)
