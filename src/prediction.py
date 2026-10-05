"""
Crop prediction using the existing models/crop_model.pkl.

Safety net: if the pickle file is missing, cannot be loaded, or does not match the
CSV columns, a RandomForest is trained from data/agriculture_data.csv and saved as
models/crop_model_auto.pkl (your original model file is never overwritten).
"""

import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.data_processing import (
    DATA_PATH,
    get_feature_columns,
    get_target_column,
    load_data,
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "crop_model.pkl"
AUTO_MODEL_PATH = BASE_DIR / "models" / "crop_model_auto.pkl"

_BUNDLE = None  # simple cache so the model loads only once


def _load_pickle(path):
    """Load a pickle file with joblib, falling back to plain pickle."""
    try:
        obj = joblib.load(path)
    except Exception:
        with open(path, "rb") as file:
            obj = pickle.load(file)

    if isinstance(obj, dict):  # some people save {"model": model, ...}
        obj = obj.get("model", obj)
    if not hasattr(obj, "predict"):
        raise ValueError("The pickle file does not contain a valid model.")
    return obj


def _train_backup_model():
    """Train a RandomForest from the CSV and save it."""
    df = load_data(DATA_PATH)
    target = get_target_column(df)
    features = get_feature_columns(df, target)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(df[features].values, df[target].astype(str).values)

    AUTO_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, AUTO_MODEL_PATH)
    return model, features


def _build_bundle():
    """Load crop_model.pkl and work out which features it needs."""
    df = load_data(DATA_PATH)
    target = get_target_column(df)
    csv_features = get_feature_columns(df, target)
    sorted_labels = sorted(df[target].astype(str).unique())

    note = ""
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("crop_model.pkl not found")

        model = _load_pickle(MODEL_PATH)

        if hasattr(model, "feature_names_in_"):
            features = [str(name) for name in model.feature_names_in_]
            use_names = True
        else:
            features = csv_features
            use_names = False
            expected = getattr(model, "n_features_in_", len(features))
            if expected != len(features):
                raise ValueError(
                    f"Model expects {expected} features but the CSV has {len(features)}."
                )
    except Exception as error:
        model, features = _train_backup_model()
        use_names = False
        note = (
            f"Your crop_model.pkl could not be used ({error}). "
            "A backup model was trained from the CSV file instead."
        )

    return {
        "model": model,
        "features": features,
        "use_names": use_names,
        "sorted_labels": sorted_labels,
        "note": note,
    }


def get_model_bundle():
    """Return (and cache) the model together with its feature list."""
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = _build_bundle()
    return _BUNDLE


def _decode_label(model, raw_label, sorted_labels):
    """Turn the model output into a crop name."""
    if isinstance(raw_label, (str, np.str_)):
        return str(raw_label)

    # Numeric output: map it to crop names if the counts match.
    classes = list(getattr(model, "classes_", []))
    if classes and len(classes) == len(sorted_labels):
        mapping = dict(zip(sorted(classes), sorted_labels))
        return mapping.get(raw_label, str(raw_label))
    return str(raw_label)


def predict_crop(inputs):
    """
    inputs: dict like {"N": 90, "P": 42, ...}
    Returns: {"best_crop": str, "top_predictions": [(crop, percent), ...], "inputs": dict}
    """
    bundle = get_model_bundle()
    model = bundle["model"]
    features = bundle["features"]

    lookup = {str(key).lower(): value for key, value in inputs.items()}
    row = []
    for feature in features:
        if feature.lower() not in lookup:
            raise ValueError(f"Missing value for '{feature}'.")
        row.append(float(lookup[feature.lower()]))

    if bundle["use_names"]:
        data = pd.DataFrame([row], columns=features)
    else:
        data = np.array([row])

    raw_prediction = model.predict(data)[0]
    best_crop = _decode_label(model, raw_prediction, bundle["sorted_labels"])

    top_predictions = []
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(data)[0]
        classes = list(model.classes_)
        best_indexes = np.argsort(probabilities)[::-1][:3]
        for index in best_indexes:
            name = _decode_label(model, classes[index], bundle["sorted_labels"])
            top_predictions.append((name, float(probabilities[index]) * 100))

    return {
        "best_crop": best_crop,
        "top_predictions": top_predictions,
        "inputs": {feature: value for feature, value in zip(features, row)},
    }