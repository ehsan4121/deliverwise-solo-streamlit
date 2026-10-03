"""Prediction helpers shared by the Streamlit interface and local tests."""

from pathlib import Path
from typing import Union

import joblib
import pandas as pd


FEATURES = [
    "item_count",
    "seller_count",
    "total_price",
    "total_freight",
    "approval_delay_hours",
    "promised_days",
    "approval_hour",
    "approval_weekday",
    "approval_month",
    "customer_state",
    "seller_state",
]


def load_pipeline(model_path: Union[str, Path]):
    """Load the trusted DeliverWise pipeline from disk."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file was not found: {model_path}")
    model = joblib.load(model_path)
    if not hasattr(model, "predict_proba"):
        raise TypeError("The loaded object does not provide predict_proba().")

    # scikit-learn releases before the training version expect this fitted
    # LogisticRegression attribute during prediction. The parameter was later
    # removed, so add its previous default when an older runtime needs it.
    classifier = getattr(model, "named_steps", {}).get("classifier")
    if classifier is not None and not hasattr(classifier, "multi_class"):
        classifier.multi_class = "auto"
    return model


def make_input(values: dict) -> pd.DataFrame:
    """Create one validated model row in the exact training-column order."""
    missing = [name for name in FEATURES if name not in values]
    if missing:
        raise ValueError(f"Missing input fields: {', '.join(missing)}")

    row = {name: values[name] for name in FEATURES}
    numeric_nonnegative = FEATURES[:6]
    for name in numeric_nonnegative:
        if float(row[name]) < 0:
            raise ValueError(f"{name} cannot be negative.")
    if not 0 <= int(row["approval_hour"]) <= 23:
        raise ValueError("approval_hour must be between 0 and 23.")
    if not 0 <= int(row["approval_weekday"]) <= 6:
        raise ValueError("approval_weekday must be between 0 and 6.")
    if not 1 <= int(row["approval_month"]) <= 12:
        raise ValueError("approval_month must be between 1 and 12.")
    for name in ["customer_state", "seller_state"]:
        row[name] = str(row[name]).strip().upper()
        if not row[name]:
            raise ValueError(f"{name} cannot be empty.")
    return pd.DataFrame([row], columns=FEATURES)


def predict_late(model, values: dict, threshold: float = 0.50) -> dict:
    """Return the late-delivery probability and threshold-based class."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")
    frame = make_input(values)
    probability = float(model.predict_proba(frame)[0, 1])
    return {
        "probability": probability,
        "predicted_late": probability >= threshold,
        "threshold": float(threshold),
        "input": frame,
    }
