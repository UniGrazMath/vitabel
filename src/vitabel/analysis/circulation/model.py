"""Pre-trained classification model for the detection of circulation (ROSC)."""

from __future__ import annotations

import functools
import importlib.resources
import logging


import joblib


logger = logging.getLogger("vitabel")


@functools.cache
def load_circulation_model():
    """Load the (pre-trained) classification model and the corresponding
    scaler used by :func:`predict_circulation` from the package assets.

    Returns
    -------
    A tuple ``(model, scaler)``.
    """
    asset_dir = importlib.resources.files("vitabel") / "assets"
    with (asset_dir / "circ-classification_model.joblib").open("rb") as fd:
        model = joblib.load(fd)
    with (asset_dir / "circ-classification_scaler.joblib").open("rb") as fd:
        scaler = joblib.load(fd)
    return model, scaler
