"""Single-channel inference: raw sEMG channel -> per-model Normal/Fatigue verdict.

This is the "đưa tín hiệu vào, đầu ra là có mỏi hay không" flow: take one EMG
channel, extract the same 14 features as the pipeline, and ask each trained model
whether the channel is Normal (0) or Fatigue (1), plus its P(Fatigue) confidence.

Models come from ``run_pipeline().results`` — each ``ModelResult`` carries its
``fitted_estimator`` (fit on the full training set) and ``features_used`` (the
feature subset that model expects, e.g. top-3 for SVM, all 14 otherwise).
"""
from __future__ import annotations

import numpy as np

from . import config
from . import feature_extraction as fe
from .evaluate import ModelResult


def _p_fatigue(estimator, X: np.ndarray) -> float | None:
    """Positive-class (Fatigue) probability, or None if unavailable."""
    if hasattr(estimator, "predict_proba"):
        return float(estimator.predict_proba(X)[0, 1])
    if hasattr(estimator, "decision_function"):
        # Map the signed margin to (0, 1) so it reads like a confidence.
        return float(1.0 / (1.0 + np.exp(-estimator.decision_function(X)[0])))
    return None


def predict_channel(x: np.ndarray, results: list[ModelResult]) -> list[dict]:
    """Predict Normal/Fatigue for one channel with every trained model.

    Returns one dict per model: ``model``, ``pred`` (0/1), ``pred_name``,
    ``p_fatigue`` (float in [0, 1] or None).
    """
    feats = fe.extract_channel_features(x)
    out: list[dict] = []
    for r in results:
        if r.fitted_estimator is None:
            continue
        X = np.array([[feats[f] for f in r.features_used]], dtype=float)
        pred = int(r.fitted_estimator.predict(X)[0])
        out.append({
            "model": r.name,
            "pred": pred,
            "pred_name": config.LABEL_NAMES[pred],
            "p_fatigue": _p_fatigue(r.fitted_estimator, X),
        })
    return out
