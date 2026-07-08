"""Evaluation: confusion matrix + metrics (no hardcoded TP/FP), CV and ROC.

Positive class = 1 (Fatigue). All metrics are derived from the real confusion
matrix / predictions, fixing the MATLAB bug where TP/FP were hardcoded.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import config
from .models import ModelSpec


@dataclass
class ModelResult:
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    cv_accuracy: float
    auc: float | None
    confusion: np.ndarray
    roc: tuple[np.ndarray, np.ndarray] | None = None   # (fpr, tpr)
    features_used: list[str] = field(default_factory=list)
    fitted_estimator: object | None = None   # model fit on full train, for inference


def _metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1]),
    }


def _scores(estimator, X) -> np.ndarray | None:
    """Positive-class scores for ROC, from predict_proba or decision_function."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    return None


def evaluate_model(
    spec: ModelSpec,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    features_used: list[str],
) -> ModelResult:
    est = clone(spec.estimator)

    # Cross-validated accuracy on the training set (StratifiedKFold, capped by
    # the smallest class count so tiny demo subsets don't crash).
    min_class = int(np.min(np.bincount(y_train.astype(int))))
    n_splits = max(2, min(config.CV_FOLDS, min_class))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=config.RANDOM_STATE)
    cv_pred = cross_val_predict(clone(spec.estimator), X_train, y_train, cv=skf)
    cv_acc = float(accuracy_score(y_train, cv_pred))

    # Fit on full train, evaluate on the held-out subject.
    est.fit(X_train, y_train)
    y_pred = est.predict(X_test)
    m = _metrics(y_test, y_pred)

    auc, roc = None, None
    scores = _scores(est, X_test)
    if scores is not None and len(np.unique(y_test)) == 2:
        auc = float(roc_auc_score(y_test, scores))
        fpr, tpr, _ = roc_curve(y_test, scores, pos_label=1)
        roc = (fpr, tpr)

    return ModelResult(
        name=spec.name,
        accuracy=m["accuracy"],
        precision=m["precision"],
        recall=m["recall"],
        f1=m["f1"],
        cv_accuracy=cv_acc,
        auc=auc,
        confusion=m["confusion"],
        roc=roc,
        features_used=features_used,
        fitted_estimator=est,
    )
