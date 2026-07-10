"""Evaluation: confusion matrix + metrics (no hardcoded TP/FP), CV and ROC.

Positive class = 1 (Fatigue). All metrics are derived from the real confusion
matrix / predictions, fixing the MATLAB bug where TP/FP were hardcoded.

Handles class imbalance and hyperparameter selection entirely on the
**training subjects** (never touches the held-out test subject):
1. ``GridSearchCV`` (scoring=F-beta, positive=Fatigue) picks
   ``spec.param_grid`` hyperparameters, if any.
2. Out-of-fold CV predictions with those hyperparameters give both
   ``cv_accuracy`` and out-of-fold Fatigue probabilities, from which
   ``_best_threshold`` picks the decision threshold maximizing train F-beta
   (instead of the default 0.5) — this is what actually rescues Recall for
   the now-small Fatigue class, on top of the per-model class-imbalance
   handling in ``models.build_models``.

``FBETA`` (>1) weighs Recall over Precision: missing a genuinely fatigued
channel (false negative) is clinically worse than one extra false alarm, so
both the hyperparameter search and the threshold are optimized for it instead
of plain F1 — while still reporting F1/Precision/Recall/AUC unchanged so the
trade-off stays visible in the UI.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict

from . import config
from .models import ModelSpec

FBETA = 2.0   # >1 -> weigh Recall over Precision (see module docstring)
_fbeta_scorer = make_scorer(fbeta_score, beta=FBETA, pos_label=1, zero_division=0)


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
    best_params: dict = field(default_factory=dict)
    threshold: float = 0.5   # P(Fatigue) cut-off; tuned to maximize train F1


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


def _best_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Threshold on `scores` maximizing F-beta (see FBETA) for Fatigue."""
    best_t, best_score = 0.5, -1.0
    for t in np.unique(scores):
        pred = (scores >= t).astype(int)
        score = fbeta_score(y_true, pred, beta=FBETA, pos_label=1, zero_division=0)
        if score > best_score:
            best_score, best_t = score, float(t)
    return best_t


def evaluate_model(
    spec: ModelSpec,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    features_used: list[str],
) -> ModelResult:
    # Cross-validated accuracy on the training set (StratifiedKFold, capped by
    # the smallest class count so tiny demo subsets don't crash).
    min_class = int(np.min(np.bincount(y_train.astype(int))))
    n_splits = max(2, min(config.CV_FOLDS, min_class))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=config.RANDOM_STATE)

    best_params: dict = {}
    if spec.param_grid:
        search = GridSearchCV(clone(spec.estimator), spec.param_grid,
                              scoring=_fbeta_scorer, cv=skf, n_jobs=1)
        search.fit(X_train, y_train)
        best_params = search.best_params_

    def _tuned_clone() -> object:
        est = clone(spec.estimator)
        if best_params:
            est.set_params(**best_params)
        return est

    cv_pred = cross_val_predict(_tuned_clone(), X_train, y_train, cv=skf)
    cv_acc = float(accuracy_score(y_train, cv_pred))

    cv_scores = None
    if hasattr(spec.estimator, "predict_proba"):
        cv_scores = cross_val_predict(_tuned_clone(), X_train, y_train, cv=skf,
                                      method="predict_proba")[:, 1]
    threshold = _best_threshold(y_train, cv_scores) if cv_scores is not None else 0.5

    # Fit on full train (tuned hyperparameters), evaluate on the held-out subject.
    est = _tuned_clone()
    est.fit(X_train, y_train)

    scores = _scores(est, X_test)
    y_pred = (scores >= threshold).astype(int) if scores is not None else est.predict(X_test)
    m = _metrics(y_test, y_pred)

    auc, roc = None, None
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
        best_params=best_params,
        threshold=threshold,
    )
