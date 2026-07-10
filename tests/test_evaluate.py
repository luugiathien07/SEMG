"""Regression tests for src/evaluate.py — metric/confusion-matrix consistency
and F-beta threshold selection (the levers used to trade Precision for
Recall on the imbalanced Fatigue class)."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from src.evaluate import _best_threshold, evaluate_model
from src.models import ModelSpec


def _synthetic_binary_data(seed=0, n=200, n_features=4, weights=(0.75, 0.25)):
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n, n_features=n_features, n_informative=n_features,
        n_redundant=0, n_clusters_per_class=1, weights=list(weights),
        random_state=seed,
    )
    return X, y


class TestEvaluateModelConsistency:
    def test_metrics_match_confusion_matrix(self):
        X_train, y_train = _synthetic_binary_data(seed=1)
        X_test, y_test = _synthetic_binary_data(seed=2)
        spec = ModelSpec(
            name="DT", estimator=DecisionTreeClassifier(random_state=0),
            feature_set="all", supports_proba=True, param_grid={},
        )
        res = evaluate_model(spec, X_train, y_train, X_test, y_test,
                             features_used=["f0", "f1", "f2", "f3"])

        tn, fp, fn, tp = res.confusion.ravel()
        total = tn + fp + fn + tp
        assert total == len(y_test)

        acc = (tn + tp) / total
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

        assert res.accuracy == pytest.approx(acc)
        assert res.precision == pytest.approx(prec)
        assert res.recall == pytest.approx(rec)
        assert res.f1 == pytest.approx(f1)

    def test_confusion_matrix_is_2x2_labels_0_1(self):
        X_train, y_train = _synthetic_binary_data(seed=1)
        X_test, y_test = _synthetic_binary_data(seed=2)
        spec = ModelSpec(
            name="DT", estimator=DecisionTreeClassifier(random_state=0),
            feature_set="all", supports_proba=True, param_grid={},
        )
        res = evaluate_model(spec, X_train, y_train, X_test, y_test,
                             features_used=["f0", "f1", "f2", "f3"])
        assert res.confusion.shape == (2, 2)


class TestBestThreshold:
    def test_module_fbeta_threshold_not_higher_than_plain_f1(self):
        # Scores where lowering the threshold trades Precision for Recall:
        # true positives spread across the score range, one false positive
        # sitting at a high score.
        y_true = np.array([0, 0, 0, 1, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.9, 0.3, 0.5, 0.7, 0.95])

        from sklearn.metrics import fbeta_score

        def best_for_beta(beta):
            best_t, best_s = 0.5, -1.0
            for t in np.unique(scores):
                pred = (scores >= t).astype(int)
                s = fbeta_score(y_true, pred, beta=beta, pos_label=1, zero_division=0)
                if s > best_s:
                    best_s, best_t = s, float(t)
            return best_t

        # _best_threshold uses the module-level FBETA (>1, favors Recall),
        # so it should never pick a higher cut-off than plain-F1 (beta=1).
        t_module = _best_threshold(y_true, scores)
        t_beta1 = best_for_beta(1.0)
        assert t_module <= t_beta1
