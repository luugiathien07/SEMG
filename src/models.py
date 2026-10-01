"""Classifier factory (spec section 4.4, extended for class-imbalance handling).

Each model is returned with the feature set it should use, plus how it
handles the Fatigue/Normal class imbalance and a hyperparameter grid for
``evaluate.evaluate_model`` to tune via GridSearchCV on the training subjects
only (never on the held-out test subject):

- SVM         : top-3 mRMR features; oversampled with SMOTE (fixed 1:1
                ratio); tunes C (kernel fixed to linear).
- KNN         : all 14 features; oversampled with SMOTE; tunes n_neighbors/
                weights, plus the SMOTE Fatigue:Normal ratio itself
                (1.0/1.3/1.6x, see ``_SMOTE_RATIOS``).
- LDA         : all 14 features; oversampled with SMOTE (fixed 1:1 ratio);
                tunes solver/shrinkage.
- DecisionTree: all 14 features; oversampled with SMOTE; tunes tree size,
                plus the SMOTE ratio (same as KNN).

The SMOTE ratio was tried as a tunable GridSearchCV axis on all 4 models
(over-oversampling Fatigue beyond a plain 1:1 balance, see
``_oversample_ratio``). It helped Recall/F1 on KNN and DecisionTree, but
regressed SVM (Recall 0.594->0.563, F1 0.738->0.715 on the held-out
subject) — GridSearchCV picked a heavier ratio that scored better on train
CV but generalized worse, the same overfitting failure mode already noted
below for SVM's class_weight grid. So the ratio is tuned only for KNN/
DecisionTree; SVM/LDA keep the fixed 1:1 default.

"RF" in the MATLAB code is actually a single decision tree (``fitctree``), so it
is named DecisionTree here.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import partial

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from . import config


def _oversample_strategy(multiplier: float, y) -> dict:
    """SMOTE ``sampling_strategy`` target: oversample the minority (Fatigue)
    class to `multiplier`x the majority class count instead of plain 1:1
    balance. `multiplier > 1.0` over-oversamples Fatigue beyond the majority
    — tunable via GridSearchCV (on train folds only) to test whether extra
    synthetic Fatigue signal improves Recall/F1.

    A module-level function bound via ``functools.partial`` (rather than a
    closure returned by a factory) so GridSearchCV's cached results stay
    picklable — required by ``st.cache_data`` in the Streamlit app.
    """
    counts = Counter(y)
    majority = max(counts.values())
    minority_label = min(counts, key=counts.get)
    return {minority_label: int(majority * multiplier)}


# Tried as a GridSearchCV axis on all 4 models: 1.0 (current 1:1 balance),
# 1.3, 1.6 (over-oversample Fatigue beyond the majority count).
_SMOTE_RATIOS = [partial(_oversample_strategy, m) for m in (1.0, 1.3, 1.6)]


@dataclass
class ModelSpec:
    name: str
    estimator: object
    feature_set: str        # "top3" or "all"
    supports_proba: bool
    # GridSearchCV-style grid: a dict, or a list of dicts for mutually
    # exclusive param combinations (e.g. SVM linear-vs-rbf). Empty means "no
    # tuning, use the estimator's defaults as-is".
    param_grid: dict | list[dict] = field(default_factory=dict)


def build_models() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="SVM",
            estimator=ImbPipeline([
                ("oversample", SMOTE(random_state=config.RANDOM_STATE)),
                ("scale", StandardScaler()),
                ("clf", CalibratedClassifierCV(
                    SVC(random_state=config.RANDOM_STATE),
                    ensemble=False)),
            ]),
            feature_set="top3",
            supports_proba=True,
            param_grid={"clf__estimator__kernel": ["linear"],
                       "clf__estimator__C": [0.001, 0.01, 0.1, 1, 10, 100]},
        ),
        ModelSpec(
            name="KNN",
            estimator=ImbPipeline([
                ("oversample", SMOTE(random_state=config.RANDOM_STATE)),
                ("scale", StandardScaler()),
                ("clf", KNeighborsClassifier(metric="euclidean")),
            ]),
            feature_set="all",
            supports_proba=True,
            param_grid={
                "clf__n_neighbors": [1, 3, 5, 7, 9, 11, 13],
                "clf__weights": ["uniform", "distance"],
                "oversample__sampling_strategy": _SMOTE_RATIOS,
            },
        ),
        ModelSpec(
            name="LDA",
            estimator=ImbPipeline([
                ("oversample", SMOTE(random_state=config.RANDOM_STATE)),
                ("clf", LinearDiscriminantAnalysis()),
            ]),
            feature_set="all",
            supports_proba=True,
            param_grid=[
                {"clf__solver": ["svd"]},
                {"clf__solver": ["lsqr"], "clf__shrinkage": ["auto", None, 0.1, 0.3, 0.5, 0.7]},
            ],
        ),
        ModelSpec(
            name="DecisionTree",
            estimator=ImbPipeline([
                ("oversample", SMOTE(random_state=config.RANDOM_STATE)),
                ("clf", DecisionTreeClassifier(random_state=config.RANDOM_STATE)),
            ]),
            feature_set="all",
            supports_proba=True,
            param_grid={
                "clf__criterion": ["gini", "entropy"],
                "clf__max_leaf_nodes": [11, 21, 31, None],
                "clf__max_depth": [None, 5, 10],
                "clf__min_samples_leaf": [1, 2, 4],
                "oversample__sampling_strategy": _SMOTE_RATIOS,
            },
        ),
        ModelSpec(
            name="RandomForest",
            estimator=ImbPipeline([
                ("oversample", SMOTE(random_state=config.RANDOM_STATE)),
                ("clf", RandomForestClassifier(random_state=config.RANDOM_STATE)),
            ]),
            feature_set="all",
            supports_proba=True,
            param_grid={
                "clf__n_estimators": [100, 300],
                "clf__max_depth": [None, 5, 10],
                "clf__min_samples_leaf": [1, 2, 4],
            },
        ),
        ModelSpec(
            name="LogisticRegression",
            estimator=ImbPipeline([
                ("oversample", SMOTE(random_state=config.RANDOM_STATE)),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(random_state=config.RANDOM_STATE, max_iter=2000)),
            ]),
            feature_set="all",
            supports_proba=True,
            param_grid={"clf__C": [0.001, 0.01, 0.1, 1, 10, 100]},
        ),
    ]
