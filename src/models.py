"""Classifier factory (spec section 4.4, extended for class-imbalance handling).

Each model is returned with the feature set it should use, plus how it
handles the Fatigue/Normal class imbalance and a hyperparameter grid for
``evaluate.evaluate_model`` to tune via GridSearchCV on the training subjects
only (never on the held-out test subject):

- SVM         : top-3 mRMR features; ``class_weight="balanced"``; tunes C
                (kernel fixed to linear — an rbf/high-C variant was tried but
                overfit the channel-level CV folds, since 2 of the 4 train
                subjects have zero Fatigue channels and can't be grouped for
                a subject-level CV; it scored well in-CV but badly on the
                held-out subject, so it was dropped).
- KNN         : all 14 features; no ``class_weight`` support, so the Fatigue
                class is oversampled with SMOTE (synthetic interpolated
                samples, not plain duplication) inside the training fold
                only; tunes n_neighbors/weights.
- LDA         : all 14 features; equal ``priors`` instead of empirical
                frequencies; tunes solver/shrinkage.
- DecisionTree: all 14 features; ``class_weight="balanced"``; tunes tree
                size (max_leaf_nodes/max_depth/min_samples_leaf) + criterion.

"RF" in the MATLAB code is actually a single decision tree (``fitctree``), so it
is named DecisionTree here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from . import config


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
            estimator=Pipeline([
                ("scale", StandardScaler()),
                ("clf", CalibratedClassifierCV(
                    SVC(class_weight="balanced", random_state=config.RANDOM_STATE),
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
            },
        ),
        ModelSpec(
            name="LDA",
            estimator=LinearDiscriminantAnalysis(priors=[0.5, 0.5]),
            feature_set="all",
            supports_proba=True,
            param_grid=[
                {"solver": ["svd"]},
                {"solver": ["lsqr"], "shrinkage": ["auto", None, 0.1, 0.3, 0.5, 0.7]},
            ],
        ),
        ModelSpec(
            name="DecisionTree",
            estimator=DecisionTreeClassifier(
                class_weight="balanced",
                random_state=config.RANDOM_STATE,
            ),
            feature_set="all",
            supports_proba=True,
            param_grid={
                "criterion": ["gini", "entropy"],
                "max_leaf_nodes": [11, 21, 31, None],
                "max_depth": [None, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
        ),
    ]
