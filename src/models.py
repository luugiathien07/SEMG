"""Classifier factory matching the MATLAB hyperparameters (spec section 4.4).

Each model is returned together with the feature set it should use:
- SVM         : top-3 mRMR features (linear kernel, standardized)
- KNN/LDA/Tree: all 14 features

"RF" in the MATLAB code is actually a single decision tree (``fitctree``), so it
is named DecisionTree here.
"""
from __future__ import annotations

from dataclasses import dataclass

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


def build_models() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="SVM",
            estimator=Pipeline([
                ("scale", StandardScaler()),
                ("clf", CalibratedClassifierCV(
                    SVC(kernel="linear", C=1.0,
                        random_state=config.RANDOM_STATE),
                    ensemble=False)),
            ]),
            feature_set="top3",
            supports_proba=True,
        ),
        ModelSpec(
            name="KNN",
            estimator=Pipeline([
                ("scale", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=1, metric="euclidean",
                                             weights="uniform")),
            ]),
            feature_set="all",
            supports_proba=True,
        ),
        ModelSpec(
            name="LDA",
            estimator=LinearDiscriminantAnalysis(solver="svd"),
            feature_set="all",
            supports_proba=True,
        ),
        ModelSpec(
            name="DecisionTree",
            estimator=DecisionTreeClassifier(
                criterion="gini",
                max_leaf_nodes=21,          # MaxNumSplits=20 -> 21 leaves
                random_state=config.RANDOM_STATE,
            ),
            feature_set="all",
            supports_proba=True,
        ),
    ]
