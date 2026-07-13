"""End-to-end orchestration: load -> extract -> select -> train -> evaluate.

Runs the BME 2024 paper's leave-one-subject-out protocol on the full dataset
(``dataset/full_emg_data``, one folder per subject): train on every subject
except ``config.TEST_SUBJECT`` (default 6) and test on that held-out subject.

Classifiers are the paper-faithful pipelines from ``loso.build_loso_models``
(KNN 1-NN / KNN 5-NN / SVM linear / LDA — each standardized, using all 14
features, no SMOTE or threshold tuning). With the project's MVC>60 labeling this
reproduces the headline held-out subject-6 result: KNN(1-NN) Acc=0.977,
F1=0.955, AUC=0.984.

Run as a CLI (``python -m src.pipeline``) to print the metrics table, or import
``run_pipeline`` / ``build_feature_table`` from the app.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
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
from . import feature_selection as fs
from . import loso
from .evaluate import ModelResult


def build_feature_table(use_cache: bool = True) -> pd.DataFrame:
    """One row per (file, valid channel): 14 features + metadata columns.

    Delegates extraction to ``loso.build_feature_table`` (all subjects, both
    labeling columns) and exposes the project's MVC>60 labeling as ``label`` so
    the rest of the pipeline and the Streamlit app stay parameter-free.
    Columns: FEATURE_NAMES + [subject, condition, file, channel,
    label_mvc, label_paper, label].
    """
    df = loso.build_feature_table(use_cache=use_cache).copy()
    df["label"] = df["label_mvc"]
    return df


def split_train_test(df: pd.DataFrame):
    """Leave-one-subject-out split into train/test feature frames."""
    train = df[df["subject"].isin(config.TRAIN_SUBJECTS)].reset_index(drop=True)
    test = df[df["subject"] == config.TEST_SUBJECT].reset_index(drop=True)
    return train, test


@dataclass
class PipelineOutput:
    feature_table: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame
    ranked_features: list[str]
    relevance: dict[str, float]
    top_k: list[str]
    results: list[ModelResult]


def _evaluate_clean(name, estimator, X_train, y_train, X_test, y_test,
                    features_used: list[str]) -> ModelResult:
    """Fit the paper-faithful model on train, evaluate on the held-out subject.

    No resampling / threshold tuning: predictions use the model's default 0.5
    cut-off, and ROC scores are the model's P(Fatigue). Positive class = 1.
    """
    # Cross-validated train accuracy (context for over/under-fit), capped by the
    # smallest class count so it never crashes on tiny folds.
    min_class = int(np.min(np.bincount(y_train.astype(int))))
    n_splits = max(2, min(config.CV_FOLDS, min_class))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=config.RANDOM_STATE)
    cv_pred = cross_val_predict(clone(estimator), X_train, y_train, cv=skf)
    cv_acc = float(accuracy_score(y_train, cv_pred))

    est = clone(estimator)
    est.fit(X_train, y_train)
    y_pred = est.predict(X_test)
    scores = (est.predict_proba(X_test)[:, 1]
              if hasattr(est, "predict_proba") else None)

    auc, roc = None, None
    if scores is not None and len(np.unique(y_test)) == 2:
        auc = float(roc_auc_score(y_test, scores))
        fpr, tpr, _ = roc_curve(y_test, scores, pos_label=1)
        roc = (fpr, tpr)

    return ModelResult(
        name=name,
        accuracy=float(accuracy_score(y_test, y_pred)),
        precision=float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        recall=float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        f1=float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        cv_accuracy=cv_acc,
        auc=auc,
        confusion=confusion_matrix(y_test, y_pred, labels=[0, 1]),
        roc=roc,
        features_used=features_used,
        fitted_estimator=est,
        best_params={},
        threshold=0.5,
    )


def run_pipeline(use_cache: bool = True) -> PipelineOutput:
    df = build_feature_table(use_cache=use_cache)
    train, test = split_train_test(df)

    X_train = train[config.FEATURE_NAMES]
    y_train = train["label"].to_numpy()
    X_test = test[config.FEATURE_NAMES]
    y_test = test["label"].to_numpy()

    ranked, relevance = fs.mrmr_rank(X_train, y_train)
    top_k = ranked[:config.TOP_K_FEATURES]

    feats = list(config.FEATURE_NAMES)
    results: list[ModelResult] = []
    for name, estimator in loso.build_loso_models().items():
        results.append(_evaluate_clean(
            name, estimator,
            X_train.to_numpy(), y_train, X_test.to_numpy(), y_test,
            features_used=feats,
        ))

    return PipelineOutput(
        feature_table=df, train=train, test=test,
        ranked_features=ranked, relevance=relevance, top_k=top_k,
        results=results,
    )


def _print_report(out: PipelineOutput) -> None:
    df = out.feature_table
    print("=" * 68)
    print("EMG FATIGUE CLASSIFICATION — LOSO PIPELINE")
    print("=" * 68)
    print(f"Samples (file x channel): {len(df)}")
    print(f"  Normal={int((df.label == 0).sum())}  "
          f"Fatigue={int((df.label == 1).sum())}")
    print(f"  Files={df.file.nunique()}  Subjects={sorted(df.subject.unique())}")
    print(f"Train subjects {config.TRAIN_SUBJECTS}: {len(out.train)} samples "
          f"(N={int((out.train.label==0).sum())}, "
          f"F={int((out.train.label==1).sum())})")
    print(f"Test subject {config.TEST_SUBJECT}: {len(out.test)} samples "
          f"(N={int((out.test.label==0).sum())}, "
          f"F={int((out.test.label==1).sum())})")
    print("-" * 68)
    print(f"mRMR ranking: {out.ranked_features}")
    print(f"Top-{config.TOP_K_FEATURES}: {out.top_k}")
    print("-" * 68)
    header = f"{'Model':<14}{'Acc':>8}{'Prec':>8}{'Rec':>8}{'F1':>8}{'CV-Acc':>8}{'AUC':>8}"
    print(header)
    for r in out.results:
        auc = f"{r.auc:.3f}" if r.auc is not None else "  -  "
        print(f"{r.name:<14}{r.accuracy:>8.3f}{r.precision:>8.3f}"
              f"{r.recall:>8.3f}{r.f1:>8.3f}{r.cv_accuracy:>8.3f}{auc:>8}")
    print("=" * 68)
    print(f"Paper reference (BME 2024): KNN F1~{config.PAPER_REFERENCE['KNN_F1']}, "
          f"AUC~{config.PAPER_REFERENCE['KNN_AUC']}")


if __name__ == "__main__":
    _print_report(run_pipeline(use_cache=False))
