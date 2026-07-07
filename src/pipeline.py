"""End-to-end orchestration: load -> extract -> select -> train -> evaluate.

Run as a CLI (`python -m emg_fatigue_demo.src.pipeline`) to print the metrics
table, or import ``run_pipeline`` / ``build_feature_table`` from the app.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from . import data_loader as dl
from . import feature_extraction as fe
from . import feature_selection as fs
from .evaluate import ModelResult, evaluate_model
from .models import build_models


def build_feature_table(use_cache: bool = True) -> pd.DataFrame:
    """One row per (file, valid channel): 14 features + metadata columns.

    Columns: FEATURE_NAMES + [subject, condition, label, file, channel].
    """
    if use_cache and config.FEATURE_CACHE.exists():
        return pd.read_parquet(config.FEATURE_CACHE)

    rows: list[dict] = []
    for info in dl.list_files():
        channels = dl.load_channels(info.path)
        mask = dl.valid_channel_mask(channels)
        for ch_idx in np.where(mask)[0]:
            feats = fe.extract_channel_features(channels[:, ch_idx])
            feats.update(
                subject=info.subject,
                condition=info.condition,
                label=info.label,
                file=info.name,
                channel=int(ch_idx),
            )
            rows.append(feats)

    df = pd.DataFrame(rows)
    # Guard against any degenerate channel producing non-finite features.
    df[config.FEATURE_NAMES] = df[config.FEATURE_NAMES].replace(
        [np.inf, -np.inf], np.nan)
    df = df.dropna(subset=config.FEATURE_NAMES).reset_index(drop=True)

    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.FEATURE_CACHE, index=False)
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


def run_pipeline(use_cache: bool = True) -> PipelineOutput:
    df = build_feature_table(use_cache=use_cache)
    train, test = split_train_test(df)

    X_train_full = train[config.FEATURE_NAMES]
    y_train = train["label"].to_numpy()
    X_test_full = test[config.FEATURE_NAMES]
    y_test = test["label"].to_numpy()

    ranked, relevance = fs.mrmr_rank(X_train_full, y_train)
    top_k = ranked[:config.TOP_K_FEATURES]

    results: list[ModelResult] = []
    for spec in build_models():
        feats = top_k if spec.feature_set == "top3" else config.FEATURE_NAMES
        res = evaluate_model(
            spec,
            X_train_full[feats].to_numpy(),
            y_train,
            X_test_full[feats].to_numpy(),
            y_test,
            features_used=list(feats),
        )
        results.append(res)

    return PipelineOutput(
        feature_table=df, train=train, test=test,
        ranked_features=ranked, relevance=relevance, top_k=top_k,
        results=results,
    )


def _print_report(out: PipelineOutput) -> None:
    df = out.feature_table
    print("=" * 68)
    print("EMG FATIGUE CLASSIFICATION — DEMO PIPELINE")
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
    print(f"Top-{config.TOP_K_FEATURES} (SVM): {out.top_k}")
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
