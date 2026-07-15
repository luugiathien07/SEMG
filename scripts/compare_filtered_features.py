"""Experiment: does bandpass+notch filtering (src.signal_processing.filter_signal,
the same mfcv.preprocess() filter now used in the Usecase 1 demo) help or hurt
the fatigue classifier, if applied before feature extraction instead of on raw
channels?

Builds a second feature table identical to ``loso.build_feature_table()`` except
each channel is filtered first, then runs the same LOSO protocol
(``loso.run_loso``) and the same single paper-matching split
(``pipeline.split_train_test`` / ``config.TEST_SUBJECT``) on both tables side by
side. Does not modify ``loso.py``, ``pipeline.py``, or ``feature_extraction.py``
— purely additive, reads the existing raw feature cache and builds a new,
separate cache for the filtered version.

    python -m scripts.compare_filtered_features            # cached (build filtered table once)
    python -m scripts.compare_filtered_features --rebuild  # force re-extraction of both tables

Writes the comparison report to ``compare_filtered_vs_raw_results.txt`` at the
project root.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import config, data_loader as dl, feature_extraction as fe, loso  # noqa: E402
from src import signal_processing as sp  # noqa: E402

FILTERED_CACHE = loso.FULL_DATA_DIR / "cache" / "loso_features_filtered.parquet"


def build_filtered_feature_table(use_cache: bool = True) -> pd.DataFrame:
    """Same as ``loso.build_feature_table()``, except every channel is run
    through ``signal_processing.filter_signal`` (bandpass 20-400Hz + notch
    50Hz) before ``feature_extraction.extract_channel_features`` — otherwise
    identical extraction/labeling logic, so the two tables are directly
    comparable column-for-column."""
    if use_cache and FILTERED_CACHE.exists():
        return pd.read_parquet(FILTERED_CACHE)

    rows: list[dict] = []
    paths = sorted(loso.FULL_DATA_DIR.rglob("*_emg.csv"))
    for i, path in enumerate(paths):
        m = loso._NAME_RE.match(path.stem)
        if not m:
            continue
        subject, condition = int(m.group(1)), m.group(2)
        channels = dl.load_channels(path)
        mask = dl.valid_channel_mask(channels)
        print(f"[{i + 1}/{len(paths)}] filtering + extracting {path.stem} "
              f"({int(mask.sum())} valid channels)")
        for ch in np.where(mask)[0]:
            filtered = sp.filter_signal(channels[:, ch], config.FS)
            feats = fe.extract_channel_features(filtered)
            feats.update(
                subject=subject,
                condition=condition,
                file=path.stem,
                channel=int(ch),
                label_mvc=loso.label_mvc(condition),
                label_paper=loso.label_paper(condition),
            )
            rows.append(feats)

    df = pd.DataFrame(rows)
    df[config.FEATURE_NAMES] = df[config.FEATURE_NAMES].replace(
        [np.inf, -np.inf], np.nan)
    df = df.dropna(subset=config.FEATURE_NAMES).reset_index(drop=True)

    FILTERED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FILTERED_CACHE, index=False)
    return df


def _fmt(v) -> str:
    return "  nan " if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:6.3f}"


def _single_split_metrics(df: pd.DataFrame) -> dict[str, dict]:
    """Paper-matching single split: train on config.TRAIN_SUBJECTS, test on
    config.TEST_SUBJECT — mirrors pipeline.run_pipeline's headline numbers,
    using loso.build_loso_models (same estimators pipeline.py evaluates)."""
    train = df[df["subject"].isin(config.TRAIN_SUBJECTS)]
    test = df[df["subject"] == config.TEST_SUBJECT]
    X_tr = train[config.FEATURE_NAMES].to_numpy()
    y_tr = train["label_mvc"].to_numpy()
    X_te = test[config.FEATURE_NAMES].to_numpy()
    y_te = test["label_mvc"].to_numpy()

    out: dict[str, dict] = {}
    for name, model in loso.build_loso_models().items():
        est = clone(model)
        est.fit(X_tr, y_tr)
        y_pred = est.predict(X_te)
        scores = (est.predict_proba(X_te)[:, 1]
                  if hasattr(est, "predict_proba") else None)
        out[name] = loso._metrics(y_te, y_pred, scores, pos=1)
    return out


def report(df_raw: pd.DataFrame, df_filt: pd.DataFrame, out_lines: list[str]) -> None:
    def emit(s: str = "") -> None:
        print(s)
        out_lines.append(s)

    emit("=" * 90)
    emit(f"SINGLE PAPER-MATCHING SPLIT — train={config.TRAIN_SUBJECTS}  "
         f"test={config.TEST_SUBJECT}  (label=mvc, positive=Fatigue)")
    emit("=" * 90)
    raw_m = _single_split_metrics(df_raw)
    filt_m = _single_split_metrics(df_filt)
    emit(f"  {'Model':<14}{'':>2}{'Acc(raw)':>9}{'Acc(filt)':>10}{'  ':>2}"
         f"{'F1(raw)':>9}{'F1(filt)':>10}{'  ':>2}{'AUC(raw)':>9}{'AUC(filt)':>10}")
    for name in raw_m:
        r, f = raw_m[name], filt_m[name]
        emit(f"  {name:<14}{'':>2}{_fmt(r['acc']):>9}{_fmt(f['acc']):>10}{'  ':>2}"
             f"{_fmt(r['f1']):>9}{_fmt(f['f1']):>10}{'  ':>2}{_fmt(r['auc']):>9}{_fmt(f['auc']):>10}")
    emit(f"\n  Paper reference: KNN F1={config.PAPER_REFERENCE['KNN_F1']}, "
         f"AUC~{config.PAPER_REFERENCE['KNN_AUC']}")
    emit("")

    for scheme in ("mvc", "paper"):
        emit("=" * 90)
        emit(f"POOLED LOSO — label={scheme}, positive=Fatigue(1), all {len(loso.LABELERS)} labeling(s) shown")
        emit("=" * 90)
        res_raw = loso.run_loso(df_raw, scheme)
        res_filt = loso.run_loso(df_filt, scheme)
        emit(f"  {'Model':<14}{'':>2}{'Acc(raw)':>9}{'Acc(filt)':>10}{'  ':>2}"
             f"{'Prec(raw)':>10}{'Prec(filt)':>11}{'  ':>2}"
             f"{'Rec(raw)':>9}{'Rec(filt)':>10}{'  ':>2}"
             f"{'F1(raw)':>9}{'F1(filt)':>10}{'  ':>2}{'AUC(raw)':>9}{'AUC(filt)':>10}")
        for name in res_raw.models:
            r = res_raw.pooled[name][1]
            f = res_filt.pooled[name][1]
            emit(
                f"  {name:<14}{'':>2}{_fmt(r['acc']):>9}{_fmt(f['acc']):>10}{'  ':>2}"
                f"{_fmt(r['prec']):>10}{_fmt(f['prec']):>11}{'  ':>2}"
                f"{_fmt(r['rec']):>9}{_fmt(f['rec']):>10}{'  ':>2}"
                f"{_fmt(r['f1']):>9}{_fmt(f['f1']):>10}{'  ':>2}"
                f"{_fmt(r['auc']):>9}{_fmt(f['auc']):>10}"
            )
        emit("")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true",
                    help="force re-extraction of both feature tables (ignore caches)")
    args = ap.parse_args()

    print("Building raw feature table (cached in loso.build_feature_table)...")
    df_raw = loso.build_feature_table(use_cache=not args.rebuild)
    print("Building filtered feature table...")
    df_filt = build_filtered_feature_table(use_cache=not args.rebuild)

    out_lines: list[str] = []
    report(df_raw, df_filt, out_lines)

    out_path = PROJECT_ROOT / "compare_filtered_vs_raw_results.txt"
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"\nSaved report to {out_path}")


if __name__ == "__main__":
    main()
