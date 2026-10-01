#!/usr/bin/env python3
"""
train_classifiers.py -- healthy_arm vs impaired_arm classification on
PhysioMio, comparing 5 models with leave-one-PATIENT-out (LOPO) cross-
validation (48 groups), using the 448-dim (64 channels x 7 classical
features) feature table from extract_features.py.

Exploratory/technical comparison only -- not a validated clinical study
(no IRB review here; see PhysioMio LICENSE Sec. 2 on Research Use / IRB
approval, and Sec. 5 on citation/no-raw-data-in-public-outputs, both
already respected: this script only ever touches the pre-computed,
de-identified feature parquet, never raw signal, and dataset/physiomio/ is
gitignored).

Run from repo root: `python3 scripts/physiomio_paper/train_classifiers.py`
Output: scripts/physiomio_paper/model_comparison.csv
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES_PATH = Path(__file__).resolve().parent / "features_healthy_vs_impaired.parquet"
OUT_PATH = Path(__file__).resolve().parent / "model_comparison.csv"


def main():
    df = pd.read_parquet(FEATURES_PATH)
    print(f"Loaded {FEATURES_PATH}: {df.shape}")
    print(df["arm_type"].value_counts())
    print(f"{df['patient'].nunique()} patients")

    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    # SVM (RBF) deliberately omitted: libsvm's ~O(n^2)-O(n^3) scaling in
    # SAMPLE COUNT made a single LOPO fold (~19,300 train samples) take
    # well over 3 hours combined across 48 folds with no partial output,
    # even without probability=True -- impractical at this dataset size.

    meta_cols = {"patient", "arm_type", "session", "movement_type",
                 "age_in_years", "days_after_stroke", "impaired_arm_side",
                 "dominant_arm"}
    feat_cols = [c for c in df.columns if c not in meta_cols]
    print(f"{len(feat_cols)} features")

    X = df[feat_cols].to_numpy()
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = (df["arm_type"] == "impaired_arm").astype(int).to_numpy()
    groups = df["patient"].to_numpy()

    models = {
        "KNN (k=7)": KNeighborsClassifier(n_neighbors=7),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=14,
                                                  class_weight="balanced", random_state=0,
                                                  n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=0),
    }

    logo = LeaveOneGroupOut()
    n_groups = len(np.unique(groups))
    print(f"\n=== LOPO (leave-one-patient-out), {n_groups} patients, "
          f"{len(y)} windows ===")

    results = []
    for name, model in models.items():
        t0 = time.time()
        accs, f1s, aucs = [], [], []
        for train_idx, test_idx in logo.split(X, y, groups):
            ytr, yte = y[train_idx], y[test_idx]
            if len(np.unique(ytr)) < 2:
                continue  # this patient's data was the only source of one class
            scaler = StandardScaler().fit(X[train_idx])
            Xtr, Xte = scaler.transform(X[train_idx]), scaler.transform(X[test_idx])
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            accs.append(accuracy_score(yte, pred))
            f1s.append(f1_score(yte, pred, zero_division=0))
            if len(np.unique(yte)) > 1:
                if hasattr(model, "predict_proba"):
                    score = model.predict_proba(Xte)[:, 1]
                else:
                    score = model.decision_function(Xte)
                aucs.append(roc_auc_score(yte, score))
        dt = time.time() - t0
        results.append(dict(
            model=name,
            mean_acc=np.mean(accs), sd_acc=np.std(accs),
            mean_f1=np.mean(f1s), sd_f1=np.std(f1s),
            mean_auc=np.nanmean(aucs) if aucs else np.nan,
            sd_auc=np.nanstd(aucs) if aucs else np.nan,
            n_folds=len(accs), n_auc_folds=len(aucs), fit_time_s=dt,
        ))
        print(f"{name:22s}: acc={np.mean(accs):.3f}+/-{np.std(accs):.3f}  "
              f"F1={np.mean(f1s):.3f}+/-{np.std(f1s):.3f}  "
              f"AUC={np.nanmean(aucs) if aucs else float('nan'):.3f}+/-"
              f"{np.nanstd(aucs) if aucs else float('nan'):.3f}  "
              f"({dt:.1f}s, {len(accs)} folds)")

    res = pd.DataFrame(results)
    res.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
