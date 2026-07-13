"""Leave-One-Subject-Out (LOSO) training & evaluation over the full dataset.

Reproduces the BME 2024 paper's protocol ("Detection of Muscles Fatigue Through
Surface EMG Signals", Huu et al.): for each subject in turn, train on all the
*other* subjects and test on the held-out one. Reports metrics **per held-out
subject (per fold)** and **pooled** (predictions from every fold concatenated
into one confusion matrix) so the single headline number in the paper can be
placed in context.

Two design choices are made explicit and selectable, because both matter a lot:

1. Labeling scheme (``LABELERS``)
   - ``"mvc"``   : %MVC in the filename > ``config.FATIGUE_MVC_THRESHOLD`` (60)
                   -> Fatigue.  This is the project's confirmed labeling
                   (90% MVC -> Fatigue, 10%-after-fatigue -> Normal) and the
                   ICACE 2019 physiological criterion (fatigue when force
                   exceeds 60% MVC).
   - ``"paper"`` : the BME 2024 experimental design — the 70% MVC exhaustion
                   test and the 10%-MVC post-fatigue recording are Fatigue;
                   the pre-fatigue 10/20/40/60/90% MVC tests are Normal.

2. Positive-class convention for the reported metrics
   Fatigue(1) is the clinically correct positive class, so that is the primary
   report. The paper's MATLAB code (``KNNClassification.m``) indexes the
   confusion matrix as if Normal(0) were positive, which inflates F1 toward the
   majority class; that convention is reported alongside for comparison, not as
   the headline.

Models mirror the paper (SVM linear / LDA / KNN). KNN follows the reference
MATLAB exactly: 1-nearest-neighbour, Euclidean, features standardized, **all 14
features** (the mRMR block in ``KNNClassification.m`` is commented out). A
5-NN variant is included as a more stable alternative. No SMOTE / threshold
tuning here — those live in ``models.py`` for the Streamlit demo; LOSO keeps the
paper's plain protocol so the numbers are directly comparable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config
from . import data_loader as dl
from . import feature_extraction as fe
from . import feature_selection as fs

# --- Where to read the raw per-subject folders and cache extracted features ---
FULL_DATA_DIR = config.PROJECT_ROOT / "dataset" / "full_emg_data"
LOSO_CACHE = FULL_DATA_DIR / "cache" / "loso_features.parquet"

_NAME_RE = re.compile(r"^Sujet_(\d+)_(.+)_emg$", re.IGNORECASE)
_NUM_RE = re.compile(r"\d+")


# --------------------------------------------------------------------------- #
# Labeling
# --------------------------------------------------------------------------- #
def label_mvc(condition: str) -> int:
    """Project labeling: %MVC in the condition > threshold (60) -> Fatigue."""
    m = _NUM_RE.search(condition)
    mvc = int(m.group()) if m else 0
    return 1 if mvc > config.FATIGUE_MVC_THRESHOLD else 0


def label_paper(condition: str) -> int:
    """BME 2024 design: the 70% exhaustion test and the post-fatigue recording
    (``*fatigue*`` in the condition) are Fatigue; pre-fatigue %MVC are Normal."""
    c = condition.lower()
    if "fatigue" in c:
        return 1
    m = _NUM_RE.search(c)
    mvc = int(m.group()) if m else 0
    return 1 if mvc == 70 else 0


LABELERS = {"mvc": label_mvc, "paper": label_paper}
LABEL_COLS = {"mvc": "label_mvc", "paper": "label_paper"}


# --------------------------------------------------------------------------- #
# Feature table (one row per valid channel of every *_emg.csv, all subjects)
# --------------------------------------------------------------------------- #
def build_feature_table(use_cache: bool = True) -> pd.DataFrame:
    """14 features per valid channel for every subject, with both label columns.

    Columns: FEATURE_NAMES + [subject, condition, file, channel,
    label_mvc, label_paper].
    """
    if use_cache and LOSO_CACHE.exists():
        return pd.read_parquet(LOSO_CACHE)

    rows: list[dict] = []
    for path in sorted(FULL_DATA_DIR.rglob("*_emg.csv")):
        m = _NAME_RE.match(path.stem)
        if not m:
            continue
        subject, condition = int(m.group(1)), m.group(2)
        channels = dl.load_channels(path)
        mask = dl.valid_channel_mask(channels)
        for ch in np.where(mask)[0]:
            feats = fe.extract_channel_features(channels[:, ch])
            feats.update(
                subject=subject,
                condition=condition,
                file=path.stem,
                channel=int(ch),
                label_mvc=label_mvc(condition),
                label_paper=label_paper(condition),
            )
            rows.append(feats)

    df = pd.DataFrame(rows)
    df[config.FEATURE_NAMES] = df[config.FEATURE_NAMES].replace(
        [np.inf, -np.inf], np.nan)
    df = df.dropna(subset=config.FEATURE_NAMES).reset_index(drop=True)

    LOSO_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(LOSO_CACHE, index=False)
    return df


# --------------------------------------------------------------------------- #
# Models (paper-faithful, no resampling/threshold tuning)
# --------------------------------------------------------------------------- #
def build_loso_models() -> dict[str, Pipeline]:
    """Standardized pipelines mirroring the paper's classifiers."""
    return {
        # Reference MATLAB KNN: 1-NN, Euclidean, standardized, all 14 features.
        "KNN(1NN)": Pipeline([
            ("scale", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=1, metric="euclidean")),
        ]),
        "KNN(5NN)": Pipeline([
            ("scale", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5, metric="euclidean")),
        ]),
        "SVM(linear)": Pipeline([
            ("scale", StandardScaler()),
            ("clf", SVC(kernel="linear", probability=True,
                        random_state=config.RANDOM_STATE)),
        ]),
        "LDA": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis()),
        ]),
    }


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _metrics(y_true, y_pred, scores, pos: int) -> dict:
    """Standard metrics for the given positive class. ``scores`` are P(class=1)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "prec": float(precision_score(y_true, y_pred, pos_label=pos, zero_division=0)),
        "rec": float(recall_score(y_true, y_pred, pos_label=pos, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=pos, zero_division=0)),
        "auc": float("nan"),
        "n": int(len(y_true)),
        "n_pos": int(np.sum(y_true == pos)),
    }
    if scores is not None and len(np.unique(y_true)) == 2:
        s = scores if pos == 1 else -scores
        out["auc"] = float(roc_auc_score((y_true == pos).astype(int), s))
    return out


@dataclass
class LosoResult:
    label_scheme: str
    models: list[str]
    subjects: list[int]
    # per_fold[model][subject] -> metrics dict (Fatigue-positive)
    per_fold: dict = field(default_factory=dict)
    # pooled[model][pos] -> metrics dict, pos in {0,1}
    pooled: dict = field(default_factory=dict)
    # aggregated y_true/y_pred/scores per model (for confusion matrices)
    pooled_arrays: dict = field(default_factory=dict)
    selected_features: list[str] = field(default_factory=list)


def evaluate_subject(
    df: pd.DataFrame,
    subject: int,
    label_scheme: str = "mvc",
) -> dict[str, dict]:
    """Single held-out-subject evaluation (the paper's actual protocol).

    Trains every model in ``build_loso_models`` on all subjects except
    ``subject`` and tests on ``subject``. Returns ``{model_name: metrics}`` with
    Fatigue(1) as the positive class. Default ``label_scheme="mvc"`` (>60% MVC ->
    Fatigue) is the labeling that reproduces the headline subject-6 numbers
    (KNN 1-NN: Acc=0.977, F1=0.955, AUC=0.984).
    """
    label_col = LABEL_COLS[label_scheme]
    tr = df[df["subject"] != subject]
    te = df[df["subject"] == subject]
    if te.empty:
        raise ValueError(f"subject {subject} has no data")
    y_tr = tr[label_col].to_numpy()
    if len(np.unique(y_tr)) < 2:
        raise ValueError(f"training set is single-class with subject {subject} held out")

    X_tr = tr[config.FEATURE_NAMES].to_numpy()
    X_te = te[config.FEATURE_NAMES].to_numpy()
    y_te = te[label_col].to_numpy()

    results: dict[str, dict] = {}
    for name, model in build_loso_models().items():
        est = clone(model)
        est.fit(X_tr, y_tr)
        y_pred = est.predict(X_te)
        scores = (est.predict_proba(X_te)[:, 1]
                  if hasattr(est, "predict_proba") else None)
        results[name] = _metrics(y_te, y_pred, scores, pos=1)
    return results


def run_loso(df: pd.DataFrame, label_scheme: str) -> LosoResult:
    """Full leave-one-subject-out over every subject with data for both classes.

    Folds whose *training* set would be single-class are skipped (nothing to
    learn); folds whose *test* subject is single-class still run and are reported
    (their per-fold F1/AUC may be degenerate, which is expected and noted).
    """
    label_col = LABEL_COLS[label_scheme]
    models = build_loso_models()
    subjects = sorted(int(s) for s in df["subject"].unique())

    per_fold = {name: {} for name in models}
    agg = {name: {"yt": [], "yp": [], "sc": []} for name in models}

    for test_s in subjects:
        tr = df[df["subject"] != test_s]
        te = df[df["subject"] == test_s]
        y_tr = tr[label_col].to_numpy()
        if len(np.unique(y_tr)) < 2:      # nothing to learn from
            for name in models:
                per_fold[name][test_s] = None
            continue
        X_tr = tr[config.FEATURE_NAMES].to_numpy()
        X_te = te[config.FEATURE_NAMES].to_numpy()
        y_te = te[label_col].to_numpy()
        for name, model in models.items():
            est = clone(model)
            est.fit(X_tr, y_tr)
            y_pred = est.predict(X_te)
            scores = (est.predict_proba(X_te)[:, 1]
                      if hasattr(est, "predict_proba") else None)
            per_fold[name][test_s] = _metrics(y_te, y_pred, scores, pos=1)
            agg[name]["yt"].append(y_te)
            agg[name]["yp"].append(y_pred)
            if scores is not None:
                agg[name]["sc"].append(scores)

    pooled = {name: {} for name in models}
    pooled_arrays = {}
    for name in models:
        yt = np.concatenate(agg[name]["yt"])
        yp = np.concatenate(agg[name]["yp"])
        sc = np.concatenate(agg[name]["sc"]) if agg[name]["sc"] else None
        pooled_arrays[name] = (yt, yp, sc)
        for pos in (1, 0):
            pooled[name][pos] = _metrics(yt, yp, sc, pos=pos)

    # mRMR top-3 on the pooled features (for reporting; classifiers use all 14).
    ranked, _ = fs.mrmr_rank(df[config.FEATURE_NAMES], df[label_col].to_numpy())
    selected = ranked[:config.TOP_K_FEATURES]

    return LosoResult(
        label_scheme=label_scheme,
        models=list(models),
        subjects=subjects,
        per_fold=per_fold,
        pooled=pooled,
        pooled_arrays=pooled_arrays,
        selected_features=selected,
    )
