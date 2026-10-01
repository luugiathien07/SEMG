#!/usr/bin/env python3
"""
fma_from_semg.py -- per-item Fugl-Meyer (FMA) grading of the paretic forearm
from HD-sEMG features on PhysioMio, under leave-one-patient-out (LOPO).

Unit of analysis: one gesture segment (one recording x one of 15 non-Rest
gestures, 4 s = four 1-s windows). Its label is the FMA item score (0/1/2)
stored in the raw files' `fma` column (tabulated in fma_items.csv). Models are
trained on 1-s windows and scored per segment (mean window score).

Feature sets (all from features_healthy_vs_impaired.parquet, 64 ch x 7):
  chan    channel-wise features (log for amplitude features), 448 dims
  summ    channel-invariant summaries of each feature over the 64 channels
          (mean, sd, p10, p90, max/mean), 35 dims; immune to array rotation
          and to the mirror image between left and right forearms
  dsumm   summ minus the same patient's healthy-arm mean for the same gesture

Models (gesture one-hot is given to every model):
  gesture       per-gesture prevalence in the training patients
  clinical      gesture + age + days after stroke + dominant arm affected
  chan / summ / dsumm / dsumm_summ / all   gradient boosting on the sets

Outputs (scripts/physiomio_paper/):
  fma_oof.csv           per-segment out-of-fold scores of every model
  fma_summary.csv       pooled + per-patient metrics
  fma_longitudinal.csv  within patient-gesture concordance across sessions
  arm_oof.csv / arm_summary.csv   healthy vs impaired arm, per recording

Run from repo root:
  sEMG_fatigue/.venv/bin/python scripts/physiomio_paper/fma_from_semg.py
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
os.environ.setdefault("OMP_NUM_THREADS", "2")

from joblib import Parallel, delayed
from scipy.stats import spearmanr, wilcoxon
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FEATURES = HERE / "features_healthy_vs_impaired.parquet"
ITEMS = HERE / "fma_items.csv"
META = REPO / "dataset" / "physiomio" / "data" / "metadata.csv"

FEATS = ["rms", "mav", "wl", "zc", "ssc", "mdf", "mnf"]
LOG_FEATS = {"rms", "mav", "wl"}
N_JOBS = 12
SEEDS = [0, 1, 2]


# ---------------------------------------------------------------- features
def load_table() -> pd.DataFrame:
    df = pd.read_parquet(FEATURES)
    meta = pd.read_csv(META)
    items = pd.read_csv(ITEMS)
    items = items[items.gesture != "Rest"]
    df = df.rename(columns={"movement_type": "gesture"})
    df = df.merge(items[["patient", "arm_type", "recording_index", "gesture", "fma"]],
                  left_on=["patient", "arm_type", "session", "gesture"],
                  right_on=["patient", "arm_type", "recording_index", "gesture"],
                  how="left").drop(columns="recording_index")
    m = meta[["patient", "arm_type", "recording_index", "days_after_stroke"]]
    df = df.drop(columns=["days_after_stroke"]).merge(
        m, left_on=["patient", "arm_type", "session"],
        right_on=["patient", "arm_type", "recording_index"]).drop(columns="recording_index")
    df["dominant_affected"] = (df.impaired_arm_side == df.dominant_arm).astype(int)
    df["seg"] = (df.patient + "|" + df.arm_type + "|" + df.session.astype(str)
                 + "|" + df.gesture)
    assert df.fma.notna().all()

    new = {}
    for f in FEATS:
        cols = [f"channel_{i:02d}_{f}" for i in range(1, 65)]
        A = df[cols].to_numpy(float)
        if f in LOG_FEATS:
            A = np.log(np.maximum(A, 1e-9))
            df[cols] = A
        new[f"s_{f}_mean"] = A.mean(1)
        new[f"s_{f}_sd"] = A.std(1)
        new[f"s_{f}_p10"] = np.percentile(A, 10, axis=1)
        new[f"s_{f}_p90"] = np.percentile(A, 90, axis=1)
        # spatial concentration; in log units for amplitude features
        new[f"s_{f}_peak"] = A.max(1) - A.mean(1) if f in LOG_FEATS else \
            A.max(1) / np.maximum(A.mean(1), 1e-9)
    summ = list(new)
    df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    ref = (df[df.arm_type == "healthy_arm"].groupby(["patient", "gesture"])[summ]
           .mean().add_prefix("ref_"))
    R = df[["patient", "gesture"]].join(ref, on=["patient", "gesture"])
    delta = {f"d{c}": df[c].to_numpy() - R[f"ref_{c}"].to_numpy() for c in summ}
    return pd.concat([df, pd.DataFrame(delta, index=df.index)], axis=1)


def feature_sets(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    gest = sorted(df.gesture.unique())
    G = [f"g_{g}" for g in gest]
    return df.join(pd.DataFrame({f"g_{g}": (df.gesture == g).astype(int) for g in gest},
                                index=df.index)), _sets(df, G)


def _sets(df: pd.DataFrame, G: list[str]) -> dict[str, list[str]]:
    clin = ["age_in_years", "days_after_stroke", "dominant_affected"]
    chan = [c for c in df.columns if c.startswith("channel_")]
    summ = [c for c in df.columns if c.startswith("s_")]
    dsumm = [c for c in df.columns if c.startswith("ds_")]
    return {
        "clinical": G + clin,
        "chan": G + chan,
        "summ": G + summ,
        "dsumm": G + dsumm,
        "dsumm_summ": G + summ + dsumm,
        "all": G + clin + summ + dsumm,
    }


# -------------------------------------------------------------- one fold
def fit_fold(df, sets, test_patient, task, seed):
    tr = df.patient != test_patient
    te = ~tr
    y = df["y"].to_numpy()
    out = {"seg": df.loc[te, "seg"].to_numpy()}
    # gesture prior
    prev = df[tr].groupby("gesture").y.mean()
    out["gesture_prior"] = df.loc[te, "gesture"].map(prev).to_numpy()
    for name, cols in sets.items():
        X = df[cols].to_numpy(float)
        if name == "clinical":
            sc = StandardScaler().fit(X[tr])
            m = LogisticRegression(max_iter=3000, C=1.0)
            m.fit(sc.transform(X[tr]), y[tr])
            out[name] = m.predict_proba(sc.transform(X[te]))[:, 1]
            continue
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                           max_leaf_nodes=15, l2_regularization=1.0,
                                           random_state=seed)
        m.fit(X[tr], y[tr])
        out[name] = m.predict_proba(X[te])[:, 1]
        if task == "fma" and name in ("dsumm_summ", "all"):
            r = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                              max_leaf_nodes=15, l2_regularization=1.0,
                                              random_state=seed)
            r.fit(X[tr], df.loc[tr, "fma"].to_numpy())
            out[f"{name}_reg"] = -r.predict(X[te])  # high = more impaired
    return pd.DataFrame(out)


def run_lopo(df, sets, task):
    cache = HERE / f"_{task}_window_oof.parquet"
    if cache.exists():
        W = pd.read_parquet(cache)
        score_cols = [c for c in W.columns if c not in ("seg", "seed")]
        return W.groupby(["seed", "seg"])[score_cols].mean().groupby("seg").mean(), score_cols
    pats = sorted(df.patient.unique())
    frames = []
    for seed in SEEDS:
        t0 = time.time()
        R = Parallel(n_jobs=N_JOBS)(delayed(fit_fold)(df, sets, p, task, seed) for p in pats)
        R = pd.concat(R)
        R["seed"] = seed
        frames.append(R)
        print(f"  [{task}] seed {seed}: {time.time() - t0:.0f}s", flush=True)
    W = pd.concat(frames)
    W.to_parquet(cache, index=False)
    score_cols = [c for c in W.columns if c not in ("seg", "seed")]
    # window -> segment (mean), then average over seeds
    S = W.groupby(["seed", "seg"])[score_cols].mean().groupby("seg").mean()
    return S, score_cols


# ---------------------------------------------------------------- metrics
def strat_auc(D, col, by):
    num = den = 0.0
    for _, g in D.groupby(by):
        if g.y.nunique() < 2:
            continue
        n = g.y.sum() * (1 - g.y).sum()
        num += roc_auc_score(g.y, g[col]) * n
        den += n
    return num / den if den else np.nan


def main():
    t0 = time.time()
    df = load_table()
    df, sets = feature_sets(df)
    print(f"{len(df)} windows, {df.seg.nunique()} segments, {df.patient.nunique()} patients")
    seg_meta = (df.groupby("seg")[["patient", "arm_type", "session", "gesture", "fma",
                                   "days_after_stroke"]].first())

    # ===== Task A: FMA item < 2 on the impaired arm =====
    I = df[df.arm_type == "impaired_arm"].copy()
    I["y"] = (I.fma < 2).astype(int)
    S, cols = run_lopo(I, sets, "fma")
    D = S.join(seg_meta)
    D["y"] = (D.fma < 2).astype(int)
    D.to_csv(HERE / "fma_oof.csv")

    rows = []
    for c in cols:
        r = dict(model=c, auc=roc_auc_score(D.y, D[c]),
                 auc_within_gesture=strat_auc(D, c, "gesture"),
                 auc_within_patient=strat_auc(D, c, "patient"),
                 auc_within_patient_gesture=strat_auc(D, c, ["patient", "gesture"]),
                 spearman_fma=spearmanr(-D[c], D.fma).correlation)
        pa = [roc_auc_score(g.y, g[c]) for _, g in D.groupby("patient") if g.y.nunique() == 2]
        r["per_patient_auc_median"] = float(np.median(pa))
        r["n_patients_auc"] = len(pa)
        rows.append(r)
    A = pd.DataFrame(rows)
    # per-patient AUC comparison against the gesture prior
    pp = {c: np.array([roc_auc_score(g.y, g[c]) for _, g in D.groupby("patient")
                       if g.y.nunique() == 2]) for c in cols}
    A["n_pat_better_than_gesture"] = [int((pp[c] > pp["gesture_prior"]).sum()) for c in cols]
    A["wilcoxon_p_vs_gesture"] = [np.nan if c == "gesture_prior" else
                                  wilcoxon(pp[c], pp["gesture_prior"]).pvalue for c in cols]

    # ===== longitudinal: within patient-gesture, across sessions =====
    L = []
    for (p, g), grp in D.groupby(["patient", "gesture"]):
        if grp.fma.nunique() < 2:
            continue
        v = grp.fma.to_numpy()
        for c in cols:
            s = grp[c].to_numpy()
            conc = disc = ties = 0
            for i in range(len(v)):
                for j in range(i + 1, len(v)):
                    if v[i] == v[j]:
                        continue
                    d = (s[i] - s[j]) * (v[j] - v[i])  # lower fma => higher score
                    conc += d > 0
                    disc += d < 0
                    ties += d == 0
            L.append(dict(patient=p, gesture=g, model=c, conc=conc, disc=disc, ties=ties))
    L = pd.DataFrame(L)
    L.to_csv(HERE / "fma_longitudinal.csv", index=False)
    Lc = L.groupby("model")[["conc", "disc", "ties"]].sum()
    Lc["concordance"] = (Lc.conc + 0.5 * Lc.ties) / (Lc.conc + Lc.disc + Lc.ties)
    Lc["n_pairs"] = Lc.conc + Lc.disc + Lc.ties
    Lc["n_patient_gesture"] = L.groupby("model").size()
    A = A.merge(Lc[["concordance", "n_pairs", "n_patient_gesture"]],
                left_on="model", right_index=True, how="left")
    A.to_csv(HERE / "fma_summary.csv", index=False)
    print(A.round(3).to_string())

    # ===== Task B: healthy vs impaired arm, per recording =====
    B = df.copy()
    B["y"] = (B.arm_type == "impaired_arm").astype(int)
    setsB = {k: v for k, v in sets.items() if k in ("chan", "summ")}
    setsB["side"] = [c for c in sets["clinical"] if c.startswith("g_")] + ["arm_is_left"]
    B["arm_is_left"] = np.where(B.arm_type == "impaired_arm",
                                B.impaired_arm_side == "l", B.impaired_arm_side != "l").astype(int)
    # per-patient centring over all of the patient's recordings (both arms),
    # as in the dataset paper's classifier; transductive for the test patient
    summ = [c for c in sets["summ"] if c.startswith("s_")]
    pn = B[summ] - B.groupby("patient")[summ].transform("mean")
    B = pd.concat([B, pn.add_prefix("pn_")], axis=1)
    setsB["summ_pnorm"] = setsB["side"][:-1] + [f"pn_{c}" for c in summ]
    SB, colsB = run_lopo(B, setsB, "arm")
    DB = SB.join(seg_meta)
    DB["rec"] = DB.patient + "|" + DB.arm_type + "|" + DB.session.astype(str)
    R = DB.groupby("rec")[colsB].mean().join(DB.groupby("rec")[["patient", "arm_type"]].first())
    R["y"] = (R.arm_type == "impaired_arm").astype(int)
    R.to_csv(HERE / "arm_oof.csv")
    rowsB = []
    for c in colsB:
        # paired: does the impaired recording outrank the healthy one in the same patient
        pair = []
        for _, g in R.groupby("patient"):
            h, i = g[g.y == 0][c].to_numpy(), g[g.y == 1][c].to_numpy()
            pair.append(np.mean([(a > b) + 0.5 * (a == b) for a in i for b in h]))
        rowsB.append(dict(model=c, auc_recording=roc_auc_score(R.y, R[c]),
                          auc_segment=roc_auc_score(DB.arm_type == "impaired_arm", DB[c]),
                          within_patient_pair_acc=float(np.mean(pair)),
                          patients_pair_gt_half=int(np.sum(np.array(pair) > 0.5))))
    AB = pd.DataFrame(rowsB)
    AB.to_csv(HERE / "arm_summary.csv", index=False)
    print(AB.round(3).to_string())
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
