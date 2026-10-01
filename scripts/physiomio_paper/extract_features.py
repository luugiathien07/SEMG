#!/usr/bin/env python3
"""
extract_features.py -- classical sEMG feature extraction for the PhysioMio
healthy-arm-vs-impaired-arm classification study.

For each (patient, arm_type, session) recording, slices each non-Rest gesture
segment into non-overlapping 1.0 s windows and computes, per channel (64),
the same 7-feature set already used for the P(fatigue) classifier in this
repo's usecase2 (RMS, MAV, Waveform Length, Zero Crossings, Slope Sign
Changes, MDF, MNF -- see script_thuyet_minh_usecase2.md), giving a 64*7=448-
dim feature vector per window. Labels: arm_type (healthy_arm/impaired_arm),
patient (for LOSO grouping), movement_type, and metadata.csv's clinical
covariates (age, days_after_stroke, impaired_arm side, dominant_arm).

This is a DE-IDENTIFIED, aggregate-feature output: no raw EMG samples are
written out, only per-window summary statistics, consistent with the
PhysioMio Data Use Agreement's "no raw data/identifiable information in
public outputs" clause (LICENSE, Sec. 5) -- this repo keeps dataset/physiomio/
gitignored regardless (see PhysioMio DUA, Sec. 2-4: Research Use requires
IRB approval; this script is exploratory/technical, not a submitted study).

Output: scripts/physiomio_paper/features_healthy_vs_impaired.parquet
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "dataset" / "physiomio" / "data"
OUT_PATH = Path(__file__).resolve().parent / "features_healthy_vs_impaired.parquet"

FS = 2048.0
WIN_S = 1.0
WIN_N = int(WIN_S * FS)
N_CHANNELS = 64
CHANNEL_COLS = [f"channel_{i:02d}" for i in range(1, N_CHANNELS + 1)]


def waveform_length(w: np.ndarray) -> float:
    return float(np.sum(np.abs(np.diff(w))))


def zero_crossings(w: np.ndarray, thresh: float = 1e-4) -> int:
    s = np.sign(w)
    s[s == 0] = 1
    crossings = (s[:-1] * s[1:] < 0) & (np.abs(np.diff(w)) > thresh)
    return int(np.sum(crossings))


def slope_sign_changes(w: np.ndarray, thresh: float = 1e-4) -> int:
    d = np.diff(w)
    s = np.sign(d)
    s[s == 0] = 1
    changes = (s[:-1] * s[1:] < 0) & (np.abs(d[:-1]) > thresh) & (np.abs(d[1:]) > thresh)
    return int(np.sum(changes))


def median_mean_freq(w: np.ndarray, fs: float) -> tuple[float, float]:
    freqs, psd = welch(w, fs=fs, nperseg=min(len(w), 256))
    cum = np.cumsum(psd)
    total = cum[-1]
    if total <= 0:
        return 0.0, 0.0
    mdf = float(freqs[np.searchsorted(cum, total / 2)])
    mnf = float(np.sum(freqs * psd) / total)
    return mdf, mnf


def window_features(w: np.ndarray, fs: float) -> dict:
    mdf, mnf = median_mean_freq(w, fs)
    return dict(
        rms=float(np.sqrt(np.mean(w ** 2))),
        mav=float(np.mean(np.abs(w))),
        wl=waveform_length(w),
        zc=zero_crossings(w),
        ssc=slope_sign_changes(w),
        mdf=mdf,
        mnf=mnf,
    )


def process_file(path: Path, patient: str, arm_type: str, session: int) -> list[dict]:
    df = pd.read_parquet(path)
    rows = []
    for gesture, g in df.groupby("movement_type", sort=False):
        if gesture == "Rest":
            continue
        arr = g[CHANNEL_COLS].to_numpy()  # (n, 64)
        n = arr.shape[0]
        for start in range(0, n - WIN_N + 1, WIN_N):  # non-overlapping
            seg = arr[start:start + WIN_N]  # (WIN_N, 64)
            feat = {}
            for ci, col in enumerate(CHANNEL_COLS):
                f = window_features(seg[:, ci], FS)
                for k, v in f.items():
                    feat[f"{col}_{k}"] = v
            feat.update(patient=patient, arm_type=arm_type, session=session,
                        movement_type=gesture)
            rows.append(feat)
    return rows


def main():
    meta_path = DATA_DIR / "metadata.csv"
    meta = pd.read_csv(meta_path)
    print(f"metadata.csv: {len(meta)} recordings")

    all_rows = []
    t0 = time.time()
    for i, row in meta.iterrows():
        fp = DATA_DIR / row["file_path"]
        if not fp.exists():
            continue
        rows = process_file(fp, row["patient"], row["arm_type"], row["recording_index"])
        for r in rows:
            r["age_in_years"] = row["age_in_years"]
            r["days_after_stroke"] = row["days_after_stroke"]
            r["impaired_arm_side"] = row["impaired_arm"]
            r["dominant_arm"] = row["dominant_arm"]
        all_rows.extend(rows)
        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{len(meta)}] files, {len(all_rows)} windows so far, "
                  f"{time.time() - t0:.0f}s elapsed")

    out = pd.DataFrame(all_rows)
    out.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}: {out.shape}")
    print(out["arm_type"].value_counts())
    print(out["patient"].nunique(), "patients")


if __name__ == "__main__":
    main()
