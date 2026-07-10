"""Sliding-window feature family used by the endurance-scoring engine.

Ported from usecase2_references/fatigue_classifier_rehab.py. Deliberately a
different (smaller) feature set than src/feature_extraction.py's 14 features:
this one targets short sliding windows over a sustained contraction, not a
whole-channel classification of one file.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

from . import config

FEAT_NAMES = ["RMS", "MAV", "WL", "ZC", "SSC", "MDF", "MNF"]


def features(w: np.ndarray) -> list[float]:
    """RMS, MAV, waveform length, zero-crossings, slope-sign-changes, MDF, MNF."""
    diff = np.diff(w)
    zc = int(np.sum((w[:-1] * w[1:] < 0) & (np.abs(diff) > 1e-3)))
    ssc = int(np.sum(diff[:-1] * diff[1:] < 0))
    fr, p = welch(w, fs=config.FS, nperseg=512)
    cs = np.cumsum(p)
    mdf = float(fr[np.searchsorted(cs, cs[-1] / 2)])
    mnf = float(np.sum(fr * p) / (np.sum(p) + 1e-12))
    return [
        float(np.sqrt(np.mean(w ** 2))), float(np.mean(np.abs(w))),
        float(np.sum(np.abs(diff))), zc, ssc, mdf, mnf,
    ]


def windowize(
    sig: np.ndarray, latent_fatigue: np.ndarray, win_sec: float, overlap: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slice a signal into overlapping windows and extract features + label.

    `latent_fatigue` is a same-length array in [0, 1]; a window is labeled
    fatigued (1) when the fatigue level at its center exceeds 0.5.
    """
    step = int(win_sec * config.FS)
    hop = int(step * (1 - overlap))
    X, y, t_centers = [], [], []
    for a in range(0, len(sig) - step, hop):
        w = sig[a:a + step]
        X.append(features(w))
        y.append(int(latent_fatigue[a + step // 2] > 0.5))
        t_centers.append((a + step / 2) / config.FS)
    return np.array(X), np.array(y), np.array(t_centers)
