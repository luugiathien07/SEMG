"""Endurance-score illustration: fatigue onset within a sustained contraction.

Ported from usecase2_references/fatigue_classifier_rehab.py, stripped of
print side-effects. This is a self-contained synthetic simulation (it does
NOT read the physiomio dataset) — a sustained isometric contraction with a
known latent fatigue curve, used only to illustrate the concept "endurance
score = time until fatigue onset" as a companion to the session-based
recovery tracking in data_loader.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from . import config
from .features import windowize

WIN_SEC, OVERLAP = 1.0, 0.5
RMS0, MDF0 = 20.0, 105.0        # baseline levels before fatigue onset
RANDOM_STATE = 11


@dataclass
class SessionResult:
    session: int
    t_half: float
    endurance_sec: float
    pct_nonfatigue: float


@dataclass
class SessionResultWithProba:
    """SessionResult enriched with per-window fatigue probability and time."""
    session: int
    t_half: float
    endurance_sec: float
    pct_nonfatigue: float
    proba: np.ndarray        # P(fatigue) per window, shape (n_windows,)
    t_centers: np.ndarray    # time center of each window (seconds)


def session_t_half(n_sessions: int = 8, random_state: int = RANDOM_STATE) -> np.ndarray:
    """Per-session fatigue-onset time (s): early sessions fatigue fast, later
    sessions fatigue slower, illustrating recovery across sessions."""
    rng = np.random.default_rng(random_state)
    return np.linspace(6, 24, n_sessions) + rng.normal(0, 1.0, n_sessions)


def _bandpass_noise(rng: np.random.Generator, n: int, center: float, rms: float) -> np.ndarray:
    lo, hi = 20.0, float(np.clip(2 * center + 40, 120, 480))
    x = rng.standard_normal(n)
    sos = butter(4, [lo, hi], btype="band", fs=config.FS, output="sos")
    x = sosfiltfilt(sos, x)
    return (x * rms / (np.sqrt(np.mean(x ** 2)) + 1e-9)).astype(np.float32)


def simulate_sustained(
    t_half: float, rng: np.random.Generator, dur: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sustained contraction signal + its latent fatigue curve f(t) in [0, 1]."""
    n = int(dur * config.FS)
    t = np.arange(n) / config.FS
    f = 1 - np.exp(-t / (t_half / np.log(2)))       # f=0.5 at t=t_half
    step = int(WIN_SEC * config.FS)
    sig = np.zeros(n, dtype=np.float32)
    for a in range(0, n - step, step // 2):
        fc = f[a + step // 2]
        rms = RMS0 * (1 + 0.6 * fc)                 # fatigue -> amplitude up
        center = MDF0 * (1 - 0.35 * fc)              # fatigue -> MDF down
        sig[a:a + step] = _bandpass_noise(rng, step, center, rms)
    return sig, f


def run(n_sessions: int = 8, random_state: int = RANDOM_STATE) -> tuple[list[SessionResult], float]:
    """Simulate n_sessions sustained contractions, train a KNN fatigue-onset
    classifier on all windows pooled together, then score each session's
    endurance = time (s) until predicted fatigue onset."""
    rng = np.random.default_rng(random_state)
    t_halves = session_t_half(n_sessions, random_state)

    raw_sessions = []
    X_all, y_all = [], []
    for s, th in enumerate(t_halves, start=1):
        sig, f = simulate_sustained(float(th), rng)
        X, y, t_centers = windowize(sig, f, WIN_SEC, OVERLAP)
        raw_sessions.append({"session": s, "t_half": float(th), "X": X, "t_centers": t_centers})
        X_all.append(X)
        y_all.append(y)
    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=0.3, random_state=0, stratify=y_all,
    )
    scaler = StandardScaler().fit(X_tr)
    knn = KNeighborsClassifier(n_neighbors=7).fit(scaler.transform(X_tr), y_tr)
    f1 = f1_score(y_te, knn.predict(scaler.transform(X_te)))

    results = []
    for ss in raw_sessions:
        proba = knn.predict_proba(scaler.transform(ss["X"]))[:, 1]
        fatigued = proba >= 0.5
        onset_idx = int(np.argmax(fatigued)) if np.any(fatigued) else len(proba) - 1
        endurance_sec = float(ss["t_centers"][onset_idx])
        pct_nonfatigue = float(np.mean(proba < 0.5) * 100)
        results.append(SessionResult(
            session=ss["session"], t_half=ss["t_half"],
            endurance_sec=endurance_sec, pct_nonfatigue=pct_nonfatigue,
        ))
    return results, float(f1)


def run_with_proba(
    n_sessions: int = 8, random_state: int = RANDOM_STATE,
) -> tuple[list[SessionResultWithProba], float]:
    """Like :func:`run`, but each result also carries the per-window fatigue
    probability ``proba`` and matching ``t_centers`` — needed for the
    in-session P(fatigue) chart."""
    rng = np.random.default_rng(random_state)
    t_halves = session_t_half(n_sessions, random_state)

    raw_sessions: list[dict] = []
    X_all, y_all = [], []
    for s, th in enumerate(t_halves, start=1):
        sig, f = simulate_sustained(float(th), rng)
        X, y, t_centers = windowize(sig, f, WIN_SEC, OVERLAP)
        raw_sessions.append({
            "session": s, "t_half": float(th),
            "X": X, "t_centers": t_centers,
        })
        X_all.append(X)
        y_all.append(y)
    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=0.3, random_state=0, stratify=y_all,
    )
    scaler = StandardScaler().fit(X_tr)
    knn = KNeighborsClassifier(n_neighbors=7).fit(scaler.transform(X_tr), y_tr)
    f1 = f1_score(y_te, knn.predict(scaler.transform(X_te)))

    results: list[SessionResultWithProba] = []
    for ss in raw_sessions:
        proba = knn.predict_proba(scaler.transform(ss["X"]))[:, 1]
        fatigued = proba >= 0.5
        onset_idx = int(np.argmax(fatigued)) if np.any(fatigued) else len(proba) - 1
        endurance_sec = float(ss["t_centers"][onset_idx])
        pct_nonfatigue = float(np.mean(proba < 0.5) * 100)
        results.append(SessionResultWithProba(
            session=ss["session"], t_half=ss["t_half"],
            endurance_sec=endurance_sec, pct_nonfatigue=pct_nonfatigue,
            proba=proba, t_centers=ss["t_centers"],
        ))
    return results, float(f1)
