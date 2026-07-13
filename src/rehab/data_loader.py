"""Read the "physiomio" rehab dataset.

Layout on disk: `dataset/physiomio/<patient>/<arm>/<session>.parquet`, each
file holding one session (concatenated gesture segments) with columns
`time, channel_01..channel_64, fma, movement_type` (see src/rehab/config.py
for schema notes). This module only reads what is already there — it does
not generate data. See usecase2_references/simulate_semg_rehab.py for a
(simplified) reference generator of illustrative data in the same family.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

from . import config


@dataclass(frozen=True)
class SessionInfo:
    patient: str
    arm: str
    session_no: int
    path: Path

    @property
    def name(self) -> str:
        return f"{self.patient}/{self.arm}/{self.path.name}"


def list_patients() -> list[str]:
    """Patient directory names under PHYSIOMIO_DIR, sorted."""
    if not config.PHYSIOMIO_DIR.is_dir():
        return []
    return sorted(p.name for p in config.PHYSIOMIO_DIR.iterdir() if p.is_dir())


def list_sessions(patient: str, arm: str) -> list[SessionInfo]:
    """Sessions for one patient/arm, ordered by session number (filename)."""
    arm_dir = config.PHYSIOMIO_DIR / patient / arm
    if not arm_dir.is_dir():
        return []
    sessions = []
    for p in sorted(arm_dir.glob("*.parquet")):
        try:
            session_no = int(p.stem)
        except ValueError:
            continue
        sessions.append(SessionInfo(patient=patient, arm=arm, session_no=session_no, path=p))
    return sorted(sessions, key=lambda s: s.session_no)


def load_session(path: Path) -> pd.DataFrame:
    """Load one session parquet file: channel_01..channel_64 + movement_type."""
    return pd.read_parquet(path)


def session_rms_mdf(df: pd.DataFrame) -> tuple[float, float]:
    """Average RMS and median-frequency across channels, excluding Rest.

    Ports `session_mdf()` from simulate_semg_rehab.py and adds the matching
    RMS computation so both recovery markers can be read off one call.
    """
    active = df[df["movement_type"] != "Rest"]
    rms_vals, mdf_vals = [], []
    for c in config.CHANNEL_COLUMNS:
        x = active[c].to_numpy()
        rms_vals.append(float(np.sqrt(np.mean(x ** 2))))
        f, p = welch(x, fs=config.FS, nperseg=1024)
        csum = np.cumsum(p)
        mdf_vals.append(float(f[np.searchsorted(csum, csum[-1] / 2)]))
    return float(np.mean(rms_vals)), float(np.mean(mdf_vals))


def compute_recovery_trend(patient: str) -> pd.DataFrame:
    """Build a per-session recovery trend for *patient*'s impaired arm.

    Returns a DataFrame with columns:
        Buổi, RMS, MDF, Symmetry (%), baseline_rms, baseline_mdf
    Symmetry = RMS_impaired / RMS_healthy × 100 (capped at 100).
    """
    # healthy baseline (average across healthy sessions)
    healthy_sessions = list_sessions(patient, "healthy_arm")
    if healthy_sessions:
        h_rms, h_mdf = [], []
        for s in healthy_sessions:
            df = load_session(s.path)
            r, m = session_rms_mdf(df)
            h_rms.append(r)
            h_mdf.append(m)
        baseline_rms = float(np.mean(h_rms))
        baseline_mdf = float(np.mean(h_mdf))
    else:
        baseline_rms, baseline_mdf = None, None

    impaired_sessions = list_sessions(patient, "impaired_arm")
    rows = []
    for s in impaired_sessions:
        df = load_session(s.path)
        rms, mdf = session_rms_mdf(df)
        symmetry = min(rms / baseline_rms * 100, 100.0) if baseline_rms else None
        rows.append({
            "Buổi": s.session_no,
            "RMS": round(rms, 2),
            "MDF": round(mdf, 1),
            "Symmetry (%)": round(symmetry, 1) if symmetry is not None else None,
            "baseline_rms": baseline_rms,
            "baseline_mdf": baseline_mdf,
        })
    return pd.DataFrame(rows)
