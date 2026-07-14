"""FastAPI backend serving UC2 data for the React frontend.

Run:
    uvicorn api_server:app --reload --port 8000
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports work.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="sEMG Demo API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy-loaded heavy modules (avoid import-time compute) ───────────
_endurance_cache: dict | None = None
_recovery_cache: dict = {}


def _get_endurance():
    global _endurance_cache
    if _endurance_cache is not None:
        return _endurance_cache

    from src.rehab import endurance as end

    results, f1 = end.run_with_proba()
    sessions = []
    for r in results:
        sessions.append({
            "session": r.session,
            "t_half": round(r.t_half, 2),
            "endurance_sec": round(r.endurance_sec, 2),
            "pct_nonfatigue": round(r.pct_nonfatigue, 2),
            "proba": [round(float(p), 4) for p in r.proba],
            "t_centers": [round(float(t), 3) for t in r.t_centers],
        })
    _endurance_cache = {"sessions": sessions, "f1": round(f1, 4)}
    return _endurance_cache


# ── UC2 Endpoints ───────────────────────────────────────────────────

@app.get("/api/uc2/endurance")
def uc2_endurance():
    """Endurance results + KNN F1 for all simulated sessions."""
    return _get_endurance()


@app.get("/api/uc2/patients")
def uc2_patients():
    from src.rehab import data_loader as rdl
    return {"patients": rdl.list_patients()}


@app.get("/api/uc2/recovery/{patient}")
def uc2_recovery(patient: str):
    if patient in _recovery_cache:
        return _recovery_cache[patient]

    from src.rehab import data_loader as rdl

    patients = rdl.list_patients()
    if patient not in patients:
        raise HTTPException(404, f"Patient '{patient}' not found")

    df = rdl.compute_recovery_trend(patient)
    if df.empty:
        return {"rows": [], "baseline_rms": None, "baseline_mdf": None}

    result = {
        "rows": df[["Buổi", "RMS", "MDF", "Symmetry (%)"]].to_dict(orient="records"),
        "baseline_rms": float(df["baseline_rms"].iloc[0]) if df["baseline_rms"].iloc[0] is not None else None,
        "baseline_mdf": float(df["baseline_mdf"].iloc[0]) if df["baseline_mdf"].iloc[0] is not None else None,
    }
    _recovery_cache[patient] = result
    return result


@app.get("/api/uc2/sessions/{patient}/{arm}")
def uc2_sessions(patient: str, arm: str):
    from src.rehab import data_loader as rdl

    sessions = rdl.list_sessions(patient, arm)
    return {
        "sessions": [
            {"session_no": s.session_no, "filename": s.path.name}
            for s in sessions
        ]
    }


@app.get("/api/uc2/signal/{patient}/{arm}/{session_no}")
def uc2_signal(patient: str, arm: str, session_no: int, channel: str = "channel_01"):
    from src.rehab import data_loader as rdl
    from src.rehab import config as rc
    import numpy as np

    sessions = rdl.list_sessions(patient, arm)
    target = next((s for s in sessions if s.session_no == session_no), None)
    if target is None:
        raise HTTPException(404, f"Session {session_no} not found")

    df = rdl.load_session(target.path)
    movements = list(df["movement_type"].unique())

    # Return metadata + first movement by default
    result = {"movements": movements, "channels": rc.CHANNEL_COLUMNS, "fs": rc.FS}

    if channel not in rc.CHANNEL_COLUMNS:
        channel = rc.CHANNEL_COLUMNS[0]

    # Return signal per movement (limited to prevent huge payloads)
    signals = {}
    for mv in movements:
        seg = df[df["movement_type"] == mv]
        x = seg[channel].to_numpy()
        t = (np.arange(len(x)) / rc.FS).tolist()
        # Downsample if too many points (> 5000)
        if len(x) > 5000:
            step = len(x) // 5000
            x = x[::step]
            t = t[::step]
        signals[mv] = {
            "time": [round(v, 4) for v in t],
            "amplitude": [round(float(v), 2) for v in x],
        }
    result["signals"] = signals
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
