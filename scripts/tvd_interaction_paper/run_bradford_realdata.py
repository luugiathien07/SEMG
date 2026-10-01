#!/usr/bin/env python3
"""
run_bradford_realdata.py -- the real-data check flagged as "what remains open"
in PAPER_TVD_Interaction.tex's Discussion: run the SAME four local
sliding-window estimators (order-1 poly / order-0 const-MLE / LAP / GCC) on
real HD-sEMG recorded during DYNAMIC contractions (Bradford, Tweedell & Leahy,
Sci Data 2023, tibialis anterior, isokinetic + isotonic, 126-electrode 9x14
grid, 4 mm IED, 2048 Hz; https://doi.org/10.17605/OSF.IO/9S3U6).

No ground-truth CV exists, so this script only produces per-window CV
estimates from all four estimators on IDENTICAL (window, column, pair) inputs;
scripts/tvd_interaction_paper/analyze_bradford_realdata.py turns them into
precision statistics.

Geometry (established empirically, see analyze script header): the 14-electrode
letter axis (MA..MN) is the fibre axis (lag scales linearly with electrode
spacing, sign consistent); the 9 digits are parallel columns. Signals are
double-differential along the fibre axis (single-differential is biased to ~10
m/s here, double-differential gives ~6 m/s) and each pair is (DD_k, DD_{k+2}),
i.e. 8 mm spacing = the IED already used in the simulations of this series.
DD_k and DD_{k+2} share one electrode (a small caveat, stated in the paper).

Estimator-independent quality gate: only windows whose (a) activity exceeds
GATE_ACT x the recording's rest baseline and (b) GCC peak correlation over the
full window >= GATE_CORR are kept, so all four estimators see the same inputs.

Usage: Motionlab/.venv/bin/python scripts/tvd_interaction_paper/run_bradford_realdata.py [files...]
"""
import sys
import csv
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.mfcv import GridConfig, preprocess, _gcc_delay  # noqa: E402
import scripts.optimizer_crlb_paper.crlb_optimizer_benchmark as optmod  # noqa: E402
import scripts.optimizer_crlb_paper.crlb_tvd_benchmark as tvdmod  # noqa: E402

FS = 2048.0
IED_PAIR_M = 0.008          # 2 x 4 mm grid pitch
REAL_CFG = GridConfig(fs=FS, ied_m=IED_PAIR_M)
# The estimators read CFG as a module global at call time (coarse_grid lives in
# crlb_optimizer_benchmark, everything else in crlb_tvd_benchmark).
optmod.CFG = REAL_CFG
tvdmod.CFG = REAL_CFG

WIN = tvdmod.N              # 600 samples -- same window as the simulations
HOP = 300
PAIR_STEP = 2               # DD_k vs DD_{k+2}
GATE_ACT = 4.0
GATE_CORR = 0.6

DATA_DIR = REPO / "scratch_external_dataset" / "bradford2023"
OUT_CSV = REPO / "journal_edit" / "tvd_bradford_realdata.csv"
SUBJECTS = ["S01", "S02", "S03", "S04", "S05", "S07"]   # the 6 subjects with hdEMG on OSF
CONDITIONS = ["ISOK30", "ISOK90", "ISOK300", "ISOT25_1", "ISOT50_1"]
DEFAULT_FILES = [f"{s}_{c}" for s in SUBJECTS for c in CONDITIONS]
LETTERS = "ABCDEFGHIJKLMN"

_DD = None                  # (n_dd, 9, T) set before forking the pool


def load_dd(name):
    subj = name.split("_")[0]
    df = pd.read_csv(DATA_DIR / subj / f"{name}.csv")
    grid = np.stack([np.stack([df[f"M{l}{d}"].values for d in range(1, 10)])
                     for l in LETTERS])                      # (14, 9, T)
    y = preprocess(grid.reshape(126, -1), REAL_CFG).reshape(14, 9, -1)
    return y[2:] - 2 * y[1:-1] + y[:-2]                       # (12, 9, T)


def window_starts(dd):
    T = dd.shape[2]
    starts = np.arange(0, T - WIN + 1, HOP)
    rms = np.array([np.sqrt(np.mean(dd[:, :, s:s + WIN] ** 2)) for s in starts])
    base = np.percentile(rms, 10)   # rest level; median fails on sustained isotonic holds
    return starts[rms > GATE_ACT * base], base


def do_window(start):
    rows = []
    seg = _DD[:, :, start:start + WIN]
    n_dd = seg.shape[0]
    for col in range(seg.shape[1]):
        for pos in range(n_dd - PAIR_STEP):
            x1, x2 = seg[pos, col], seg[pos + PAIR_STEP, col]
            lag, corr = _gcc_delay(x1, x2, max_lag=15)
            if corr < GATE_CORR:
                continue
            cv = {}
            cv["poly"] = tvdmod.sliding_window_cv_end(x1, x2)
            cv["constmle"] = tvdmod.sliding_window_constmle_cv_end(x1, x2)
            cv["lap"] = tvdmod.sliding_window_lap_cv_end(x1, x2)
            cv["gcc"] = tvdmod.gcc_cv_end_mc(np.stack([x1, x2], axis=0))
            rows.append(dict(t_s=round(start / FS, 3), col=col, pos=pos,
                             gcc_corr=round(corr, 4), gcc_lag_full=round(lag, 4),
                             **{f"cv_{m}": v for m, v in cv.items()}))
    return rows


def main():
    global _DD
    names = sys.argv[1:] or DEFAULT_FILES
    all_rows = []
    for name in names:
        t0 = time.time()
        _DD = load_dd(name)
        starts, base = window_starts(_DD)
        print(f"{name}: {_DD.shape[2] / FS:.1f}s, {len(starts)} active windows "
              f"(rest RMS {base:.1f} uV)", flush=True)
        with Pool(28) as pool:
            for rows in pool.imap_unordered(do_window, starts, chunksize=1):
                for r in rows:
                    r["file"] = name
                    all_rows.append(r)
        print(f"  -> {len(all_rows)} gated pairs so far ({time.time() - t0:.0f}s)", flush=True)

    fields = ["file", "t_s", "col", "pos", "gcc_corr", "gcc_lag_full",
              "cv_poly", "cv_constmle", "cv_lap", "cv_gcc"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print(f"Wrote {OUT_CSV} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
