#!/usr/bin/env python3
"""
run_all_methods_bradford.py -- the sixteen estimators of run_all_methods.py on
real dynamic-contraction HD-sEMG (Bradford, Tweedell & Leahy, Sci Data 2023,
tibialis anterior, isokinetic + isotonic; OSF 10.17605/OSF.IO/9S3U6).

Inputs are exactly the gated (file, window, column, pair) inputs of
journal_edit/tvd_bradford_realdata.csv (run_bradford_realdata.py): 600-sample
windows at 2048 Hz, double-differential pairs (DD_k, DD_{k+2}) along the fibre
axis, 8 mm apart, kept when active and when the full-window GCC peak
correlation is >= 0.6. The pair is oriented so that the second channel lags
(swapped when the stored full-window GCC lag is negative), because DP, DLL and
the delay clip assume a positive delay. Every estimator sees the same oriented
pair and returns the delay at the 51 instants n = 50, 60, ..., 550 used in the
simulations; the delay is converted to MFCV with IED = 8 mm and Fs = 2048 Hz.

Output: journal_edit/tvd_all_methods_bradford.npz with
  cv      (inputs, methods, 51) float32, MFCV in m/s (delay clipped to 1..12
          samples, as in the simulations)
  sec     (inputs, methods) run time in seconds
  rec, t_s, col, pos    the input keys, methods, n_eval

    Motionlab/.venv/bin/python scripts/tvd_methods_extension/run_all_methods_bradford.py [n_random_inputs]
"""
from __future__ import annotations

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "tvd_methods_extension"))
sys.path.insert(0, str(REPO / "scripts" / "tvd_interaction_paper"))

import run_bradford_realdata as rb  # noqa: E402  (sets optmod/tvdmod CFG to 2048 Hz, 8 mm)
import run_tvd_methods as base  # noqa: E402
import run_all_methods as am  # noqa: E402

# the per-window CohF estimator binds fs as a default argument; the order-0
# MLE's grid reads crlb_optimizer_benchmark.CFG, already switched by rb
base.CFG = rb.REAL_CFG
base._cohf_one.__defaults__ = (rb.FS, 256)

FS, IED = rb.FS, rb.IED_PAIR_M
OUT = REPO / "journal_edit" / "tvd_all_methods_bradford.npz"
_DD = None


def _one(args):
    i, start, col, pos, flip = args
    x1, x2 = _DD[pos, col, start:start + am.N], _DD[pos + rb.PAIR_STEP, col, start:start + am.N]
    if flip:
        x1, x2 = x2, x1
    rng = np.random.default_rng([7, i])
    cv = np.empty((len(am.METHODS), len(am.N_EVAL)), np.float32)
    dt = np.empty(len(am.METHODS))
    for k, m in enumerate(am.METHODS):
        t0 = time.perf_counter()
        th = am.est_global_sa(x1, x2, rng) if m == "global_sa" else am.EST[m](x1, x2)
        dt[k] = time.perf_counter() - t0
        th = np.clip(np.abs(np.nan_to_num(np.asarray(th, float), nan=am.THETA_MAX)),
                     am.THETA_MIN, am.THETA_MAX)
        cv[k] = IED * FS / th
    return i, cv, dt


def main():
    global _DD
    keys = pd.read_csv(REPO / "journal_edit" / "tvd_bradford_realdata.csv",
                       usecols=["file", "t_s", "col", "pos", "gcc_lag_full"])
    if len(sys.argv) > 1:  # smoke test: a random subset, written to a separate file
        keys = keys.sample(int(sys.argv[1]), random_state=0).sort_index().reset_index(drop=True)
        global OUT
        OUT = OUT.with_name("tvd_all_methods_bradford_subset.npz")
    CV = np.full((len(keys), len(am.METHODS), len(am.N_EVAL)), np.nan, np.float32)
    SEC = np.full((len(keys), len(am.METHODS)), np.nan)
    t_all = time.time()
    for fi, (name, g) in enumerate(keys.groupby("file", sort=False), 1):
        _DD = rb.load_dd(name)
        jobs = [(i, int(round(r.t_s * FS / rb.HOP)) * rb.HOP, int(r.col), int(r.pos), r.gcc_lag_full < 0)
                for i, r in zip(g.index, g.itertuples())]
        with Pool(28) as pool:
            for i, cv, dt in pool.imap_unordered(_one, jobs, chunksize=4):
                CV[i], SEC[i] = cv, dt
        print(f"[{fi}] {name}: {len(g)} inputs ({time.time() - t_all:.0f}s)", flush=True)
    np.savez_compressed(OUT, cv=CV, sec=SEC, methods=np.array(am.METHODS), n_eval=am.N_EVAL,
                        rec=keys.file.values, t_s=keys.t_s.values, col=keys.col.values,
                        pos=keys.pos.values)
    print(f"Wrote {OUT} ({len(keys)} inputs, {time.time() - t_all:.0f}s)")


if __name__ == "__main__":
    main()
