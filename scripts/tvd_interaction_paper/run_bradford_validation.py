#!/usr/bin/env python3
"""
run_bradford_validation.py -- ground-truth-free validation on the real
dynamic-contraction recordings (Bradford et al. 2023), for the same gated
(window, column, pair) inputs as run_bradford_realdata.py.

For each input, four delay models are fitted to the 600-sample window and their
residual sum of squares (RSS) between x2 and the warped x1 is recorded over the
same number of samples (540):

  g0 : global order-0  (one constant delay for the whole window)
  g1 : global order-1  (linear delay over the whole window)
  l0 : local order-0   (constant delay per 100-sample segment, 6 segments)
  l1 : local order-1   (linear delay per 100-sample segment, 6 segments)

BIC = n ln(RSS/n) + k ln(n) with k = 1, 2, 6, 12 penalizes the extra
parameters of the local models, so "local beats global" is not merely the
consequence of having more parameters. The angular position, angular velocity
and torque channels (raw analog volts of the dynamometer, microvolts as
stored) are sampled at the window centre for the CV-vs-movement checks.
"""
import sys
import csv
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "tvd_interaction_paper"))

from src.mfcv import _gcc_delay  # noqa: E402
import scripts.optimizer_crlb_paper.crlb_tvd_benchmark as tvdmod  # noqa: E402
from run_bradford_realdata import load_dd, FS, DATA_DIR, WIN, HOP, PAIR_STEP  # noqa: E402

N_SEG = tvdmod.N // tvdmod.SEG_LEN          # 6 disjoint segments
SEG = tvdmod.SEG_LEN
N_EFF = 540                                  # samples entering every RSS
IN_CSV = REPO / "journal_edit" / "tvd_bradford_realdata.csv"
OUT_CSV = REPO / "journal_edit" / "tvd_bradford_validation.csv"

_DD = None


def _fit(cost, d0, *args):
    try:
        res = minimize(cost, d0, args=args, method="BFGS", options={"maxiter": 100})
        return float(res.fun)
    except Exception:
        return np.nan


def _local_rss(x1, x2, order1):
    total = 0.0
    for s in range(N_SEG):
        a, b = s * SEG, (s + 1) * SEG
        d, _ = _gcc_delay(x1[a:b], x2[a:b], max_lag=20)
        d0 = np.array([d, 0.0]) if order1 else np.array([d])
        total += _fit(tvdmod.tvd_cost_local, d0, x1[a:b], x2[a:b])
    return total


def do_task(task):
    start, col, pos = task
    x1 = _DD[pos, col, start:start + WIN]
    x2 = _DD[pos + PAIR_STEP, col, start:start + WIN]
    d, _ = _gcc_delay(x1, x2, max_lag=20)
    rss = {
        "g0": _fit(tvdmod.tvd_cost, np.array([d]), x1, x2),
        "g1": _fit(tvdmod.tvd_cost, np.array([d, 0.0]), x1, x2),
        "l0": _local_rss(x1, x2, False),
        "l1": _local_rss(x1, x2, True),
    }
    e2 = float(np.sum(x2[30:570] ** 2))
    k = {"g0": 1, "g1": 2, "l0": N_SEG, "l1": 2 * N_SEG}
    row = dict(t_s=round(start / FS, 3), col=col, pos=pos)
    for m, v in rss.items():
        row[f"nmse_{m}"] = v / e2
        row[f"bic_{m}"] = N_EFF * np.log(max(v, 1e-30) / N_EFF) + k[m] * np.log(N_EFF)
    return row


def main():
    global _DD
    df = pd.read_csv(IN_CSV)
    rows = []
    only = set(sys.argv[1:])
    for name, g in df.groupby("file", sort=False):
        if only and name not in only:
            continue
        t0 = time.time()
        _DD = load_dd(name)
        raw = pd.read_csv(DATA_DIR / name.split("_")[0] / f"{name}.csv",
                          usecols=["angular_position", "angular_velocity", "torque"])
        tasks = [(int(round(t * FS / HOP)) * HOP, int(c), int(p))
                 for t, c, p in zip(g.t_s, g.col, g.pos)]
        with Pool(28) as pool:
            out = pool.map(do_task, tasks, chunksize=16)
        for r in out:
            centre = int(round(r["t_s"] * FS / HOP)) * HOP + WIN // 2
            r.update(file=name,
                     angle=float(raw.angular_position.iloc[centre]),
                     velocity=float(raw.angular_velocity.iloc[centre]),
                     torque=float(raw.torque.iloc[centre]))
            rows.append(r)
        print(f"{name}: {len(tasks)} inputs ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
