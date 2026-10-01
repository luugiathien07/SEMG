#!/usr/bin/env python3
"""
analyze_all_methods.py -- statistics for run_all_methods.py.

Reads journal_edit/tvd_all_methods_parts/*.npz (per-realization % CV errors)
and writes journal_edit/tvd_all_methods_summary.csv with, per source x truth x
SNR x method:
  rmse        trajectory RMSE of the % CV error (all realizations)
  bias2_share squared bias / MSE, bias taken per instant across realizations
  median_abs  median |error|
  outlier     share of (realization, instant) errors with |error| > 25 %
  var         error variance, mean over instants
  var_crlb    var / CRLB, colored source only: the two-channel scalar
              Slepian-Bangs bound for a 100-sample window at the true CV(n),
              propagated to % CV by the delta method (the bound the per-window
              order-0 estimators estimate against; a reference level for the
              others)
The multi-MU-population source is scored on n >= 150 (it is silent over its
first 100 samples). Prints the mean rank per method over the 24 cells, on RMSE
and on median |error|.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.optimizer_crlb_paper.crlb_optimizer_benchmark import crlb_theta  # noqa: E402
from scripts.optimizer_crlb_paper.crlb_tvd_benchmark import CFG, true_theta, true_theta_sinusoidal  # noqa: E402

TRUTHS = {"ramp": true_theta, "sinusoid": true_theta_sinusoidal}
rows = []
for p in sorted((REPO / "journal_edit" / "tvd_all_methods_parts").glob("*.npz")):
    z = np.load(p)
    src, truth, snr = str(z["source"]), str(z["truth"]), float(z["snr_db"])
    n_eval, E = z["n_eval"], z["err"].astype(float)          # (M, methods, n)
    keep = n_eval >= 150 if src == "multipop" else np.ones(len(n_eval), bool)
    E, n_eval = E[:, :, keep], n_eval[keep]
    crlb_pct2 = None
    if src == "colored":
        th = TRUTHS[truth](n_eval)
        cv = CFG.ied_m * CFG.fs / th
        v = np.array([crlb_theta(c, snr, n=100, fs=CFG.fs, ied_m=CFG.ied_m, n_rows=2) for c in cv])
        crlb_pct2 = (100.0 / th) ** 2 * v
    for i, meth in enumerate(z["methods"]):
        e = E[:, i, :]
        bias2 = np.mean(e.mean(axis=0) ** 2)
        var = e.var(axis=0)
        mse = np.mean(e ** 2)
        rows.append(dict(source=src, truth=truth, snr_db=snr, method=str(meth), M=e.shape[0],
                         rmse=np.sqrt(mse), bias2_share=bias2 / mse, median_abs=np.median(np.abs(e)),
                         outlier=np.mean(np.abs(e) > 25), var=var.mean(),
                         var_crlb=np.mean(var / crlb_pct2) if crlb_pct2 is not None else np.nan,
                         ms=1000 * z["time_s"][:, i].mean()))
S = pd.DataFrame(rows)
S.to_csv(REPO / "journal_edit" / "tvd_all_methods_summary.csv", index=False)
print(f"{S.groupby(['source', 'truth', 'snr_db']).ngroups} cells, M = {S.M.min()}; "
      "wrote journal_edit/tvd_all_methods_summary.csv")
for k in ("rmse", "median_abs"):
    r = S.assign(rank=S.groupby(["source", "truth", "snr_db"])[k].rank())
    print(f"\nmean rank on {k} (lower is better):")
    print(r.groupby("method")["rank"].mean().sort_values().round(2).to_string())
print("\nRMSE % by source (mean over truths and SNRs), outlier share, ms per realization:")
t = S.pivot_table(index="method", columns="source", values="rmse", aggfunc="mean")
t["outlier_%"] = 100 * S.groupby("method").outlier.mean()
t["ms"] = S.groupby("method").ms.mean()
print(t.sort_values("colored").round(2).to_string())
print("\nColored source, error variance / CRLB (100-sample, 2-channel scalar bound), by SNR:")
c = S[S.source == "colored"].pivot_table(index="method", columns="snr_db", values="var_crlb", aggfunc="mean")
print(c.round(1).to_string())
