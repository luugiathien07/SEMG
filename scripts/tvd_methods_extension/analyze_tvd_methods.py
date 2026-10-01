#!/usr/bin/env python3
"""
analyze_tvd_methods.py -- summary tables for run_tvd_methods.py.

Reads journal_edit/tvd_methods_extension_data.csv and prints, for every
source x ground truth, the trajectory RMSE of the % CV error per method and SNR,
its split into squared bias and variance (averaged over evaluation instants),
the share of clipped estimates, and the ranking of methods by mean RMSE.
The multi-MU-population source is nearly silent over its first 100 samples, so
it is also reported restricted to evaluation instants n >= 150.

    Motionlab/.venv/bin/python scripts/tvd_methods_extension/analyze_tvd_methods.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
df = pd.read_csv(REPO / "journal_edit" / "tvd_methods_extension_data.csv")
ORDER = ["poly_local", "legendre5", "gcc", "lap", "lap_kalman", "cohf", "dll", "dp"]


def summarize(d):
    g = d.assign(bias2=d.mean_err_pct ** 2).groupby(["method", "snr_db"])
    s = g.agg(bias2=("bias2", "mean"), var=("var_err_pct2", "mean"), clipped=("frac_clipped", "mean"))
    s["rmse"] = np.sqrt(s.bias2 + s["var"])
    s["bias_share"] = s.bias2 / (s.bias2 + s["var"])
    return s


print(f"M = {df.M.iloc[0]} realizations per cell; RMSE over the CV(n) trajectory, % of true CV")
rt = pd.read_csv(REPO / "journal_edit" / "tvd_methods_extension_runtime.csv").set_index("method")
cells = [(s, t, None) for s in ["colored", "muap", "multipop"] for t in ["ramp", "sinusoid"]]
cells += [("multipop", t, 150) for t in ["ramp", "sinusoid"]]
rank_rows = []
for src, truth, nmin in cells:
    d = df[(df.source == src) & (df.truth == truth)]
    if nmin:
        d = d[d.n >= nmin]
    s = summarize(d)
    tag = f"{src} x {truth}" + (f" (n >= {nmin})" if nmin else "")
    print(f"\n=== {tag} ===")
    rm = s.rmse.unstack("snr_db").reindex(ORDER)
    bs = s.bias_share.unstack("snr_db").reindex(ORDER)
    cl = s.clipped.groupby("method").mean().reindex(ORDER)
    tab = rm.round(2).astype(str)
    for c in tab.columns:
        tab[c] = rm[c].map("{:.2f}".format) + " (" + (100 * bs[c]).map("{:.0f}".format) + "%)"
    tab["clipped"] = (100 * cl).map("{:.1f}%".format)
    tab["ms/real."] = rt.reindex(ORDER).mean_ms_per_realization.map("{:.0f}".format)
    tab.columns = [f"{c:g} dB" if isinstance(c, float) else c for c in tab.columns]
    print("RMSE % (bias^2 share of MSE in brackets)")
    print(tab.to_string())
    mean_rmse = rm.mean(axis=1).sort_values()
    for r, (m, v) in enumerate(mean_rmse.items(), 1):
        rank_rows.append(dict(cell=tag, method=m, rank=r, mean_rmse=v))

rk = pd.DataFrame(rank_rows)
print("\n=== mean rank across the 6 full cells (lower is better) ===")
full = rk[~rk.cell.str.contains("n >=")]
print(full.groupby("method")["rank"].mean().sort_values().round(2).to_string())
