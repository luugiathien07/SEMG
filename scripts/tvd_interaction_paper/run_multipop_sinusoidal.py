#!/usr/bin/env python3
"""
run_multipop_sinusoidal.py -- a SECOND, independent realistic source for the
(realistic source, sinusoidal ground truth) cell, to check whether the LAP
reversal found with the single-MUAP template (run_muap_sinusoidal.py) is a
property of that specific template or survives a genuinely different
realistic source.

Source: one channel (index 4, middle of a 9-channel array) from the legacy
multi-motor-unit-population simulator already used, for the SAME purpose
(a second realistic source, substituted into this manuscript series' own
Monte Carlo loop with no new signal-generation code, per
PAPER_Multichannel_TVD.tex's own realistic_source_multichannel.py and
PAPER_Optimizer_CRLB.tex's own "Directions for future work" paragraph). This
is qualitatively different from the single-MUAP template: many
asynchronously-recruited motor units superimposed, not one MU tiled to fill
the window -- a real interference pattern, not a periodic repeat of one
waveform.

Reuses the exact same four local estimators (poly/constmle/LAP/GCC) and the
same global BFGS fit as run_muap_sinusoidal.py, so the three sources
(colored noise, single-MUAP, multi-MU-population) are directly comparable
at the same M.

Outputs:
  - tvd_multipop_sinusoidal_global_data.csv
  - tvd_multipop_sinusoidal_sliding_data.csv
"""
import sys
import time
import csv
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "multichannel_paper"))

import scripts.optimizer_crlb_paper.crlb_tvd_benchmark as tvdmod  # noqa: E402
from scripts.optimizer_crlb_paper.crlb_tvd_benchmark import (  # noqa: E402
    CFG, N, SNR_VALUES, SEG_LEN, SEG_OVERLAP,
    true_theta_sinusoidal, make_pair, run_cell, run_variant,
    sliding_window_cv_end, sliding_window_constmle_cv_end,
    sliding_window_lap_cv_end, gcc_cv_end_mc, METHOD_LABELS,
)
from realistic_source_multichannel import load_source as load_multipop_source  # noqa: E402

M_FAST = 100  # was 60 (before that, matched to M_FAST=12; see run_muap_sinusoidal.py)
M_SLOW = 50   # was 30
tvdmod.REPS_FAST = M_FAST
tvdmod.REPS_SLOW = M_SLOW


def run_global():
    source = load_multipop_source()
    assert source.shape == (N,), source.shape
    run_variant(
        variant=f"multi-MU-population x sinusoidal (global order-1 fit), M={M_FAST}/{M_SLOW}",
        theta_fn=true_theta_sinusoidal,
        source_fixed=source,
        csv_name="tvd_multipop_sinusoidal_global_data.csv",
        pdf_name="tvd_multipop_sinusoidal_global.pdf",
        plot_title="TVD, multi-MU-population source x sinusoidal ground truth",
    )


def run_sliding():
    theta_fn = true_theta_sinusoidal
    cv_ref = CFG.ied_m * CFG.fs / theta_fn(N - 1)
    source = load_multipop_source()
    methods = ("poly", "constmle", "lap", "gcc")
    agg = {m: {} for m in methods}
    agg_time = {m: {} for m in methods}
    rows = []

    print(f"\n--- variant: multi-MU-population x sinusoidal, sliding-window fits "
          f"(SEG_LEN={SEG_LEN}, overlap={SEG_OVERLAP:.0%}, cv_ref={cv_ref:.4f} m/s, "
          f"M={M_FAST}) ---")
    for snr in SNR_VALUES:
        seed0 = int(snr * 137)
        ests = {m: [] for m in methods}
        times = {m: [] for m in methods}
        for r in range(M_FAST):
            rng = np.random.default_rng(seed0 + r)
            x1, x2 = make_pair(rng, snr, source, theta_fn)

            t1 = time.perf_counter()
            ests["poly"].append(sliding_window_cv_end(x1, x2))
            times["poly"].append((time.perf_counter() - t1) * 1e3)

            t1 = time.perf_counter()
            ests["constmle"].append(sliding_window_constmle_cv_end(x1, x2))
            times["constmle"].append((time.perf_counter() - t1) * 1e3)

            t1 = time.perf_counter()
            ests["lap"].append(sliding_window_lap_cv_end(x1, x2))
            times["lap"].append((time.perf_counter() - t1) * 1e3)

            t1 = time.perf_counter()
            ests["gcc"].append(gcc_cv_end_mc(np.stack([x1, x2], axis=0)))
            times["gcc"].append((time.perf_counter() - t1) * 1e3)

        for method in methods:
            e = np.asarray(ests[method])
            e = e[np.isfinite(e)]
            pct_err = 100.0 * (e - cv_ref) / cv_ref
            mean_abs = np.mean(np.abs(pct_err)) if len(pct_err) else np.nan
            rmse = np.sqrt(np.mean(pct_err ** 2)) if len(pct_err) else np.nan
            sd = np.std(pct_err) if len(pct_err) else np.nan
            t_mean = np.mean(times[method])
            agg[method][snr] = rmse
            agg_time[method][snr] = t_mean
            rows.append(dict(snr_db=snr, method=METHOD_LABELS[method],
                              mean_abs_pct=mean_abs, rmse_pct=rmse, sd_pct=sd,
                              time_ms=t_mean, n_valid=len(pct_err)))
        print(f"snr={snr:>4} done: " +
              "   ".join(f"{m}={agg[m][snr]:.3f}% ({agg_time[m][snr]:.2f} ms)"
                         for m in methods))

    global_bfgs = {}
    for snr in SNR_VALUES:
        seed0 = int(snr * 137)
        res = run_cell(snr, seed0, theta_fn, cv_ref, source_fixed=source)
        global_bfgs[snr] = res["BFGS"]["rmse_pct"]

    csv_path = REPO / "journal_edit" / "tvd_multipop_sinusoidal_sliding_data.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["snr_db", "method", "mean_abs_pct",
                                           "rmse_pct", "sd_pct", "time_ms", "n_valid"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {csv_path}")

    print("\n=== (multi-MU-population, sinusoidal): global fit vs. four local strategies (RMSE) ===")
    for snr in SNR_VALUES:
        print(f"  SNR={snr:>4} dB: global-fit BFGS={global_bfgs[snr]:.3f}%   "
              f"local-poly(order-1)={agg['poly'][snr]:.3f}%   "
              f"local-const(order-0,MLE)={agg['constmle'][snr]:.3f}%   "
              f"LAP(Gilliam2018)={agg['lap'][snr]:.3f}%   "
              f"GCC(Knapp1976)={agg['gcc'][snr]:.3f}%")

    return global_bfgs, agg


if __name__ == "__main__":
    run_global()
    run_sliding()
    print("\nDone.")
