#!/usr/bin/env python3
"""
run_muap_sinusoidal.py -- fills in the one cell of the 2x2 (source x
ground-truth-shape) factorial design that PAPER_Optimizer_CRLB.tex does not
compute: a REALISTIC (Farina-Merletti MUAP) source combined with a
NON-POLYNOMIAL (sinusoidal, Eq. 25 of Luu et al. 2018) time-varying delay.

The other three cells already exist in that companion manuscript:
  (colored noise, linear ramp)   -- tvd_benchmark_data.csv            (baseline)
  (MUAP,          linear ramp)   -- tvd_benchmark_muap_data.csv       (main_muap)
  (colored noise, sinusoidal)    -- tvd_benchmark_sinusoidal_data.csv (main_sinusoidal)
                                     + tvd_benchmark_sinusoidal_sliding_data.csv
This script computes the missing fourth cell, (MUAP, sinusoidal), at the
SAME rigor (same REPS_FAST/REPS_SLOW, SNR grid, SEG_LEN) as the other three,
for both the global 4-optimizer comparison and the local 4-strategy
sliding-window comparison, so the four cells are directly comparable.

Outputs (M=100/50; see the M_FAST/M_SLOW comment below for history):
  - tvd_muap_sinusoidal_global_data.csv
  - tvd_muap_sinusoidal_sliding_data.csv
  - tvd_muap_sinusoidal_global.pdf
"""
import sys
import time
import csv
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import scripts.optimizer_crlb_paper.crlb_tvd_benchmark as tvdmod  # noqa: E402
from scripts.optimizer_crlb_paper.crlb_tvd_benchmark import (  # noqa: E402
    CFG, N, SNR_VALUES, SEG_LEN, SEG_OVERLAP,
    true_theta_sinusoidal, load_muap_source, make_pair, run_cell, run_variant,
    sliding_window_cv_end, sliding_window_constmle_cv_end,
    sliding_window_lap_cv_end, gcc_cv_end_mc, METHOD_LABELS,
)

OUT_DIR = REPO / "scripts" / "tvd_interaction_paper"

# M=100/50 rerun (2026-09-23): the companion manuscript's own baseline cells
# (main/main_sinusoidal/main_muap/main_sinusoidal_sliding in
# crlb_tvd_benchmark.py, via rerun_baseline_M100.py) were themselves moved
# from REPS_FAST=12/REPS_SLOW=6 to 100/50, so this cell (and the M=60/30
# rerun that preceded it, kept for provenance in git history) is rerun at
# the same M for apples-to-apples comparison within PAPER_TVD_Interaction.tex's
# own tables. REPS_FAST/REPS_SLOW are monkey-patched on the crlb_tvd_benchmark
# module itself (rather than passed as arguments) because run_variant()/
# run_cell() read them as that module's own globals at call time, not as
# parameters.
M_FAST = 100  # was 12, then 60
M_SLOW = 50   # was 6, then 30


def run_global():
    """Global single order-1 fit, 4 optimizers (BFGS/SA/GA/PSO), MUAP source."""
    tvdmod.REPS_FAST = M_FAST
    tvdmod.REPS_SLOW = M_SLOW
    source = load_muap_source()
    run_variant(
        variant=f"muap x sinusoidal (global order-1 fit), M_FAST={M_FAST}/M_SLOW={M_SLOW}",
        theta_fn=true_theta_sinusoidal,
        source_fixed=source,
        csv_name="tvd_muap_sinusoidal_global_data.csv",
        pdf_name="tvd_muap_sinusoidal_global.pdf",
        plot_title="TVD, MUAP source x sinusoidal ground truth: optimizer comparison",
    )


def run_sliding():
    """Local 4-strategy sliding-window comparison, MUAP source, matching
    main_sinusoidal_sliding()'s structure exactly but with the MUAP source
    in place of colored noise."""
    tvdmod.REPS_FAST = M_FAST
    tvdmod.REPS_SLOW = M_SLOW
    theta_fn = true_theta_sinusoidal
    cv_ref = CFG.ied_m * CFG.fs / theta_fn(N - 1)
    source = load_muap_source()
    methods = ("poly", "constmle", "lap", "gcc")
    agg = {m: {} for m in methods}
    agg_time = {m: {} for m in methods}
    rows = []

    print(f"\n--- variant: muap x sinusoidal, sliding-window fits "
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

    # Global BFGS fit against the SAME (MUAP, sinusoidal) ground truth, for
    # the same "global vs. local" comparison main_sinusoidal_sliding() makes.
    global_bfgs = {}
    for snr in SNR_VALUES:
        seed0 = int(snr * 137)
        res = run_cell(snr, seed0, theta_fn, cv_ref, source_fixed=source)
        global_bfgs[snr] = res["BFGS"]["rmse_pct"]

    csv_path = REPO / "journal_edit" / "tvd_muap_sinusoidal_sliding_data.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["snr_db", "method", "mean_abs_pct",
                                           "rmse_pct", "sd_pct", "time_ms", "n_valid"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {csv_path}")

    print("\n=== (MUAP, sinusoidal): global fit vs. four local strategies (RMSE) ===")
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
