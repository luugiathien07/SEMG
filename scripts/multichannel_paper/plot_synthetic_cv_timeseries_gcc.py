"""
Synthetic-signal counterpart of plot_realdata_cv_timeseries_gcc.py, for ALL
FOUR local multichannel TVD estimators already benchmarked in this paper
(crlb_tvd_benchmark.py's main_multichannel methods dict: poly, constmle,
clap, gcc) rather than GCC alone.

Real recordings have no known ground-truth CV, so plot_realdata_cv_timeseries_gcc.py
can only show the K=2-vs-K=11 CV(t) trace itself. Here, each estimator's own
per-SEG_LEN=100-sample-segment building block (already used internally by
sliding_window_cv_end_mc / sliding_window_constmle_cv_end_mc / clap_cv_end /
gcc_cv_end_mc, which normally discard every segment but the LAST -- the
paper's "estimated CV at window end" convention) is instead evaluated and
KEPT at every segment, applied to synthetic colored-noise EMG (shwedyk_semg)
with a KNOWN time-varying ground truth (true_theta_sinusoidal, the same
non-polynomial CV(t) used for this paper's "realistic"/tilt-realistic
tables). This gives, for K=2 and K=11 double-differential-style channel
counts:

  - CV(t) and RMSE(t): mean +/- SD estimated CV and root-mean-square error
    against the true CV(t), across Monte Carlo realizations, at every
    segment time t within the window (not just window end).
  - mean RMSE vs. SNR: RMSE(t) averaged over segment times, at each of this
    paper's four benchmarked SNR levels.

Each Monte Carlo realization draws ONE n_rows=11 multichannel column
(make_multichannel_pair) reused across all four methods and both K values,
matching how main_multichannel()/run_cell() reuse a single "cols" draw
across methods, and how the real GCC CSV derives both K=2 and K=11
estimates from the same recording's channels 1..11.

NOTE on the first segment (t~0.05s): sampling every segment (not just
window end) surfaces a boundary artifact invisible in the paper's own
window-end tables -- for early segments, computing channel k>0 requires
CubicSpline(source) extrapolated near n=0 by up to k*theta samples, which
can occasionally produce a large spurious delay lock for some
estimator/K/SNR combinations. This is a property of make_multichannel_pair's
extrapolation model, not of any one estimator, so it is kept in the
per-segment CSV for transparency but EXCLUDED from the "mean RMSE vs SNR"
summary (see EXCLUDE_FIRST_SEGMENT_FROM_SUMMARY).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "optimizer_crlb_paper"))

from crlb_tvd_benchmark import (  # noqa: E402
    CFG, N, SNR_VALUES, SEG_LEN, LAP_R, _segment_starts,
    make_multichannel_pair, shwedyk_semg, true_theta_sinusoidal,
    tvd_cost_mc, refine_bfgs_mc, coarse_grid, refine_brent, clap_theta_end,
)
from src.mfcv import _gcc_delay  # noqa: E402

OUT_DIR = PROJECT_ROOT / "journal_edit"
K_VALUES = {"K2": 2, "K11": 11}
GCC_MAX_LAG = 40  # matches gcc_cv_end_mc's own max_lag (kept identical so
                   # this script's window-end GCC value reproduces the
                   # already-published table numbers)
REPS = 15          # matches REPS_FAST, the convention this paper already
                    # uses for the (slower, BFGS-based) poly estimator
DEMO_SNR = 15.0     # dB, representative SNR shown in the CV(t)/RMSE(t) trace
THETA_FN = true_theta_sinusoidal
EXCLUDE_FIRST_SEGMENT_FROM_SUMMARY = True

METHOD_LABELS = {"poly": "poly (BFGS)", "constmle": "const-delay (Brent)",
                  "clap": "CLAP", "gcc": "GCC"}
COLORS = {"poly": "#0B2A4A", "constmle": "#3C9D45", "clap": "#8E44AD", "gcc": "#E08A1E"}
K_LINESTYLE = {"K2": "--", "K11": "-"}


def poly_segment_theta_end(seg: np.ndarray) -> float:
    delay0, _ = _gcc_delay(seg[0], seg[1], max_lag=40)
    d0 = np.array([delay0, 0.0])
    d = refine_bfgs_mc(seg, d0)
    return float(np.polyval(d[::-1], SEG_LEN - 1))


def constmle_segment_theta_end(seg: np.ndarray) -> float:
    X, freqs, t0, step = coarse_grid(seg)
    return float(refine_brent(X, freqs, t0, step))


def clap_segment_theta_end(seg: np.ndarray) -> float:
    return float(clap_theta_end(seg, R=LAP_R, W=SEG_LEN))


def gcc_segment_theta_end(seg: np.ndarray) -> float:
    K = seg.shape[0]
    delays = [_gcc_delay(seg[k], seg[k + 1], max_lag=GCC_MAX_LAG)[0] for k in range(K - 1)]
    return float(np.mean(delays))


METHOD_FNS = {
    "poly": poly_segment_theta_end,
    "constmle": constmle_segment_theta_end,
    "clap": clap_segment_theta_end,
    "gcc": gcc_segment_theta_end,
}


def segment_end_samples() -> np.ndarray:
    return np.array([start + SEG_LEN - 1 for start in _segment_starts()])


def true_cv_at_segments(theta_fn) -> np.ndarray:
    n = segment_end_samples()
    theta = theta_fn(n)
    return CFG.ied_m / (theta / CFG.fs)


def run_snr(snr_db: float, theta_fn, reps: int = REPS, seed0: int = 0):
    """dict[(method, K_label)] -> (reps, n_segments) array of estimated CV [m/s]."""
    starts = _segment_starts()
    n_seg = len(starts)
    out = {(m, k): np.full((reps, n_seg), np.nan) for m in METHOD_FNS for k in K_VALUES}
    K_max = max(K_VALUES.values())
    for r in range(reps):
        rng = np.random.default_rng(seed0 + r)
        src = shwedyk_semg(N, CFG.fs, rng)
        cols_full = make_multichannel_pair(rng, snr_db, src, theta_fn, n_rows=K_max)
        for k_label, K in K_VALUES.items():
            cols = cols_full[:K]
            for si, start in enumerate(starts):
                seg = cols[:, start:start + SEG_LEN]
                for m_label, fn in METHOD_FNS.items():
                    theta = fn(seg)
                    if not np.isfinite(theta) or abs(theta) < 1e-9:
                        continue
                    out[(m_label, k_label)][r, si] = CFG.ied_m / (abs(theta) / CFG.fs)
    return out


def main() -> None:
    t_s = segment_end_samples() / CFG.fs
    cv_true = true_cv_at_segments(THETA_FN)
    n_seg = len(t_s)
    summary_slice = slice(1, None) if EXCLUDE_FIRST_SEGMENT_FROM_SUMMARY else slice(None)

    rows = []
    per_snr_estimates = {}
    for snr in SNR_VALUES:
        seed0 = int(snr * 137)
        print(f"--- synthetic sliding-window, all methods, SNR={snr:.0f} dB ---")
        est = run_snr(snr, THETA_FN, seed0=seed0)
        per_snr_estimates[snr] = est
        for (m_label, k_label), arr in est.items():
            cv_mean = np.nanmean(arr, axis=0)
            cv_std = np.nanstd(arr, axis=0)
            rmse = np.sqrt(np.nanmean((arr - cv_true[None, :]) ** 2, axis=0))
            for si in range(n_seg):
                rows.append(dict(snr_db=snr, method=m_label, K=k_label, t_s=round(float(t_s[si]), 4),
                                  cv_mean=float(cv_mean[si]), cv_std=float(cv_std[si]),
                                  cv_true=float(cv_true[si]), rmse=float(rmse[si])))
            mean_rmse = np.nanmean(rmse[summary_slice])
            print(f"  {m_label:9s} {k_label}: mean RMSE (excl. 1st seg={EXCLUDE_FIRST_SEGMENT_FROM_SUMMARY}) "
                  f"= {mean_rmse:.4f} m/s")

    csv_path = OUT_DIR / "tvd_multichannel_synthetic_allmethods_data.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["snr_db", "method", "K", "t_s", "cv_mean", "cv_std", "cv_true", "rmse"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {csv_path}")

    # --- Figure 1: CV(t) [top row] and RMSE(t) [bottom row], K=2 vs K=11 columns, DEMO_SNR ---
    demo_est = per_snr_estimates[DEMO_SNR]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for j, k_label in enumerate(K_VALUES):
        ax_cv, ax_rmse = axes[0][j], axes[1][j]
        ax_cv.plot(t_s, cv_true, "k--", lw=1.4, label="ground truth")
        for m_label, color in COLORS.items():
            arr = demo_est[(m_label, k_label)]
            cv_mean = np.nanmean(arr, axis=0)
            cv_std = np.nanstd(arr, axis=0)
            ax_cv.plot(t_s, cv_mean, "o-", color=color, ms=3, lw=1.2, label=METHOD_LABELS[m_label])
            ax_cv.fill_between(t_s, cv_mean - cv_std, cv_mean + cv_std, color=color, alpha=0.15)

            rmse = np.sqrt(np.nanmean((arr - cv_true[None, :]) ** 2, axis=0))
            ax_rmse.plot(t_s, rmse, "o-", color=color, ms=3, lw=1.2, label=METHOD_LABELS[m_label])

        ax_cv.set_title(f"{k_label}", fontsize=10)
        ax_cv.grid(alpha=0.2)
        ax_rmse.grid(alpha=0.2)
        ax_rmse.set_xlabel("Time [s]")
    axes[0][0].set_ylabel("CV [m/s]")
    axes[1][0].set_ylabel("RMSE [m/s]")
    axes[0][0].legend(fontsize=7, framealpha=0.9)
    fig.suptitle(f"Synthetic sliding-window CV(t) and RMSE(t), all four estimators (SNR={DEMO_SNR:.0f} dB, "
                 f"mean ± SD over {REPS} realizations)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out1 = OUT_DIR / "tvd_multichannel_synthetic_allmethods_timeseries.pdf"
    fig.savefig(out1)
    print(f"Wrote {out1}")

    # --- Figure 2: mean RMSE (averaged over t, excl. 1st segment) vs. SNR, K=2 vs K=11 ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for j, k_label in enumerate(K_VALUES):
        ax = axes2[j]
        for m_label, color in COLORS.items():
            mean_rmse_by_snr = []
            for snr in SNR_VALUES:
                arr = per_snr_estimates[snr][(m_label, k_label)]
                rmse = np.sqrt(np.nanmean((arr - cv_true[None, :]) ** 2, axis=0))
                mean_rmse_by_snr.append(np.nanmean(rmse[summary_slice]))
            ax.plot(SNR_VALUES, mean_rmse_by_snr, "o-", color=color, lw=1.6, ms=5,
                     label=METHOD_LABELS[m_label])
        ax.set_xlabel("SNR [dB]")
        ax.set_title(f"{k_label}", fontsize=10)
        ax.grid(alpha=0.2)
    axes2[0].set_ylabel("Mean RMSE over window [m/s]\n(excl. 1st segment)" if EXCLUDE_FIRST_SEGMENT_FROM_SUMMARY
                         else "Mean RMSE over window [m/s]")
    axes2[0].legend(fontsize=8, framealpha=0.9)
    fig2.suptitle("Synthetic sliding-window CV: mean RMSE vs. SNR, all four estimators", fontsize=11)
    fig2.tight_layout(rect=[0, 0, 1, 0.94])
    out2 = OUT_DIR / "tvd_multichannel_synthetic_allmethods_rmse_vs_snr.pdf"
    fig2.savefig(out2)
    print(f"Wrote {out2}")


if __name__ == "__main__":
    main()
