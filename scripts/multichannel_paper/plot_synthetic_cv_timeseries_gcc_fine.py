"""
Fine-grained (hop=1 sample, i.e. maximally overlapping) version of the
synthetic GCC-pooled CV(t)/RMSE(t) trace.

plot_synthetic_cv_timeseries_gcc.py samples CV(t) at the SEG_LEN=100,
SEG_OVERLAP=50% segment grid already used throughout crlb_tvd_benchmark.py
(_segment_starts(), step=50 samples -> 11 points per 600-sample window):
that grid is the one the paper's own window-end tables are built on, but it
is too coarse to show a smooth CV(t)/RMSE(t) curve. This script instead
slides the SAME SEG_LEN=100-sample GCC-pooled window one sample at a time
(step=1, 501 points per window) -- feasible only for GCC, since it is a
closed-form per-segment estimator (~0.02-0.2ms/segment); the other three
local estimators in this paper (poly's BFGS fit alone costs ~20-100ms per
segment) would take on the order of an hour per SNR sweep at this
resolution and are therefore not included here (see
plot_synthetic_cv_timeseries_gcc.py for the coarse-grid, all-four-estimator
comparison instead).
"""
from __future__ import annotations

import csv
import functools
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "optimizer_crlb_paper"))

from crlb_tvd_benchmark import (  # noqa: E402
    CFG, N, SNR_VALUES, SEG_LEN,
    make_multichannel_pair, shwedyk_semg, true_theta_sinusoidal,
)
import crlb_optimizer_benchmark as _crlb_opt  # noqa: E402
# source_psd_two_sided(n, fs) only depends on the (fixed) window length and
# sample rate, not on cv_true/snr_db/n_rows -- but crlb_theta() recomputes
# its 500-realization PSD estimate on every call with no caching of its own.
# Since the local-CRLB(t) overlay below calls crlb_theta() once per segment
# (~500 times per K), this same (n=SEG_LEN, fs=CFG.fs) PSD would otherwise be
# re-estimated from scratch ~500 times; memoizing it here is what keeps that
# affordable.
_crlb_opt.source_psd_two_sided = functools.lru_cache(maxsize=8)(_crlb_opt.source_psd_two_sided)
from crlb_optimizer_benchmark import crlb_theta  # noqa: E402
from src.mfcv import _gcc_delay  # noqa: E402

OUT_DIR = PROJECT_ROOT / "journal_edit"
K_VALUES = {"K2": 2, "K11": 11}
GCC_MAX_LAG = 40  # matches gcc_cv_end_mc's own max_lag elsewhere in this paper
HOP = 1            # samples -- "sliding window with 1-sample overlap step"
REPS = 500         # GCC is cheap enough to afford more reps than the
                    # all-methods comparison's REPS_FAST=15
DEMO_SNR = 15.0
THETA_FN = true_theta_sinusoidal
EXCLUDE_FIRST_SEGMENT_FROM_SUMMARY = True  # same boundary-extrapolation
                                            # artifact documented in
                                            # plot_synthetic_cv_timeseries_gcc.py
BOUNDARY_CUTOFF_S = 0.07  # at hop=1 the artifact isn't confined to a single
                           # segment (unlike the coarse, step=50 grid): channel
                           # k up to K_max-1=10 needs CubicSpline(source)
                           # extrapolated by up to k*theta(n)<=~53 samples near
                           # n=0, so RMSE(t) stays elevated (checked: >1 m/s)
                           # out to t~=0.064s at every SNR tested here; 0.07s
                           # is used as a uniform, slightly conservative cutoff

COLORS = {"K2": "#4C72B0", "K11": "#DD8452"}


def fine_segment_starts() -> list[int]:
    return list(range(0, N - SEG_LEN + 1, HOP))


def gcc_pooled_delay(seg: np.ndarray) -> float:
    K = seg.shape[0]
    delays = [_gcc_delay(seg[k], seg[k + 1], max_lag=GCC_MAX_LAG)[0] for k in range(K - 1)]
    return float(np.mean(delays))


def segment_end_samples() -> np.ndarray:
    return np.array([start + SEG_LEN - 1 for start in fine_segment_starts()])


def true_cv_at_segments(theta_fn) -> np.ndarray:
    n = segment_end_samples()
    theta = theta_fn(n)
    return CFG.ied_m / (theta / CFG.fs)


def local_crlb_sd_cv(cv_true_val: float, snr_db: float, K: int) -> float:
    """Approximate, POINTWISE local CRLB on CV [m/s] at one segment: crlb_theta
    (constant-delay Slepian-Bangs bound) evaluated at that segment's own
    instantaneous true CV, converted from Var(theta_hat) [samples^2] to
    Var(CV_hat) by the delta method (CV=ied_m*fs/theta,
    dCV/dtheta=-ied_m*fs/theta^2). This is the SAME approximation
    PAPER_Optimizer_CRLB.tex's Table tab:tvd_sliding already uses for local
    order-0/LAP/GCC estimators under a non-constant (sinusoidal) ground
    truth: treat each SEG_LEN-sample segment as if the delay were constant
    at that segment's own local value -- not an exact time-varying CRLB
    (none exists for this K-channel model), but the same convention already
    established elsewhere in this manuscript series."""
    theta_true = CFG.ied_m / cv_true_val * CFG.fs
    var_theta = crlb_theta(cv_true_val, snr_db, n=SEG_LEN, n_rows=K)
    if not np.isfinite(var_theta):
        return np.nan
    dcv_dtheta = CFG.ied_m * CFG.fs / theta_true ** 2
    return float(dcv_dtheta * np.sqrt(var_theta))


def run_snr(snr_db: float, theta_fn, reps: int = REPS, seed0: int = 0) -> dict[str, np.ndarray]:
    starts = fine_segment_starts()
    n_seg = len(starts)
    out = {k: np.full((reps, n_seg), np.nan) for k in K_VALUES}
    K_max = max(K_VALUES.values())
    for r in range(reps):
        rng = np.random.default_rng(seed0 + r)
        src = shwedyk_semg(N, CFG.fs, rng)
        cols_full = make_multichannel_pair(rng, snr_db, src, theta_fn, n_rows=K_max)
        for k_label, K in K_VALUES.items():
            cols = cols_full[:K]
            for si, start in enumerate(starts):
                seg = cols[:, start:start + SEG_LEN]
                theta = gcc_pooled_delay(seg)
                if abs(theta) < 1e-9:
                    continue
                out[k_label][r, si] = CFG.ied_m / (abs(theta) / CFG.fs)
    return out


def main() -> None:
    t_s = segment_end_samples() / CFG.fs
    cv_true = true_cv_at_segments(THETA_FN)
    summary_mask = (t_s >= BOUNDARY_CUTOFF_S) if EXCLUDE_FIRST_SEGMENT_FROM_SUMMARY else np.ones_like(t_s, dtype=bool)

    print("--- precomputing local CRLB(t) (delta-method, per K/SNR/segment) ---")
    crlb_sd = {snr: {k: np.array([local_crlb_sd_cv(cv, snr, K) for cv in cv_true])
                      for k, K in K_VALUES.items()}
               for snr in SNR_VALUES}

    rows = []
    per_snr_estimates = {}
    for snr in SNR_VALUES:
        seed0 = int(snr * 137)
        print(f"--- synthetic GCC, hop={HOP} sample, SNR={snr:.0f} dB ---")
        est = run_snr(snr, THETA_FN, seed0=seed0)
        per_snr_estimates[snr] = est
        for k_label in K_VALUES:
            cv_mean = np.nanmean(est[k_label], axis=0)
            cv_std = np.nanstd(est[k_label], axis=0)
            rmse = np.sqrt(np.nanmean((est[k_label] - cv_true[None, :]) ** 2, axis=0))
            crlb_k = crlb_sd[snr][k_label]
            for si in range(len(t_s)):
                rows.append(dict(snr_db=snr, K=k_label, t_s=round(float(t_s[si]), 4),
                                  cv_mean=float(cv_mean[si]), cv_std=float(cv_std[si]),
                                  cv_true=float(cv_true[si]), rmse=float(rmse[si]),
                                  crlb_sd_cv=float(crlb_k[si])))
            print(f"  {k_label}: mean RMSE (t>={BOUNDARY_CUTOFF_S}s) = {np.nanmean(rmse[summary_mask]):.4f} m/s"
                  f"  |  mean local-CRLB SD = {np.nanmean(crlb_k[summary_mask]):.4f} m/s")

    csv_path = OUT_DIR / "tvd_multichannel_synthetic_gcc_fine_data.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["snr_db", "K", "t_s", "cv_mean", "cv_std", "cv_true", "rmse", "crlb_sd_cv"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {csv_path}")

    # --- Figure 1: CV(t) and RMSE(t) at DEMO_SNR, hop=1 ---
    # The boundary-extrapolation artifact (see BOUNDARY_CUTOFF_S) dwarfs the
    # informative part of the trace at full scale, so the displayed figure
    # is cropped to t>=BOUNDARY_CUTOFF_S (the full trace, incl. the spike,
    # remains in the CSV for transparency).
    demo_est = per_snr_estimates[DEMO_SNR]
    post_cutoff = t_s >= BOUNDARY_CUTOFF_S
    t_plot = t_s[post_cutoff]
    fig, (ax_cv, ax_rmse, ax_sd) = plt.subplots(3, 1, figsize=(7, 8.5), sharex=True)
    ax_cv.plot(t_plot, cv_true[post_cutoff], "k--", lw=1.2, label="ground truth")
    for k_label, color in COLORS.items():
        cv_mean = np.nanmean(demo_est[k_label], axis=0)
        cv_std = np.nanstd(demo_est[k_label], axis=0)
        ax_cv.plot(t_plot, cv_mean[post_cutoff], "-", color=color, lw=1.1, label=k_label)
        ax_cv.fill_between(t_plot, (cv_mean - cv_std)[post_cutoff], (cv_mean + cv_std)[post_cutoff],
                            color=color, alpha=0.2)

        rmse = np.sqrt(np.nanmean((demo_est[k_label] - cv_true[None, :]) ** 2, axis=0))
        ax_rmse.plot(t_plot, rmse[post_cutoff], "-", color=color, lw=1.1, label=k_label)

        # Variance (not RMSE) is the quantity crlb_theta actually bounds: RMSE
        # also includes GCC's own bias against the (non-constant) ground
        # truth, which no variance-only CRLB accounts for -- plotting RMSE
        # against CRLB would repeat the MAE/RMSE-vs-CRLB mismatch already
        # flagged for this figure and corrected here.
        ax_sd.plot(t_plot, (cv_std ** 2)[post_cutoff], "-", color=color, lw=1.1, label=f"{k_label} Var")
        ax_sd.plot(t_plot, (crlb_sd[DEMO_SNR][k_label] ** 2)[post_cutoff], "--", color=color, lw=1.2, alpha=0.85,
                   label=f"{k_label} local CRLB")

    ax_cv.set_ylabel("CV [m/s]")
    ax_cv.set_title(f"Synthetic GCC-pooled CV(t), K=2 vs K=11, hop=1 sample\n"
                     f"(SNR={DEMO_SNR:.0f} dB, mean ± SD over {REPS} realizations, "
                     f"first {BOUNDARY_CUTOFF_S*1000:.0f}ms omitted)", fontsize=8)
    ax_cv.legend(fontsize=7, framealpha=0.9)
    ax_cv.grid(alpha=0.2)

    ax_rmse.set_ylabel("RMSE [m/s]")
    ax_rmse.set_title("RMSE(t) vs. ground truth (accuracy: bias + variance, no CRLB)", fontsize=9)
    ax_rmse.legend(fontsize=7, framealpha=0.9)
    ax_rmse.grid(alpha=0.2)

    ax_sd.set_xlabel("Time [s]")
    ax_sd.set_ylabel("Variance [m/s$^2$]")
    ax_sd.set_yscale("log")
    ax_sd.set_title("Variance(t) vs. local CRLB (dashed) -- precision only, the quantity CRLB actually bounds", fontsize=9)
    ax_sd.legend(fontsize=6.5, framealpha=0.9, ncol=2)
    ax_sd.grid(alpha=0.2, which="both")
    fig.tight_layout()
    out1 = OUT_DIR / "tvd_multichannel_synthetic_gcc_fine_timeseries.pdf"
    fig.savefig(out1)
    print(f"Wrote {out1}")

    # --- Figure 2: mean RMSE vs. SNR (left, accuracy, no CRLB) and mean SD vs.
    # SNR with local-CRLB (right, precision -- the quantity CRLB bounds) ---
    fig2, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10, 4))
    for k_label, color in COLORS.items():
        mean_rmse_by_snr = [
            np.nanmean(np.sqrt(np.nanmean(
                (per_snr_estimates[snr][k_label] - cv_true[None, :]) ** 2, axis=0))[summary_mask])
            for snr in SNR_VALUES
        ]
        mean_var_by_snr = [
            np.nanmean((np.nanstd(per_snr_estimates[snr][k_label], axis=0) ** 2)[summary_mask])
            for snr in SNR_VALUES
        ]
        mean_crlb_by_snr = [np.nanmean((crlb_sd[snr][k_label] ** 2)[summary_mask]) for snr in SNR_VALUES]
        ax_l.plot(SNR_VALUES, mean_rmse_by_snr, "o-", color=color, lw=1.6, ms=5, label=k_label)
        ax_r.plot(SNR_VALUES, mean_var_by_snr, "o-", color=color, lw=1.6, ms=5, label=f"{k_label} Var")
        ax_r.plot(SNR_VALUES, mean_crlb_by_snr, "--", color=color, lw=1.4, alpha=0.85,
                  label=f"{k_label} local CRLB")
    ax_l.set_xlabel("SNR [dB]")
    ax_l.set_ylabel(f"Mean RMSE over window [m/s]\n(t≥{BOUNDARY_CUTOFF_S}s)")
    ax_l.set_title("Accuracy: mean RMSE vs. SNR", fontsize=9)
    ax_l.legend(fontsize=8, framealpha=0.9)
    ax_l.grid(alpha=0.2)

    ax_r.set_yscale("log")
    ax_r.set_xlabel("SNR [dB]")
    ax_r.set_ylabel(f"Mean variance over window [m/s$^2$]\n(t≥{BOUNDARY_CUTOFF_S}s)")
    ax_r.set_title("Precision: mean variance vs. SNR (dashed: local CRLB)", fontsize=9)
    ax_r.legend(fontsize=7, framealpha=0.9)
    ax_r.grid(alpha=0.2, which="both")

    fig2.suptitle("Synthetic GCC-pooled CV, hop=1 sample", fontsize=10)
    fig2.tight_layout(rect=[0, 0, 1, 0.94])
    out2 = OUT_DIR / "tvd_multichannel_synthetic_gcc_fine_rmse_vs_snr.pdf"
    fig2.savefig(out2)
    print(f"Wrote {out2}")


if __name__ == "__main__":
    main()
