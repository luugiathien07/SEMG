"""
Sliding-window CV(t)/RMSE(t) trace for the 2-channel (scalar) GCC baseline
already benchmarked, at window end only, in Table tab:tvd_sliding / Figure
fig:tvd_sliding of PAPER_Optimizer_CRLB.tex (main_sinusoidal_sliding(),
gcc_cv_end_mc applied to a 2-channel stack). That table reports a single
mean-abs-%-error number per SNR, at the window's last SEG_LEN=100-sample
segment; this script instead evaluates the SAME per-segment GCC estimator
(_gcc_delay, cross-correlation + parabolic sub-sample interpolation) at
EVERY window position, sliding one sample at a time (hop=1, matching this
manuscript series' original gcc_glissant.m convention -- see
scripts/multichannel_paper/plot_synthetic_cv_timeseries_gcc_fine.py, the
K-channel analogue of this script), against the same non-polynomial Eq. (25)
ground truth (true_theta_sinusoidal) used throughout that table.

Unlike the K-channel case, only a single channel pair exists here (K=2, one
GCC delay per segment, no pooling), so there is no boundary-extrapolation
artifact from averaging across channels with growing k*theta offsets: the
only extrapolation is x2's own single-channel warp by theta(n)<~5.3 samples,
checked empirically below to confirm it does not need the same exclusion
window plot_synthetic_cv_timeseries_gcc_fine.py applies for K up to 11.
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
    make_pair, shwedyk_semg, true_theta_sinusoidal,
)
import crlb_optimizer_benchmark as _crlb_opt  # noqa: E402
# see scripts/multichannel_paper/plot_synthetic_cv_timeseries_gcc_fine.py's
# identical comment: crlb_theta()'s own 500-realization PSD estimate has no
# caching, and is re-derived on every call unless memoized here.
_crlb_opt.source_psd_two_sided = functools.lru_cache(maxsize=8)(_crlb_opt.source_psd_two_sided)
from crlb_optimizer_benchmark import crlb_theta  # noqa: E402
from src.mfcv import _gcc_delay  # noqa: E402

OUT_DIR = PROJECT_ROOT / "journal_edit"
GCC_MAX_LAG = 40  # matches gcc_cv_end_mc's own max_lag, i.e. Table tab:tvd_sliding's convention
HOP = 1
REPS = 500
DEMO_SNR = 15.0
THETA_FN = true_theta_sinusoidal
BOUNDARY_CUTOFF_S = 0.055  # a much smaller version of the K-channel case's
                            # artifact (see module docstring): only the first
                            # 1-2 samples show elevated RMSE here (single
                            # channel pair, <=5.3-sample extrapolation), but
                            # excluded for consistency with the K-channel
                            # script's methodology


def fine_segment_starts() -> list[int]:
    return list(range(0, N - SEG_LEN + 1, HOP))


def gcc_segment_theta(x1_seg: np.ndarray, x2_seg: np.ndarray) -> float:
    return _gcc_delay(x1_seg, x2_seg, max_lag=GCC_MAX_LAG)[0]


def segment_end_samples() -> np.ndarray:
    return np.array([start + SEG_LEN - 1 for start in fine_segment_starts()])


def true_cv_at_segments(theta_fn) -> np.ndarray:
    n = segment_end_samples()
    theta = theta_fn(n)
    return CFG.ied_m / (theta / CFG.fs)


def local_crlb_sd_cv(cv_true_val: float, snr_db: float) -> float:
    """Pointwise local CRLB on CV [m/s] at K=2, delta-method conversion from
    crlb_theta's Var(theta_hat) -- the SAME bound (and the same
    treat-the-segment-as-locally-constant approximation) Table tab:tvd_sliding
    already uses for local order-0/LAP/GCC under this paper's sinusoidal
    ground truth, evaluated here at every segment instead of only reported
    once as a single SNR-dependent range."""
    theta_true = CFG.ied_m / cv_true_val * CFG.fs
    var_theta = crlb_theta(cv_true_val, snr_db, n=SEG_LEN, n_rows=2)
    if not np.isfinite(var_theta):
        return np.nan
    dcv_dtheta = CFG.ied_m * CFG.fs / theta_true ** 2
    return float(dcv_dtheta * np.sqrt(var_theta))


def run_snr(snr_db: float, theta_fn, reps: int = REPS, seed0: int = 0) -> np.ndarray:
    """(reps, n_segments) array of estimated CV [m/s]."""
    starts = fine_segment_starts()
    n_seg = len(starts)
    out = np.full((reps, n_seg), np.nan)
    for r in range(reps):
        rng = np.random.default_rng(seed0 + r)
        src = shwedyk_semg(N, CFG.fs, rng)
        x1, x2 = make_pair(rng, snr_db, src, theta_fn)
        for si, start in enumerate(starts):
            x1_seg = x1[start:start + SEG_LEN]
            x2_seg = x2[start:start + SEG_LEN]
            theta = gcc_segment_theta(x1_seg, x2_seg)
            if abs(theta) < 1e-9:
                continue
            out[r, si] = CFG.ied_m / (abs(theta) / CFG.fs)
    return out


def main() -> None:
    t_s = segment_end_samples() / CFG.fs
    cv_true = true_cv_at_segments(THETA_FN)

    print("--- precomputing local CRLB(t) (delta-method, per SNR/segment) ---")
    crlb_sd = {snr: np.array([local_crlb_sd_cv(cv, snr) for cv in cv_true]) for snr in SNR_VALUES}

    rows = []
    per_snr_estimates = {}
    for snr in SNR_VALUES:
        seed0 = int(snr * 137)
        print(f"--- scalar (K=2) GCC, hop={HOP} sample, SNR={snr:.0f} dB ---")
        est = run_snr(snr, THETA_FN, seed0=seed0)
        per_snr_estimates[snr] = est
        cv_mean = np.nanmean(est, axis=0)
        cv_std = np.nanstd(est, axis=0)
        rmse = np.sqrt(np.nanmean((est - cv_true[None, :]) ** 2, axis=0))
        for si in range(len(t_s)):
            rows.append(dict(snr_db=snr, t_s=round(float(t_s[si]), 4),
                              cv_mean=float(cv_mean[si]), cv_std=float(cv_std[si]),
                              cv_true=float(cv_true[si]), rmse=float(rmse[si]),
                              crlb_sd_cv=float(crlb_sd[snr][si])))
        # sanity-check the first few segments for the boundary-extrapolation
        # artifact documented for the K-channel case (should be absent/small here)
        print(f"  RMSE, first 5 segments: {rmse[:5].round(3)}")
        mask = t_s >= BOUNDARY_CUTOFF_S
        print(f"  mean RMSE (t>={BOUNDARY_CUTOFF_S}s) = {np.nanmean(rmse[mask]):.4f} m/s"
              f"  |  mean local-CRLB SD = {np.nanmean(crlb_sd[snr][mask]):.4f} m/s")

    csv_path = OUT_DIR / "tvd_sliding_gcc_fine_data.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["snr_db", "t_s", "cv_mean", "cv_std", "cv_true", "rmse", "crlb_sd_cv"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {csv_path}")

    # --- Figure 1: CV(t) and RMSE(t) at DEMO_SNR, hop=1 ---
    demo_est = per_snr_estimates[DEMO_SNR]
    post_cutoff = t_s >= BOUNDARY_CUTOFF_S
    t_plot = t_s[post_cutoff]
    fig, (ax_cv, ax_rmse, ax_sd) = plt.subplots(3, 1, figsize=(7, 8.5), sharex=True)
    cv_mean = np.nanmean(demo_est, axis=0)
    cv_std = np.nanstd(demo_est, axis=0)
    ax_cv.plot(t_plot, cv_true[post_cutoff], "k--", lw=1.2, label="ground truth")
    ax_cv.plot(t_plot, cv_mean[post_cutoff], "-", color="#E08A1E", lw=1.1, label="GCC (K=2)")
    ax_cv.fill_between(t_plot, (cv_mean - cv_std)[post_cutoff], (cv_mean + cv_std)[post_cutoff],
                        color="#E08A1E", alpha=0.2)

    rmse = np.sqrt(np.nanmean((demo_est - cv_true[None, :]) ** 2, axis=0))
    ax_rmse.plot(t_plot, rmse[post_cutoff], "-", color="#E08A1E", lw=1.1, label="GCC (K=2)")

    # Variance (not RMSE) is the quantity crlb_theta actually bounds -- RMSE
    # also includes GCC's own bias against this non-constant ground truth,
    # which no variance-only CRLB accounts for.
    ax_sd.plot(t_plot, (cv_std ** 2)[post_cutoff], "-", color="#E08A1E", lw=1.1, label="GCC Var")
    ax_sd.plot(t_plot, (crlb_sd[DEMO_SNR] ** 2)[post_cutoff], "--", color="#0B2A4A", lw=1.2, label="local CRLB")

    ax_cv.set_ylabel("CV [m/s]")
    ax_cv.set_title(f"Scalar (K=2) GCC-pooled CV(t), hop=1 sample\n"
                     f"(SNR={DEMO_SNR:.0f} dB, mean ± SD over {REPS} realizations)", fontsize=8)
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
    ax_sd.legend(fontsize=7, framealpha=0.9)
    ax_sd.grid(alpha=0.2, which="both")
    fig.tight_layout()
    out1 = OUT_DIR / "tvd_sliding_gcc_fine_timeseries.pdf"
    fig.savefig(out1)
    print(f"Wrote {out1}")

    # --- Figure 2: mean RMSE vs. SNR (left, accuracy, no CRLB) and mean SD vs.
    # SNR with local-CRLB (right, precision -- the quantity CRLB bounds) ---
    fig2, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10, 4))
    mean_rmse_by_snr = []
    mean_var_by_snr = []
    mean_crlb_by_snr = []
    for snr in SNR_VALUES:
        rmse_snr = np.sqrt(np.nanmean((per_snr_estimates[snr] - cv_true[None, :]) ** 2, axis=0))
        mean_rmse_by_snr.append(np.nanmean(rmse_snr[post_cutoff]))
        mean_var_by_snr.append(np.nanmean((np.nanstd(per_snr_estimates[snr], axis=0) ** 2)[post_cutoff]))
        mean_crlb_by_snr.append(np.nanmean((crlb_sd[snr] ** 2)[post_cutoff]))
    ax_l.plot(SNR_VALUES, mean_rmse_by_snr, "o-", color="#E08A1E", lw=1.6, ms=5, label="GCC (K=2)")
    ax_l.set_xlabel("SNR [dB]")
    ax_l.set_ylabel(f"Mean RMSE over window [m/s]\n(t≥{BOUNDARY_CUTOFF_S}s)")
    ax_l.set_title("Accuracy: mean RMSE vs. SNR", fontsize=9)
    ax_l.legend(fontsize=8, framealpha=0.9)
    ax_l.grid(alpha=0.2)

    ax_r.plot(SNR_VALUES, mean_var_by_snr, "o-", color="#E08A1E", lw=1.6, ms=5, label="GCC Var")
    ax_r.plot(SNR_VALUES, mean_crlb_by_snr, "--", color="#0B2A4A", lw=1.4, label="local CRLB")
    ax_r.set_yscale("log")
    ax_r.set_xlabel("SNR [dB]")
    ax_r.set_ylabel(f"Mean variance over window [m/s$^2$]\n(t≥{BOUNDARY_CUTOFF_S}s)")
    ax_r.set_title("Precision: mean variance vs. SNR (dashed: local CRLB)", fontsize=9)
    ax_r.legend(fontsize=8, framealpha=0.9)
    ax_r.grid(alpha=0.2, which="both")

    fig2.suptitle("Scalar (K=2) GCC-pooled CV, hop=1 sample", fontsize=10)
    fig2.tight_layout(rect=[0, 0, 1, 0.94])
    fig2.tight_layout()
    out2 = OUT_DIR / "tvd_sliding_gcc_fine_rmse_vs_snr.pdf"
    fig2.savefig(out2)
    print(f"Wrote {out2}")


if __name__ == "__main__":
    main()
