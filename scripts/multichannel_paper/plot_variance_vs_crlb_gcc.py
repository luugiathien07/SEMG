"""
Empirical variance of the GCC-pooled delay estimate vs. the analytic
K-channel Cramer-Rao Lower Bound (Slepian-Bangs, crlb_theta() in
crlb_optimizer_benchmark.py -- the SAME bound already used elsewhere in this
manuscript series for the local order-0/LAP/GCC sliding-window comparison,
here evaluated as a function of K rather than only at K=2), as a function of
channel count K=2..11, one line per SNR.

crlb_theta()'s Slepian-Bangs derivation assumes a KNOWN CONSTANT delay
(Sigma(f) built from a fixed theta_true, not a time-varying one), so this
script matches that assumption exactly: a constant-CV (CV0=4.5 m/s) source,
SEG_LEN=100-sample window (crlb_theta's own default local-estimator window
size elsewhere in this paper), GCC-pooled per-segment estimate
(gcc_pooled_delay, same K-1 adjacent-pair-averaged estimator used
throughout scripts/multichannel_paper/). The 100-sample analysis window is
taken from the MIDDLE of the underlying 600-sample signal (n=250..349) so
that no channel needs the near-n=0 cubic-spline extrapolation documented as
a boundary artifact in plot_synthetic_cv_timeseries_gcc_fine.py.
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
    CFG, N, SNR_VALUES, SEG_LEN, CV0,
    make_multichannel_pair, shwedyk_semg,
)
from crlb_optimizer_benchmark import crlb_theta  # noqa: E402
from src.mfcv import _gcc_delay  # noqa: E402

OUT_DIR = PROJECT_ROOT / "journal_edit"
K_LIST = list(range(2, 12))
GCC_MAX_LAG = 40
REPS = 500
MID_START = (N - SEG_LEN) // 2  # 250, avoids n=0 extrapolation for all K<=13


def const_theta_fn(n: np.ndarray) -> np.ndarray:
    theta0 = CFG.ied_m / CV0 * CFG.fs
    return np.full_like(np.asarray(n, dtype=float), theta0)


def gcc_pooled_delay(seg: np.ndarray) -> float:
    K = seg.shape[0]
    delays = [_gcc_delay(seg[k], seg[k + 1], max_lag=GCC_MAX_LAG)[0] for k in range(K - 1)]
    return float(np.mean(delays))


def run_snr_k(snr_db: float, K: int, reps: int = REPS, seed0: int = 0) -> np.ndarray:
    """(reps,) array of estimated theta [samples], from the middle SEG_LEN segment."""
    thetas = np.full(reps, np.nan)
    for r in range(reps):
        rng = np.random.default_rng(seed0 + r)
        src = shwedyk_semg(N, CFG.fs, rng)
        cols = make_multichannel_pair(rng, snr_db, src, const_theta_fn, n_rows=K)
        seg = cols[:, MID_START:MID_START + SEG_LEN]
        thetas[r] = gcc_pooled_delay(seg)
    return thetas


def main() -> None:
    theta0 = CFG.ied_m / CV0 * CFG.fs
    rows = []
    empirical_by_snr_k = {snr: [] for snr in SNR_VALUES}
    crlb_by_snr_k = {snr: [] for snr in SNR_VALUES}

    for snr in SNR_VALUES:
        seed0 = int(snr * 137)
        print(f"--- GCC vs CRLB, constant CV0={CV0} m/s, SNR={snr:.0f} dB ---")
        for K in K_LIST:
            thetas = run_snr_k(snr, K, seed0=seed0 + K * 1000)
            thetas = thetas[np.isfinite(thetas)]
            var_emp = float(np.var(thetas)) if len(thetas) else np.nan
            var_crlb = crlb_theta(CV0, snr, n=SEG_LEN, n_rows=K)
            empirical_by_snr_k[snr].append(var_emp)
            crlb_by_snr_k[snr].append(var_crlb)
            ratio = var_emp / var_crlb if np.isfinite(var_crlb) and var_crlb > 0 else np.nan
            rows.append(dict(snr_db=snr, K=K, var_empirical_samples2=var_emp,
                              var_crlb_samples2=var_crlb, ratio_to_crlb=ratio))
            print(f"  K={K:2d}: Var_emp={var_emp:.5f}  CRLB={var_crlb:.5f}  ratio={ratio:.2f}x")

    csv_path = OUT_DIR / "tvd_multichannel_gcc_variance_vs_crlb_data.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["snr_db", "K", "var_empirical_samples2",
                                           "var_crlb_samples2", "ratio_to_crlb"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {csv_path}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    cmap = plt.get_cmap("viridis")
    for i, snr in enumerate(SNR_VALUES):
        color = cmap(i / max(1, len(SNR_VALUES) - 1))
        ax1.plot(K_LIST, empirical_by_snr_k[snr], "o-", color=color, lw=1.6, ms=5,
                  label=f"empirical, SNR={snr:.0f} dB")
        ax1.plot(K_LIST, crlb_by_snr_k[snr], "--", color=color, lw=1.2, alpha=0.8)
    ax1.set_yscale("log")
    ax1.set_xlabel("K (channels)")
    ax1.set_ylabel("Var(theta_hat) [samples²]")
    ax1.set_title("Empirical variance (solid) vs. CRLB (dashed)", fontsize=9)
    ax1.set_xticks(K_LIST)
    ax1.legend(fontsize=6.5, framealpha=0.9)
    ax1.grid(alpha=0.2, which="both")

    for i, snr in enumerate(SNR_VALUES):
        color = cmap(i / max(1, len(SNR_VALUES) - 1))
        ratio = np.array(empirical_by_snr_k[snr]) / np.array(crlb_by_snr_k[snr])
        ax2.plot(K_LIST, ratio, "o-", color=color, lw=1.6, ms=5, label=f"SNR={snr:.0f} dB")
    ax2.axhline(1.0, color="k", ls="--", lw=1.0)
    ax2.set_xlabel("K (channels)")
    ax2.set_ylabel("Var(theta_hat) / CRLB")
    ax2.set_title("Ratio to CRLB (1.0 = efficient)", fontsize=9)
    ax2.set_xticks(K_LIST)
    ax2.legend(fontsize=8, framealpha=0.9)
    ax2.grid(alpha=0.2)

    fig.suptitle(f"GCC-pooled variance vs. multichannel CRLB (Slepian-Bangs), "
                 f"constant CV0={CV0} m/s, SEG_LEN={SEG_LEN}-sample window", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = OUT_DIR / "tvd_multichannel_gcc_variance_vs_crlb.pdf"
    fig.savefig(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
