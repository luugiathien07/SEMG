"""
Extends plot_synthetic_cv_timeseries_gcc_fine.py's K=2-vs-K=11 comparison to
a full channel-count sweep, K=2,3,...,11: mean RMSE (hop=1 sample, matching
gcc_glissant.m's pas=1 convention, t>=BOUNDARY_CUTOFF_S to exclude the
boundary-extrapolation artifact documented there) as a function of K, one
line per SNR -- the direct "does pooling more channels help" question,
answered against a known ground truth and swept continuously in K rather
than only at the two endpoints K=2 and K=11 used elsewhere in this paper's
GCC-pooled sliding-window figures.

Each Monte Carlo realization still draws ONE n_rows=11 column
(make_multichannel_pair) and every K<=11 reuses that realization's first K
channels, so all K values at a given (SNR, realization) are directly
comparable (same underlying noise/source draw), not independent samples.
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
    CFG, N, SNR_VALUES, SEG_LEN,
    make_multichannel_pair, shwedyk_semg, true_theta_sinusoidal,
)
from src.mfcv import _gcc_delay  # noqa: E402

OUT_DIR = PROJECT_ROOT / "journal_edit"
K_LIST = list(range(2, 12))  # 2..11
GCC_MAX_LAG = 40
HOP = 1
REPS = 500
THETA_FN = true_theta_sinusoidal
BOUNDARY_CUTOFF_S = 0.07


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


def run_snr(snr_db: float, theta_fn, reps: int = REPS, seed0: int = 0) -> dict[int, np.ndarray]:
    """dict[K] -> (reps, n_segments) array of estimated CV [m/s]."""
    starts = fine_segment_starts()
    n_seg = len(starts)
    out = {K: np.full((reps, n_seg), np.nan) for K in K_LIST}
    K_max = max(K_LIST)
    for r in range(reps):
        rng = np.random.default_rng(seed0 + r)
        src = shwedyk_semg(N, CFG.fs, rng)
        cols_full = make_multichannel_pair(rng, snr_db, src, theta_fn, n_rows=K_max)
        for K in K_LIST:
            cols = cols_full[:K]
            for si, start in enumerate(starts):
                seg = cols[:, start:start + SEG_LEN]
                theta = gcc_pooled_delay(seg)
                if abs(theta) < 1e-9:
                    continue
                out[K][r, si] = CFG.ied_m / (abs(theta) / CFG.fs)
    return out


def main() -> None:
    t_s = segment_end_samples() / CFG.fs
    cv_true = true_cv_at_segments(THETA_FN)
    summary_mask = t_s >= BOUNDARY_CUTOFF_S

    rows = []
    mean_rmse_by_snr_k = {snr: [] for snr in SNR_VALUES}
    for snr in SNR_VALUES:
        seed0 = int(snr * 137)
        print(f"--- synthetic GCC, hop={HOP}, K=2..{max(K_LIST)}, SNR={snr:.0f} dB ---")
        est = run_snr(snr, THETA_FN, seed0=seed0)
        for K in K_LIST:
            rmse = np.sqrt(np.nanmean((est[K] - cv_true[None, :]) ** 2, axis=0))
            mean_rmse = float(np.nanmean(rmse[summary_mask]))
            mean_rmse_by_snr_k[snr].append(mean_rmse)
            rows.append(dict(snr_db=snr, K=K, mean_rmse=mean_rmse))
            print(f"  K={K:2d}: mean RMSE = {mean_rmse:.4f} m/s")

    csv_path = OUT_DIR / "tvd_multichannel_synthetic_gcc_rmse_vs_k_data.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["snr_db", "K", "mean_rmse"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {csv_path}")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    cmap = plt.get_cmap("viridis")
    for i, snr in enumerate(SNR_VALUES):
        color = cmap(i / max(1, len(SNR_VALUES) - 1))
        ax.plot(K_LIST, mean_rmse_by_snr_k[snr], "o-", color=color, lw=1.6, ms=5,
                 label=f"SNR={snr:.0f} dB")
    ax.set_xlabel("K (channels)")
    ax.set_ylabel(f"Mean RMSE over window [m/s]\n(t≥{BOUNDARY_CUTOFF_S}s)")
    ax.set_title("Synthetic GCC-pooled CV, hop=1 sample: mean RMSE vs. K", fontsize=9)
    ax.set_xticks(K_LIST)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out = OUT_DIR / "tvd_multichannel_synthetic_gcc_rmse_vs_k.pdf"
    fig.savefig(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
