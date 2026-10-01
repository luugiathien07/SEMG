"""
Compares four members of the classical Generalized Cross-Correlation (GCC)
family (Knapp & Carter 1976) -- which differ only in the frequency-domain
WEIGHTING applied to the cross-spectrum before the inverse FFT -- on the
exact same constant-delay, K-channel, variance-vs-CRLB test as
plot_variance_vs_crlb_gcc.py:

  - CC   (plain, what every other script in this paper calls "GCC" /
          _gcc_delay in src/mfcv.py): W(f) = 1, no weighting.
  - PHAT (Phase Transform): W(f) = 1/|Gx1x2(f)|, keeps phase only.
  - SCOT (Smoothed Coherence Transform): W(f) = 1/sqrt(Gx1x1(f) Gx2x2(f)).
  - Roth: W(f) = 1/Gx1x1(f) (asymmetric: whitens only the reference channel).

Two well-known GCC variants are deliberately NOT included: the
Hannan-Thomson/ML weighting and the Eckart (SNR-optimal) filter both need a
SMOOTHED coherence or noise-PSD estimate (Gxy/sqrt(Gxx Gyy) computed from
several averaged sub-segments) to be well-defined -- from a single 100-sample
raw periodogram (this benchmark's SEG_LEN, matching every other local
estimator in this paper), the single-snapshot coherence estimate is
identically 1 at every frequency, which makes the ML weight's
1/(1-|gamma|^2) term blow up. gcc_glissant.m's own "CohF"/Eckart
implementation sidesteps this the same way real deployments would: cpsd()
with Welch averaging over a much longer signal. Reproducing that fairly
would need a second, separate benchmark (a longer window, not this one's
100-sample local-estimator convention), so it is left out here rather than
computed on a window it is not well-posed for.

All four weightings above are cheap (one extra elementwise array op per
delay call on top of _gcc_delay's own FFT cross-correlation + parabolic
sub-sample interpolation), so all four are benchmarked at every (SNR, K)
cell already used for the plain-GCC-vs-CRLB result.
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

OUT_DIR = PROJECT_ROOT / "journal_edit"
K_LIST = list(range(2, 12))
GCC_MAX_LAG = 40
REPS = 500
MID_START = (N - SEG_LEN) // 2
EPS = 1e-12

WEIGHTINGS = ["cc", "phat", "scot", "roth"]
LABELS = {"cc": "GCC (plain CC)", "phat": "GCC-PHAT", "scot": "GCC-SCOT", "roth": "GCC-Roth"}
COLORS = {"cc": "#E08A1E", "phat": "#0B2A4A", "scot": "#8E44AD", "roth": "#3C9D45"}


def gcc_family_delay(x1: np.ndarray, x2: np.ndarray, weighting: str, max_lag: int = GCC_MAX_LAG) -> float:
    x1 = x1 - x1.mean()
    x2 = x2 - x2.mean()
    n = len(x1)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    X1 = np.fft.rfft(x1, nfft)
    X2 = np.fft.rfft(x2, nfft)
    Gx1x2 = X1 * np.conj(X2)
    if weighting == "cc":
        W = 1.0
    elif weighting == "phat":
        W = 1.0 / np.maximum(np.abs(Gx1x2), EPS)
    elif weighting == "scot":
        Gx1x1 = np.abs(X1) ** 2
        Gx2x2 = np.abs(X2) ** 2
        W = 1.0 / np.maximum(np.sqrt(Gx1x1 * Gx2x2), EPS)
    elif weighting == "roth":
        Gx1x1 = np.abs(X1) ** 2
        W = 1.0 / np.maximum(Gx1x1, EPS)
    else:
        raise ValueError(weighting)

    R = np.fft.irfft(Gx1x2 * W, nfft)
    R = np.concatenate((R[-max_lag:], R[: max_lag + 1]))
    lags = np.arange(-max_lag, max_lag + 1)

    i = int(np.argmax(R))
    if 0 < i < len(R) - 1:
        y0, y1, y2 = R[i - 1], R[i], R[i + 1]
        den = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    else:
        frac = 0.0
    return -float(lags[i] + frac)


def const_theta_fn(n: np.ndarray) -> np.ndarray:
    theta0 = CFG.ied_m / CV0 * CFG.fs
    return np.full_like(np.asarray(n, dtype=float), theta0)


def pooled_delay(seg: np.ndarray, weighting: str) -> float:
    K = seg.shape[0]
    delays = [gcc_family_delay(seg[k], seg[k + 1], weighting, max_lag=GCC_MAX_LAG) for k in range(K - 1)]
    return float(np.mean(delays))


def run_snr_k(snr_db: float, K: int, weighting: str, reps: int = REPS, seed0: int = 0) -> np.ndarray:
    thetas = np.full(reps, np.nan)
    for r in range(reps):
        rng = np.random.default_rng(seed0 + r)
        src = shwedyk_semg(N, CFG.fs, rng)
        cols = make_multichannel_pair(rng, snr_db, src, const_theta_fn, n_rows=K)
        seg = cols[:, MID_START:MID_START + SEG_LEN]
        thetas[r] = pooled_delay(seg, weighting)
    return thetas


def main() -> None:
    rows = []
    var_by_w_snr_k = {w: {snr: [] for snr in SNR_VALUES} for w in WEIGHTINGS}
    crlb_by_snr_k = {snr: [] for snr in SNR_VALUES}

    for snr in SNR_VALUES:
        seed0 = int(snr * 137)
        for K in K_LIST:
            var_crlb = crlb_theta(CV0, snr, n=SEG_LEN, n_rows=K)
            crlb_by_snr_k[snr].append(var_crlb)
            print(f"--- SNR={snr:.0f} dB, K={K} ---")
            for w in WEIGHTINGS:
                thetas = run_snr_k(snr, K, w, seed0=seed0 + K * 1000)
                thetas = thetas[np.isfinite(thetas)]
                var_emp = float(np.var(thetas)) if len(thetas) else np.nan
                ratio = var_emp / var_crlb if np.isfinite(var_crlb) and var_crlb > 0 else np.nan
                var_by_w_snr_k[w][snr].append(var_emp)
                rows.append(dict(snr_db=snr, K=K, weighting=LABELS[w],
                                  var_empirical_samples2=var_emp, var_crlb_samples2=var_crlb,
                                  ratio_to_crlb=ratio))
                print(f"  {LABELS[w]:16s}: Var_emp={var_emp:.5f}  ratio={ratio:.2f}x")

    csv_path = OUT_DIR / "tvd_multichannel_gcc_family_compare_data.csv"
    with open(csv_path, "w", newline="") as f:
        w_csv = csv.DictWriter(f, fieldnames=["snr_db", "K", "weighting", "var_empirical_samples2",
                                               "var_crlb_samples2", "ratio_to_crlb"])
        w_csv.writeheader()
        for r in rows:
            w_csv.writerow(r)
    print(f"Wrote {csv_path}")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    axes = axes.ravel()
    for ax, snr in zip(axes, SNR_VALUES):
        for w in WEIGHTINGS:
            ratio = np.array(var_by_w_snr_k[w][snr]) / np.array(crlb_by_snr_k[snr])
            ax.plot(K_LIST, ratio, "o-", color=COLORS[w], lw=2.0, ms=6, label=LABELS[w])
        ax.axhline(1.0, color="k", ls="--", lw=1.2)
        ax.set_yscale("log")
        ax.set_title(f"SNR={snr:.0f} dB", fontsize=12)
        ax.set_xticks(K_LIST)
        ax.tick_params(labelsize=10)
        ax.grid(alpha=0.25, which="both")
    for ax in axes[2:]:
        ax.set_xlabel("K (channels)", fontsize=11)
    for ax in axes[::2]:
        ax.set_ylabel("Var(theta_hat) / CRLB (log scale)", fontsize=11)
    axes[0].legend(fontsize=10, framealpha=0.9, loc="upper left")
    fig.suptitle("GCC family (CC/PHAT/SCOT/Roth): ratio to multichannel CRLB, "
                 f"constant CV0={CV0} m/s, SEG_LEN={SEG_LEN}-sample window", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT_DIR / "tvd_multichannel_gcc_family_compare.pdf"
    fig.savefig(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
