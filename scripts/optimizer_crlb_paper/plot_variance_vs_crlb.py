#!/usr/bin/env python3
"""
plot_variance_vs_crlb.py -- redraws every estimator-vs-CRLB figure of
PAPER_Optimizer_CRLB.tex (and the two global-fit figures of
PAPER_TVD_Interaction.tex) from the existing journal_edit/*_data.csv files,
pairing the CRLB with the empirical error VARIANCE rather than with RMSE.

The CRLB bounds Var(theta_hat); RMSE^2 = Var + bias^2, so RMSE is only
comparable to it when the estimator is unbiased. Both axes here are on the
variance scale in %CV^2: empirical Var = sd_pct^2 (the CSV column), and the
bound is the square of the %-SD CRLB used elsewhere in these scripts.

No Monte Carlo is rerun; only the analytic CRLBs are recomputed.

Figures written (journal_edit/):
  crlb_vs_snr.pdf                         scalar, six optimizers, CV=4.5 m/s
  tvd_vs_snr.pdf                          TVD, linear ramp, colored noise
  tvd_vs_snr_muap.pdf                     TVD, linear ramp, MUAP source
  tvd_vs_snr_sinusoidal.pdf               TVD, sinusoidal, global fit
  tvd_vs_snr_sinusoidal_sliding.pdf       TVD, sinusoidal, global vs local
  tvd_muap_sinusoidal_global.pdf          (interaction paper)
  tvd_multipop_sinusoidal_global.pdf      (interaction paper)
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.optimizer_crlb_paper.crlb_optimizer_benchmark import (  # noqa: E402
    crlb_pct_error_sd, crlb_theta,
)
from scripts.optimizer_crlb_paper.crlb_tvd_benchmark import (  # noqa: E402
    CFG, CV0, N, SEG_LEN, SNR_VALUES, crlb_linear_fim, crlb_sinusoidal_theta_pct,
    true_theta, true_theta_sinusoidal,
)

OUT = REPO / "journal_edit"
YLABEL = r"Error variance in CV [%$^2$]  (log scale)"


def read_var(csv_name, key="method", filt=None):
    """{method: {snr: Var [%^2]}} from a *_data.csv (Var = sd_pct^2)."""
    out = defaultdict(dict)
    with open(OUT / csv_name) as f:
        for r in csv.DictReader(f):
            if filt and not filt(r):
                continue
            out[r[key]][float(r["snr_db"])] = float(r["sd_pct"]) ** 2
    return out


def crlb_var_poly(theta_fn, window_len, local_end):
    """Order-1 polynomial CRLB on theta at sample local_end, as %CV^2."""
    theta_end = theta_fn(N - 1)
    out = {}
    for snr in SNR_VALUES:
        var_d0, var_d1 = crlb_linear_fim(snr, cv_true=CV0, window_len=window_len)
        var_theta = var_d0 + local_end ** 2 * var_d1
        out[snr] = (100.0 / theta_end) ** 2 * var_theta
    return out


def crlb_var_const(theta_fn, window_len):
    """Order-0 (constant delay) CRLB, K=2 channels, as %CV^2."""
    theta_end = theta_fn(N - 1)
    return {snr: (100.0 / theta_end) ** 2
            * crlb_theta(CV0, snr, n=window_len, fs=CFG.fs, ied_m=CFG.ied_m, n_rows=2)
            for snr in SNR_VALUES}


def finish(ax, fig, title, pdf_name, legend_fs=8):
    ax.set_yscale("log")
    ax.set_xlabel("SNR [dB]")
    ax.set_ylabel(YLABEL)
    ax.set_title(title, loc="left", fontsize=10)
    ax.grid(alpha=0.2, which="both")
    ax.legend(fontsize=legend_fs, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / pdf_name)
    plt.close(fig)
    print(f"Wrote {OUT / pdf_name}")


def plot_scalar():
    var = read_var("optim_benchmark_data.csv", filt=lambda r: float(r["cv_true"]) == 4.5)
    colors = {"grid+golden": "#0B2A4A", "Brent": "#028090", "Newton": "#3C9D45",
              "SA": "#E08A1E", "GA": "#B23A48", "PSO": "#6A4C93"}
    snrs = sorted(next(iter(var.values())))
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for m, c in colors.items():
        ax.plot(snrs, [var[m][s] for s in snrs], "o-", color=c, label=m, lw=1.8, ms=5)
    ax.plot(snrs, [crlb_pct_error_sd(4.5, s) ** 2 for s in snrs], "k--",
            label="CRLB (CV=4.5 m/s)", lw=1.6)
    finish(ax, fig, "Optimizer comparison vs. Cramer-Rao bound (CV=4.5 m/s)",
           "crlb_vs_snr.pdf")


TVD_COLORS = {"BFGS": "#0B2A4A", "SA": "#E08A1E", "GA": "#B23A48", "PSO": "#6A4C93"}


def plot_tvd_global(csv_name, pdf_name, title, theta_fn, sinusoidal_bound=False):
    var = read_var(csv_name)
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for m, c in TVD_COLORS.items():
        ax.plot(SNR_VALUES, [var[m][s] for s in SNR_VALUES], "o-", color=c,
                label=m, lw=1.8, ms=5)
    poly = crlb_var_poly(theta_fn, N, N - 1)
    ax.plot(SNR_VALUES, [poly[s] for s in SNR_VALUES], "k--",
            label="CRLB, order-1 polynomial model", lw=1.6)
    if sinusoidal_bound:
        ax.plot(SNR_VALUES,
                [crlb_sinusoidal_theta_pct(s, window_len=N, n_eval=N - 1, cv_op=CV0) ** 2
                 for s in SNR_VALUES],
                "-.", color="#8E44AD", label="CRLB, sinusoidal model", lw=1.6)
    finish(ax, fig, title, pdf_name)


def plot_sliding():
    glob = read_var("tvd_benchmark_sinusoidal_data.csv")["BFGS"]
    loc = read_var("tvd_benchmark_sinusoidal_sliding_data.csv")
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(SNR_VALUES, [glob[s] for s in SNR_VALUES], "o-", color="#B23A48",
            label="Single global order-1 fit (BFGS)", lw=1.8, ms=5)
    series = [("MLE-sliding-poly(BFGS)", "#0B2A4A", f"Local poly, order-1 ({SEG_LEN}-smp, BFGS)"),
              ("MLE-sliding-constdelay(Brent)", "#3C9D45",
               f"Local const-delay, order-0 ({SEG_LEN}-smp, grid+Brent)"),
              ("LAP(Gilliam2018)", "#8E44AD", f"Local All-Pass, Gilliam et al. 2018 ({SEG_LEN}-smp)"),
              ("GCC(Knapp1976)", "#E08A1E", f"Local GCC (Knapp and Carter 1976, {SEG_LEN}-smp)")]
    for key, c, label in series:
        ax.plot(SNR_VALUES, [loc[key][s] for s in SNR_VALUES], "o-", color=c,
                label=label, lw=1.8, ms=5)
    for bound, c, label in (
            (crlb_var_poly(true_theta_sinusoidal, N, N - 1), "#B23A48", f"CRLB, {N}-sample window"),
            (crlb_var_poly(true_theta_sinusoidal, SEG_LEN, SEG_LEN - 1), "#0B2A4A",
             f"CRLB, {SEG_LEN}-sample, order-1"),
            (crlb_var_const(true_theta_sinusoidal, SEG_LEN), "#3C9D45",
             f"CRLB, {SEG_LEN}-sample, order-0")):
        ax.plot(SNR_VALUES, [bound[s] for s in SNR_VALUES], "--", color=c, label=label, lw=1.2)
    finish(ax, fig, "Non-polynomial ground truth: global vs. local strategies",
           "tvd_vs_snr_sinusoidal_sliding.pdf", legend_fs=7)


if __name__ == "__main__":
    plot_scalar()
    plot_tvd_global("tvd_benchmark_data.csv", "tvd_vs_snr.pdf",
                    "TVD, linear ramp: optimizer comparison", true_theta)
    plot_tvd_global("tvd_benchmark_muap_data.csv", "tvd_vs_snr_muap.pdf",
                    "TVD, MUAP source, linear ramp", true_theta)
    plot_tvd_global("tvd_benchmark_sinusoidal_data.csv", "tvd_vs_snr_sinusoidal.pdf",
                    "TVD, non-polynomial ground truth: global order-1 fit",
                    true_theta_sinusoidal, sinusoidal_bound=True)
    plot_sliding()
    plot_tvd_global("tvd_muap_sinusoidal_global_data.csv", "tvd_muap_sinusoidal_global.pdf",
                    "TVD, MUAP source x sinusoidal ground truth", true_theta_sinusoidal)
    plot_tvd_global("tvd_multipop_sinusoidal_global_data.csv",
                    "tvd_multipop_sinusoidal_global.pdf",
                    "TVD, multi-MU-population source x sinusoidal ground truth",
                    true_theta_sinusoidal)
