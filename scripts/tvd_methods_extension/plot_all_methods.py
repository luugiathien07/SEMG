#!/usr/bin/env python3
"""
plot_all_methods.py -- figures and paired statistics for the 16-estimator
TVD benchmark (PAPER_TVD_Benchmark.tex).

Reads journal_edit/tvd_all_methods_summary.csv (analyze_all_methods.py) and
the per-realization errors in journal_edit/tvd_all_methods_parts/*.npz.
Writes to journal_edit/:
  bench_rmse_vs_snr.pdf        RMSE vs SNR, 3 sources x 2 truths, 7 estimators
  bench_var_crlb.pdf           error variance / CRLB, colored source, vs SNR
  bench_bias_profile.pdf       mean % error along the window, sinusoidal truth
  bench_speed_accuracy.pdf     mean rank on RMSE vs ms per realization
  bench_paired_data.csv        paired bootstrap of RMSE differences against
                               poly_local, per cell (95% CI, B=2000)
  bench_rank_data.csv          mean rank per method on RMSE and median |error|,
                               overall and per truth

    Motionlab/.venv/bin/python scripts/tvd_methods_extension/plot_all_methods.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "journal_edit"
PARTS = OUT / "tvd_all_methods_parts"

S = pd.read_csv(OUT / "tvd_all_methods_summary.csv")

KEY = ["poly_local", "dp", "lap_kalman", "constmle", "legendre5", "gcc", "global_bfgs"]
LABEL = {"poly_local": "Local order-1", "dp": "DP", "lap_kalman": "LAP+Kalman",
         "constmle": "Local order-0 MLE", "legendre5": "Legendre-5", "gcc": "GCC (CC)",
         "global_bfgs": "Global order-1", "global_sa": "Global order-1 (SA)", "lap": "LAP",
         "cohf": "CohF", "dll": "DLL", "gcc_phat": "GCC-PHAT", "gcc_scot": "GCC-SCOT",
         "gcc_roth": "GCC-Roth", "gcc_ht": "GCC-HT", "gcc_eckart": "GCC-Eckart"}
# reference categorical palette (dataviz skill), fixed order, with markers as
# secondary encoding for print and colour-vision deficiency
COLOR = dict(zip(KEY, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]))
MARK = dict(zip(KEY, ["o", "s", "^", "D", "v", "P", "X"]))
SRC = {"colored": "Colored noise", "muap": "Single MUAP", "multipop": "Multi-MU population"}
TRU = {"ramp": "linear ramp", "sinusoid": "sinusoidal"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4,
    "lines.markersize": 4, "pdf.fonttype": 42,
})


def _grid(ax):
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- RMSE vs SNR
fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), sharex=True, sharey=True)
for j, src in enumerate(SRC):
    for i, tru in enumerate(TRU):
        ax = axes[i, j]
        d = S[(S.source == src) & (S.truth == tru)]
        for m in KEY:
            r = d[d.method == m].sort_values("snr_db")
            ax.plot(r.snr_db, r.rmse, color=COLOR[m], marker=MARK[m], label=LABEL[m])
        ax.set_yscale("log")
        ax.set_title(f"{SRC[src]}, {TRU[tru]}", color=INK)
        _grid(ax)
        if i == 1:
            ax.set_xlabel("SNR (dB)")
        if j == 0:
            ax.set_ylabel("Trajectory RMSE (%)")
axes[0, 0].set_xticks([10, 15, 20, 25])
axes[0, 0].set_yticks([1, 2, 5, 10, 20])
axes[0, 0].set_yticklabels(["1", "2", "5", "10", "20"])
h, lab = axes[0, 0].get_legend_handles_labels()
fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(OUT / "bench_rmse_vs_snr.pdf", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------ variance / CRLB
fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.6), sharey=True)
for ax, tru in zip(axes, TRU):
    d = S[(S.source == "colored") & (S.truth == tru)]
    for m in KEY:
        r = d[d.method == m].sort_values("snr_db")
        ax.plot(r.snr_db, r.var_crlb, color=COLOR[m], marker=MARK[m], label=LABEL[m])
    ax.axhline(1.0, color=MUTED, lw=1, ls="--")
    ax.set_yscale("log")
    ax.set_title(f"Colored noise, {TRU[tru]}", color=INK)
    ax.set_xlabel("SNR (dB)")
    ax.set_xticks([10, 15, 20, 25])
    _grid(ax)
axes[0].set_ylabel("Error variance / CRLB")
axes[1].legend(loc="upper left", frameon=False, bbox_to_anchor=(1.0, 1.0))
fig.tight_layout()
fig.savefig(OUT / "bench_var_crlb.pdf", bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------- bias profile
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
for ax, src in zip(axes, SRC):
    z = np.load(PARTS / f"{src}_sinusoid_20dB_M500.npz")
    meths = [str(x) for x in z["methods"]]
    n, E = z["n_eval"], z["err"].astype(float)
    keep = n >= 150 if src == "multipop" else np.ones(len(n), bool)
    for m in KEY:
        ax.plot(n[keep], E[:, meths.index(m), keep].mean(axis=0), color=COLOR[m],
                marker=MARK[m], markevery=5, label=LABEL[m])
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_title(f"{SRC[src]}, sinusoidal, 20 dB", color=INK)
    ax.set_xlabel("Sample $n$")
    _grid(ax)
axes[0].set_ylabel("Mean MFCV error (%)")
h, lab = axes[0].get_legend_handles_labels()
fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.17, 1, 1))
fig.savefig(OUT / "bench_bias_profile.pdf", bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------------- ranks
rows = []
for scope, d in [("all", S), ("ramp", S[S.truth == "ramp"]), ("sinusoid", S[S.truth == "sinusoid"])]:
    for k in ("rmse", "median_abs"):
        r = d.assign(rank=d.groupby(["source", "truth", "snr_db"])[k].rank())
        for m, v in r.groupby("method")["rank"].mean().items():
            rows.append(dict(scope=scope, metric=k, method=m, mean_rank=round(v, 3)))
R = pd.DataFrame(rows)
R.to_csv(OUT / "bench_rank_data.csv", index=False)

# ----------------------------------------------------------- speed/accuracy
rk = R[(R.scope == "all") & (R.metric == "rmse")].set_index("method").mean_rank
ms = S.groupby("method").ms.mean()
fig, ax = plt.subplots(figsize=(4.6, 3.2))
ax.scatter(ms[rk.index], rk, s=28, color="#2a78d6", edgecolor="white", linewidth=1, zorder=3)
off = {"gcc_scot": (4, -9), "gcc_phat": (4, 3), "global_sa": (4, -8), "lap": (4, -8),
       "legendre5": (-12, -10), "constmle": (-30, 6), "global_bfgs": (-20, 6), "poly_local": (-55, 4)}
for m in rk.index:
    ax.annotate(LABEL[m], (ms[m], rk[m]), xytext=off.get(m, (4, 2)), textcoords="offset points",
                fontsize=6.5, color=INK)
ax.set_xscale("log")
ax.set_xlabel("Mean run time per realization (ms, log scale)")
ax.set_ylabel("Mean rank on RMSE over 24 cells")
ax.invert_yaxis()
_grid(ax)
fig.tight_layout()
fig.savefig(OUT / "bench_speed_accuracy.pdf", bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------- paired bootstrap vs ref
rng = np.random.default_rng(0)
B, REF = 2000, "poly_local"
rows = []
for p in sorted(PARTS.glob("*.npz")):
    z = np.load(p)
    src, tru, snr = str(z["source"]), str(z["truth"]), float(z["snr_db"])
    meths = [str(x) for x in z["methods"]]
    n, E = z["n_eval"], z["err"].astype(float)
    keep = n >= 150 if src == "multipop" else np.ones(len(n), bool)
    E = E[:, :, keep]
    msq = np.mean(E ** 2, axis=2)                     # (M, methods): per-realization MSE
    M = msq.shape[0]
    idx = rng.integers(0, M, size=(B, M))
    boot = np.sqrt(msq[idx].mean(axis=1))             # (B, methods)
    iref = meths.index(REF)
    for i, m in enumerate(meths):
        if m == REF:
            continue
        diff = boot[:, i] - boot[:, iref]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        rows.append(dict(source=src, truth=tru, snr_db=snr, method=m,
                         rmse=np.sqrt(msq[:, i].mean()), rmse_ref=np.sqrt(msq[:, iref].mean()),
                         diff=np.sqrt(msq[:, i].mean()) - np.sqrt(msq[:, iref].mean()),
                         ci_lo=lo, ci_hi=hi))
P = pd.DataFrame(rows)
P.to_csv(OUT / "bench_paired_data.csv", index=False)
P["verdict"] = np.where(P.ci_hi < 0, "better", np.where(P.ci_lo > 0, "worse", "tie"))
print("cells where each method is better / tied / worse than poly_local (95% paired bootstrap):")
print(P.groupby("method").verdict.value_counts().unstack(fill_value=0).to_string())
print("\ncells where a method beats poly_local:")
print(P[P.verdict == "better"][["source", "truth", "snr_db", "method", "rmse", "rmse_ref", "ci_lo", "ci_hi"]]
      .round(2).to_string(index=False))
print("\nmean rank (RMSE / median_abs) by scope:")
print(R.pivot_table(index="method", columns=["metric", "scope"], values="mean_rank").round(2)
      .sort_values(("rmse", "all")).to_string())
print("\nms per realization:")
print(ms.sort_values().round(1).to_string())
