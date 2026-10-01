#!/usr/bin/env python3
"""
plot_all_methods_bradford.py -- real-data figures for PAPER_TVD_Benchmark.tex
(Bradford 2023 recordings), from journal_edit/tvd_all_methods_bradford.npz,
bench_bradford_precision.csv and tvd_all_methods_summary.csv.

  bench_bradford_example.pdf    one recorded window: the oriented pair of one
                                column, the MFCV trajectory of six estimators in
                                that column, and the window-level MFCV of every
                                accepted column per estimator. The window is
                                chosen by rule: among (window, pair) groups with
                                at least seven columns accepted by every
                                estimator of the comparison set, the one whose
                                local order-1 cross-column spread is closest to
                                that estimator's median over all groups.
  bench_bradford_peaklock.pdf   distribution of the fractional part of the
                                accepted delays (distance to the nearest whole
                                sample), GCC-HT / GCC-Eckart vs the others
  bench_bradford_sim_vs_real.pdf  real-data cross-column spread against the
                                simulated error SD, one panel per source model

    Motionlab/.venv/bin/python scripts/tvd_methods_extension/plot_all_methods_bradford.py
"""
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

warnings.filterwarnings("ignore", "All-NaN slice")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "tvd_interaction_paper"))
OUT = REPO / "journal_edit"
IED_FS = 0.008 * 2048.0

LABEL = {"poly_local": "Local order-1", "dp": "DP", "lap_kalman": "LAP+Kalman",
         "constmle": "Local order-0 MLE", "legendre5": "Legendre-5", "gcc": "GCC (CC)",
         "global_bfgs": "Global order-1", "global_sa": "Global order-1 (SA)", "lap": "LAP",
         "cohf": "CohF", "dll": "DLL", "gcc_ht": "GCC-HT", "gcc_eckart": "GCC-Eckart",
         "gcc_phat": "GCC-PHAT", "gcc_scot": "GCC-SCOT", "gcc_roth": "GCC-Roth"}
SIX = ["poly_local", "gcc", "dp", "global_bfgs", "lap_kalman", "gcc_eckart"]
COLOR = dict(zip(SIX, ["#2a78d6", "#008300", "#eb6834", "#4a3aa7", "#1baf7a", "#e87ba4"]))
MARK = dict(zip(SIX, ["o", "P", "s", "X", "^", "v"]))
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 1.3, "lines.markersize": 4, "pdf.fonttype": 42,
})


def _grid(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


z = np.load(OUT / "tvd_all_methods_bradford.npz", allow_pickle=True)
CV = z["cv"].astype(float)
METH = [str(m) for m in z["methods"]]
N_EVAL = z["n_eval"]
keys = pd.DataFrame({"rec": z["rec"].astype(str), "t_s": z["t_s"], "col": z["col"], "pos": z["pos"]})
ok = (CV >= 2) & (CV <= 10)
res = pd.read_csv(OUT / "bench_bradford_precision.csv", index_col=0)
SET = [m for m in METH if res.acceptance[m] >= 0.5]
idx = [METH.index(m) for m in SET]
win = np.nanmedian(np.where(ok, CV, np.nan), axis=2)
win_ok = ok.mean(axis=2) >= 0.5


def rel_spread(v):
    v = np.asarray(v, float)
    return 100 * 1.4826 * np.median(np.abs(v - np.median(v))) / np.median(v)


# ------------------------------------------------------------- example window
cand = []
for (rec, t, pos), g in keys.groupby(["rec", "t_s", "pos"]):
    ii = g.index.values
    ii = ii[win_ok[np.ix_(ii, idx)].all(axis=1)]
    if len(ii) >= 7:
        cand.append((rel_spread(win[ii, METH.index("poly_local")]), rec, t, pos, ii))
target = res.loc["poly_local", "spread_window"]
sp, rec, t, pos, ii = min(cand, key=lambda c: abs(c[0] - target))
print(f"example: {rec} t={t:.3f}s pos={pos}, {len(ii)} columns ({len(cand)} candidate groups), local order-1 spread {sp:.1f}% (median {target:.2f}%)")

import run_bradford_realdata as rb  # noqa: E402
dd = rb.load_dd(rec)
start = int(round(t * rb.FS / rb.HOP)) * rb.HOP
col_show = int(keys.col[ii[np.argsort(win[ii, METH.index("poly_local")])[len(ii) // 2]]])   # median column
x1, x2 = dd[pos, col_show, start:start + 600], dd[pos + rb.PAIR_STEP, col_show, start:start + 600]
lag = pd.read_csv(OUT / "tvd_bradford_realdata.csv").set_index(["file", "t_s", "col", "pos"]).gcc_lag_full
if lag.loc[(rec, t, col_show, pos)] < 0:
    x1, x2 = x2, x1

fig = plt.figure(figsize=(7.2, 4.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.25], width_ratios=[1.15, 1], hspace=0.45, wspace=0.28)
ax = fig.add_subplot(gs[0, :])
tt = 1000 * np.arange(600) / rb.FS
sc = np.std(x1) * 4
ax.plot(tt, x1 / sc + 1.2, color=INK, lw=0.8)
ax.plot(tt, x2 / sc, color=MUTED, lw=0.8)
ax.set_yticks([1.2, 0])
ax.set_yticklabels(["channel 1", "channel 2"])
ax.set_xlabel("Time within window (ms)")
ax.set_title(f"(a) Recorded pair, {rec.replace('_', ' ')}, t = {t:.1f} s, column {col_show + 1}", color=INK, loc="left")
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)

ax = fig.add_subplot(gs[1, 0])
row = ii[keys.col.values[ii] == col_show][0]
for m in SIX:
    ax.plot(1000 * N_EVAL / rb.FS, CV[row, METH.index(m)], color=COLOR[m], marker=MARK[m],
            markevery=5, label=LABEL[m])
ax.set_xlabel("Time within window (ms)")
ax.set_ylabel("MFCV (m/s)")
ax.set_title(f"(b) MFCV trajectory, column {col_show + 1}", color=INK, loc="left")
_grid(ax)

ax = fig.add_subplot(gs[1, 1])
lab_sp = []
for j, m in enumerate(SIX):
    v = win[ii, METH.index(m)]
    ax.scatter(v, np.full(len(v), j), s=14, color=COLOR[m], marker=MARK[m], edgecolor="white", linewidth=0.5)
    lab_sp.append(f"{rel_spread(v):.1f}%")
ax.set_yticks(range(len(SIX)))
ax.set_yticklabels([LABEL[m] for m in SIX])
ax.invert_yaxis()
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks(range(len(SIX)))
ax2.set_yticklabels(lab_sp)
ax2.tick_params(axis="y", length=0, colors=MUTED)
ax2.spines["right"].set_visible(False)
ax.set_xlabel(f"Window-level MFCV of the {len(ii)} columns (m/s)")
ax.set_title("(c) Across columns (spread at right)", color=INK, loc="left")
_grid(ax)
h, lab = fig.axes[1].get_legend_handles_labels()
fig.legend(h, lab, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.subplots_adjust(bottom=0.16)
fig.savefig(OUT / "bench_bradford_example.pdf", bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------- peak locking
bins = np.linspace(0, 0.5, 11)
fig, ax = plt.subplots(figsize=(4.4, 2.8))
others = [m for m in SET if m not in ("gcc_ht", "gcc_eckart")]
H = []
for m in others:
    c = CV[:, METH.index(m)][ok[:, METH.index(m)]]
    th = IED_FS / c
    H.append(np.histogram(np.abs(th - np.round(th)), bins=bins)[0] / len(th))
H = np.array(H)
ctr = 0.5 * (bins[:-1] + bins[1:])
ax.fill_between(ctr, H.min(0), H.max(0), color="#c3c2b7", alpha=0.6, lw=0,
                label=f"Other {len(others)} estimators (range)")
for m, colr, mk in [("gcc_ht", "#eb6834", "s"), ("gcc_eckart", "#2a78d6", "o")]:
    c = CV[:, METH.index(m)][ok[:, METH.index(m)]]
    th = IED_FS / c
    ax.plot(ctr, np.histogram(np.abs(th - np.round(th)), bins=bins)[0] / len(th), color=colr,
            marker=mk, label=LABEL[m])
ax.axhline(0.1, color=MUTED, lw=0.8, ls="--")
ax.text(0.495, 0.103, "no clustering", ha="right", va="bottom", fontsize=6.5, color=MUTED)
ax.set_xlabel("Distance of the delay from the nearest whole sample")
ax.set_ylabel("Share of accepted estimates")
ax.set_xlim(0, 0.5)
_grid(ax)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "bench_bradford_peaklock.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- sim vs real
S = pd.read_csv(OUT / "tvd_all_methods_summary.csv")
SRC = {"colored": "Colored noise", "muap": "Single MUAP", "multipop": "Multi-MU population"}
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
for ax, src in zip(axes, SRC):
    sd = np.sqrt(S[S.source == src].groupby("method")["var"].mean())
    x, y = sd[SET], res.loc[SET, "spread_window"]
    r, p = spearmanr(x, y)
    ax.scatter(x, y, s=22, color="#2a78d6", edgecolor="white", linewidth=0.8, zorder=3)
    for m in SET:
        if m in ("gcc_ht", "gcc_eckart", "lap", "lap_kalman", "poly_local", "dp"):
            dy = -8 if m == "gcc_eckart" else 2
            ax.annotate(LABEL[m], (x[m], y[m]), xytext=(3, dy), textcoords="offset points", fontsize=6, color=INK)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([2, 5, 10, 20, 50]))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(["2", "5", "10", "20", "50"]))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_title(f"{SRC[src]}\n$\\rho$ = {r:+.2f} (p = {p:.2f})", color=INK)
    ax.set_xlabel("Simulated error SD (%)")
    _grid(ax)
axes[0].set_ylabel("Real cross-column spread (%)")
fig.tight_layout()
fig.savefig(OUT / "bench_bradford_sim_vs_real.pdf", bbox_inches="tight")
plt.close(fig)
print("wrote bench_bradford_example.pdf, bench_bradford_peaklock.pdf, bench_bradford_sim_vs_real.pdf")
