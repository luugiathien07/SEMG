"""Landscape graphical abstract for PAPER_Optimizer_CRLB (MBEC).

Reads journal_edit/optim_benchmark_data.csv (constant delay) and
journal_edit/tvd_benchmark_data.csv (time-varying delay). Run from the repository root:
    Motionlab/.venv/bin/python scripts/optimizer_crlb_paper/make_graphical_abstract_landscape.py
Writes journal_edit/graphical_abstract_landscape.{pdf,tif}.
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import FancyBboxPatch

OUT = Path("journal_edit")
INK, MUTED, GRID = "#1f2328", "#59636e", "#d0d7de"
ACCENT, NEUTRAL = "#2a78d6", "#8c959f"
SERIES = {"BFGS": "#2a78d6", "GA": "#eb6834", "PSO": "#1baf7a", "SA": "#eda100"}
MARKERS = {"BFGS": "o", "GA": "s", "PSO": "^", "SA": "D"}
NAMES = {"BFGS": "BFGS", "GA": "Genetic alg.", "PSO": "Particle swarm", "SA": "Sim. annealing"}

times = defaultdict(list)
with open(OUT / "optim_benchmark_data.csv") as f:
    for r in csv.DictReader(f):
        times[r["method"]].append(float(r["time_ms"]))
labels = {"grid+golden": "Golden-section", "Brent": "Brent", "Newton": "Newton-Raphson",
          "SA": "Sim. annealing", "GA": "Genetic alg.", "PSO": "Particle swarm"}
order = ["Brent", "Newton", "grid+golden", "GA", "SA", "PSO"]
mean_t = [sum(times[m]) / len(times[m]) for m in order]

tvd = defaultdict(dict)
with open(OUT / "tvd_benchmark_data.csv") as f:
    for r in csv.DictReader(f):
        tvd[r["method"]][float(r["snr_db"])] = float(r["rmse_pct"])

# MBEC graphical abstract, landscape: 37.63 x 32.93 (width x height), drawn at 4x.
W, H = 37.63 * 4 / 25.4, 32.93 * 4 / 25.4
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "text.color": INK,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.labelcolor": MUTED})
fig = plt.figure(figsize=(W, H))
fig.patch.set_facecolor("white")

fig.text(0.5, 0.97, "Which optimizer for ML conduction-velocity estimation?",
         ha="center", va="top", fontsize=9.5, weight="bold")

# Pipeline strip
steps = ["Multichannel sEMG\n(K = 13)", "80-point\ncoarse grid", "Local refinement\n(6 optimizers)",
         "Error variance\nvs. CRLB"]
x0, bw, gap, y, bh = 0.035, 0.205, 0.035, 0.80, 0.09
for i, s in enumerate(steps):
    x = x0 + i * (bw + gap)
    hl = i == 2
    fig.patches.append(FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0.005,rounding_size=0.01",
                                      transform=fig.transFigure, fc="#e8f1fc" if hl else "#f6f8fa",
                                      ec=ACCENT if hl else GRID, lw=1.0))
    fig.text(x + bw / 2, y + bh / 2, s, ha="center", va="center", fontsize=6.8,
             weight="bold" if hl else "normal")
    if i < len(steps) - 1:
        fig.text(x + bw + gap / 2, y + bh / 2, "→", ha="center", va="center", fontsize=10, color=MUTED)

# Left panel: constant delay, time per call
fig.text(0.035, 0.735, "Constant delay", fontsize=8, weight="bold", color=ACCENT, va="top")
fig.text(0.035, 0.695, "All six equally accurate (variance 2.0-2.7× CRLB)",
         fontsize=6.5, color=MUTED, va="top")
ax = fig.add_axes([0.2, 0.25, 0.27, 0.40])
ypos = list(range(len(order)))[::-1]
ax.barh(ypos, mean_t, color=[ACCENT if m in ("Brent", "Newton") else NEUTRAL for m in order], height=0.62)
ax.set_xscale("log")
ax.set_xlim(0.5, 250)
ax.set_xticks([1, 10, 100])
ax.set_xticklabels(["1", "10", "100"])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_yticks(ypos)
ax.set_yticklabels([labels[m] for m in order], fontsize=6.6, color=INK)
ax.set_xlabel("Time per call [ms], log scale", fontsize=6.3)
for yy, t in zip(ypos, mean_t):
    ax.text(t * 1.12, yy, f"{t:.2g}", va="center", fontsize=6.2, color=INK)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="x", color=GRID, lw=0.5)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)

# Right panel: time-varying delay, RMSE vs SNR
fig.text(0.54, 0.735, "Time-varying delay: ranking reverses", fontsize=8, weight="bold",
         color=ACCENT, va="top")
fig.text(0.54, 0.695, "BFGS and GA keep improving; SA levels off", fontsize=6.5, color=MUTED, va="top")
ax2 = fig.add_axes([0.6, 0.25, 0.25, 0.40])
for m in ["BFGS", "GA", "PSO", "SA"]:
    snr = sorted(tvd[m])
    ax2.plot(snr, [tvd[m][s] for s in snr], color=SERIES[m], marker=MARKERS[m], ms=3.5, lw=1.4)
    ax2.text(snr[-1] + 0.8, tvd[m][snr[-1]] * {"BFGS": 0.86, "GA": 1.14, "PSO": 1.0, "SA": 1.0}[m],
             NAMES[m], fontsize=6.0, va="center", color=INK)
ax2.set_yscale("log")
ax2.set_ylim(0.7, 20)
ax2.set_yticks([1, 2, 5, 10, 20])
ax2.set_yticklabels(["1", "2", "5", "10", "20"])
ax2.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax2.set_xticks([10, 15, 20, 25])
ax2.set_xlim(9, 26)
ax2.set_xlabel("SNR [dB]", fontsize=6.3)
ax2.set_ylabel("RMSE of CV [%]", fontsize=6.3)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)
ax2.grid(axis="y", color=GRID, lw=0.5)
ax2.set_axisbelow(True)

# Take-home strip
fig.patches.append(FancyBboxPatch((0.035, 0.03), 0.93, 0.1, boxstyle="round,pad=0.005,rounding_size=0.01",
                                  transform=fig.transFigure, fc="#f6f8fa", ec=GRID, lw=0.8))
fig.text(0.5, 0.08, "Use Brent (2.6× faster than golden-section, no accuracy cost) for constant delay.\n"
         "For dynamic contraction the delay model matters more: a non-polynomial delay gives a 34% bias\n"
         "for every optimizer, removed by local instead of global polynomial fitting.",
         ha="center", va="center", fontsize=6.6, linespacing=1.35)

fig.savefig(OUT / "graphical_abstract_landscape.pdf")
fig.savefig(OUT / "graphical_abstract_landscape.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
print("wrote", OUT / "graphical_abstract_landscape.pdf", OUT / "graphical_abstract_landscape.tif")
