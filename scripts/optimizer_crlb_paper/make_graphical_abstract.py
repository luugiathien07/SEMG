"""Graphical abstract for PAPER_Optimizer_CRLB (MBEC), from journal_edit/optim_benchmark_data.csv.

Run from the repository root:
    Motionlab/.venv/bin/python scripts/optimizer_crlb_paper/make_graphical_abstract.py
Writes journal_edit/graphical_abstract.{pdf,tif}.
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
ACCENT, NEUTRAL = "#1f6feb", "#8c959f"

times = defaultdict(list)
with open(OUT / "optim_benchmark_data.csv") as f:
    for r in csv.DictReader(f):
        times[r["method"]].append(float(r["time_ms"]))
labels = {"grid+golden": "Golden-section", "Brent": "Brent", "Newton": "Newton-Raphson",
          "SA": "Simulated annealing", "GA": "Genetic algorithm", "PSO": "Particle swarm"}
order = ["Brent", "Newton", "grid+golden", "GA", "SA", "PSO"]
mean_t = [sum(times[m]) / len(times[m]) for m in order]

# MBEC graphical abstract: 32.93 x 37.63 (width x height), drawn at 4x for legibility.
W, H = 32.93 * 4 / 25.4, 37.63 * 4 / 25.4
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "text.color": INK,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": INK})
fig = plt.figure(figsize=(W, H))
fig.patch.set_facecolor("white")

fig.text(0.5, 0.965, "Which optimizer for ML conduction-velocity estimation?",
         ha="center", va="top", fontsize=9, weight="bold")

# Pipeline strip
steps = ["Multichannel\nsEMG (K = 13)", "80-point\ncoarse grid", "Local\nrefinement", "CV vs.\nCRLB"]
x0, bw, gap, y, bh = 0.04, 0.2, 0.04, 0.80, 0.09
for i, s in enumerate(steps):
    x = x0 + i * (bw + gap)
    hl = i == 2
    fig.patches.append(FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0.005,rounding_size=0.01",
                                      transform=fig.transFigure, fc="#ddf4ff" if hl else "#f6f8fa",
                                      ec=ACCENT if hl else GRID, lw=1.0))
    fig.text(x + bw / 2, y + bh / 2, s, ha="center", va="center", fontsize=7,
             weight="bold" if hl else "normal")
    if i < len(steps) - 1:
        fig.text(x + bw + gap / 2, y + bh / 2, "→", ha="center", va="center", fontsize=10, color=MUTED)
fig.text(0.5, 0.765, "Monte Carlo: colored-noise and Farina-Merletti MUAP sources, SNR 10-25 dB",
         ha="center", va="top", fontsize=6.5, color=MUTED)

# Bar chart: time per call (single measure, one axis)
ax = fig.add_axes([0.36, 0.43, 0.58, 0.28])
ypos = range(len(order))[::-1]
colors = [ACCENT if m in ("Brent", "Newton") else NEUTRAL for m in order]
ax.barh(list(ypos), mean_t, color=colors, height=0.62)
ax.set_xscale("log")
ax.set_yticks(list(ypos))
ax.set_yticklabels([labels[m] for m in order], fontsize=7)
ax.set_xlabel("Time per refinement call [ms], log scale", fontsize=6.8, color=MUTED)
for yy, t in zip(ypos, mean_t):
    ax.text(t * 1.12, yy, f"{t:.2g}", va="center", fontsize=6.5, color=INK)
ax.set_xlim(0.5, 200)
ax.set_xticks([1, 10, 100])
ax.set_xticklabels(["1", "10", "100"])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="x", color=GRID, lw=0.5)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)
fig.text(0.04, 0.735, "Constant delay: all six equally accurate (error variance 2.0-2.7× CRLB)",
         fontsize=7, weight="bold", va="top")

# Findings
lines = [
    ("Constant delay", "Brent / Newton-Raphson: 2-2.6× faster than golden-section,\nno accuracy cost. Population methods: 10-26× slower."),
    ("Time-varying delay", "Ranking reverses: BFGS and GA stay within ≈3× CRLB;\nPSO occasionally diverges; SA levels off near 4% error."),
    ("Non-polynomial delay", "34% SNR-independent bias for every optimizer,\nremoved by local instead of global polynomial fitting."),
]
yy = 0.34
for head, body in lines:
    fig.text(0.04, yy, head, fontsize=7.2, weight="bold", color=ACCENT, va="top")
    fig.text(0.04, yy - 0.028, body, fontsize=6.8, va="top", linespacing=1.3)
    yy -= 0.105

fig.savefig(OUT / "graphical_abstract.pdf")
fig.savefig(OUT / "graphical_abstract.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
print("wrote", OUT / "graphical_abstract.pdf", OUT / "graphical_abstract.tif")
