"""
Regenerates tvd_bradford_modelselection_angle.pdf (PAPER_TVD_Interaction.tex,
Section results-realchecks, Tables tab:bic, tab:angle and tab:torque / Figure
fig:bradford-checks) from journal_edit/tvd_bradford_realdata.csv,
tvd_bradford_validation.csv, tvd_bradford_validation_angle.csv and
tvd_bradford_validation_torque.csv.

Three panels:
  left   -- BIC-selected delay model: share of inputs won by each of the four
            models (global/local x order-0/order-1), all inputs vs. isokinetic
            vs. isotonic (matches Table tab:bic).
  middle -- per-recording Spearman rho between window-median MFCV and joint
            angle, GCC vs. local polynomial fit, isokinetic vs. isotonic
            (matches Table tab:angle).
  right  -- per-recording Spearman rho between window-median MFCV and
            instantaneous torque, GCC, isotonic recordings only (matches
            Table tab:torque).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "journal_edit"

MODELS = ["g0", "g1", "l0", "l1"]
MODEL_LABELS = {"g0": "Global\norder-0", "g1": "Global\norder-1",
                "l0": "Local\norder-0", "l1": "Local\norder-1"}
MODEL_COLORS = {"g0": "#4C72B0", "g1": "#55A868", "l0": "#C44E52", "l1": "#8172B2"}


def load_bic() -> pd.DataFrame:
    R = pd.read_csv(OUT_DIR / "tvd_bradford_realdata.csv")
    V = pd.read_csv(OUT_DIR / "tvd_bradford_validation.csv")
    D = R.merge(V, on=["file", "t_s", "col", "pos"], how="inner")
    D["cond"] = D.file.str.split("_", n=1).str[1]
    D["kind"] = np.where(D.cond.str.startswith("ISOK"), "isokinetic", "isotonic")
    B = D[[f"bic_{m}" for m in MODELS]].to_numpy()
    ok = np.isfinite(B).all(axis=1)
    D = D[ok].copy()
    D["winner"] = np.array(MODELS)[B[ok].argmin(axis=1)]
    return D


def panel_bic(ax) -> None:
    D = load_bic()
    groups = [("All", D), ("Isokinetic", D[D.kind == "isokinetic"]), ("Isotonic", D[D.kind == "isotonic"])]
    x = np.arange(len(groups))
    width = 0.8 / len(MODELS)
    for j, m in enumerate(MODELS):
        shares = [g.winner.eq(m).mean() * 100 for _, g in groups]
        ax.bar(x + (j - (len(MODELS) - 1) / 2) * width, shares, width=width * 0.95,
               color=MODEL_COLORS[m], label=MODEL_LABELS[m].replace("\n", " "))
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("Share of inputs with lowest BIC [\\%]")
    ax.set_title("BIC-selected delay model", fontsize=9)
    ax.axhline(25, color="0.6", lw=0.7, ls="--", zorder=0)
    ax.legend(loc="upper right", fontsize=6.5, framealpha=0.9)


def panel_angle(ax) -> None:
    A = pd.read_csv(OUT_DIR / "tvd_bradford_validation_angle.csv")
    kinds = ["isokinetic", "isotonic"]
    ests = [("rho_gcc", "GCC", "#DD8452"), ("rho_poly", "Local poly", "#4C72B0")]
    positions, data, colors, ticklabels = [], [], [], []
    pos = 0
    for kind in kinds:
        g = A[A.kind == kind]
        for col, label, color in ests:
            v = g[col].dropna().to_numpy()
            data.append(v)
            positions.append(pos)
            colors.append(color)
            ticklabels.append(f"{label}\n({kind})")
            pos += 1
        pos += 0.6
    bp = ax.boxplot(data, positions=positions, widths=0.7, showfliers=True, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for med in bp["medians"]:
        med.set_color("black")
    for p, v in zip(positions, data):
        jitter = (np.random.RandomState(0).rand(len(v)) - 0.5) * 0.25
        ax.scatter(np.full(len(v), p) + jitter, v, s=10, color="black", alpha=0.5, zorder=3)
    ax.axhline(0, color="0.4", lw=0.8, ls="--", zorder=0)
    ax.set_xticks(positions)
    ax.set_xticklabels(ticklabels, fontsize=7)
    ax.set_ylabel("Spearman $\\rho$ (MFCV vs. joint angle)")
    ax.set_title("Per-recording MFCV--angle association", fontsize=9)


def panel_torque(ax) -> None:
    T = pd.read_csv(OUT_DIR / "tvd_bradford_validation_torque.csv")
    v = T.rho.dropna().to_numpy()
    bp = ax.boxplot([v], positions=[0], widths=0.5, showfliers=True, patch_artist=True)
    bp["boxes"][0].set_facecolor("#DD8452")
    bp["boxes"][0].set_alpha(0.7)
    bp["medians"][0].set_color("black")
    jitter = (np.random.RandomState(0).rand(len(v)) - 0.5) * 0.2
    ax.scatter(np.zeros(len(v)) + jitter, v, s=14, color="black", alpha=0.6, zorder=3)
    ax.axhline(0, color="0.4", lw=0.8, ls="--", zorder=0)
    ax.set_xticks([0])
    ax.set_xticklabels([f"GCC\n(isotonic, n={len(v)})"], fontsize=7)
    ax.set_ylabel("Spearman $\\rho$ (MFCV vs. torque)")
    ax.set_title("Per-recording MFCV--torque association", fontsize=9)


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4))
    panel_bic(axes[0])
    panel_angle(axes[1])
    panel_torque(axes[2])
    fig.tight_layout()
    out = OUT_DIR / "tvd_bradford_modelselection_angle.pdf"
    fig.savefig(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
