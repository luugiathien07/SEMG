"""
Regenerates tvd_bradford_crosscol_precision.pdf (PAPER_TVD_Interaction.tex,
Section results-real, Table tab:real / Figure fig:bradford-precision) from
journal_edit/tvd_bradford_realdata.csv.

Recomputes the cross-column robust relative spread of CV at m = 3, 4, 5
minimum accepted columns per (file, t_s, pos) group -- same definition as
analyze_bradford_realdata.py's step (1) -- and shows it as a grouped boxplot,
one group per m, one box per estimator, matching Table tab:real.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "journal_edit"
CSV = OUT_DIR / "tvd_bradford_realdata.csv"

METHODS = ["poly", "constmle", "gcc", "lap"]
LABELS = {"poly": "Local poly", "constmle": "Local const.", "lap": "LAP", "gcc": "GCC"}
COLORS = {"poly": "#4C72B0", "constmle": "#55A868", "lap": "#C44E52", "gcc": "#DD8452"}
CV_MIN, CV_MAX = 2.0, 10.0
M_VALUES = [3, 4, 5]


def robust_rel_spread(v: np.ndarray) -> float:
    med = np.median(v)
    return 1.4826 * np.median(np.abs(v - med)) / med * 100.0


def crosscol_spread(df: pd.DataFrame, min_cols: int) -> pd.DataFrame:
    rows = []
    for (f, t, pos), g in df.groupby(["file", "t_s", "pos"]):
        if g.col.nunique() < min_cols:
            continue
        rows.append({m: robust_rel_spread(g[f"cv_{m}"].to_numpy()) for m in METHODS})
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(CSV)
    ok_all = np.all([df[f"cv_{m}"].between(CV_MIN, CV_MAX) for m in METHODS], axis=0)
    d = df[ok_all]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    n_est = len(METHODS)
    width = 0.8 / n_est
    for j, m in enumerate(METHODS):
        data = [crosscol_spread(d, mv)[m].to_numpy() for mv in M_VALUES]
        positions = [i + (j - (n_est - 1) / 2) * width for i in range(len(M_VALUES))]
        bp = ax.boxplot(data, positions=positions, widths=width * 0.9, showfliers=False,
                         patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor(COLORS[m])
            patch.set_alpha(0.7)
        for med in bp["medians"]:
            med.set_color("black")
        bp["boxes"][0].set_label(LABELS[m])

    ax.set_xticks(range(len(M_VALUES)))
    ax.set_xticklabels([f"$m={mv}$" for mv in M_VALUES])
    ax.set_ylabel("Cross-column robust relative spread of MFCV [\\%]")
    ax.set_xlabel("Minimum accepted columns per group")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    out = OUT_DIR / "tvd_bradford_crosscol_precision.pdf"
    fig.savefig(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
