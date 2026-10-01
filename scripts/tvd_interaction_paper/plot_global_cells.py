"""
Regenerates tvd_global_cells_comparison.pdf (PAPER_TVD_Interaction.tex,
Section results-global, Table tab:global / Figure fig:global-cells) from the
four per-cell benchmark CSVs already in journal_edit/: BFGS RMSE vs. SNR for
all four cells of the source x ground-truth design, the same numbers as
Table tab:global.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "journal_edit"

CELLS = [
    ("tvd_benchmark_data.csv", "Colored/linear", "#4C72B0", "o-"),
    ("tvd_benchmark_muap_data.csv", "MUAP/linear", "#55A868", "s-"),
    ("tvd_benchmark_sinusoidal_data.csv", "Colored/sinus.", "#DD8452", "^-"),
    ("tvd_muap_sinusoidal_global_data.csv", "MUAP/sinus. (new cell)", "#C44E52", "D-"),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for fname, label, color, style in CELLS:
        df = pd.read_csv(OUT_DIR / fname)
        d = df[df.method == "BFGS"].sort_values("snr_db")
        lw = 2.2 if "new cell" in label else 1.3
        ax.plot(d.snr_db, d.rmse_pct, style, color=color, lw=lw, ms=5, label=label)

    ax.set_yscale("log")
    ax.set_xlabel("SNR [dB]")
    ax.set_ylabel("RMSE in recovered MFCV [\\%] (log scale)")
    ax.set_title("Global order-1 fit, BFGS: all four cells", fontsize=9)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.2, which="both")
    fig.tight_layout()

    out = OUT_DIR / "tvd_global_cells_comparison.pdf"
    fig.savefig(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
