"""
Regenerates two figures for PAPER_TVD_Interaction.tex's local-strategy
results, one per source-comparison table, both as 2x2 small multiples (one
panel per local strategy) of RMSE vs. SNR:

  tvd_local_colored_vs_muap.pdf   -- Section results-local, Table tab:local
                                      (colored-noise M=12 vs. single-MUAP M=12)
  tvd_local_source_comparison.pdf -- Section results-multipop, Table tab:multipop
                                      (colored M=12, single-MUAP M=60,
                                      multi-motor-unit-population M=60)

Reads the same sliding-window benchmark CSVs the tables themselves are built
from; no numbers are recomputed here.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "journal_edit"

METHOD_MAP = {
    "MLE-sliding-poly(BFGS)": "Local poly (BFGS)",
    "MLE-sliding-constdelay(Brent)": "Local const. (Brent)",
    "LAP(Gilliam2018)": "LAP",
    "GCC(Knapp1976)": "GCC",
}
METHOD_ORDER = ["MLE-sliding-poly(BFGS)", "MLE-sliding-constdelay(Brent)",
                "LAP(Gilliam2018)", "GCC(Knapp1976)"]


def load(fname: str) -> pd.DataFrame:
    return pd.read_csv(OUT_DIR / fname)


def small_multiples(sources: list[tuple[str, str, str]], title: str, out_name: str) -> None:
    """sources: list of (csv filename, legend label, color)."""
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
    dfs = [(load(fname), label, color) for fname, label, color in sources]

    for ax, method in zip(axes.flat, METHOD_ORDER):
        for df, label, color in dfs:
            d = df[df.method == method].sort_values("snr_db")
            ax.plot(d.snr_db, d.rmse_pct, "o-", color=color, ms=4, lw=1.3, label=label)
        ax.set_title(METHOD_MAP[method], fontsize=9)
        ax.set_xlabel("SNR [dB]")
        ax.set_ylabel("RMSE [\\%]")
        ax.grid(alpha=0.2)

    axes.flat[0].legend(fontsize=7, framealpha=0.9)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = OUT_DIR / out_name
    fig.savefig(out)
    print(f"Wrote {out}")


def main() -> None:
    small_multiples(
        [("tvd_benchmark_sinusoidal_sliding_data.csv", "Colored ($M{=}12$)", "#4C72B0"),
         ("tvd_muap_sinusoidal_sliding_data.csv", "Single-MUAP ($M{=}12$)", "#DD8452")],
        "Local sliding-window fits: colored vs. single-MUAP source",
        "tvd_local_colored_vs_muap.pdf",
    )
    small_multiples(
        [("tvd_benchmark_sinusoidal_sliding_data.csv", "Colored ($M{=}12$)", "#4C72B0"),
         ("tvd_muap_sinusoidal_sliding_data_M60.csv", "Single-MUAP ($M{=}60$)", "#DD8452"),
         ("tvd_multipop_sinusoidal_sliding_data.csv", "Multi-MU-pop. ($M{=}60$)", "#55A868")],
        "Local sliding-window fits: three sources",
        "tvd_local_source_comparison.pdf",
    )


if __name__ == "__main__":
    main()
