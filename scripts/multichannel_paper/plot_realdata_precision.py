"""
Regenerates tvd_multichannel_realdata_precision.pdf (PAPER_Multichannel_TVD.tex,
Section results-realdata, Table tab:realdata / Figure fig:realdata) from the
current double-differential real-data CSVs.

2x2 grid: rows = pooling scheme (order-0 joint MLE, GCC-pooled per-adjacent-
pair average), columns = precision metric (within-recording temporal SD,
cross-column SD), each panel a paired K=2-vs-K=11 boxplot with connecting
lines, matching the original single-scheme figure's style.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "journal_edit"


def paired_temporal(df: pd.DataFrame) -> pd.DataFrame:
    acc = df[df.accepted == 1]
    temporal = acc.groupby(["subject", "condition", "col", "K"]).agg(
        n=("cv_ms", "size"), var=("cv_ms", "var")).reset_index()
    piv = temporal.pivot_table(index=["subject", "condition", "col"], columns="K", values=["n", "var"])
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    paired = piv.dropna(subset=["var_K2", "var_K11"])
    return paired[(paired.n_K2 >= 4) & (paired.n_K11 >= 4)]


def paired_crosscol(df: pd.DataFrame) -> pd.DataFrame:
    acc = df[df.accepted == 1]
    colmean = acc.groupby(["subject", "condition", "col", "K"]).agg(
        n=("cv_ms", "size"), mean_cv=("cv_ms", "mean")).reset_index()
    colmean_ok = colmean[colmean.n >= 4]
    grp = colmean_ok.groupby(["subject", "condition", "K"]).agg(
        ncols=("col", "nunique"), var=("mean_cv", "var")).reset_index()
    grp_full = grp[grp.ncols == 4]
    pivc = grp_full.pivot_table(index=["subject", "condition"], columns="K", values="var")
    return pivc.dropna(subset=["K2", "K11"])


def panel(ax, paired: pd.DataFrame, col2: str, col11: str, title: str, ylabel: str):
    k2, k11 = paired[col2].values, paired[col11].values
    for a, b in zip(k2, k11):
        ax.plot([0, 1], [a, b], color="0.7", lw=0.6, zorder=1)
    bp = ax.boxplot([k2, k11], positions=[0, 1], widths=0.4, showfliers=False,
                     patch_artist=True, zorder=2)
    for patch, color in zip(bp["boxes"], ["#4C72B0", "#DD8452"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["K=2", "K=11"])
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} (n={len(paired)})", fontsize=9)


def main() -> None:
    mle = pd.read_csv(OUT_DIR / "tvd_multichannel_realdata.csv")
    gcc = pd.read_csv(OUT_DIR / "tvd_multichannel_realdata_gcc.csv")

    mle_temp = paired_temporal(mle)
    mle_cross = paired_crosscol(mle)
    gcc_temp = paired_temporal(gcc)
    gcc_cross = paired_crosscol(gcc)

    fig, axes = plt.subplots(2, 2, figsize=(7, 6.5))
    panel(axes[0, 0], mle_temp, "var_K2", "var_K11",
          "Order-0 MLE: within-recording temporal variance", "CV Variance [m/s$^2$]")
    panel(axes[0, 1], mle_cross, "K2", "K11",
          "Order-0 MLE: cross-column variance", "CV Variance [m/s$^2$]")
    panel(axes[1, 0], gcc_temp, "var_K2", "var_K11",
          "GCC-pooled: within-recording temporal variance", "CV Variance [m/s$^2$]")
    panel(axes[1, 1], gcc_cross, "K2", "K11",
          "GCC-pooled: cross-column variance", "CV Variance [m/s$^2$]")
    fig.tight_layout()
    out = OUT_DIR / "tvd_multichannel_realdata_precision.pdf"
    fig.savefig(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
