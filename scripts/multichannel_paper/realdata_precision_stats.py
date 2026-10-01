"""
realdata_precision_stats.py -- reproduces every statistic in
PAPER_Multichannel_TVD.tex's Table tab:realdata and Section results-realdata's
prose (acceptance rate, temporal precision, cross-column precision, K=2 vs
K=11), from the already-generated raw per-window CSVs
(tvd_multichannel_realdata.csv = order-0 joint MLE,
tvd_multichannel_realdata_gcc.csv = GCC-pooled), computing VARIANCE (not SD)
of CV as the precision statistic, with the same paired Wilcoxon signed-rank
test and win/loss counts used in the paper.

This is a from-scratch re-derivation of numbers that a previous session
computed ad hoc (no script for them existed in this repo) -- written here so
the variance-based version is reproducible, not just squared from the old
SD-based medians.

Run from repo root: `python3 scripts/multichannel_paper/realdata_precision_stats.py`
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, wilcoxon

OUT_DIR = Path(__file__).resolve().parents[2] / "journal_edit"


def acceptance_stats(df: pd.DataFrame) -> dict:
    tab = df.groupby("K")["accepted"].agg(["sum", "count"])
    k2_acc, k2_n = tab.loc["K2", "sum"], tab.loc["K2", "count"]
    k11_acc, k11_n = tab.loc["K11", "sum"], tab.loc["K11", "count"]
    cont = np.array([[k2_acc, k2_n - k2_acc], [k11_acc, k11_n - k11_acc]])
    chi2, p, _, _ = chi2_contingency(cont, correction=False)
    return dict(k2_rate=k2_acc / k2_n, k11_rate=k11_acc / k11_n,
                n_k2=int(k2_n), n_k11=int(k11_n), chi2=chi2, p=p)


def temporal_precision(df: pd.DataFrame) -> dict:
    acc = df[df.accepted == 1]
    g = acc.groupby(["subject", "condition", "col", "K"]).agg(
        n=("cv_ms", "size"), var=("cv_ms", "var")).reset_index()
    piv = g.pivot_table(index=["subject", "condition", "col"], columns="K", values=["n", "var"])
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    paired = piv.dropna(subset=["var_K2", "var_K11"])
    paired = paired[(paired.n_K2 >= 4) & (paired.n_K11 >= 4)]
    k2, k11 = paired["var_K2"].to_numpy(), paired["var_K11"].to_numpy()
    stat, p = wilcoxon(k2, k11)
    return dict(n=len(paired), med_k2=np.median(k2), med_k11=np.median(k11),
                p=p, frac_k11_loses=float(np.mean(k11 > k2)))


def crosscolumn_precision(df: pd.DataFrame) -> dict:
    acc = df[df.accepted == 1]
    colmean = acc.groupby(["subject", "condition", "col", "K"]).agg(
        n=("cv_ms", "size"), mean_cv=("cv_ms", "mean")).reset_index()
    colmean_ok = colmean[colmean.n >= 4]
    grp = colmean_ok.groupby(["subject", "condition", "K"]).agg(
        ncols=("col", "nunique"), var=("mean_cv", "var")).reset_index()
    grp_full = grp[grp.ncols == 4]
    piv = grp_full.pivot_table(index=["subject", "condition"], columns="K", values="var")
    paired = piv.dropna(subset=["K2", "K11"])
    k2, k11 = paired["K2"].to_numpy(), paired["K11"].to_numpy()
    stat, p = wilcoxon(k2, k11)
    return dict(n=len(paired), med_k2=np.median(k2), med_k11=np.median(k11),
                p=p, frac_k11_wins=float(np.mean(k11 < k2)))


def report(name: str, df: pd.DataFrame) -> None:
    print(f"\n=== {name} ===")
    a = acceptance_stats(df)
    print(f"Acceptance: K2={a['k2_rate']:.1%} (n={a['n_k2']})  "
          f"K11={a['k11_rate']:.1%} (n={a['n_k11']})  chi2 p={a['p']:.2e}")
    t = temporal_precision(df)
    print(f"Temporal variance [m/s^2]: n={t['n']}  median K2={t['med_k2']:.4f}  "
          f"median K11={t['med_k11']:.4f}  Wilcoxon p={t['p']:.4f}  "
          f"K11 loses in {t['frac_k11_loses']:.0%} of recordings")
    c = crosscolumn_precision(df)
    print(f"Cross-column variance [m/s^2]: n={c['n']}  median K2={c['med_k2']:.4f}  "
          f"median K11={c['med_k11']:.4f}  Wilcoxon p={c['p']:.4f}  "
          f"K11 wins in {c['frac_k11_wins']:.0%} of groups")


def main() -> None:
    mle = pd.read_csv(OUT_DIR / "tvd_multichannel_realdata.csv")
    gcc = pd.read_csv(OUT_DIR / "tvd_multichannel_realdata_gcc.csv")
    report("Order-0 MLE (joint fit)", mle)
    report("GCC-pooled (pairwise average)", gcc)


if __name__ == "__main__":
    main()
