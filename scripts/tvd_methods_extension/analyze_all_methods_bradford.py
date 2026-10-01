#!/usr/bin/env python3
"""
analyze_all_methods_bradford.py -- precision of the sixteen estimators on the
Bradford 2023 recordings (journal_edit/tvd_all_methods_bradford.npz, from
run_all_methods_bradford.py), and agreement of the real-data ranking with the
simulated ranking under each of the three source models.

No ground truth exists, so only precision is scored:
  acceptance     share of MFCV estimates inside the physiological gate
                 [2, 10] m/s (src.mfcv.QualityGate), over all instants
  cross-column   the 9 grid columns are parallel lines a few mm apart on the
  spread         same muscle, so within one (recording, window, pair position)
                 the true MFCV is essentially common to all columns. The robust
                 relative spread 1.4826*MAD/median (%) of the accepted column
                 estimates measures estimator noise. Two levels:
                   window   one value per input = median MFCV over its 51
                            instants, spread across columns
                   instant  spread across columns at each of the 51 instants,
                            then the median over instants
Comparisons are made on identical inputs: a (group, level) enters only if at
least MIN_COLS columns are accepted by every estimator in the comparison set,
which is the estimators with >= 50% acceptance overall. Subject S07 (poor
signal quality, see PAPER_TVD_Interaction) is reported but excluded from the
subject-level counts.

Sim-vs-real: Spearman correlation, over the comparison set, between the
real-data spread and the simulated error standard deviation (sqrt of the error
variance in tvd_all_methods_summary.csv, averaged over truths and SNRs) for each
source model separately.

Writes journal_edit/bench_bradford_precision.csv (per estimator) and
journal_edit/bench_bradford_subject.csv (per subject x estimator), and prints
the statistics quoted in PAPER_TVD_Benchmark.tex.

    Motionlab/.venv/bin/python scripts/tvd_methods_extension/analyze_all_methods_bradford.py [min_cols]
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

warnings.filterwarnings("ignore", "All-NaN slice")

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "journal_edit"
MIN_COLS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
CV_MIN, CV_MAX = 2.0, 10.0
IED_FS = 0.008 * 2048.0     # m/s per sample of delay

z = np.load(OUT / "tvd_all_methods_bradford.npz", allow_pickle=True)
CV = z["cv"].astype(float)                                  # (inputs, methods, 51)
METH = [str(m) for m in z["methods"]]
keys = pd.DataFrame({"rec": z["rec"].astype(str), "t_s": z["t_s"], "col": z["col"], "pos": z["pos"]})
keys["subject"] = keys.rec.str.split("_").str[0]
ok = (CV >= CV_MIN) & (CV <= CV_MAX)
print(f"{len(keys)} inputs, {keys.rec.nunique()} recordings, "
      f"{keys.groupby(['rec', 't_s', 'pos']).ngroups} (window, pair) groups")

acc = pd.Series(ok.mean(axis=(0, 2)), index=METH)
med_cv = pd.Series([np.median(CV[:, i][ok[:, i]]) for i in range(len(METH))], index=METH)
SET = [m for m in METH if acc[m] >= 0.5]
print(f"comparison set ({len(SET)} estimators with >= 50% acceptance): {SET}")
idx = [METH.index(m) for m in SET]


def rel_spread(v, axis):
    med = np.nanmedian(v, axis=axis, keepdims=True)
    mad = np.nanmedian(np.abs(v - med), axis=axis)
    return 100 * 1.4826 * mad / np.squeeze(med, axis=axis)


win_cv = np.nanmedian(np.where(ok, CV, np.nan), axis=2)     # window level, accepted instants
win_ok = ok.mean(axis=2) >= 0.5                             # input accepted if >= half its instants are
rows_w, rows_i = [], []
for (rec, t, pos), g in keys.groupby(["rec", "t_s", "pos"]):
    ii = g.index.values
    if len(ii) < MIN_COLS:
        continue
    subj = rec.split("_")[0]
    # window level: columns accepted by every estimator of the set
    cols = win_ok[np.ix_(ii, idx)].all(axis=1)
    if cols.sum() >= MIN_COLS:
        v = win_cv[np.ix_(ii[cols], idx)]                   # (cols, set)
        rows_w.append(dict(rec=rec, subject=subj, **dict(zip(SET, rel_spread(v, 0)))))
    # instant level
    o = ok[np.ix_(ii, idx)].all(axis=1)                     # (cols, 51)
    good = o.sum(axis=0) >= MIN_COLS
    if good.any():
        v = np.where(o[:, None, :], CV[np.ix_(ii, idx)], np.nan)[:, :, good]   # (cols, set, n)
        rows_i.append(dict(rec=rec, subject=subj, **dict(zip(SET, np.nanmedian(rel_spread(v, 0), axis=1)))))
W, I = pd.DataFrame(rows_w), pd.DataFrame(rows_i)

res = pd.DataFrame({"acceptance": acc, "median_cv": med_cv})
res["spread_window"] = W[SET].median()
res["spread_instant"] = I[SET].median()
res["sec"] = pd.Series(np.nanmean(z["sec"], axis=0), index=METH)
res.round(4).to_csv(OUT / "bench_bradford_precision.csv", index_label="method")
print(f"\ngroups with >= {MIN_COLS} common columns: window level {len(W)}, instant level {len(I)}")
print(res.sort_values("spread_window").round(3).to_string())

REF = "poly_local"
print(f"\npaired Wilcoxon against {REF} over groups (window level): median difference, p")
for m in SET:
    if m != REF:
        d = W[m] - W[REF]
        print(f"  {m:12s} {d.median():+7.2f}  p={wilcoxon(d).pvalue:.2g}")

usable = W[W.subject != "S07"]
S = usable.groupby("subject")[SET].median()
S.round(3).to_csv(OUT / "bench_bradford_subject.csv")
print("\nper-subject median window-level spread (S07 excluded):")
print(S.round(2).T.to_string())
best = S.idxmin(axis=1)
print("best estimator per subject:", best.to_dict())
print(f"subjects in which each estimator is more precise than {REF}:")
print((S.lt(S[REF], axis=0)).sum().drop(REF).to_string())
rec_med = usable.groupby("rec")[SET].median()
print(f"recordings ({len(rec_med)}) in which each estimator is more precise than {REF}:")
print((rec_med.lt(rec_med[REF], axis=0)).sum().drop(REF).to_string())

print("\nsimulated error SD vs real-data spread, Spearman over the comparison set:")
Ssum = pd.read_csv(OUT / "tvd_all_methods_summary.csv")
for src in ["colored", "muap", "multipop"]:
    sd = np.sqrt(Ssum[Ssum.source == src].groupby("method")["var"].mean())
    for lvl, col in [("window", "spread_window"), ("instant", "spread_instant")]:
        r, p = spearmanr(sd[SET], res.loc[SET, col])
        print(f"  {src:9s} vs real {lvl:7s}: rho = {r:+.2f} (p = {p:.2g})")
    nine = [m for m in SET if m not in ("gcc_ht", "gcc_eckart", "lap", "lap_kalman")]
    r9, p9 = spearmanr(sd[nine], res.loc[nine, "spread_window"])
    print(f"  {src:9s} vs real window, without HT/Eckart/LAP/LAP+Kalman: rho = {r9:+.2f} (p = {p9:.2g})")
    rr, pr = spearmanr(Ssum[Ssum.source == src].groupby("method").rmse.mean()[SET], res.loc[SET, "spread_window"])
    print(f"  {src:9s} RMSE vs real window  : rho = {rr:+.2f} (p = {pr:.2g})")

# ------------------------------------------------------------------ figure
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABEL = {"poly_local": "Local order-1", "dp": "DP", "lap_kalman": "LAP+Kalman",
         "constmle": "Local order-0 MLE", "legendre5": "Legendre-5", "gcc": "GCC (CC)",
         "global_bfgs": "Global order-1", "global_sa": "Global order-1 (SA)", "lap": "LAP",
         "cohf": "CohF", "dll": "DLL", "gcc_ht": "GCC-HT", "gcc_eckart": "GCC-Eckart"}
order = res.loc[SET, "spread_window"].sort_values().index[::-1]
fig, ax = plt.subplots(figsize=(4.8, 3.6))
y = np.arange(len(order))
for j, subj in enumerate(S.index):
    ax.scatter(S.loc[subj, order], y + (j - 2) * 0.08, s=10, color="#8a8984", zorder=2,
               label="Subject median" if j == 0 else None)
ax.scatter(res.loc[order, "spread_window"], y, s=40, marker="D", color="#2a78d6",
           edgecolor="white", linewidth=1, zorder=3, label="All groups (median)")
ax.set_yticks(y)
ax.set_yticklabels([LABEL[m] for m in order], fontsize=7)
ax.set_xlabel("Cross-column robust relative spread of MFCV (%)", fontsize=8)
ax.tick_params(axis="x", labelsize=7)
ax.grid(True, axis="x", color="#e4e3df", lw=0.6)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(fontsize=7, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
fig.tight_layout()
fig.savefig(OUT / "bench_bradford_precision.pdf", bbox_inches="tight")
print("wrote journal_edit/bench_bradford_precision.pdf")

# ------------------------------------------- peak locking and agreement checks
print("\nshare of accepted estimates within 0.1 sample of an integer lag (0.20 if uniform):")
for i, m in enumerate(METH):
    c = CV[:, i][ok[:, i]]
    th = IED_FS / c
    print(f"  {m:12s} {np.mean(np.abs(th - np.round(th)) < 0.1):.3f}")
rec_all = pd.DataFrame(win_cv, columns=METH).assign(rec=keys.rec).groupby("rec").median()
rec_all = rec_all[~rec_all.index.str.startswith("S07")]
print(f"\nSpearman, over {len(rec_all)} recordings, of recording-median MFCV against {REF}:")
for m in SET:
    if m != REF:
        print(f"  {m:12s} {spearmanr(rec_all[m], rec_all[REF])[0]:+.2f}")
