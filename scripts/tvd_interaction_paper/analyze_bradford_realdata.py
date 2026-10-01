#!/usr/bin/env python3
"""
analyze_bradford_realdata.py -- precision statistics for the real-data check
(journal_edit/tvd_bradford_realdata.csv, from run_bradford_realdata.py).

No ground-truth CV exists, so accuracy cannot be scored. What CAN be scored,
identically for all four estimators, is precision:

 (1) CROSS-COLUMN dispersion (primary). The 9 grid columns are parallel lines
     ~ a few mm apart on the same muscle, so within one (window, pair-position)
     the true CV is essentially common to all columns. The robust relative
     spread of the 9 estimates (1.4826*MAD / median) measures how much
     estimator noise, not physiology, moves the result. Paired per
     (file, window, pos) across estimators -> Wilcoxon signed-rank.
 (2) TEMPORAL dispersion (secondary). Within one (file, col, pos), the robust
     relative spread of the estimates over consecutive active windows. This
     mixes in genuine CV change during the movement, so it is reported but
     not used for the main claim.

Only estimates inside the physiological gate [2, 10] m/s are used (the same
acceptance range as src.mfcv.QualityGate); acceptance rate is reported too,
because an estimator that is "precise" only because it rejects most windows
is not better.
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "journal_edit" / "tvd_bradford_realdata.csv"
METHODS = {"poly": "Local poly (BFGS)", "constmle": "Local const. (Brent)",
           "lap": "LAP", "gcc": "GCC"}
CV_MIN, CV_MAX = 2.0, 10.0
MIN_COLS = int(sys.argv[1]) if len(sys.argv) > 1 else 5   # min accepted columns per (window, pair) group


def robust_rel_spread(v):
    v = np.asarray(v, float)
    med = np.median(v)
    return 1.4826 * np.median(np.abs(v - med)) / med * 100.0


def main():
    df = pd.read_csv(CSV)
    df["subject"] = df.file.str.split("_").str[0]
    print(f"{len(df)} gated (window, column, pair) inputs, files: {sorted(df.file.unique())}")
    print(f"windows: {df.groupby(['file', 't_s']).ngroups}\n")

    print("=== acceptance ([2,10] m/s) and median CV, per estimator ===")
    for m, lab in METHODS.items():
        cv = df[f"cv_{m}"]
        ok = cv.between(CV_MIN, CV_MAX)
        print(f"  {lab:22s} accepted {ok.mean():6.1%}   median CV (accepted) {cv[ok].median():.2f} m/s")

    ok_all = np.all([df[f"cv_{m}"].between(CV_MIN, CV_MAX) for m in METHODS], axis=0)
    d = df[ok_all]
    print(f"\n{len(d)} inputs ({len(d) / len(df):.1%}) accepted by ALL four -> used for paired stats\n")

    # (1) cross-column
    rows = []
    for (f, t, pos), g in d.groupby(["file", "t_s", "pos"]):
        if g.col.nunique() < MIN_COLS:
            continue
        rows.append(dict(file=f, t_s=t, pos=pos, n=len(g),
                         **{m: robust_rel_spread(g[f"cv_{m}"]) for m in METHODS}))
    cc = pd.DataFrame(rows)
    print(f"=== (1) cross-column robust relative spread [%] (n={len(cc)} window x pos groups, >= {MIN_COLS} columns) ===")
    for m, lab in METHODS.items():
        print(f"  {lab:22s} median {cc[m].median():6.2f}   mean {cc[m].mean():6.2f}")
    print("  paired Wilcoxon (row estimator lower spread than column estimator?):")
    for a, b in combinations(METHODS, 2):
        s, p = wilcoxon(cc[a], cc[b])
        print(f"    {a:8s} vs {b:8s}: median diff {np.median(cc[a] - cc[b]):+6.2f} pp, "
              f"{a} lower in {np.mean(cc[a] < cc[b]):.0%} of groups, p={p:.2e}")
    print("  per file (median spread):")
    print(cc.groupby("file")[list(METHODS)].median().round(2).to_string())

    # (2) temporal
    rows = []
    for (f, c, pos), g in d.groupby(["file", "col", "pos"]):
        if len(g) < 6:
            continue
        rows.append(dict(file=f, col=c, pos=pos, n=len(g),
                         **{m: robust_rel_spread(g[f"cv_{m}"]) for m in METHODS}))
    tt = pd.DataFrame(rows)
    print(f"\n=== (2) temporal robust relative spread [%] (n={len(tt)} file x col x pos series, >= 6 windows) ===")
    for m, lab in METHODS.items():
        print(f"  {lab:22s} median {tt[m].median():6.2f}")
    for a, b in combinations(METHODS, 2):
        s, p = wilcoxon(tt[a], tt[b])
        print(f"    {a:8s} vs {b:8s}: {a} lower in {np.mean(tt[a] < tt[b]):.0%}, p={p:.2e}")

    # (3) honest-unit-of-replication tests. Windows overlap 50% and columns /
    # pair positions share the same muscle, so the group-level p-values above
    # treat dependent samples as independent. The defensible units are the
    # recording and, above it, the subject: take the median cross-column
    # spread within each, then test across those.
    cc["subject"] = cc.file.str.split("_").str[0]
    print("\n=== (3) cross-column spread aggregated to independent units ===")
    for unit in ("file", "subject"):
        agg = cc.groupby(unit)[list(METHODS)].median()
        print(f"  unit = {unit} (n={len(agg)}): median of unit medians")
        print("   ", "  ".join(f"{m}={agg[m].median():.2f}" for m in METHODS))
        for a, b in combinations(METHODS, 2):
            s, p = wilcoxon(agg[a], agg[b])
            print(f"    {a:8s} vs {b:8s}: {a} lower in {int((agg[a] < agg[b]).sum())}/{len(agg)} units, p={p:.3f}")
    print("\n  per-subject median cross-column spread [%] and median accepted CV [m/s]:")
    per_subj = cc.groupby("subject")[list(METHODS)].median().round(2)
    per_subj["n_groups"] = cc.groupby("subject").size()
    per_subj["cv_med"] = d.groupby("subject").cv_gcc.median().round(2)
    print(per_subj.to_string())

    cc.to_csv(REPO / "journal_edit" / "tvd_bradford_crosscol_spread.csv", index=False)


if __name__ == "__main__":
    main()
