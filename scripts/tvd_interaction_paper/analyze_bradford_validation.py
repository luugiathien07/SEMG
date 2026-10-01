#!/usr/bin/env python3
"""
analyze_bradford_validation.py -- three ground-truth-free validations on the
real dynamic-contraction recordings (inputs: journal_edit/tvd_bradford_realdata.csv
and journal_edit/tvd_bradford_validation.csv).

 V1  Delay-model selection. Per window, which of {global order-0, global
     order-1, local order-0, local order-1} has the lowest BIC? Tests the
     paper's central premise that a single global low-order polynomial is not
     an adequate description of the delay over a window of real dynamic
     contraction, i.e. that local models are preferred even after the
     parameter penalty. Aggregated to recording and subject before testing.
 V2  CV vs. contraction level. Median accepted CV per recording, compared
     between isotonic 25 % and 50 % MVIC and across isokinetic speeds
     (30 / 90 / 300 deg/s), within subject.
 V3  CV vs. joint angle. Within isokinetic recordings, Spearman correlation
     between the window-median CV and the dynamometer angle signal, restricted
     to windows moving in the dominant direction (raw analog units, so only
     the sign and strength of the association are interpretable).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

REPO = Path(__file__).resolve().parents[2]
R = pd.read_csv(REPO / "journal_edit" / "tvd_bradford_realdata.csv")
V = pd.read_csv(REPO / "journal_edit" / "tvd_bradford_validation.csv")
D = R.merge(V.drop(columns=[]), on=["file", "t_s", "col", "pos"], how="inner")
D["subject"] = D.file.str.split("_").str[0]
D["cond"] = D.file.str.split("_", n=1).str[1]
D["kind"] = np.where(D.cond.str.startswith("ISOK"), "isokinetic", "isotonic")
print(f"{len(D)} inputs, {D.file.nunique()} recordings, {D.subject.nunique()} subjects\n")

# ------------------------------------------------------------------ V1
M = ["g0", "g1", "l0", "l1"]
NAMES = {"g0": "global order-0", "g1": "global order-1", "l0": "local order-0", "l1": "local order-1"}
B = D[[f"bic_{m}" for m in M]].to_numpy()
ok = np.isfinite(B).all(axis=1)
D1 = D[ok].copy()
B = B[ok]
D1["winner"] = np.array(M)[B.argmin(axis=1)]
D1["local_better_o1"] = D1.bic_l1 < D1.bic_g1          # same order, local vs global
D1["local_better_o0"] = D1.bic_l0 < D1.bic_g0
D1["any_local_better"] = np.minimum(D1.bic_l0, D1.bic_l1) < np.minimum(D1.bic_g0, D1.bic_g1)

print("=== V1: BIC-selected delay model, share of inputs ===")
sh = D1.winner.value_counts(normalize=True).reindex(M).fillna(0)
print("  all:        " + "  ".join(f"{NAMES[m]}={sh[m]:.1%}" for m in M))
for kind, g in D1.groupby("kind"):
    sh = g.winner.value_counts(normalize=True).reindex(M).fillna(0)
    print(f"  {kind:11s} " + "  ".join(f"{NAMES[m]}={sh[m]:.1%}" for m in M) + f"   (n={len(g)})")
print("  median NMSE (residual / x2 energy): " +
      "  ".join(f"{NAMES[m]}={D1[f'nmse_{m}'].median():.3f}" for m in M))
for c in ("local_better_o1", "local_better_o0", "any_local_better"):
    print(f"  share with {c:18s}: all={D1[c].mean():.1%}  " +
          "  ".join(f"{k}={g[c].mean():.1%}" for k, g in D1.groupby('kind')))

print("\n  per-recording share where local order-1 beats global order-1 (BIC):")
rec = D1.groupby("file").local_better_o1.mean()
sub = D1.groupby("subject").local_better_o1.mean()
print(f"    recordings: median {rec.median():.1%}, min {rec.min():.1%}, max {rec.max():.1%}, "
      f">50% in {(rec > 0.5).sum()}/{len(rec)}")
print("    subjects  : " + "  ".join(f"{s}={v:.1%}" for s, v in sub.items()))
rec_any = D1.groupby("file").any_local_better.mean()
sub_any = D1.groupby("subject").any_local_better.mean()
print(f"  any local model beats best global model: recordings median {rec_any.median():.1%}, "
      f">50% in {(rec_any > 0.5).sum()}/{len(rec_any)}; subjects " +
      "  ".join(f"{s}={v:.1%}" for s, v in sub_any.items()))
try:
    st, p = wilcoxon(rec - 0.5)
    print(f"  Wilcoxon (recording-level share vs 50%): p={p:.4f}")
except Exception as e:
    print("  Wilcoxon n/a", e)

# ------------------------------------------------------------------ V2
print("\n=== V2: median accepted CV [m/s] per recording, by estimator ===")
def med_cv(g, col):
    v = g[col]; v = v[v.between(2, 10)]
    return v.median() if len(v) >= 20 else np.nan
rows = []
for f, g in D.groupby("file"):
    rows.append(dict(file=f, subject=g.subject.iloc[0], cond=g.cond.iloc[0], n=len(g),
                     **{m: med_cv(g, f"cv_{m}") for m in ("poly", "constmle", "lap", "gcc")},
                     torque_uV=g.torque.median()))
P = pd.DataFrame(rows)
print(P.round(2).to_string(index=False))

def paired(cond_a, cond_b, est):
    a = P[P.cond == cond_a].set_index("subject")[est]
    b = P[P.cond == cond_b].set_index("subject")[est]
    j = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    return j
print("\n  isotonic 50% vs 25% MVIC, paired within subject (positive = higher CV at 50%):")
for est in ("gcc", "constmle", "poly", "lap"):
    j = paired("ISOT25_1", "ISOT50_1", est)
    d = j.b - j.a
    print(f"    {est:9s} n={len(j)}  diffs " + " ".join(f"{x:+.2f}" for x in d) +
          f"  -> higher at 50% in {(d > 0).sum()}/{len(d)} subjects")
print("\n  isokinetic speed, median CV per subject (gcc):")
for s, g in P[P.cond.str.startswith("ISOK")].groupby("subject"):
    order = g.set_index("cond").gcc.reindex(["ISOK30", "ISOK90", "ISOK300"])
    print(f"    {s}: 30={order['ISOK30']:.2f}  90={order['ISOK90']:.2f}  300={order['ISOK300']:.2f}")

# ------------------------------------------------------------------ V3
print("\n=== V3: window-median CV vs dynamometer angle (Spearman), isokinetic + isotonic ===")
rows = []
for f, g in D.groupby("file"):
    est = {}
    for m in ("gcc", "poly"):
        gg = g[g[f"cv_{m}"].between(2, 10)]
        w = gg.groupby("t_s").agg(cv=(f"cv_{m}", "median"), n=(f"cv_{m}", "size"),
                                   angle=("angle", "first"), vel=("velocity", "first"))
        w = w[w.n >= 3]
        if len(w) < 8:
            est[m] = (np.nan, np.nan, len(w)); continue
        if g.kind.iloc[0] == "isokinetic":
            dom = np.sign(w.vel.median())
            if dom != 0:
                w = w[np.sign(w.vel) == dom]
        if len(w) < 8 or w.angle.nunique() < 4:
            est[m] = (np.nan, np.nan, len(w)); continue
        rho, p = spearmanr(w.angle, w.cv)
        est[m] = (rho, p, len(w))
    rows.append(dict(file=f, subject=g.subject.iloc[0], kind=g.kind.iloc[0],
                     rho_gcc=est["gcc"][0], p_gcc=est["gcc"][1], n_gcc=est["gcc"][2],
                     rho_poly=est["poly"][0], n_poly=est["poly"][2]))
A = pd.DataFrame(rows)
print(A.round(3).to_string(index=False))
for kind, g in A.groupby("kind"):
    for est in ("gcc", "poly"):
        r = g[f"rho_{est}"].dropna()
        print(f"  {kind:11s} {est:5s}: n_rec={len(r)}, median rho={r.median():+.2f}, "
              f"positive in {(r > 0).sum()}/{len(r)}, negative in {(r < 0).sum()}/{len(r)}")
sub = A.dropna(subset=["rho_gcc"]).groupby("subject").rho_gcc.median()
print("  per-subject median rho (gcc, all recordings): " + "  ".join(f"{s}={v:+.2f}" for s, v in sub.items()))

P.to_csv(REPO / "journal_edit" / "tvd_bradford_validation_cv_by_recording.csv", index=False)
A.to_csv(REPO / "journal_edit" / "tvd_bradford_validation_angle.csv", index=False)

# ------------------------------------------------------------------ V4
# Within-recording MFCV vs. instantaneous torque (Spearman), GCC, isotonic
# only: unlike V2 (which compares the two block-level 25%/50% MVIC targets
# using one summary CV per recording), this uses every retained window's own
# torque reading, so it can see force-CV covariation that V2's two-point
# comparison cannot. Isokinetic recordings are excluded here because torque
# there varies with joint angle through the moment arm, which V3 already
# attributes to angle.
print("\n=== V4: window-median CV vs instantaneous torque (Spearman), GCC, isotonic only ===")
rows = []
for f, g in D[D.kind == "isotonic"].groupby("file"):
    gg = g[g.cv_gcc.between(2, 10)]
    w = gg.groupby("t_s").agg(cv=("cv_gcc", "median"), n=("cv_gcc", "size"), torque=("torque", "first"))
    w = w[w.n >= 3]
    if len(w) < 8:
        rows.append(dict(file=f, subject=g.subject.iloc[0], n=len(w), rho=np.nan, p=np.nan))
        continue
    rho, p = spearmanr(w.torque, w.cv)
    rows.append(dict(file=f, subject=g.subject.iloc[0], n=len(w), rho=rho, p=p))
T = pd.DataFrame(rows)
print(T.round(3).to_string(index=False))
r = T.rho.dropna()
print(f"  n_recordings={len(r)}, median rho={r.median():+.3f}, positive in {(r > 0).sum()}/{len(r)}, "
      f"negative in {(r < 0).sum()}/{len(r)}, significant (p<0.05) in {(T.p < 0.05).sum()}/{len(r)}")
sub = T.dropna(subset=["rho"]).groupby("subject").rho.median()
print("  per-subject median rho: " + "  ".join(f"{s}={v:+.2f}" for s, v in sub.items()))

T.to_csv(REPO / "journal_edit" / "tvd_bradford_validation_torque.csv", index=False)
