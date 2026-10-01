#!/usr/bin/env python3
"""
analyze_fma.py -- statistics and figures for PAPER_PhysioMio_FMA.tex from the
out-of-fold scores written by fma_from_semg.py.

  fma_metrics_ci.csv   pooled / within-gesture / within-patient AUC and
                       longitudinal concordance (same-day, different-day
                       pairs), with 95% patient-cluster bootstrap intervals
  fma_gesture.csv      per-gesture prevalence and within-gesture AUC
  figures (journal_edit/): physiomio_prevalence.pdf, physiomio_patients.pdf,
                       physiomio_features.pdf, physiomio_longitudinal.pdf

Run from repo root:
  sEMG_fatigue/.venv/bin/python scripts/physiomio_paper/analyze_fma.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import logit
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIG = REPO / "journal_edit"
N_BOOT = 2000
RNG = np.random.default_rng(0)

MODELS = ["gesture_prior", "clinical", "chan", "summ", "dsumm", "dsumm_summ", "all",
          "chan+prior"]
LABEL = {"gesture_prior": "Gesture prior", "clinical": "Clinical",
         "chan": "Channel features", "summ": "Summary features",
         "dsumm": r"$\Delta$-summary", "dsumm_summ": r"Summary + $\Delta$",
         "all": r"Summary + $\Delta$ + clinical", "chan+prior": "Channel + prior (fused)"}


def load():
    D = pd.read_csv(HERE / "fma_oof.csv")
    D["y"] = (D.fma < 2).astype(int)
    eps = 1e-4
    D["chan+prior"] = (logit(D.chan.clip(eps, 1 - eps))
                       + logit(D.gesture_prior.clip(eps, 1 - eps)))
    return D


# ----------------------------------------------------------------- metrics
def strat_auc(D, col, by):
    num = den = 0.0
    for _, g in D.groupby(by):
        n1 = g.y.sum()
        n0 = len(g) - n1
        if n1 == 0 or n0 == 0:
            continue
        num += roc_auc_score(g.y, g[col]) * n1 * n0
        den += n1 * n0
    return num / den if den else np.nan


def pair_table(D):
    """All pairs of segments of the same patient and gesture with different FMA."""
    rows = []
    for (p, g), G in D.groupby(["patient", "gesture"]):
        if G.fma.nunique() < 2:
            continue
        G = G.reset_index(drop=True)
        for i in range(len(G)):
            for j in range(i + 1, len(G)):
                if G.fma[i] == G.fma[j]:
                    continue
                worse, better = (i, j) if G.fma[i] < G.fma[j] else (j, i)
                r = dict(patient=p, gesture=g,
                         same_day=G.days_after_stroke[i] == G.days_after_stroke[j],
                         improved_later=G.days_after_stroke[better] > G.days_after_stroke[worse])
                for c in MODELS:
                    d = G[c][worse] - G[c][better]
                    r[c] = 1.0 if d > 0 else (0.0 if d < 0 else 0.5)
                rows.append(r)
    return pd.DataFrame(rows)


def metrics(D, P):
    out = {}
    for c in MODELS:
        out[(c, "auc")] = roc_auc_score(D.y, D[c])
        out[(c, "auc_within_gesture")] = strat_auc(D, c, "gesture")
        out[(c, "auc_within_patient")] = strat_auc(D, c, "patient")
        out[(c, "conc_diff_day")] = P.loc[~P.same_day, c].mean()
        out[(c, "conc_same_day")] = P.loc[P.same_day, c].mean()
    return out


def bootstrap(D, P):
    pats = D.patient.unique()
    Dg = dict(tuple(D.groupby("patient")))
    Pg = dict(tuple(P.groupby("patient")))
    est = metrics(D, P)
    def one(s):
        Db = pd.concat([Dg[p].assign(patient=f"{p}#{k}") for k, p in enumerate(s)])
        Pb = pd.concat([Pg[p].assign(patient=f"{p}#{k}") for k, p in enumerate(s) if p in Pg])
        return metrics(Db, Pb)
    draws = [RNG.choice(pats, len(pats), replace=True) for _ in range(N_BOOT)]
    B = pd.DataFrame(Parallel(n_jobs=16)(delayed(one)(s) for s in draws))
    rows = []
    for c in MODELS:
        r = dict(model=c)
        for m in ["auc", "auc_within_gesture", "auc_within_patient",
                  "conc_diff_day", "conc_same_day"]:
            r[m] = est[(c, m)]
            r[f"{m}_lo"], r[f"{m}_hi"] = np.nanpercentile(B[(c, m)], [2.5, 97.5])
        rows.append(r)
    return pd.DataFrame(rows), B


# ----------------------------------------------------------------- figures
def fig_prevalence(D):
    G = []
    for g, s in D.groupby("gesture"):
        G.append(dict(gesture=g, n=len(s), prevalence=s.y.mean(),
                      auc_chan=roc_auc_score(s.y, s.chan) if s.y.nunique() == 2 else np.nan,
                      auc_summ=roc_auc_score(s.y, s.summ) if s.y.nunique() == 2 else np.nan))
    G = pd.DataFrame(G).sort_values("prevalence")
    G.to_csv(HERE / "fma_gesture.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    y = np.arange(len(G))
    ax[0].barh(y, G.prevalence, color="0.55")
    ax[0].set_yticks(y, G.gesture)
    ax[0].set_xlabel("Fraction of segments with FMA item < 2")
    ax[0].set_title("(a) Item difficulty", loc="left", fontsize=10)
    ax[1].plot(G.auc_chan, y, "o", color="C0", label="Channel features")
    ax[1].plot(G.auc_summ, y, "s", mfc="none", color="C1", label="Summary features")
    ax[1].axvline(0.5, color="0.6", lw=0.8, ls="--")
    ax[1].set_xlabel("AUC within gesture")
    ax[1].set_title("(b) Discrimination within each gesture", loc="left", fontsize=10)
    ax[1].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "physiomio_prevalence.pdf")
    plt.close(fig)
    return G


def fig_patients(D):
    rows = []
    for p, s in D.groupby("patient"):
        r = dict(patient=p, mean_fma=s.fma.mean(), n=len(s))
        for c in ["chan", "gesture_prior", "chan+prior"]:
            r[f"score_{c}"] = s[c].mean() if c != "chan+prior" else np.nan
            r[f"auc_{c}"] = roc_auc_score(s.y, s[c]) if s.y.nunique() == 2 else np.nan
        rows.append(r)
    R = pd.DataFrame(rows)
    R.to_csv(HERE / "fma_patient.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    rho = spearmanr(R.mean_fma, R.score_chan).correlation
    ax[0].plot(R.mean_fma, R.score_chan, "o", color="C0", ms=4)
    ax[0].set_xlabel("Mean FMA item score of the patient (impaired arm)")
    ax[0].set_ylabel("Mean predicted P(item < 2), channel features")
    ax[0].set_title(f"(a) Between patients, Spearman $\\rho={rho:.2f}$", loc="left", fontsize=10)
    ok = R.auc_chan.notna()
    ax[1].plot(R.auc_gesture_prior[ok], R.auc_chan[ok], "o", color="C0", ms=4)
    ax[1].plot([0.3, 1], [0.3, 1], color="0.6", lw=0.8, ls="--")
    ax[1].set_xlabel("Within-patient AUC, gesture prior")
    ax[1].set_ylabel("Within-patient AUC, channel features")
    ax[1].set_title("(b) Within patients", loc="left", fontsize=10)
    ax[1].set_xlim(0.3, 1.01)
    ax[1].set_ylim(0.3, 1.01)
    fig.tight_layout()
    fig.savefig(FIG / "physiomio_patients.pdf")
    plt.close(fig)
    return R, rho


def fig_features():
    """Channel-mean amplitude and mean frequency of the impaired arm relative to
    the same patient's healthy arm, by FMA item score."""
    df = pd.read_parquet(HERE / "features_healthy_vs_impaired.parquet")
    items = pd.read_csv(HERE / "fma_items.csv")
    df = df.rename(columns={"movement_type": "gesture"}).merge(
        items[["patient", "arm_type", "recording_index", "gesture", "fma"]],
        left_on=["patient", "arm_type", "session", "gesture"],
        right_on=["patient", "arm_type", "recording_index", "gesture"])
    rms = np.log(df[[f"channel_{i:02d}_rms" for i in range(1, 65)]].to_numpy()).mean(1)
    mnf = df[[f"channel_{i:02d}_mnf" for i in range(1, 65)]].to_numpy().mean(1)
    rmsn = df[[f"channel_{i:02d}_rms" for i in range(1, 65)]].to_numpy()
    conc = np.log(rmsn.max(1) / rmsn.mean(1))
    df = df.assign(lrms=rms, mnf_mean=mnf, conc=conc)
    seg = df.groupby(["patient", "arm_type", "session", "gesture"])[
        ["lrms", "mnf_mean", "conc", "fma"]].mean().reset_index()
    ref = seg[seg.arm_type == "healthy_arm"].groupby(["patient", "gesture"])[
        ["lrms", "mnf_mean", "conc"]].mean()
    imp = seg[seg.arm_type == "impaired_arm"].join(ref, on=["patient", "gesture"], rsuffix="_ref")
    imp["d_rms_db"] = 20 / np.log(10) * (imp.lrms - imp.lrms_ref)
    imp["d_mnf"] = imp.mnf_mean - imp.mnf_mean_ref
    imp["d_conc"] = imp.conc - imp.conc_ref
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.6))
    stats = {}
    for k, (col, lab) in enumerate([("d_rms_db", "Amplitude re healthy arm (dB)"),
                                    ("d_mnf", "Mean frequency re healthy arm (Hz)")]):
        data = [imp.loc[imp.fma == v, col].dropna().to_numpy() for v in (0, 1, 2)]
        ax[k].boxplot(data, showfliers=False, widths=0.6, medianprops=dict(color="k"))
        ax[k].axhline(0, color="0.6", lw=0.8, ls="--")
        ax[k].set_xticks([1, 2, 3], ["0", "1", "2"])
        ax[k].set_xlabel("FMA item score")
        ax[k].set_ylabel(lab, fontsize=9)
        ax[k].set_title(f"({'abc'[k]})", loc="left", fontsize=10)
        stats[col] = [np.median(d) for d in data] + [len(d) for d in data]
    imp["d_conc"].groupby(imp.fma).median().to_csv(HERE / "fma_conc_medians.csv")
        # healthy-arm items for scale
    healthy = seg[seg.arm_type == "healthy_arm"]
    fig.tight_layout()
    fig.savefig(FIG / "physiomio_features.pdf")
    plt.close(fig)
    S = pd.DataFrame(stats, index=["med0", "med1", "med2", "n0", "n1", "n2"]).T
    S.to_csv(HERE / "fma_feature_medians.csv")
    return S, len(healthy)


def fig_longitudinal(P, CI):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    show = ["clinical", "chan", "summ", "dsumm", "all"]
    x = np.arange(len(show))
    for k, (m, mk) in enumerate([("conc_diff_day", "o"), ("conc_same_day", "s")]):
        v = CI.set_index("model").loc[show]
        ax.errorbar(x + (k - 0.5) * 0.2, v[m], yerr=[v[m] - v[f"{m}_lo"], v[f"{m}_hi"] - v[m]],
                    fmt=mk, capsize=3, color=f"C{k}",
                    label=["Sessions on different days", "Sessions on the same day"][k])
    ax.axhline(0.5, color="0.6", lw=0.8, ls="--")
    ax.set_xticks(x, [LABEL[m] for m in show], fontsize=8, rotation=15)
    ax.set_ylabel("Within patient-gesture concordance")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "physiomio_longitudinal.pdf")
    plt.close(fig)


def main():
    D = load()
    P = pair_table(D)
    print(f"{len(D)} segments, {D.patient.nunique()} patients, prevalence {D.y.mean():.3f}")
    print(f"pairs: {len(P)} ({P.same_day.sum()} same day), "
          f"{P.groupby(['patient', 'gesture']).ngroups} patient-gestures, "
          f"{P.patient.nunique()} patients; improved later in "
          f"{P.loc[~P.same_day, 'improved_later'].mean():.3f} of different-day pairs")

    # in-sample gesture prior: an optimistic version free of the leave-one-out shift
    prev = D.groupby("gesture").y.mean()
    print(f"in-sample gesture prior AUC {roc_auc_score(D.y, D.gesture.map(prev)):.3f}, "
          f"within patient {strat_auc(D.assign(gp=D.gesture.map(prev)), 'gp', 'patient'):.3f}")

    CI, B = bootstrap(D, P)
    CI.to_csv(HERE / "fma_metrics_ci.csv", index=False)
    print(CI.round(3).to_string())
    # paired bootstrap: chan minus gesture prior within patient, chan+prior minus prior
    for a, b in [("chan", "gesture_prior"), ("chan+prior", "gesture_prior"),
                 ("chan", "summ"), ("summ", "dsumm"), ("chan", "clinical")]:
        for m in ["auc", "auc_within_patient", "conc_diff_day"]:
            d = B[(a, m)] - B[(b, m)]
            print(f"  {a} - {b} [{m}]: {CI.set_index('model').loc[a, m] - CI.set_index('model').loc[b, m]:+.3f} "
                  f"[{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]")

    G = fig_prevalence(D)
    print(G.round(3).to_string())
    R, rho = fig_patients(D)
    ok = R.auc_chan.notna()
    print(f"patients with both classes {ok.sum()}; chan > prior in {(R.auc_chan > R.auc_gesture_prior)[ok].sum()}, "
          f"wilcoxon p={wilcoxon(R.auc_chan[ok], R.auc_gesture_prior[ok]).pvalue:.3g}; "
          f"median chan {R.auc_chan[ok].median():.3f}, prior {R.auc_gesture_prior[ok].median():.3f}; "
          f"between-patient rho {rho:.3f}")
    S, nh = fig_features()
    print(S.round(2).to_string())
    fig_longitudinal(P, CI)

    A = pd.read_csv(HERE / "arm_summary.csv") if (HERE / "arm_summary.csv").exists() else None
    if A is not None:
        print(A.round(3).to_string())
        arm_by_day()


def arm_by_day():
    """Paretic-vs-unaffected pair ordering, split by whether the paretic
    recording was made on the same day as the unaffected-arm reference."""
    R = pd.read_csv(HERE / "arm_oof.csv")
    meta = pd.read_csv(REPO / "dataset" / "physiomio" / "data" / "metadata.csv")
    R[["patient", "arm", "sess"]] = R.rec.str.split("|", expand=True)
    R["sess"] = R.sess.astype(int)
    R = R.merge(meta[["patient", "arm_type", "recording_index", "days_after_stroke"]],
                left_on=["patient", "arm", "sess"],
                right_on=["patient", "arm_type", "recording_index"], suffixes=("", "_m"))
    rows = []
    for c in ["chan", "summ", "summ_pnorm"]:
        for p, g in R.groupby("patient"):
            h = g[g.arm == "healthy_arm"]
            for _, i in g[g.arm == "impaired_arm"].iterrows():
                rows.append(dict(model=c, patient=p,
                                 same_day=i.days_after_stroke == h.days_after_stroke.min(),
                                 acc=np.mean([(i[c] > x) + 0.5 * (i[c] == x) for x in h[c]])))
    T = pd.DataFrame(rows).groupby(["model", "same_day"]).acc.agg(["mean", "size"])
    T.to_csv(HERE / "arm_by_day.csv")
    print(T.round(3).to_string())


if __name__ == "__main__":
    main()
