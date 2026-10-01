#!/usr/bin/env python3
"""
plot_physiomio_realdata.py -- figures of the recorded HD-sEMG for
PAPER_PhysioMio_FMA.tex. Only derived quantities are plotted (per-electrode
RMS maps, channel-averaged spectra, per-session summaries); no raw signal, in
line with the PhysioMio data use agreement.

  physiomio_maps.pdf        4 x 16 RMS maps of one patient, unaffected arm and
                            paretic sessions, for three gestures
  physiomio_spectra.pdf     channel-averaged power spectra by FMA item score
  physiomio_trajectories.pdf  per-session FMA sum, relative amplitude and
                            predicted risk for the patients whose FMA sum
                            changes most

Derived per-segment arrays are cached in scripts/physiomio_paper/_segment_maps.npz
(local only; not for distribution).

Run from repo root:
  sEMG_fatigue/.venv/bin/python scripts/physiomio_paper/plot_physiomio_realdata.py
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.signal import welch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = REPO / "dataset" / "physiomio" / "data"
FIG = REPO / "journal_edit"
CACHE = HERE / "_segment_maps.npz"
FS = 2048.0
CH = [f"channel_{i:02d}" for i in range(1, 65)]
NPERSEG = 512
LINES = [(46, 54), (162, 170), (330, 338)]  # power-line and interference peaks (Hz)


def one(arg):
    rec, path = arg
    t = pq.read_table(DATA / path, columns=CH + ["movement_type"]).to_pandas()
    out = []
    for g, s in t.groupby("movement_type", sort=False):
        x = s[CH].to_numpy(float)
        x = x - x.mean(0)
        rms = np.sqrt(np.mean(x ** 2, 0))
        f, P = welch(x, fs=FS, nperseg=NPERSEG, axis=0)
        out.append((rec, g, rms, P.mean(1)))
    return out, f


def build_cache():
    meta = pd.read_csv(DATA / "metadata.csv")
    with ProcessPoolExecutor(16) as ex:
        res = list(ex.map(one, enumerate(meta.file_path)))
    f = res[0][1]
    rows = [r for rs, _ in res for r in rs]
    np.savez_compressed(CACHE, rec=np.array([r[0] for r in rows]),
                        gesture=np.array([r[1] for r in rows]),
                        rms=np.stack([r[2] for r in rows]),
                        psd=np.stack([r[3] for r in rows]), f=f)


def load():
    if not CACHE.exists():
        build_cache()
    Z = np.load(CACHE)
    meta = pd.read_csv(DATA / "metadata.csv")
    S = pd.DataFrame({"rec": Z["rec"], "gesture": Z["gesture"]})
    S = S.join(meta[["patient", "arm_type", "recording_index", "days_after_stroke"]], on="rec")
    items = pd.read_csv(HERE / "fma_items.csv")[["rec", "gesture", "fma"]]
    S = S.merge(items, on=["rec", "gesture"], how="left")
    return S, Z["rms"], Z["psd"], Z["f"]


def fig_maps(S, RMS):
    """One patient: unaffected arm and paretic sessions, three gestures."""
    imp = S[(S.arm_type == "impaired_arm") & (S.gesture != "Rest")]
    # patient with the widest within-patient spread of FMA over sessions
    spread = imp.groupby("patient").apply(
        lambda g: g.groupby("recording_index").fma.sum().agg(lambda v: v.max() - v.min()))
    nsess = imp.groupby("patient").recording_index.nunique()
    pat = spread[nsess >= 4].idxmax()
    gestures = ["MassFlexion", "DiameterGrasp", "PinchGraspPinkie"]
    P = S[S.patient == pat]
    h = P[(P.arm_type == "healthy_arm")].recording_index.min()
    sess = sorted(P[P.arm_type == "impaired_arm"].recording_index.unique())
    cols = [("healthy_arm", h)] + [("impaired_arm", s) for s in
                                   [sess[0], sess[len(sess) // 2], sess[-1]]]
    fig, ax = plt.subplots(len(gestures), len(cols), figsize=(10, 5.2))
    for i, g in enumerate(gestures):
        maps = []
        for arm, r in cols:
            k = P.index[(P.arm_type == arm) & (P.recording_index == r) & (P.gesture == g)][0]
            maps.append((RMS[k].reshape(4, 16) * 1e3, P.loc[k]))  # mV -> uV
        allv = np.concatenate([m.ravel() for m, _ in maps])
        norm = LogNorm(vmin=np.percentile(allv, 2), vmax=np.percentile(allv, 98), clip=True)
        for j, (m, row) in enumerate(maps):
            a = ax[i, j]
            im = a.imshow(m, cmap="viridis", norm=norm, aspect="auto")
            a.set_xticks([])
            a.set_yticks([])
            arm = "Unaffected" if row.arm_type == "healthy_arm" else "Paretic"
            a.set_title(f"{arm}, day {int(row.days_after_stroke)}, FMA {int(row.fma)}", fontsize=8)
            if j == 0:
                a.set_ylabel(g, fontsize=8)
        cb = fig.colorbar(im, ax=ax[i, :].tolist(), fraction=0.02, pad=0.01)
        cb.set_label(r"RMS ($\mu$V)", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    fig.savefig(FIG / "physiomio_maps.pdf", bbox_inches="tight")
    plt.close(fig)
    return pat, cols


def fig_spectra(S, PSD, f):
    """Channel-averaged PSD, mean over patients of the per-patient mean, by score."""
    G = S[S.gesture != "Rest"].copy()
    G["grp"] = np.where(G.arm_type == "healthy_arm", "Unaffected arm",
                        "Paretic, FMA " + G.fma.astype("Int64").astype(str))
    band = (f >= 20) & (f <= 450)
    for lo, hi in LINES:
        band &= ~((f >= lo) & (f <= hi))
    G["mnf_m"] = (PSD[G.index][:, band] * f[band]).sum(1) / PSD[G.index][:, band].sum(1)
    ref = G[G.arm_type == "healthy_arm"].groupby(["patient", "gesture"]).mnf_m.mean()
    G = G.join(ref.rename("mnf_ref"), on=["patient", "gesture"])
    G["d_mnf_m"] = G.mnf_m - G.mnf_ref
    order = ["Unaffected arm", "Paretic, FMA 2", "Paretic, FMA 1", "Paretic, FMA 0"]
    colors = ["0.2", "C0", "C1", "C3"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    out = {}
    logP = np.log10(PSD + 1e-30)
    for grp, c in zip(order, colors):
        idx = G.index[G.grp == grp]
        per_pat = pd.DataFrame(logP[idx]).groupby(G.loc[idx, "patient"].to_numpy()).mean()
        m = per_pat.mean().to_numpy()
        se = per_pat.std().to_numpy() / np.sqrt(len(per_pat))
        ax[0].plot(f, 10 * m, color=c, lw=1.2, label=f"{grp} ({len(per_pat)} patients)")
        ax[0].fill_between(f, 10 * (m - 1.96 * se), 10 * (m + 1.96 * se), color=c, alpha=0.15, lw=0)
        # normalized spectrum: shape only
        Pn = PSD[idx] / PSD[idx].sum(1, keepdims=True)
        pn = pd.DataFrame(Pn).groupby(G.loc[idx, "patient"].to_numpy()).mean().mean().to_numpy()
        ax[1].plot(f, pn / (f[1] - f[0]), color=c, lw=1.2)
        mnf = np.sum(f * pn)
        out[grp] = dict(n_patients=len(per_pat), n_segments=len(idx), mnf_hz=mnf,
                        mnf_masked_hz=pd.Series(G.loc[idx, "mnf_m"].to_numpy()).groupby(
                            G.loc[idx, "patient"].to_numpy()).mean().mean(),
                        d_mnf_masked_hz=np.nanmedian(G.loc[idx, "d_mnf_m"]),
                        power_db_20_450=10 * np.log10(PSD[idx][:, (f >= 20) & (f <= 450)].sum(1)).mean())
    for a in ax:
        for lo, hi in LINES:
            a.axvspan(lo, hi, color="0.85", lw=0, zorder=0)
        a.set_xlim(10, 500)
        a.set_xlabel("Frequency (Hz)")
    ax[0].set_ylabel(r"Power (dB re 1 mV$^2$/Hz)")
    ax[1].set_ylabel("Normalized power (1/Hz)")
    ax[0].set_title("(a) Absolute spectrum", loc="left", fontsize=10)
    ax[1].set_title("(b) Spectral shape", loc="left", fontsize=10)
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "physiomio_spectra.pdf")
    plt.close(fig)
    T = pd.DataFrame(out).T
    T.to_csv(HERE / "fma_spectra_summary.csv")
    return T


def fig_trajectories(S, RMS):
    """Per-session FMA sum, amplitude re unaffected arm, and predicted risk."""
    G = S[S.gesture != "Rest"].copy()
    G["lrms"] = np.log(RMS[G.index].mean(1))
    ref = G[G.arm_type == "healthy_arm"].groupby(["patient", "gesture"]).lrms.mean()
    I = G[G.arm_type == "impaired_arm"].join(ref.rename("ref"), on=["patient", "gesture"])
    I["d_db"] = 20 / np.log(10) * (I.lrms - I.ref)
    oof = pd.read_csv(HERE / "fma_oof.csv")
    oof = oof[["patient", "session", "gesture", "chan"]].rename(columns={"session": "recording_index"})
    I = I.merge(oof, on=["patient", "recording_index", "gesture"])
    sess = I.groupby(["patient", "recording_index"]).agg(
        day=("days_after_stroke", "first"), fma_sum=("fma", "sum"),
        d_db=("d_db", "mean"), risk=("chan", "mean"), n=("fma", "size")).reset_index()
    sess = sess[sess.n == 15]
    ch = sess.groupby("patient").fma_sum.agg(lambda v: v.max() - v.min())
    ns = sess.groupby("patient").size()
    pats = ch[ns >= 4].sort_values(ascending=False).index[:6]
    fig, ax = plt.subplots(2, 3, figsize=(10, 5.4), sharex=False)
    rows = []
    for a, p in zip(ax.ravel(), pats):
        d = sess[sess.patient == p].sort_values(["day", "recording_index"])
        a.plot(d.day, d.fma_sum, "o-", color="k", ms=4, label="FMA sum (of 30)")
        a.set_ylim(-1, 31)
        a.set_xlabel("Days after stroke", fontsize=8)
        a.tick_params(labelsize=7)
        b = a.twinx()
        b.plot(d.day, d.risk, "s--", color="C3", ms=3, label="Predicted risk")
        b.set_ylim(0, 1)
        b.tick_params(labelsize=7, colors="C3")
        a.set_title(f"Patient {p.replace('patient', '')}", fontsize=9)
        from scipy.stats import spearmanr
        rows.append(dict(patient=p, n_sessions=len(d), fma_min=d.fma_sum.min(), fma_max=d.fma_sum.max(),
                         rho_risk=spearmanr(d.fma_sum, d.risk).correlation,
                         rho_amp=spearmanr(d.fma_sum, d.d_db).correlation,
                         rho_day=spearmanr(d.fma_sum, d.day).correlation))
    ax[0, 0].set_ylabel("FMA item sum", fontsize=8)
    ax[1, 0].set_ylabel("FMA item sum", fontsize=8)
    fig.legend([plt.Line2D([], [], color="k", marker="o"),
                plt.Line2D([], [], color="C3", marker="s", ls="--")],
               ["FMA item sum (0-30)", "Mean predicted risk, channel model"],
               loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG / "physiomio_trajectories.pdf")
    plt.close(fig)
    R = pd.DataFrame(rows)
    # all patients with >= 4 complete sessions
    allr = []
    from scipy.stats import spearmanr
    for p, d in sess.groupby("patient"):
        if len(d) >= 4 and d.fma_sum.nunique() > 1:
            allr.append(dict(patient=p, rho_risk=spearmanr(d.fma_sum, d.risk).correlation,
                             rho_amp=spearmanr(d.fma_sum, d.d_db).correlation,
                             rho_day=spearmanr(d.fma_sum, d.day).correlation))
    A = pd.DataFrame(allr)
    A.to_csv(HERE / "fma_session_trajectories.csv", index=False)
    return R, A, sess


def main():
    S, RMS, PSD, f = load()
    pat, cols = fig_maps(S, RMS)
    print("map patient", pat, cols)
    T = fig_spectra(S, PSD, f)
    print(T.round(2).to_string())
    R, A, sess = fig_trajectories(S, RMS)
    print(R.round(2).to_string())
    print(f"{len(A)} patients with >=4 complete sessions and varying FMA sum")
    for c in ["rho_risk", "rho_amp", "rho_day"]:
        v = A[c].dropna()
        print(f"  {c}: median {v.median():.2f}, negative in {(v < 0).sum()}/{len(v)}, "
              f"positive in {(v > 0).sum()}/{len(v)}")
    print("session-level pooled (FMA sum vs risk):",
          round(pd.Series(sess.fma_sum).corr(sess.risk, method="spearman"), 3), len(sess))


if __name__ == "__main__":
    main()
