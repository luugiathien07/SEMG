#!/usr/bin/env python3
"""
run_tvd_methods.py -- eight two-channel time-varying-delay (TVD) estimators on
three sources x two ground truths x four SNRs, scored on the whole CV(n)
trajectory against the known delay.

Sources (all from crlb_tvd_benchmark / the multichannel paper, unchanged):
  colored   Shwedyk-PSD colored noise, redrawn every realization
  muap      one Farina-Merletti MUAP channel, tiled (fixed source)
  multipop  one channel of the multi-motor-unit-population simulator (fixed)
Ground truths: linear CV ramp 4.5->5.5 m/s (true_theta) and the
non-polynomial inverse-sinusoid of Eq. (25) in Luu et al. 2018
(true_theta_sinusoidal), both over the N=600-sample window.

Estimators (every one returns theta_hat at the same evaluation instants
N_EVAL = 50, 60, ..., 550, i.e. centers of full 100-sample windows):
  poly_local   local order-1 polynomial, cubic-spline warp, BFGS, per window
               (the pipeline's own local fit, crlb_tvd_benchmark)
  gcc          cross-correlation + parabolic interpolation, per window
  lap          single-scale Local All-Pass filter (Gilliam et al. 2018)
  cohf         Fourier phase coherency (Leclerc et al. 2008; PAPERV7 Eq. 6):
               Welch cross-spectrum of 3 half-length Hann segments, delay =
               coherence-weighted slope of the unwrapped phase in a fixed band
  legendre5    one global order-5 Legendre-polynomial delay model
               (Boualem et al. 2015), same warp cost, BFGS, initialized by
               projecting the GCC trajectory onto the basis
  dll          delay-locked-loop / normalized-gradient tracker in the spirit
               of Xu et al. 2017, run forward and backward and averaged
  dp           dynamic-programming path through the local normalized
               cross-correlation surface with a smoothness penalty
               (Gupta et al. 2010)
  lap_kalman   LAP estimates smoothed by a random-walk RTS Kalman smoother
               (Jelfs & Gilliam 2019)

Tuning constants (DLL step, DP penalty, Kalman q/r, CohF band) are fixed a
priori below and are the same for every source, truth and SNR.

Run from the repository root:
    Motionlab/.venv/bin/python scripts/tvd_methods_extension/run_tvd_methods.py [M]
Output: journal_edit/tvd_methods_extension_data.csv (one row per
source x truth x SNR x method x evaluation instant, with mean and variance of
the % CV error over the M realizations) and ..._runtime.csv.
"""
from __future__ import annotations

import csv
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from numpy.polynomial import legendre as L
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "multichannel_paper"))

from src.mfcv import _gcc_delay  # noqa: E402
from tests.test_mfcv import shwedyk_semg  # noqa: E402
from scripts.optimizer_crlb_paper.crlb_tvd_benchmark import (  # noqa: E402
    CFG, N, SNR_VALUES, SEG_LEN, true_theta, true_theta_sinusoidal,
    load_muap_source, make_pair, refine_bfgs_local, lap_theta_end, LAP_R,
)
from realistic_source_multichannel import load_source as load_multipop_source  # noqa: E402

M_DEFAULT = 500
HALF = SEG_LEN // 2
N_EVAL = np.arange(HALF, N - HALF + 1, 10)          # 50..550
THETA_MIN, THETA_MAX = 1.0, 12.0                    # clip: CV in [1.3, 16] m/s
METHODS = ["poly_local", "gcc", "lap", "cohf", "legendre5", "dll", "dp", "lap_kalman"]
SOURCES = ["colored", "muap", "multipop"]
TRUTHS = {"ramp": true_theta, "sinusoid": true_theta_sinusoidal}

# a-priori constants, identical for every cell
COHF_BAND = (20.0, 350.0)   # Hz; source PSDs peak at 60-150 Hz
DLL_MU = 0.02               # normalized step
DLL_ALPHA = 0.02            # power-estimate forgetting factor
DP_LAGS = np.arange(1.0, 9.0 + 1e-9, 0.05)
DP_LAMBDA = 2.0             # penalty per (sample of delay change)^2 between eval instants
KAL_Q, KAL_R = 0.02, 0.5    # random-walk process / measurement variance (samples^2)
LEG_ORDER = 5


# ---------------------------------------------------------------------------
# per-window estimators
# ---------------------------------------------------------------------------

def _windows(x1, x2):
    for c in N_EVAL:
        yield x1[c - HALF:c + HALF], x2[c - HALF:c + HALF]


def est_gcc(x1, x2):
    return np.array([_gcc_delay(a, b, max_lag=20)[0] for a, b in _windows(x1, x2)])


def est_lap(x1, x2):
    return np.array([lap_theta_end(a, b, R=LAP_R, W=SEG_LEN) for a, b in _windows(x1, x2)])


def est_poly_local(x1, x2):
    out = []
    for a, b in _windows(x1, x2):
        d0 = np.array([_gcc_delay(a, b, max_lag=20)[0], 0.0])
        d = refine_bfgs_local(a, b, d0)
        out.append(np.polyval(d[::-1], HALF))  # delay at the window center
    return np.array(out)


def _cohf_one(a, b, fs=CFG.fs, nfft=256):
    seg = len(a) // 2
    hop = seg // 2
    w = np.hanning(seg)
    s12 = s11 = s22 = 0
    for st in range(0, len(a) - seg + 1, hop):  # 3 segments, 50% overlap
        A = np.fft.rfft(w * (a[st:st + seg] - a[st:st + seg].mean()), nfft)
        B = np.fft.rfft(w * (b[st:st + seg] - b[st:st + seg].mean()), nfft)
        s12 = s12 + A * np.conj(B)
        s11 = s11 + np.abs(A) ** 2
        s22 = s22 + np.abs(B) ** 2
    f = np.fft.rfftfreq(nfft, 1 / fs)
    coh = s12 / np.sqrt(s11 * s22 + 1e-30)
    band = (f >= COHF_BAND[0]) & (f <= COHF_BAND[1])
    ph = np.unwrap(np.angle(coh[np.r_[0, np.where(band)[0]]]))[1:]  # unwrap from DC
    wgt = np.abs(coh[band]) ** 2
    fb = f[band]
    # X1 X2* = |S|^2 e^{+j 2 pi f theta/fs} for x2 = x1 delayed by theta
    slope = np.sum(wgt * fb * ph) / np.sum(wgt * fb ** 2)
    return slope * fs / (2 * np.pi)


def est_cohf(x1, x2):
    return np.array([_cohf_one(a, b) for a, b in _windows(x1, x2)])


# ---------------------------------------------------------------------------
# whole-window estimators
# ---------------------------------------------------------------------------

_T = 2.0 * np.arange(N) / (N - 1) - 1.0


def _leg_cost(c, x1, x2):
    theta = L.legval(_T, c)
    x1w = CubicSpline(np.arange(N), x1, extrapolate=True)(np.arange(N) - theta)
    lo, hi = int(0.05 * N), int(0.95 * N)
    return float(np.sum((x2[lo:hi] - x1w[lo:hi]) ** 2))


def est_legendre5(x1, x2):
    g = est_gcc(x1, x2)
    c0 = L.legfit(_T[N_EVAL], g, LEG_ORDER)
    res = minimize(_leg_cost, c0, args=(x1, x2), method="BFGS", options={"maxiter": 200})
    return L.legval(_T[N_EVAL], res.x)


def _dll_pass(x1, x2, theta0):
    n = np.arange(N)
    sp = CubicSpline(n, x1, extrapolate=True)
    dsp = sp.derivative()
    th = np.empty(N)
    t = theta0
    p = np.mean(np.gradient(x1) ** 2)
    for k in range(N):
        e = x2[k] - sp(k - t)
        g = dsp(k - t)
        p = (1 - DLL_ALPHA) * p + DLL_ALPHA * g * g
        # e = x2(k) - x1(k - t), de/dt = x1'(k - t): gradient step on e^2
        t = float(np.clip(t - DLL_MU * e * g / (p + 1e-12), THETA_MIN, THETA_MAX))
        th[k] = t
    return th


def est_dll(x1, x2):
    t0 = _gcc_delay(x1[:SEG_LEN], x2[:SEG_LEN], max_lag=20)[0]
    fwd = _dll_pass(x1, x2, t0)
    # backward pass: time-reverse both channels; the delay changes sign
    t1 = _gcc_delay(x1[-SEG_LEN:], x2[-SEG_LEN:], max_lag=20)[0]
    bwd = _dll_pass(x2[::-1], x1[::-1], t1)[::-1]
    return 0.5 * (fwd + bwd)[N_EVAL]


def est_dp(x1, x2):
    int_lags = np.arange(0, 11)
    surf = []
    for a, b in _windows(x1, x2):
        a = a - a.mean()
        b = b - b.mean()
        den = np.sqrt(np.sum(a * a) * np.sum(b * b)) + 1e-30
        r = np.array([np.sum(a[:len(a) - l] * b[l:]) for l in int_lags]) / den
        surf.append(CubicSpline(int_lags, r)(DP_LAGS))
    S = np.array(surf)                       # (n_eval, n_lags)
    n_e, n_l = S.shape
    D = (DP_LAGS[:, None] - DP_LAGS[None, :]) ** 2 * DP_LAMBDA
    score = S[0].copy()
    back = np.zeros((n_e, n_l), dtype=int)
    for i in range(1, n_e):
        tot = score[None, :] - D              # rows: current lag, cols: previous lag
        back[i] = np.argmax(tot, axis=1)
        score = S[i] + tot[np.arange(n_l), back[i]]
    path = np.empty(n_e, dtype=int)
    path[-1] = int(np.argmax(score))
    for i in range(n_e - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return DP_LAGS[path]


def est_lap_kalman(x1, x2):
    z = est_lap(x1, x2)
    n = len(z)
    xf, pf = np.empty(n), np.empty(n)
    x, p = z[0], KAL_R
    for i in range(n):
        if i:
            p = p + KAL_Q
        k = p / (p + KAL_R)
        x = x + k * (z[i] - x)
        p = (1 - k) * p
        xf[i], pf[i] = x, p
    xs = xf.copy()
    for i in range(n - 2, -1, -1):
        c = pf[i] / (pf[i] + KAL_Q)
        xs[i] = xf[i] + c * (xs[i + 1] - xf[i])
    return xs


EST = {"poly_local": est_poly_local, "gcc": est_gcc, "lap": est_lap, "cohf": est_cohf,
       "legendre5": est_legendre5, "dll": est_dll, "dp": est_dp, "lap_kalman": est_lap_kalman}


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

_FIXED = {}


def _source(name, rng):
    if name == "colored":
        return shwedyk_semg(N, CFG.fs, rng)
    if name not in _FIXED:
        _FIXED[name] = load_muap_source() if name == "muap" else load_multipop_source()
    return _FIXED[name]


def _one(args):
    src, truth, snr, m = args
    rng = np.random.default_rng([SOURCES.index(src), list(TRUTHS).index(truth), int(snr), m])
    s = _source(src, rng)
    x1, x2 = make_pair(rng, snr, s, TRUTHS[truth])
    th_true = TRUTHS[truth](N_EVAL)
    cv_true = CFG.ied_m * CFG.fs / th_true
    out = {}
    for meth, f in EST.items():
        t0 = time.perf_counter()
        th = np.asarray(f(x1, x2), dtype=float)
        dt = time.perf_counter() - t0
        bad = ~np.isfinite(th) | (np.abs(th) < THETA_MIN) | (np.abs(th) > THETA_MAX)
        th = np.clip(np.abs(np.nan_to_num(th, nan=THETA_MAX)), THETA_MIN, THETA_MAX)
        err = 100.0 * (CFG.ied_m * CFG.fs / th - cv_true) / cv_true
        out[meth] = (err, bad, dt)
    return (src, truth, snr, m), out


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else M_DEFAULT
    jobs = [(s, t, snr, m) for s in SOURCES for t in TRUTHS for snr in SNR_VALUES for m in range(M)]
    acc = {}
    t0 = time.time()
    with Pool(28) as pool:
        for i, (key, out) in enumerate(pool.imap_unordered(_one, jobs, chunksize=2)):
            src, truth, snr, _ = key
            for meth, (err, bad, dt) in out.items():
                a = acc.setdefault((src, truth, snr, meth), {"err": [], "bad": [], "dt": []})
                a["err"].append(err)
                a["bad"].append(bad)
                a["dt"].append(dt)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(jobs)} realizations ({time.time() - t0:.0f}s)", flush=True)
    out_dir = REPO / "journal_edit"
    with open(out_dir / "tvd_methods_extension_data.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "truth", "snr_db", "method", "n", "M", "mean_err_pct",
                    "var_err_pct2", "frac_clipped"])
        for (src, truth, snr, meth), a in sorted(acc.items()):
            E, B = np.array(a["err"]), np.array(a["bad"])
            for j, n in enumerate(N_EVAL):
                w.writerow([src, truth, snr, meth, int(n), E.shape[0], f"{E[:, j].mean():.6g}",
                            f"{E[:, j].var():.6g}", f"{B[:, j].mean():.4g}"])
    with open(out_dir / "tvd_methods_extension_runtime.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "mean_ms_per_realization"])
        for meth in METHODS:
            dts = [d for (s, t, snr, m), a in acc.items() if m == meth for d in a["dt"]]
            w.writerow([meth, f"{1000 * np.mean(dts):.3f}"])
    print(f"Wrote journal_edit/tvd_methods_extension_data.csv (M={M}, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
