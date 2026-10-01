#!/usr/bin/env python3
"""
run_all_methods.py -- one Monte Carlo over every two-channel delay estimator
used across the manuscript series, on identical realizations.

Same design as run_tvd_methods.py (3 sources x 2 ground truths x 4 SNRs,
estimates at N_EVAL = 50..550, whole-trajectory scoring), with more
estimators and with the raw per-realization errors kept so that robust
statistics (median, outlier share) can be computed afterwards:

  from run_tvd_methods.py : poly_local, gcc (= plain CC), lap, cohf, legendre5,
                            dll, dp, lap_kalman
  GCC family (Knapp & Carter 1976), per 100-sample window:
      gcc_phat, gcc_scot, gcc_roth   single-periodogram weightings, as in
                                     compare_gcc_family.py
      gcc_ht, gcc_eckart             need a smoothed coherence / noise PSD,
                                     taken from the same 3-segment Welch
                                     estimate as cohf
  constmle     local order-0 delay by the grid + Brent search of the scalar
               case (crlb_optimizer_benchmark), per window
  global_bfgs  one order-1 polynomial over the whole window, BFGS
               (crlb_tvd_benchmark.refine_bfgs)
  global_sa    the same model, simulated annealing (refine_sa)

Every realization of a (source, truth, SNR) cell is shared by all estimators.
Cells are written to journal_edit/tvd_all_methods_parts/ as they finish, so an
interrupted run resumes where it stopped.

    Motionlab/.venv/bin/python scripts/tvd_methods_extension/run_all_methods.py [M] [workers]
"""
from __future__ import annotations

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "tvd_methods_extension"))
sys.path.insert(0, str(REPO / "scripts" / "multichannel_paper"))

import run_tvd_methods as base  # noqa: E402
from run_tvd_methods import (  # noqa: E402
    CFG, N, N_EVAL, HALF, SNR_VALUES, SOURCES, TRUTHS, THETA_MIN, THETA_MAX, _windows,
)
from scripts.optimizer_crlb_paper.crlb_optimizer_benchmark import coarse_grid, refine_brent  # noqa: E402
from scripts.optimizer_crlb_paper.crlb_tvd_benchmark import (  # noqa: E402
    make_pair, d0_init, refine_bfgs, refine_sa,
)
from compare_gcc_family import gcc_family_delay  # noqa: E402

PARTS = REPO / "journal_edit" / "tvd_all_methods_parts"


def _welch(a, b, nfft=256):
    seg = len(a) // 2
    w = np.hanning(seg)
    g11 = g22 = g12 = 0
    for st in range(0, len(a) - seg + 1, seg // 2):
        A = np.fft.rfft(w * (a[st:st + seg] - a[st:st + seg].mean()), nfft)
        B = np.fft.rfft(w * (b[st:st + seg] - b[st:st + seg].mean()), nfft)
        g11 = g11 + np.abs(A) ** 2
        g22 = g22 + np.abs(B) ** 2
        g12 = g12 + A * np.conj(B)
    return g11, g22, g12, nfft


def _peak(W, g12, nfft, max_lag=20):
    r = np.fft.irfft(W * g12, nfft)
    r = np.concatenate((r[-max_lag:], r[:max_lag + 1]))
    lags = np.arange(-max_lag, max_lag + 1)
    i = int(np.argmax(r))
    frac = 0.0
    if 0 < i < len(r) - 1:
        y0, y1, y2 = r[i - 1], r[i], r[i + 1]
        den = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    return -(lags[i] + frac)  # same sign convention as src.mfcv._gcc_delay


def _gcc_smoothed(a, b, kind):
    g11, g22, g12, nfft = _welch(a, b)
    m12 = np.abs(g12) + 1e-30
    if kind == "ht":  # Hannan-Thomson / ML weighting
        coh2 = np.clip(m12 ** 2 / (g11 * g22 + 1e-30), 0, 0.999)
        W = coh2 / (m12 * (1 - coh2))
    else:  # Eckart: Gss / (Gn1n1 Gn2n2), Gss ~ |G12|
        n1 = np.maximum(g11 - m12, 1e-3 * g11)
        n2 = np.maximum(g22 - m12, 1e-3 * g22)
        W = m12 / (n1 * n2)
    return _peak(W, g12, nfft)


def est_family(kind):
    def f(x1, x2):
        return np.array([gcc_family_delay(a, b, kind, max_lag=20) for a, b in _windows(x1, x2)])
    return f


def est_smoothed(kind):
    def f(x1, x2):
        return np.array([_gcc_smoothed(a, b, kind) for a, b in _windows(x1, x2)])
    return f


def est_constmle(x1, x2):
    out = []
    for a, b in _windows(x1, x2):
        X, freqs, t0, step = coarse_grid(np.stack([a, b], axis=0))
        out.append(refine_brent(X, freqs, t0, step))
    return np.array(out)


def est_global_bfgs(x1, x2):
    d = refine_bfgs(x1, x2, d0_init(x1, x2))
    return np.polyval(d[::-1], N_EVAL)


def est_global_sa(x1, x2, rng):
    d = refine_sa(x1, x2, d0_init(x1, x2), rng)
    return np.polyval(d[::-1], N_EVAL)


EST = dict(base.EST)
EST.update({
    "gcc_phat": est_family("phat"), "gcc_scot": est_family("scot"),
    "gcc_roth": est_family("roth"), "gcc_ht": est_smoothed("ht"),
    "gcc_eckart": est_smoothed("eckart"), "constmle": est_constmle,
    "global_bfgs": est_global_bfgs, "global_sa": None,  # needs rng, handled below
})
METHODS = list(EST)


def _one(args):
    src, truth, snr, m = args
    rng = np.random.default_rng([SOURCES.index(src), list(TRUTHS).index(truth), int(snr), m])
    s = base._source(src, rng)
    x1, x2 = make_pair(rng, snr, s, TRUTHS[truth])
    cv_true = CFG.ied_m * CFG.fs / TRUTHS[truth](N_EVAL)
    sa_rng = np.random.default_rng([99, SOURCES.index(src), list(TRUTHS).index(truth), int(snr), m])
    err = np.empty((len(METHODS), len(N_EVAL)), dtype=np.float32)
    dt = np.empty(len(METHODS))
    for i, meth in enumerate(METHODS):
        t0 = time.perf_counter()
        th = est_global_sa(x1, x2, sa_rng) if meth == "global_sa" else EST[meth](x1, x2)
        dt[i] = time.perf_counter() - t0
        th = np.clip(np.abs(np.nan_to_num(np.asarray(th, float), nan=THETA_MAX)), THETA_MIN, THETA_MAX)
        err[i] = 100.0 * (CFG.ied_m * CFG.fs / th - cv_true) / cv_true
    return m, err, dt


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    PARTS.mkdir(parents=True, exist_ok=True)
    cells = [(s, t, snr) for s in SOURCES for t in TRUTHS for snr in SNR_VALUES]
    t_all = time.time()
    with Pool(workers) as pool:
        for ci, (src, truth, snr) in enumerate(cells, 1):
            out = PARTS / f"{src}_{truth}_{int(snr)}dB_M{M}.npz"
            if out.exists():
                print(f"[{ci}/{len(cells)}] {out.name} exists, skipped", flush=True)
                continue
            t0 = time.time()
            E = np.empty((M, len(METHODS), len(N_EVAL)), dtype=np.float32)
            T = np.empty((M, len(METHODS)))
            for m, err, dt in pool.imap_unordered(_one, [(src, truth, snr, m) for m in range(M)], chunksize=4):
                E[m], T[m] = err, dt
            np.savez_compressed(out, err=E, time_s=T, methods=np.array(METHODS), n_eval=N_EVAL,
                                source=src, truth=truth, snr_db=snr)
            print(f"[{ci}/{len(cells)}] {src} x {truth} @ {snr:g} dB done in {time.time() - t0:.0f}s "
                  f"(total {time.time() - t_all:.0f}s)", flush=True)
    print(f"All {len(cells)} cells in {PARTS}")


if __name__ == "__main__":
    main()
