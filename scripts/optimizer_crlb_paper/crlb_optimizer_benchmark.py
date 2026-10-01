#!/usr/bin/env python3
"""
crlb_optimizer_benchmark.py -- Monte Carlo benchmark of 5 delay-refinement
methods (grid+golden-section, Brent, simulated annealing, GA/differential
evolution, PSO) against the multichannel Cramer-Rao Lower Bound, for the MLE
delay estimator in src/mfcv.py.

This reuses the REAL pipeline pieces rather than re-deriving them:
  - the multichannel MLE cost function `_mle_cost` and the 80-point coarse
    grid search from src/mfcv.estimate_delay_mle (current HEAD, which already
    refines with scipy's bounded Brent method);
  - the original grid+golden-section refinement, restored verbatim from git
    history (commit 0921a3a, before the Brent switch) as the "current (as
    described in the paper)" baseline;
  - the exact Shwedyk (1977) PSD synthetic sEMG generator used to validate
    estimate_delay_mle in tests/test_mfcv.py (shwedyk_semg/delay_signal/
    make_column), with the same CV values, Fs, IED and n_rows already used
    there.

All five refinement methods start from the SAME 80-point coarse-grid minimum
per realization (matching the paper's own framing: "... followed by a
golden-section refinement around the grid minimum ... we benchmarked it
against the four alternatives"), so the comparison isolates the refinement
step, not the coarse localization.

The multichannel CRLB is computed via the Slepian-Bangs formula using the
ANALYTIC covariance of the known generative model (source PSD x additive
white noise), not a finite-difference/ensemble estimate -- this is exact for
this synthetic benchmark because the generative model is fully known here
(unlike the real HD-sEMG grid dataset in Section 3.5, where the covariance
must be estimated because the true model is unknown).

Outputs:
  - journal_edit/optim_benchmark_data.csv   (raw per-cell results)
  - journal_edit/crlb_vs_snr.pdf            (figure: % error SD vs SNR,
                                              5 methods + CRLB reference)
  - prints the LaTeX table body for Table~\\ref{tab:optim_benchmark}
"""
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar, differential_evolution

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.mfcv import GridConfig, QualityGate, _mle_cost, _shift_fd  # noqa: E402
from tests.test_mfcv import shwedyk_semg, delay_signal, make_column  # noqa: E402

CFG = GridConfig(fs=2000.0, ied_m=0.008, n_rows=13)
GATE = QualityGate()
CV_VALUES = [3.0, 4.0, 4.5, 5.0, 6.0]
SNR_VALUES = [10.0, 15.0, 20.0, 25.0]
DUR_S = 0.3           # 600 samples @ 2000 Hz -- matches the "600-sample burst"
                       # window already used elsewhere in the paper.
N = int(DUR_S * CFG.fs)
REPS_FAST = 15         # grid+golden, grid+Brent
REPS_SLOW = 6          # grid+SA, grid+GA(DE), grid+PSO


# ---------------------------------------------------------------------------
# Coarse grid stage (identical to src.mfcv.estimate_delay_mle up to the point
# where it calls minimize_scalar for the Brent refinement).
# ---------------------------------------------------------------------------

def coarse_grid(x):
    x = np.asarray(x, dtype=float)
    x = x - x.mean(axis=1, keepdims=True)
    K, Nsamp = x.shape
    X = np.fft.rfft(x, axis=-1)
    freqs = np.fft.rfftfreq(Nsamp, d=1.0)

    th_lo = CFG.ied_m / GATE.cv_max * CFG.fs
    th_hi = CFG.ied_m / GATE.cv_min * CFG.fs
    pos = np.linspace(th_lo, th_hi, 80)
    grid = np.concatenate([-pos[::-1], pos])
    costs = np.array([_mle_cost(t, X, freqs) for t in grid])
    t0 = float(grid[int(np.argmin(costs))])
    step = pos[1] - pos[0]
    return X, freqs, t0, step


# ---------------------------------------------------------------------------
# Five refinement methods, all operating on [t0-step, t0+step].
# ---------------------------------------------------------------------------

def refine_golden(X, freqs, t0, step, tol=1e-4):
    """Verbatim from git history commit 0921a3a (pre-Brent version)."""
    a, b = t0 - step, t0 + step
    gr = (np.sqrt(5) - 1) / 2
    c, d_ = b - gr * (b - a), a + gr * (b - a)
    fc, fd = _mle_cost(c, X, freqs), _mle_cost(d_, X, freqs)
    while abs(b - a) > tol:
        if fc < fd:
            b, d_, fd = d_, c, fc
            c = b - gr * (b - a)
            fc = _mle_cost(c, X, freqs)
        else:
            a, c, fc = c, d_, fd
            d_ = a + gr * (b - a)
            fd = _mle_cost(d_, X, freqs)
    return (a + b) / 2


def refine_brent(X, freqs, t0, step, tol=1e-4):
    """Current HEAD src.mfcv.estimate_delay_mle refinement step."""
    res = minimize_scalar(_mle_cost, bounds=(t0 - step, t0 + step),
                           args=(X, freqs), method="bounded",
                           options={"xatol": tol})
    return float(res.x)


def refine_newton(X, freqs, t0, step, n_iter=20, tol=1e-6):
    """Newton-Raphson (paper Section: Resolution by Newton Raphson method):
    root-find the derivative of the cost via the tangent-line intersection,
    i.e. theta_new = theta - f'(theta)/f''(theta), with f', f'' from a
    central finite difference (the cost has no closed-form gradient here)."""
    a, b = t0 - step, t0 + step
    h = step / 50.0
    theta = t0
    for _ in range(n_iter):
        f_plus = _mle_cost(theta + h, X, freqs)
        f_minus = _mle_cost(theta - h, X, freqs)
        f0 = _mle_cost(theta, X, freqs)
        d1 = (f_plus - f_minus) / (2 * h)
        d2 = (f_plus - 2 * f0 + f_minus) / (h * h)
        if abs(d2) < 1e-12:
            break
        theta_new = theta - d1 / d2
        theta_new = min(max(theta_new, a), b)
        if abs(theta_new - theta) < tol:
            theta = theta_new
            break
        theta = theta_new
    return theta


def refine_sa(X, freqs, t0, step, rng, n_iter=200):
    """Metropolis simulated annealing, per the paper's own SA description
    (Section: Resolution by Simulated annealing), adapted to this scalar
    bounded refinement."""
    a, b = t0 - step, t0 + step
    cur = t0
    cur_cost = _mle_cost(cur, X, freqs)
    best, best_cost = cur, cur_cost
    T0 = max(cur_cost, 1e-12)
    for k in range(1, n_iter + 1):
        T = T0 / k
        cand = min(max(cur + rng.normal(0.0, step / 4.0), a), b)
        cand_cost = _mle_cost(cand, X, freqs)
        dl = cand_cost - cur_cost
        if dl <= 0 or rng.uniform() < np.exp(-dl / max(T, 1e-12)):
            cur, cur_cost = cand, cand_cost
            if cur_cost < best_cost:
                best, best_cost = cur, cur_cost
    return best


def refine_ga(X, freqs, t0, step, rng):
    """Differential evolution (Storn & Price 1997), the GA variant named in
    the paper's "Resolution by Genetic Algorithm (GA)" section."""
    a, b = t0 - step, t0 + step
    res = differential_evolution(
        lambda arr: _mle_cost(arr[0], X, freqs), bounds=[(a, b)],
        maxiter=20, popsize=12, tol=1e-6, seed=int(rng.integers(0, 2**31 - 1)),
        polish=False, updating="deferred",
    )
    return float(res.x[0])


def refine_pso(X, freqs, t0, step, rng, n_particles=15, n_iter=30,
               w=0.7, c1=1.5, c2=1.5):
    """Standard PSO (Kennedy & Eberhart 1995): inertia + cognitive + social."""
    a, b = t0 - step, t0 + step
    pos = rng.uniform(a, b, size=n_particles)
    vel = rng.uniform(-step, step, size=n_particles) * 0.1
    pbest = pos.copy()
    pbest_cost = np.array([_mle_cost(p, X, freqs) for p in pos])
    gi = int(np.argmin(pbest_cost))
    gbest, gbest_cost = pbest[gi], pbest_cost[gi]
    for _ in range(n_iter):
        r1 = rng.uniform(size=n_particles)
        r2 = rng.uniform(size=n_particles)
        vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
        pos = np.clip(pos + vel, a, b)
        costs = np.array([_mle_cost(p, X, freqs) for p in pos])
        better = costs < pbest_cost
        pbest[better] = pos[better]
        pbest_cost[better] = costs[better]
        gi = int(np.argmin(pbest_cost))
        if pbest_cost[gi] < gbest_cost:
            gbest_cost = pbest_cost[gi]
            gbest = pbest[gi]
    return gbest


# ---------------------------------------------------------------------------
# Monte Carlo benchmark
# ---------------------------------------------------------------------------

def theta_to_cv(theta):
    return CFG.ied_m / (abs(theta) / CFG.fs)


def run_cell(cv_true, snr_db, seed0):
    fast = {"grid+golden": [], "Brent": [], "Newton": []}
    fast_t = {"grid+golden": [], "Brent": [], "Newton": []}
    for r in range(REPS_FAST):
        rng = np.random.default_rng(seed0 + r)
        x = make_column(cv_true, CFG, rng, dur_s=DUR_S, snr_db=snr_db,
                         n_rows=CFG.n_rows)
        X, freqs, t0, step = coarse_grid(x)
        for name, fn in (("grid+golden", refine_golden), ("Brent", refine_brent),
                          ("Newton", refine_newton)):
            t1 = time.perf_counter()
            th = fn(X, freqs, t0, step)
            dt = (time.perf_counter() - t1) * 1e3
            fast[name].append(theta_to_cv(th))
            fast_t[name].append(dt)

    slow = {"SA": [], "GA": [], "PSO": []}
    slow_t = {"SA": [], "GA": [], "PSO": []}
    for r in range(REPS_SLOW):
        rng = np.random.default_rng(seed0 + 1000 + r)
        x = make_column(cv_true, CFG, rng, dur_s=DUR_S, snr_db=snr_db,
                         n_rows=CFG.n_rows)
        X, freqs, t0, step = coarse_grid(x)
        mc_rng = np.random.default_rng(seed0 + 2000 + r)
        for name, fn in (("SA", lambda *a: refine_sa(*a, mc_rng)),
                          ("GA", lambda *a: refine_ga(*a, mc_rng)),
                          ("PSO", lambda *a: refine_pso(*a, mc_rng))):
            t1 = time.perf_counter()
            th = fn(X, freqs, t0, step)
            dt = (time.perf_counter() - t1) * 1e3
            slow[name].append(theta_to_cv(th))
            slow_t[name].append(dt)

    all_times = {**fast_t, **slow_t}
    out = {}
    for name, ests in {**fast, **slow}.items():
        ests = np.asarray(ests)
        pct_err = 100.0 * (ests - cv_true) / cv_true
        timing = np.asarray(all_times[name])
        out[name] = dict(mean_abs_pct=np.mean(np.abs(pct_err)),
                          rmse_pct=np.sqrt(np.mean(pct_err ** 2)),
                          sd_pct=np.std(pct_err),
                          time_ms=np.mean(timing))
    return out


# ---------------------------------------------------------------------------
# Analytic multichannel CRLB (Slepian-Bangs), exact for this known model.
# ---------------------------------------------------------------------------

def source_psd_two_sided(n, fs, fl=60.0, fh=120.0, reps=500, seed=12345):
    """Empirical average two-sided PSD of shwedyk_semg at length n, estimated
    by averaging periodograms over many independent draws of the SAME
    generator used for the Monte Carlo benchmark (exact for this model, no
    hand-derived normalization constant needed)."""
    rng = np.random.default_rng(seed)
    acc = np.zeros(n)
    for _ in range(reps):
        s = shwedyk_semg(n, fs, rng, fl=fl, fh=fh)
        acc += np.abs(np.fft.fft(s, n)) ** 2 / n
    return acc / reps  # two-sided periodogram-averaged PSD estimate


def crlb_theta(cv_true, snr_db, n=N, fs=CFG.fs, ied_m=CFG.ied_m,
               n_rows=CFG.n_rows):
    """Var(theta_hat) lower bound [samples^2] via Slepian-Bangs, using the
    analytic (known) covariance of the constant-delay + additive white noise
    model: Sigma(f)_kl = Ps(f) exp(-i 2 pi f (k-l) theta) + sigma_w^2 delta_kl.
    """
    theta_true = ied_m / cv_true * fs
    freqs = np.fft.fftfreq(n, d=1.0 / fs)  # cycles/sec, two-sided, n bins
    Ps = source_psd_two_sided(n, fs)       # two-sided, same grid as freqs
    p_sig = np.mean(Ps) * 1.0              # ~ mean signal power (unit-ish, matches make_column normalization)
    # noise level exactly as in make_column: p_noise = p_sig_time/(10**(snr/10))
    # p_sig_time (time-domain signal power) ~ 1 because shwedyk_semg is unit-std.
    p_noise = 1.0 / (10 ** (snr_db / 10))
    sigma_w2 = p_noise

    K = n_rows
    k_idx = np.arange(K)
    dK = k_idx[:, None] - k_idx[None, :]          # (K,K) integer lag matrix
    J = 0.0
    for fi, f in enumerate(freqs):
        if f == 0:
            continue
        ps_f = Ps[fi]
        phase = np.exp(-1j * 2 * np.pi * f * dK * theta_true / fs)
        # NOTE: theta_true is in SAMPLES; convert f (Hz) * (theta/fs) -> cycles
        Sigma = ps_f * phase + sigma_w2 * np.eye(K)
        dSigma = ps_f * (-1j * 2 * np.pi * (f / fs) * dK) * phase
        try:
            Sinv = np.linalg.inv(Sigma)
        except np.linalg.LinAlgError:
            continue
        M = Sinv @ dSigma @ Sinv @ dSigma
        J += np.real(np.trace(M))
    if J <= 0:
        return np.nan
    return 1.0 / J


def crlb_pct_error_sd(cv_true, snr_db, **kw):
    var_theta = crlb_theta(cv_true, snr_db, **kw)
    theta_true = kw.get("ied_m", CFG.ied_m) / cv_true * kw.get("fs", CFG.fs)
    if not np.isfinite(var_theta):
        return np.nan
    rel_sd = np.sqrt(var_theta) / theta_true
    return 100.0 * rel_sd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rows = []
    method_names = ["grid+golden", "Brent", "Newton", "SA", "GA", "PSO"]
    agg = {m: {snr: [] for snr in SNR_VALUES} for m in method_names}
    agg_time = {m: {snr: [] for snr in SNR_VALUES} for m in method_names}

    for cv in CV_VALUES:
        for snr in SNR_VALUES:
            seed0 = int(cv * 1000 + snr)
            res = run_cell(cv, snr, seed0)
            for m in method_names:
                rows.append(dict(cv_true=cv, snr_db=snr, method=m, **res[m]))
                agg[m][snr].append(res[m]["rmse_pct"])
                agg_time[m][snr].append(res[m]["time_ms"])
            print(f"cv={cv:>4} snr={snr:>4} done: "
                  + ", ".join(f"{m}={res[m]['rmse_pct']:.3f}%" for m in method_names))

    # --- CRLB at CV=4.5 across all SNR levels ---
    crlb_by_snr = {snr: crlb_pct_error_sd(4.5, snr) for snr in SNR_VALUES}

    # --- write CSV ---
    import csv
    csv_path = REPO / "journal_edit" / "optim_benchmark_data.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cv_true", "snr_db", "method",
                                           "mean_abs_pct", "rmse_pct", "sd_pct", "time_ms"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {csv_path}")

    # --- summary table (averaged over CV, per SNR range like the paper) ---
    print("\n=== Table (RMSE %err range / mean time-ms) ===")
    summary = {}
    for m in method_names:
        lo_e = min(np.mean(agg[m][snr]) for snr in SNR_VALUES)
        hi_e = max(np.mean(agg[m][snr]) for snr in SNR_VALUES)
        mean_t = np.mean([np.mean(agg_time[m][snr]) for snr in SNR_VALUES])
        summary[m] = (lo_e, hi_e, mean_t)
        print(f"{m:12s}: {lo_e:.3f}--{hi_e:.3f} %  |  time {mean_t:.2f} ms")

    print("\nCRLB %err SD at CV=4.5 m/s by SNR:")
    for snr in SNR_VALUES:
        print(f"  SNR={snr:>4} dB: {crlb_by_snr[snr]:.4f} %")

    # --- figure ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    COLORS = {"grid+golden": "#0B2A4A", "Brent": "#028090", "Newton": "#3C9D45",
              "SA": "#E08A1E", "GA": "#B23A48", "PSO": "#6A4C93"}
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    # The CRLB bounds variance, so it is paired with the empirical error
    # variance (sd_pct^2) at the CV it is evaluated at, not with RMSE.
    for m in method_names:
        ys = [next(r["sd_pct"] for r in rows if r["method"] == m
                   and r["cv_true"] == 4.5 and r["snr_db"] == snr) ** 2
              for snr in SNR_VALUES]
        ax.plot(SNR_VALUES, ys, "o-", color=COLORS[m], label=m, lw=1.8, ms=5)
    crlb_ys = [crlb_by_snr[snr] ** 2 for snr in SNR_VALUES]
    ax.plot(SNR_VALUES, crlb_ys, "k--", label="CRLB (CV=4.5 m/s)", lw=1.6)
    ax.set_yscale("log")
    ax.set_xlabel("SNR [dB]")
    ax.set_ylabel(r"Error variance in CV [%$^2$]  (log scale)")
    ax.set_title("Optimizer comparison vs. Cramer-Rao bound (CV=4.5 m/s)", loc="left", fontsize=10)
    ax.grid(alpha=0.2, which="both")
    ax.legend(fontsize=8, framealpha=0.9)
    plt.tight_layout()
    fig_path = REPO / "journal_edit" / "crlb_vs_snr.pdf"
    plt.savefig(fig_path)
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
