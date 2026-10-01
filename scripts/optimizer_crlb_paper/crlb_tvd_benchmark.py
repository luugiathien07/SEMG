#!/usr/bin/env python3
"""
crlb_tvd_benchmark.py -- Optimizer comparison for the TIME-VARYING delay
(TVD) case: theta(n) = d0 + d1*n (order-1 polynomial), a 2-channel model
x1(n) = s(n) + w1(n), x2(n) = s(n - theta(n)) + w2(n), matching the original
paper's TVD polynomial model (Section: Adaptation / Resolution by Newton
Raphson) and the real MATLAB implementation in
"programme.../delais_variable/newton/.../eqm12.m" +
"Newton_lineaire.m", which this reproduces faithfully:
  - cost(d) = sum over the MIDDLE 90% of the window of
              (x2(n) - warp(x1, theta(n)))^2
  - theta(n) = d0 + d1*n + ... (polyval-style, order given by len(d)-1)
  - ground truth in Newton_lineaire.m is itself a per-slice LINEAR delay
    ramp (order-1), which is what is reproduced as the BASELINE variant here.
  - the real code resolves d via MATLAB's fminunc, whose default algorithm
    is quasi-Newton (BFGS) -- reproduced here as scipy's BFGS, not
    reimplemented from scratch, since this is what "Newton-type" resolution
    means for a >1-D parameter vector in the original code.

Unlike the constant-delay case, golden-section and Brent's method do not
generalize to a >1-D parameter vector (they are 1-D bounded methods), so
they are excluded here by construction, not benchmarked and found worse --
this asymmetry is itself part of what this script demonstrates.

Optimizers compared: BFGS (quasi-Newton, the incumbent/real-code choice),
simulated annealing, a genetic algorithm (differential evolution), and
particle swarm optimization, each generalized to the 2-D coefficient vector
[d0, d1].

CRLB: approximated via a linear-basis-expansion of the SAME scalar Fisher
information used in the constant-delay CRLB (crlb_optimizer_benchmark.py),
under the standard approximation that the instantaneous Fisher information
is locally constant over one analysis window (valid for a modest delay
ramp): FIM[i,j] = J_scalar * sum_n phi_i(n) phi_j(n), phi = [1, n].

Three variants (--variant {baseline,sinusoidal,muap,all}), isolating one
factor at a time relative to the baseline:
  - baseline:   Shwedyk-PSD colored-noise source + linear CV ramp (CV0->CV1).
                Unchanged from the original version of this script -- exact
                output filenames/numbers reproduce journal_edit/tvd_*.
  - sinusoidal: SAME colored-noise source, but the ground-truth delay is the
                "inverse sinusoidal TVD" of Eq. (25) in Luu et al. (2018,
                Fluct. Noise Lett. 17(2):1850015) -- a genuinely
                non-polynomial delay, re-derived for this benchmark's own
                Fs/IED/window (see true_theta_sinusoidal docstring). Tests
                whether the optimizer ranking survives fitting an order-1
                polynomial to a delay that is NOT actually polynomial.
  - muap:       SAME linear CV ramp, but the source waveform is a real
                Farina-Merletti MUAP template (see load_muap_source) instead
                of colored noise. Tests whether the optimizer ranking
                survives warping a realistic, structured (non-Gaussian)
                waveform instead of colored noise -- the TVD analogue of
                crlb_muap_benchmark.py's scalar-case robustness check.

Outputs:
  - journal_edit/tvd_benchmark_data.csv        (baseline)
  - journal_edit/tvd_vs_snr.pdf                (baseline)
  - journal_edit/tvd_benchmark_sinusoidal_data.csv
  - journal_edit/tvd_vs_snr_sinusoidal.pdf
  - journal_edit/tvd_benchmark_muap_data.csv
  - journal_edit/tvd_vs_snr_muap.pdf
"""
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize, differential_evolution

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.mfcv import GridConfig  # noqa: E402
from tests.test_mfcv import shwedyk_semg  # noqa: E402
from scripts.optimizer_crlb_paper.crlb_optimizer_benchmark import (  # noqa: E402
    crlb_theta, coarse_grid, refine_brent,
)

CFG = GridConfig(fs=2000.0, ied_m=0.008, n_rows=13)
N = 600                 # samples, matches the "600-sample burst" convention
                         # used elsewhere in the paper for dynamic contraction
CV0 = 4.5                # m/s, center conduction velocity (CRLB operating point)
CV1 = 5.5                # m/s, CV at the end of the window (linear ramp)
SNR_VALUES = [10.0, 15.0, 20.0, 25.0]
REPS_FAST = 12           # BFGS
REPS_SLOW = 6            # SA, GA, PSO


def true_theta(n):
    """Linear CV ramp CV0->CV1 over the window, converted to a delay ramp
    theta(n) [samples] between channel 1 and channel 2 (IED apart). This is
    the BASELINE ground truth (matches Newton_lineaire.m, see module
    docstring)."""
    cv_n = CV0 + (CV1 - CV0) * (n / (N - 1))
    return CFG.ied_m / cv_n * CFG.fs


# --- Sinusoidal (non-polynomial) ground truth, reproducing Eq. (25) of
# Luu et al. (2018) -- see true_theta_sinusoidal docstring for the
# re-derivation of (K, B, C, omega) for this benchmark's own setup.
_SIN_CV_LOW, _SIN_CV_HIGH = 3.0, 6.0   # m/s, the CV grid already used
                                        # elsewhere in this paper (see
                                        # crlb_optimizer_benchmark.py)
_SIN_B, _SIN_C = 3.0, 1.0              # (B+C)/(B-C) = 2 = CV_HIGH/CV_LOW
_SIN_K = CFG.ied_m * CFG.fs * (_SIN_B - _SIN_C) / _SIN_CV_LOW  # samples
_SIN_OMEGA = 2.0 * np.pi / N            # one full cycle over the window,
                                         # matching J25's own choice of one
                                         # cycle over its own window


def true_theta_sinusoidal(n):
    """Reproduces the FUNCTIONAL FORM of Eq. (25) in Luu et al. (2018,
    Fluct. Noise Lett. 17(2):1850015):

        theta(n) = K / (B + C*sin(omega*n))

    the "inverse sinusoidal TVD" used there as a genuinely non-polynomial
    ground truth, to test how well a fixed-order polynomial fit approximates
    it. J25 used its own Fs=2048 Hz, IED=5 mm, a 5 s (10240-sample) window
    and (K,B,C) tuned to that setup, giving a CV excursion of 2-8 m/s. Here
    the same functional form is re-derived for this benchmark's own
    Fs=2000 Hz, IED=8 mm, N=600-sample window: one full sinusoid cycle over
    the window (matching J25's own choice of one cycle over its own window),
    and (B,C)=(3,1) chosen so CV(n) sweeps exactly the [3, 6] m/s range
    already used for the other benchmarks in this paper (the CV grid in
    crlb_optimizer_benchmark.py), rather than J25's own [2, 8] m/s range.
    """
    return _SIN_K / (_SIN_B + _SIN_C * np.sin(_SIN_OMEGA * n))


def crlb_sinusoidal_fim(snr_db, window_len=N, cv_op=CV0):
    """FIM for the TRUE, correctly-specified sinusoidal parameter vector
    p=(K,B,C) of true_theta_sinusoidal (theta(n)=K/(B+C*sin(omega*n))), at
    the true parameter values, under the SAME local-constant-information
    approximation as crlb_linear_fim (same j_density, same self-consistency
    argument -- see that function's docstring). This answers a different
    question from crlb_linear_fim's order-1 bound: not 'how well can an
    order-1 polynomial be fit to this window' (a misspecified-model
    question, blind to the resulting bias) but 'if the sinusoidal form
    itself were known and (K,B,C) estimated directly, what is the
    noise-limited floor' -- a correctly-specified-model bound that, unlike
    the polynomial one, is NOT tautologically insensitive to the
    non-polynomial shape of the true delay."""
    var_theta = crlb_theta(cv_op, snr_db, n=window_len, fs=CFG.fs,
                            ied_m=CFG.ied_m, n_rows=2)
    if not np.isfinite(var_theta) or var_theta <= 0:
        return None
    j_density = 1.0 / var_theta / window_len
    n = np.arange(window_len)
    D = _SIN_B + _SIN_C * np.sin(_SIN_OMEGA * n)
    dK = 1.0 / D
    dB = -_SIN_K / D ** 2
    dC = -_SIN_K * np.sin(_SIN_OMEGA * n) / D ** 2
    Psi = np.stack([dK, dB, dC], axis=0)  # (3, window_len)
    FIM = j_density * (Psi @ Psi.T)
    try:
        return np.linalg.inv(FIM)
    except np.linalg.LinAlgError:
        return None


def crlb_sinusoidal_theta_pct(snr_db, window_len=N, n_eval=None, cv_op=CV0):
    """CRLB on theta(n_eval) [as %CV error] under the correctly-specified
    sinusoidal model, via the delta method: Var(theta(n)) ~= psi(n)^T
    COV(K,B,C) psi(n), psi(n) = d theta(n)/d(K,B,C) -- the direct analogue,
    for this parameterization, of crlb_linear_fim's phi(n)^T FIM^-1 phi(n)
    for the polynomial case."""
    if n_eval is None:
        n_eval = window_len - 1
    COV = crlb_sinusoidal_fim(snr_db, window_len, cv_op=cv_op)
    if COV is None:
        return np.nan
    D = _SIN_B + _SIN_C * np.sin(_SIN_OMEGA * n_eval)
    psi = np.array([1.0 / D, -_SIN_K / D ** 2, -_SIN_K * np.sin(_SIN_OMEGA * n_eval) / D ** 2])
    var_theta_end = float(psi @ COV @ psi)
    if var_theta_end < 0 or not np.isfinite(var_theta_end):
        return np.nan
    theta_true = true_theta_sinusoidal(n_eval)
    return 100.0 * np.sqrt(var_theta_end) / theta_true


MUAP_CHANNEL = 6  # middle channel of the 13-channel array (0-indexed)


def load_muap_source():
    """Realistic single-channel source waveform: one channel (the array's
    middle element, index 6 of 13) of the Farina-Merletti MUAP template
    already used for the scalar-case robustness check
    (crlb_muap_benchmark.py, muap_cv4p5.mat), tiled to fill the 600-sample
    TVD analysis window -- a periodic train of one real MUAP shape,
    standing in for a motor unit firing repetitively, in place of
    Shwedyk-PSD colored noise. This isolates the effect of a realistic,
    structured (non-Gaussian) source waveform from the effect of the
    ground-truth delay shape (tested separately by true_theta_sinusoidal
    above): this variant keeps the SAME linear-ramp ground truth as the
    baseline (true_theta) and only changes the source.
    """
    p = REPO / "scripts" / "optimizer_crlb_paper" / "muap_cv4p5.mat"
    d = sio.loadmat(p, squeeze_me=True, struct_as_record=False)
    template = d["MUAP"][MUAP_CHANNEL].astype(float)
    reps = int(np.ceil(N / len(template)))
    return np.tile(template, reps)[:N]


def make_pair(rng, snr_db, source, theta_fn):
    """x2(n) = s(n - theta(n)): build via cubic-spline resampling of the
    source s at (n - theta(n)) -- matches eqm12.m's Delay_Modeling_Var role,
    just implemented with a standard interpolator instead of a custom
    filter. `source` is either freshly generated per call (baseline,
    sinusoidal variants) or a fixed array reused across realizations (muap
    variant); `theta_fn` selects the ground-truth delay shape."""
    s = source
    n = np.arange(N)
    th = theta_fn(n)
    x1 = s.copy()
    spline = CubicSpline(n, s, extrapolate=True)
    x2 = spline(n - th)
    p_sig = np.mean(s ** 2)
    p_noise = p_sig / (10 ** (snr_db / 10))
    x1 = x1 + rng.standard_normal(N) * np.sqrt(p_noise)
    x2 = x2 + rng.standard_normal(N) * np.sqrt(p_noise)
    return x1, x2


def tvd_cost(d, x1, x2):
    """Faithful port of eqm12.m: warp x1 by the candidate polynomial delay
    and compare to x2 over the middle 90% of the window."""
    n = np.arange(N)
    theta = np.polyval(d[::-1], n)  # d = [d0, d1, ...] ascending, matches eqm12.m
    spline = CubicSpline(n, x1, extrapolate=True)
    x1_warped = spline(n - theta)
    deb = int(round(0.05 * N))
    fin = int(round(0.95 * N))
    return float(np.sum((x2[deb:fin] - x1_warped[deb:fin]) ** 2))


def d0_init(x1, x2):
    """Coarse initialization: constant-delay cross-correlation estimate for
    d0, zero for the slope -- analogous to Newton_lineaire.m's two-point
    slice initialization, simplified to a single-window coarse start."""
    from src.mfcv import _gcc_delay
    delay0, _ = _gcc_delay(x1, x2, max_lag=40)
    return np.array([delay0, 0.0])


def refine_bfgs(x1, x2, d0):
    res = minimize(tvd_cost, d0, args=(x1, x2), method="BFGS",
                    options={"maxiter": 100})
    return res.x


def refine_sa(x1, x2, d0, rng, n_iter=150):
    bounds = np.array([[d0[0] - 5, d0[0] + 5], [d0[1] - 0.05, d0[1] + 0.05]])
    cur = d0.copy()
    cur_cost = tvd_cost(cur, x1, x2)
    best, best_cost = cur.copy(), cur_cost
    T0 = max(cur_cost, 1e-12)
    step = np.array([0.5, 0.005])
    for k in range(1, n_iter + 1):
        T = T0 / k
        cand = cur + rng.normal(0, 1, size=2) * step
        cand = np.clip(cand, bounds[:, 0], bounds[:, 1])
        cand_cost = tvd_cost(cand, x1, x2)
        dl = cand_cost - cur_cost
        if dl <= 0 or rng.uniform() < np.exp(-dl / max(T, 1e-12)):
            cur, cur_cost = cand, cand_cost
            if cur_cost < best_cost:
                best, best_cost = cur.copy(), cur_cost
    return best


def refine_ga(x1, x2, d0, rng):
    bounds = [(d0[0] - 5, d0[0] + 5), (d0[1] - 0.05, d0[1] + 0.05)]
    res = differential_evolution(tvd_cost, bounds, args=(x1, x2), maxiter=20,
                                  popsize=12, tol=1e-6,
                                  seed=int(rng.integers(0, 2**31 - 1)),
                                  polish=False, updating="deferred")
    return res.x


def refine_pso(x1, x2, d0, rng, n_particles=15, n_iter=30, w=0.7, c1=1.5, c2=1.5):
    lo = d0 - np.array([5, 0.05])
    hi = d0 + np.array([5, 0.05])
    pos = rng.uniform(lo, hi, size=(n_particles, 2))
    vel = rng.uniform(-1, 1, size=(n_particles, 2)) * (hi - lo) * 0.1
    pbest = pos.copy()
    pbest_cost = np.array([tvd_cost(p, x1, x2) for p in pos])
    gi = int(np.argmin(pbest_cost))
    gbest, gbest_cost = pbest[gi].copy(), pbest_cost[gi]
    for _ in range(n_iter):
        r1 = rng.uniform(size=(n_particles, 1))
        r2 = rng.uniform(size=(n_particles, 1))
        vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
        pos = np.clip(pos + vel, lo, hi)
        costs = np.array([tvd_cost(p, x1, x2) for p in pos])
        better = costs < pbest_cost
        pbest[better] = pos[better]
        pbest_cost[better] = costs[better]
        gi = int(np.argmin(pbest_cost))
        if pbest_cost[gi] < gbest_cost:
            gbest_cost = pbest_cost[gi]
            gbest = pbest[gi].copy()
    return gbest


def d_to_cv_end(d):
    """Estimated CV at the end of the window from the fitted linear delay,
    for a single summary error metric comparable to the constant-delay
    work. (Name kept general: this is the fitted polynomial's implied CV at
    n=N-1, compared against whichever ground truth's own CV at n=N-1 is
    passed as `cv_ref` in run_cell -- exact for the baseline/muap variants'
    linear ground truth, and a well-defined single-point comparison for the
    sinusoidal variant's non-polynomial ground truth.)"""
    theta_end = np.polyval(d[::-1], N - 1)
    if abs(theta_end) < 1e-9:
        return np.nan
    return CFG.ied_m / (abs(theta_end) / CFG.fs)


def crlb_linear_fim(snr_db, cv_true=CV0, window_len=None):
    """FIM for [d0, d1] under the local-constant-Fisher-information
    approximation: FIM = j_density * [[W, S1],[S1, S2]], phi=[1, n], for a
    window of `window_len` samples (default: the full N-sample window, for
    the single-global-fit variants). Pass window_len=SEG_LEN to get the
    CRLB actually applicable to the sliding-window estimator, which only
    ever sees SEG_LEN samples -- not N -- when fitting any one segment;
    using the N-sample bound there would understate its achievable variance
    and is not the estimator actually benchmarked.

    j_density = J_total(window_len) / window_len, NOT J_total itself: the
    scalar CRLB machinery (crlb_theta) returns the TOTAL Fisher information
    already aggregated over the whole `window_len`-sample window for a
    SINGLE constant-delay parameter (order-0). Re-using J_total directly as
    if it were a per-sample density before summing Sum_n phi_k(n)phi_l(n)
    (itself an O(window_len) quantity) double-counts the window length.
    Sanity check: setting phi(n)=[1] only (order-0) must reproduce the
    plain scalar CRLB, Var(d0) = 1/FIM_00 = 1/(j_density * window_len) =
    1/(J_total/window_len * window_len) = 1/J_total = var_theta -- which
    only holds with the /window_len division included below."""
    if window_len is None:
        window_len = N
    var_theta = crlb_theta(cv_true, snr_db, n=window_len, fs=CFG.fs,
                            ied_m=CFG.ied_m, n_rows=2)  # 2-channel case
    if not np.isfinite(var_theta) or var_theta <= 0:
        return np.nan, np.nan
    j_density = 1.0 / var_theta / window_len
    n = np.arange(window_len)
    W = float(window_len)
    S1, S2 = float(np.sum(n)), float(np.sum(n ** 2))
    FIM = j_density * np.array([[W, S1], [S1, S2]])
    try:
        COV = np.linalg.inv(FIM)
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    return COV[0, 0], COV[1, 1]  # Var(d0), Var(d1)


def run_cell(snr_db, seed0, theta_fn, cv_ref, source_fixed=None):
    fast = {"BFGS": []}
    fast_t = {"BFGS": []}
    for r in range(REPS_FAST):
        rng = np.random.default_rng(seed0 + r)
        src = source_fixed if source_fixed is not None else shwedyk_semg(N, CFG.fs, rng)
        x1, x2 = make_pair(rng, snr_db, src, theta_fn)
        d0 = d0_init(x1, x2)
        t1 = time.perf_counter()
        d = refine_bfgs(x1, x2, d0)
        dt = (time.perf_counter() - t1) * 1e3
        fast["BFGS"].append(d_to_cv_end(d))
        fast_t["BFGS"].append(dt)

    slow = {"SA": [], "GA": [], "PSO": []}
    slow_t = {"SA": [], "GA": [], "PSO": []}
    for r in range(REPS_SLOW):
        rng = np.random.default_rng(seed0 + 1000 + r)
        src = source_fixed if source_fixed is not None else shwedyk_semg(N, CFG.fs, rng)
        x1, x2 = make_pair(rng, snr_db, src, theta_fn)
        d0 = d0_init(x1, x2)
        mc_rng = np.random.default_rng(seed0 + 2000 + r)
        for name, fn in (("SA", lambda *a: refine_sa(*a, mc_rng)),
                          ("GA", lambda *a: refine_ga(*a, mc_rng)),
                          ("PSO", lambda *a: refine_pso(*a, mc_rng))):
            t1 = time.perf_counter()
            d = fn(x1, x2, d0)
            dt = (time.perf_counter() - t1) * 1e3
            slow[name].append(d_to_cv_end(d))
            slow_t[name].append(dt)

    all_times = {**fast_t, **slow_t}
    out = {}
    for name, ests in {**fast, **slow}.items():
        ests = np.asarray(ests)
        ests = ests[np.isfinite(ests)]
        pct_err = 100.0 * (ests - cv_ref) / cv_ref
        timing = np.asarray(all_times[name])
        out[name] = dict(mean_abs_pct=np.mean(np.abs(pct_err)) if len(pct_err) else np.nan,
                          rmse_pct=np.sqrt(np.mean(pct_err ** 2)) if len(pct_err) else np.nan,
                          sd_pct=np.std(pct_err) if len(pct_err) else np.nan,
                          time_ms=np.mean(timing),
                          n_valid=len(pct_err))
    return out


def run_variant(variant, theta_fn, source_fixed, csv_name, pdf_name, plot_title,
                 err_label=r"Error variance in CV at window end [%$^2$]  (log scale)",
                 extra_crlb_curves=None):
    method_names = ["BFGS", "SA", "GA", "PSO"]
    cv_ref = CFG.ied_m * CFG.fs / theta_fn(N - 1)
    rows = []
    agg = {m: {snr: [] for snr in SNR_VALUES} for m in method_names}
    agg_time = {m: {snr: [] for snr in SNR_VALUES} for m in method_names}

    print(f"\n--- variant: {variant} (cv_ref at window end = {cv_ref:.4f} m/s) ---")
    for snr in SNR_VALUES:
        seed0 = int(snr * 137)
        res = run_cell(snr, seed0, theta_fn, cv_ref, source_fixed=source_fixed)
        for m in method_names:
            rows.append(dict(snr_db=snr, method=m, **res[m]))
            agg[m][snr].append(res[m]["rmse_pct"])
            agg_time[m][snr].append(res[m]["time_ms"])
        print(f"snr={snr:>4} done: " +
              ", ".join(f"{m}={res[m]['rmse_pct']:.3f}% "
                        f"({res[m]['n_valid']}/{REPS_FAST if m=='BFGS' else REPS_SLOW} valid)"
                        for m in method_names))

    import csv
    csv_path = REPO / "journal_edit" / csv_name
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["snr_db", "method", "mean_abs_pct",
                                           "rmse_pct", "sd_pct", "time_ms", "n_valid"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {csv_path}")

    print(f"\n=== TVD summary [{variant}] (RMSE %err at window end, true CV={cv_ref:.4f} m/s) ===")
    for m in method_names:
        vals = [np.mean(agg[m][snr]) for snr in SNR_VALUES]
        t_mean = np.mean([np.mean(agg_time[m][snr]) for snr in SNR_VALUES])
        print(f"{m:6s}: " + ", ".join(f"{v:.3f}%" for v in vals) + f"  |  time {t_mean:.2f} ms")

    print("\nCRLB (order-1 polynomial model) SD[d1] converted to %err in CV by SNR:")
    crlb_pct = {}
    for snr in SNR_VALUES:
        var_d0, var_d1 = crlb_linear_fim(snr, cv_true=CV0)
        theta_end_true = theta_fn(N - 1)
        sd_theta_end = np.sqrt(var_d0 + (N - 1) ** 2 * var_d1) if np.isfinite(var_d1) else np.nan
        pct = 100.0 * sd_theta_end / theta_end_true if np.isfinite(sd_theta_end) else np.nan
        crlb_pct[snr] = pct
        print(f"  SNR={snr:>4} dB: {pct:.4f} %")

    if extra_crlb_curves:
        for label, curve in extra_crlb_curves:
            print(f"\nCRLB [{label}] by SNR:")
            for snr in SNR_VALUES:
                print(f"  SNR={snr:>4} dB: {curve[snr]:.4f} %")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    COLORS = {"BFGS": "#0B2A4A", "SA": "#E08A1E", "GA": "#B23A48", "PSO": "#6A4C93"}
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    # The CRLB bounds variance: pair it with the empirical error variance
    # (sd_pct^2), both on the %CV^2 scale, not with RMSE.
    for m in method_names:
        ys = [next(r["sd_pct"] for r in rows if r["method"] == m and r["snr_db"] == snr) ** 2
              for snr in SNR_VALUES]
        ax.plot(SNR_VALUES, ys, "o-", color=COLORS[m], label=m, lw=1.8, ms=5)
    ax.plot(SNR_VALUES, [crlb_pct[s] ** 2 for s in SNR_VALUES], "k--",
            label="CRLB, order-1 polynomial model", lw=1.6)
    if extra_crlb_curves:
        EXTRA_STYLES = ["-.", ":"]
        EXTRA_COLORS = ["#8E44AD", "#028090"]
        for i, (label, curve) in enumerate(extra_crlb_curves):
            ax.plot(SNR_VALUES, [curve[s] ** 2 for s in SNR_VALUES],
                    EXTRA_STYLES[i % 2], color=EXTRA_COLORS[i % 2],
                    label=label, lw=1.6)
    ax.set_yscale("log")
    ax.set_xlabel("SNR [dB]")
    ax.set_ylabel(err_label)
    ax.set_title(plot_title, loc="left", fontsize=11)
    ax.grid(alpha=0.2, which="both")
    ax.legend(fontsize=8, framealpha=0.9)
    plt.tight_layout()
    fig_path = REPO / "journal_edit" / pdf_name
    plt.savefig(fig_path)
    print(f"Wrote {fig_path}")


def main():
    """Baseline variant: Shwedyk-PSD colored-noise source + linear CV ramp.
    Unchanged output filenames/numbers relative to the original
    (pre-multi-variant) version of this script."""
    run_variant(
        variant="baseline",
        theta_fn=true_theta,
        source_fixed=None,
        csv_name="tvd_benchmark_data.csv",
        pdf_name="tvd_vs_snr.pdf",
        plot_title="Time-varying delay: optimizer comparison",
    )


def main_sinusoidal():
    """Same colored-noise source as baseline, but the ground truth is the
    non-polynomial 'inverse sinusoidal TVD' of Eq. (25), Luu et al. (2018) --
    see true_theta_sinusoidal. Also computes the CRLB under the CORRECTLY
    specified sinusoidal model (K,B,C estimated directly, see
    crlb_sinusoidal_theta_pct) alongside the misspecified order-1
    polynomial CRLB, to separate 'how much of the gap is avoidable bias
    from fitting the wrong functional form' from 'how much is the
    irreducible noise floor even with the right form'."""
    sinusoidal_crlb_pct = {
        snr: crlb_sinusoidal_theta_pct(snr, window_len=N, n_eval=N - 1, cv_op=CV0)
        for snr in SNR_VALUES
    }
    run_variant(
        variant="sinusoidal (Eq. 25, Luu et al. 2018)",
        theta_fn=true_theta_sinusoidal,
        source_fixed=None,
        csv_name="tvd_benchmark_sinusoidal_data.csv",
        pdf_name="tvd_vs_snr_sinusoidal.pdf",
        plot_title="TVD, non-polynomial ground truth (Eq. 25, J25)",
        extra_crlb_curves=[("CRLB, correctly-specified sinusoidal model", sinusoidal_crlb_pct)],
    )


SEG_LEN = 100      # samples per local segment -- matches the real-data
                   # convention already used for Newton/MLE resolution on
                   # dynamic-contraction bursts (PAPERV7.tex, "Resolution by
                   # Newton Raphson method": "low-order (1st/2nd-degree)
                   # local polynomial fits ... either on successive disjoint
                   # segments or on 50%-overlapping sliding windows")
SEG_OVERLAP = 0.5  # 50% overlap, PAPERV7's stated alternative to
                   # back-to-back segments


def _segment_starts():
    step = int(SEG_LEN * (1 - SEG_OVERLAP))
    return list(range(0, N - SEG_LEN + 1, step))


def tvd_cost_local(d, x1_seg, x2_seg):
    """Same cost as tvd_cost, but over one LOCAL segment of length SEG_LEN
    with its own local time axis n=0..SEG_LEN-1: each segment gets its own
    low-order (here order-1) polynomial, instead of a single polynomial
    spanning the whole N=600-sample window."""
    n = np.arange(SEG_LEN)
    theta = np.polyval(d[::-1], n)
    spline = CubicSpline(n, x1_seg, extrapolate=True)
    x1_warped = spline(n - theta)
    deb = int(round(0.05 * SEG_LEN))
    fin = int(round(0.95 * SEG_LEN))
    return float(np.sum((x2_seg[deb:fin] - x1_warped[deb:fin]) ** 2))


def refine_bfgs_local(x1_seg, x2_seg, d0):
    res = minimize(tvd_cost_local, d0, args=(x1_seg, x2_seg), method="BFGS",
                    options={"maxiter": 100})
    return res.x


def sliding_window_cv_end(x1, x2):
    """Sliding-window MLE (BFGS) with a local order-1 fit per segment. By
    construction of _segment_starts(), the LAST segment's local index
    SEG_LEN-1 lands exactly on the global window's last sample (N-1), so its
    fit gives the same 'estimated CV at window end' quantity used by
    d_to_cv_end() for the single-global-fit variants -- a direct,
    apples-to-apples comparison of the two fitting strategies against the
    same non-polynomial ground truth."""
    from src.mfcv import _gcc_delay
    d_last = None
    for start in _segment_starts():
        x1_seg = x1[start:start + SEG_LEN]
        x2_seg = x2[start:start + SEG_LEN]
        delay0, _ = _gcc_delay(x1_seg, x2_seg, max_lag=40)
        d0 = np.array([delay0, 0.0])
        d_last = refine_bfgs_local(x1_seg, x2_seg, d0)
    theta_end = np.polyval(d_last[::-1], SEG_LEN - 1)
    if abs(theta_end) < 1e-9:
        return np.nan
    return CFG.ied_m / (abs(theta_end) / CFG.fs)


def sliding_window_constmle_cv_end(x1, x2):
    """Sliding-window LOCAL CONSTANT-DELAY estimator via the frequency-
    domain MLE: per SEG_LEN-sample segment, a single CONSTANT delay
    (order-0 -- no local linear slope) is estimated by reusing the exact
    same ideal-phase-shift MLE already used for the scalar constant-delay
    case (crlb_optimizer_benchmark.py's coarse_grid + refine_brent). NOTE:
    despite an ideal allpass filter sharing the same e^{-j*omega*theta}
    phase-shift model as this MLE cost, this is NOT the published "local
    all-pass filter" (LAP) method of Gilliam et al. (ICASSP 2018,
    https://ieeexplore.ieee.org/document/8461390/) -- see
    sliding_window_lap_cv_end() for a faithful implementation of that
    method. This function is kept as a second, simpler order-0 baseline:
    same brute-force grid+Brent search as the scalar case, just applied
    locally instead of once over the whole window."""
    theta_end = None
    for start in _segment_starts():
        x1_seg = x1[start:start + SEG_LEN]
        x2_seg = x2[start:start + SEG_LEN]
        col = np.stack([x1_seg, x2_seg], axis=0)
        X, freqs, t0, step = coarse_grid(col)
        theta_end = refine_brent(X, freqs, t0, step)
    if theta_end is None or abs(theta_end) < 1e-9:
        return np.nan
    return CFG.ied_m / (abs(theta_end) / CFG.fs)


LAP_R = 8  # filter half-support (samples) -- upper bound on the delay
           # magnitude the LAP filter can represent; our delays (~2.7-5.3
           # samples for CV in [3,6] m/s at this Fs/IED) sit comfortably
           # inside it, so the single-scale filter below is sufficient and
           # the multi-scale refinement of Gilliam et al. Section 3.2 is
           # not needed here.


def lap_theta_end(x1, x2, R=LAP_R, W=SEG_LEN):
    """Faithful (single-scale, single-pair) Local All-Pass (LAP) delay
    estimator: Gilliam, Bingham, Blu & Jelfs, "Time-Varying Delay
    Estimation Using Common Local All-Pass Filters with Application to
    Surface Electromyography", ICASSP 2018
    (https://ieeexplore.ieee.org/document/8461390/), Eq. (4)-(8).

    A constant delay is equivalent to all-pass filtering (Concept 1:
    x2 = h*x1 with H(w)=e^{-jwt}); any all-pass filter's frequency response
    can be written H(w)=P(e^jw)/P(e^-jw) for a real FIR filter p, so
    estimating h reduces to estimating p (Concept 2, Eq. 4); p itself is
    then approximated as p0 + c1*p1, where p0 is a Gaussian and p1 = k*p0
    its first-derivative-shaped counterpart (Concept 3, Eq. 5-6), leaving
    exactly ONE free parameter c1, found by ordinary linear least squares
    over the fitting window W (last W samples of the segment here) -- no
    iterative optimization at all, unlike every other estimator in this
    paper. The delay is then read off the fitted filter's impulse response
    via Eq. (8).

    This is a genuinely different local model from both
    sliding_window_cv_end() (order-1 polynomial + cubic-spline warp) and
    sliding_window_constmle_cv_end() (order-0 grid+Brent search): it
    represents the delay as a 2-coefficient Gaussian-derivative filter
    rather than a warp of the raw samples, and solves for it in closed
    form rather than by any kind of search."""
    k = np.arange(-R, R + 1)
    sigma = R / 2.0 - 0.2
    p0 = np.exp(-(k.astype(float) ** 2) / (2.0 * sigma ** 2))
    p1 = k * p0
    x1w = np.asarray(x1[-W:], dtype=float)
    x2w = np.asarray(x2[-W:], dtype=float)
    if len(x1w) <= 2 * R:
        return np.nan
    c_p0_x1 = np.convolve(x1w, p0, mode="valid")
    c_p0_x2 = np.convolve(x2w, p0, mode="valid")
    c_p1_x1 = np.convolve(x1w, p1, mode="valid")
    c_p1_x2 = np.convolve(x2w, p1, mode="valid")
    A = c_p0_x1 - c_p0_x2
    B = c_p1_x1 + c_p1_x2
    denom = float(np.sum(B * B))
    if denom < 1e-12:
        return np.nan
    c1 = -float(np.sum(A * B)) / denom
    papp = p0 + c1 * p1
    denom2 = float(np.sum(papp))
    if abs(denom2) < 1e-9:
        return np.nan
    return 2.0 * float(np.sum(k * papp)) / denom2


def sliding_window_lap_cv_end(x1, x2):
    """LAP delay estimate for the LAST SEG_LEN-sample segment only (the
    segment covering the global window end), for the same 'estimated CV at
    window end' comparison used by the other local estimators."""
    theta_end = lap_theta_end(x1, x2, R=LAP_R, W=SEG_LEN)
    if not np.isfinite(theta_end) or abs(theta_end) < 1e-9:
        return np.nan
    return CFG.ied_m / (abs(theta_end) / CFG.fs)


METHOD_LABELS = {
    "poly": "MLE-sliding-poly(BFGS)",
    "constmle": "MLE-sliding-constdelay(Brent)",
    "lap": "LAP(Gilliam2018)",
    "gcc": "GCC(Knapp1976)",
}


def main_sinusoidal_sliding():
    """Sliding-window fits applied to the SAME non-polynomial Eq. (25)
    ground truth as main_sinusoidal(), to test whether a local-fit strategy
    reduces the large, SNR-independent bias found there with a single
    global order-1 fit over the whole window -- and to compare THREE
    different LOCAL models against each other: an order-1 local polynomial
    warped by cubic-spline interpolation (sliding_window_cv_end, matching
    PAPERV7.tex's real Newton/MLE convention); an order-0 local constant
    delay found by the same brute-force grid+Brent search as the scalar
    case (sliding_window_constmle_cv_end); and the published Local All-Pass
    (LAP) filter method of Gilliam et al., ICASSP 2018
    (sliding_window_lap_cv_end) -- a 2-coefficient Gaussian-derivative
    all-pass filter fit by closed-form linear least squares, no search at
    all. All three run on the SAME per-realization (x1, x2) pair for a
    paired comparison."""
    theta_fn = true_theta_sinusoidal
    cv_ref = CFG.ied_m * CFG.fs / theta_fn(N - 1)
    rows = []
    methods = ("poly", "constmle", "lap", "gcc")
    agg = {m: {} for m in methods}
    agg_var = {m: {} for m in methods}
    agg_time = {m: {} for m in methods}

    print(f"\n--- variant: sinusoidal, sliding-window fits "
          f"(SEG_LEN={SEG_LEN}, overlap={SEG_OVERLAP:.0%}, cv_ref={cv_ref:.4f} m/s) ---")
    for snr in SNR_VALUES:
        seed0 = int(snr * 137)
        ests = {m: [] for m in methods}
        times = {m: [] for m in methods}
        for r in range(REPS_FAST):
            rng = np.random.default_rng(seed0 + r)
            src = shwedyk_semg(N, CFG.fs, rng)
            x1, x2 = make_pair(rng, snr, src, theta_fn)

            t1 = time.perf_counter()
            ests["poly"].append(sliding_window_cv_end(x1, x2))
            times["poly"].append((time.perf_counter() - t1) * 1e3)

            t1 = time.perf_counter()
            ests["constmle"].append(sliding_window_constmle_cv_end(x1, x2))
            times["constmle"].append((time.perf_counter() - t1) * 1e3)

            t1 = time.perf_counter()
            ests["lap"].append(sliding_window_lap_cv_end(x1, x2))
            times["lap"].append((time.perf_counter() - t1) * 1e3)

            t1 = time.perf_counter()
            ests["gcc"].append(gcc_cv_end_mc(np.stack([x1, x2], axis=0)))
            times["gcc"].append((time.perf_counter() - t1) * 1e3)

        for method in methods:
            e = np.asarray(ests[method])
            e = e[np.isfinite(e)]
            pct_err = 100.0 * (e - cv_ref) / cv_ref
            mean_abs = np.mean(np.abs(pct_err)) if len(pct_err) else np.nan
            rmse = np.sqrt(np.mean(pct_err ** 2)) if len(pct_err) else np.nan
            sd = np.std(pct_err) if len(pct_err) else np.nan
            t_mean = np.mean(times[method])
            agg[method][snr] = rmse
            agg_var[method][snr] = sd ** 2
            agg_time[method][snr] = t_mean
            rows.append(dict(snr_db=snr, method=METHOD_LABELS[method],
                              mean_abs_pct=mean_abs, rmse_pct=rmse, sd_pct=sd,
                              time_ms=t_mean, n_valid=len(pct_err)))
        print(f"snr={snr:>4} done: " +
              "   ".join(f"{m}={agg[m][snr]:.3f}% ({agg_time[m][snr]:.2f} ms)"
                         for m in methods))

    # For direct comparison, also recompute the single-global-fit BFGS curve
    # against the SAME ground truth (already reported by main_sinusoidal(),
    # recomputed here so this script's output is self-contained).
    global_bfgs = {}
    global_bfgs_var = {}
    for snr in SNR_VALUES:
        seed0 = int(snr * 137)
        res = run_cell(snr, seed0, theta_fn, cv_ref, source_fixed=None)
        global_bfgs[snr] = res["BFGS"]["rmse_pct"]
        global_bfgs_var[snr] = res["BFGS"]["sd_pct"] ** 2

    import csv
    csv_path = REPO / "journal_edit" / "tvd_benchmark_sinusoidal_sliding_data.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["snr_db", "method", "mean_abs_pct",
                                           "rmse_pct", "sd_pct", "time_ms", "n_valid"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {csv_path}")

    print("\n=== sinusoidal ground truth: global fit vs. four local strategies (RMSE) ===")
    for snr in SNR_VALUES:
        print(f"  SNR={snr:>4} dB: global-fit BFGS={global_bfgs[snr]:.3f}%   "
              f"local-poly(order-1)={agg['poly'][snr]:.3f}%   "
              f"local-const(order-0,MLE)={agg['constmle'][snr]:.3f}%   "
              f"LAP(Gilliam2018)={agg['lap'][snr]:.3f}%   "
              f"GCC(Knapp1976)={agg['gcc'][snr]:.3f}%")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Different CRLBs for different parameterizations, not one: the global
    # fit had N=600 samples and 2 free parameters (d0, d1); the local
    # polynomial fit has only SEG_LEN=100 samples but still 2 free
    # parameters per segment; the local constant-delay (MLE) and LAP fits
    # both estimate, in the end, a SINGLE delay value from a SEG_LEN=100-
    # sample window -- the same statistical problem regardless of whether
    # it is solved by a grid search or by LAP's 2-coefficient closed-form
    # filter fit -- so both share the plain scalar crlb_theta() bound, with
    # no linear-basis propagation needed. Using the wrong one of these for
    # any curve would compare it against an estimator it is not.
    theta_end_true = theta_fn(N - 1)

    def _crlb_pct_poly(window_len, local_end):
        out = {}
        for snr in SNR_VALUES:
            var_d0, var_d1 = crlb_linear_fim(snr, cv_true=CV0, window_len=window_len)
            sd_theta_end = (np.sqrt(var_d0 + local_end ** 2 * var_d1)
                             if np.isfinite(var_d1) else np.nan)
            out[snr] = 100.0 * sd_theta_end / theta_end_true if np.isfinite(sd_theta_end) else np.nan
        return out

    def _crlb_pct_const(window_len):
        out = {}
        for snr in SNR_VALUES:
            var_theta = crlb_theta(CV0, snr, n=window_len, fs=CFG.fs,
                                    ied_m=CFG.ied_m, n_rows=2)
            out[snr] = (100.0 * np.sqrt(var_theta) / theta_end_true
                        if np.isfinite(var_theta) and var_theta > 0 else np.nan)
        return out

    crlb_global_pct = _crlb_pct_poly(N, N - 1)
    crlb_sliding_poly_pct = _crlb_pct_poly(SEG_LEN, SEG_LEN - 1)
    crlb_sliding_const_pct = _crlb_pct_const(SEG_LEN)

    print(f"\nCRLB, global order-1/{N}-smp vs. local order-1/{SEG_LEN}-smp "
          f"vs. local order-0 (const-delay MLE and LAP share this bound)/{SEG_LEN}-smp:")
    for snr in SNR_VALUES:
        print(f"  SNR={snr:>4} dB: global={crlb_global_pct[snr]:.4f} %   "
              f"local-poly={crlb_sliding_poly_pct[snr]:.4f} %   "
              f"local-order0={crlb_sliding_const_pct[snr]:.4f} %")

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(SNR_VALUES, [global_bfgs_var[s] for s in SNR_VALUES], "o-",
            color="#B23A48", label="Single global order-1 fit (BFGS)", lw=1.8, ms=5)
    ax.plot(SNR_VALUES, [agg_var["poly"][s] for s in SNR_VALUES], "o-",
            color="#0B2A4A", label=f"Local poly, order-1 ({SEG_LEN}-smp, BFGS)", lw=1.8, ms=5)
    ax.plot(SNR_VALUES, [agg_var["constmle"][s] for s in SNR_VALUES], "o-",
            color="#3C9D45", label=f"Local const-delay, order-0 ({SEG_LEN}-smp, grid+Brent)", lw=1.8, ms=5)
    ax.plot(SNR_VALUES, [agg_var["lap"][s] for s in SNR_VALUES], "o-",
            color="#8E44AD", label=f"Local All-Pass, Gilliam et al. 2018 ({SEG_LEN}-smp)", lw=1.8, ms=5)
    ax.plot(SNR_VALUES, [agg_var["gcc"][s] for s in SNR_VALUES], "o-",
            color="#E08A1E", label=f"Local GCC (Knapp and Carter 1976, {SEG_LEN}-smp)", lw=1.8, ms=5)
    ax.plot(SNR_VALUES, [crlb_global_pct[s] ** 2 for s in SNR_VALUES], "--",
            color="#B23A48", label=f"CRLB, {N}-sample window", lw=1.2)
    ax.plot(SNR_VALUES, [crlb_sliding_poly_pct[s] ** 2 for s in SNR_VALUES], "--",
            color="#0B2A4A", label=f"CRLB, {SEG_LEN}-sample, order-1", lw=1.2)
    ax.plot(SNR_VALUES, [crlb_sliding_const_pct[s] ** 2 for s in SNR_VALUES], "--",
            color="#3C9D45", label=f"CRLB, {SEG_LEN}-sample, order-0", lw=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("SNR [dB]")
    ax.set_ylabel(r"Error variance in CV at window end [%$^2$]  (log scale)")
    ax.set_title("Non-polynomial ground truth: global vs. three local strategies",
                  loc="left", fontsize=10)
    ax.grid(alpha=0.2, which="both")
    ax.legend(fontsize=7, framealpha=0.9)
    plt.tight_layout()
    fig_path = REPO / "journal_edit" / "tvd_vs_snr_sinusoidal_sliding.pdf"
    plt.savefig(fig_path)
    print(f"Wrote {fig_path}")


def main_variance_vs_time(theta_fn=None, source_fixed=None, variant_label="baseline",
                            pdf_name="tvd_variance_vs_time.pdf",
                            csv_name="tvd_variance_vs_time_data.csv"):
    """Variance of the estimated TVD as a function of TIME within the
    window, compared against the analytic CRLB(theta(n)) at each n, for
    several SNR levels -- the same style of plot as Fig. 2 in Luu et al.
    (2018, J25): 'The variance of the polynomial delay (blue) ... compared
    to the CRLB (red) as a function of time for different noise levels.'
    Computed for ALL FOUR optimizers (BFGS, SA, GA, PSO), not just BFGS, so
    the same 'how close to CRLB' question already asked at the window end
    (Table 2/Figure 3) can be asked across the whole window for every
    optimizer, not only the most accurate one.

    Unlike the rest of this script, which only reports a single 'estimated
    CV at window end' summary, this computes the FULL fitted delay
    trajectory theta_hat(n) = polyval(d, n) for n=0..N-1 on every Monte
    Carlo realization, then takes the variance ACROSS realizations at each
    n separately -- this is the quantity crlb_linear_fim's phi(n)^T FIM^-1
    phi(n) bound (Eq. 7) actually applies to, unlike the mean-ABSOLUTE-error
    summary used elsewhere in this script (see the paper's Section 5.1 for
    why MAE and the CRLB-bounded SD are related but different statistics)."""
    if theta_fn is None:
        theta_fn = true_theta
    n_axis = np.arange(N)
    reps = 100  # more than REPS_FAST elsewhere, to reduce Monte Carlo noise
               # in the variance estimate itself; only ONE fit per
               # realization is needed (theta_hat(n) is then read off the
               # SAME fitted d via polyval), so this costs no more than the
               # window-end benchmark already run for each optimizer

    method_fns = {
        "BFGS": lambda x1, x2, d0, rng: refine_bfgs(x1, x2, d0),
        "SA": lambda x1, x2, d0, rng: refine_sa(x1, x2, d0, rng),
        "GA": lambda x1, x2, d0, rng: refine_ga(x1, x2, d0, rng),
        "PSO": lambda x1, x2, d0, rng: refine_pso(x1, x2, d0, rng),
    }
    COLORS = {"BFGS": "#0B2A4A", "SA": "#E08A1E", "GA": "#B23A48", "PSO": "#6A4C93"}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.5), sharex=True)
    axes = axes.ravel()

    for ax, snr in zip(axes, SNR_VALUES):
        seed0 = int(snr * 137)
        var_crlb_exact = crlb_tvd_theta_curve(snr, cv_true=CV0, window_len=N, n_axis=n_axis)
        db_crlb = 10 * np.log10(np.maximum(var_crlb_exact, 1e-300))

        for method, fn in method_fns.items():
            theta_hat = np.empty((reps, N))
            for r in range(reps):
                rng = np.random.default_rng(seed0 + r)
                mc_rng = np.random.default_rng(seed0 + 5000 + r)
                src = source_fixed if source_fixed is not None else shwedyk_semg(N, CFG.fs, rng)
                x1, x2 = make_pair(rng, snr, src, theta_fn)
                d0 = d0_init(x1, x2)
                d = fn(x1, x2, d0, mc_rng)
                theta_hat[r, :] = np.polyval(d[::-1], n_axis)

            var_empirical = np.var(theta_hat, axis=0)  # samples^2, per n
            for n in n_axis:
                rows.append(dict(snr_db=snr, method=method, n=int(n),
                                  var_empirical_samples2=float(var_empirical[n]),
                                  var_crlb_samples2=float(var_crlb_exact[n])))

            db_emp = 10 * np.log10(np.maximum(var_empirical, 1e-300))
            ax.plot(n_axis, db_emp, color=COLORS[method], lw=1.4, label=method)

        ax.plot(n_axis, db_crlb, "k--", lw=1.6, label="CRLB")
        ax.set_title(f"SNR={snr:.0f} dB", loc="left", fontsize=10)
        ax.grid(alpha=0.2)

    for ax in axes[2:]:
        ax.set_xlabel("Time (samples)")
    for ax in axes[::2]:
        ax.set_ylabel("Variance of TVD (dB)")
    axes[0].legend(fontsize=7, framealpha=0.9, loc="best", ncol=2)
    fig.suptitle(f"Variance of the order-1 polynomial delay vs. CRLB, over time, "
                 f"all four optimizers ({variant_label})", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    import csv
    csv_path = REPO / "journal_edit" / csv_name
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["snr_db", "method", "n",
                                           "var_empirical_samples2", "var_crlb_samples2"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {csv_path}")

    fig_path = REPO / "journal_edit" / pdf_name
    plt.savefig(fig_path)
    print(f"Wrote {fig_path}")


def crlb_tvd_theta_curve(snr_db, cv_true, window_len, n_axis):
    """Exact phi(n)^T FIM^-1 phi(n) (Eq. 7) at every n in n_axis, including
    the off-diagonal Cov(d0,d1) term that crlb_linear_fim's two returned
    diagonal variances alone omit -- needed for a full variance-vs-time
    curve (crlb_theta_end-style code elsewhere only ever evaluates this at
    n=W-1, where the off-diagonal contribution happens to combine with the
    diagonal ones in the sqrt(var_d0 + n^2*var_d1) shortcut used there;
    that shortcut is NOT valid at every n and is not used here)."""
    var_theta = crlb_theta(cv_true, snr_db, n=window_len, fs=CFG.fs,
                            ied_m=CFG.ied_m, n_rows=2)
    if not np.isfinite(var_theta) or var_theta <= 0:
        return np.full(len(n_axis), np.nan)
    j_density = 1.0 / var_theta / window_len
    n_full = np.arange(window_len)
    S0, S1, S2 = float(window_len), float(np.sum(n_full)), float(np.sum(n_full ** 2))
    FIM = j_density * np.array([[S0, S1], [S1, S2]])
    try:
        COV = np.linalg.inv(FIM)
    except np.linalg.LinAlgError:
        return np.full(len(n_axis), np.nan)
    out = np.empty(len(n_axis))
    for i, n in enumerate(n_axis):
        phi = np.array([1.0, float(n)])
        out[i] = float(phi @ COV @ phi)
    return out


def main_muap():
    """Same linear CV ramp as baseline, but the source waveform is a real
    Farina-Merletti MUAP template (see load_muap_source) instead of colored
    noise."""
    source = load_muap_source()
    run_variant(
        variant="muap (Farina-Merletti template)",
        theta_fn=true_theta,
        source_fixed=source,
        csv_name="tvd_benchmark_muap_data.csv",
        pdf_name="tvd_vs_snr_muap.pdf",
        plot_title="TVD on a real MUAP source (Farina-Merletti): optimizer comparison",
    )


# =============================================================================
# Multi-sensor (K-channel) extension of the local TVD estimators, plus a
# tilted-array (fiber-misalignment) signal-generation option. All three local
# estimators (poly, const-delay, LAP->CLAP) generalize from the 2-channel
# case above by taking a (K, samples) column instead of a (x1, x2) pair;
# tilt only changes how that column is GENERATED (make_multichannel_pair),
# so no estimator code needs to know about it.
# =============================================================================

def make_multichannel_pair(rng, snr_db, source, theta_fn, n_rows=None, tilt_deg=0.0):
    """K-channel column x_k(n) = warp(source, k*theta(n)*cos(tilt)) + w_k(n),
    k=0..K-1 -- the direct K-channel generalization of make_pair()'s 2-channel
    model, matching x_k(n) = s(n - k*theta) + w_k(n) already used for the
    scalar (constant-delay) case throughout this paper (Eq. 1).

    tilt_deg models the electrode array being rotated by that angle away
    from the true muscle-fiber direction: the electrodes are still spaced
    IED apart PHYSICALLY, but only the component of that spacing along the
    fiber direction, IED*cos(tilt_deg), actually contributes to the
    propagation delay -- so the true delay per physical channel step
    shrinks by cos(tilt_deg), which is exactly equivalent to the estimator
    (which knows nothing about the tilt and assumes delay-per-channel =
    theta(n)) reading an apparent CV of CV_true/cos(tilt_deg): a systematic
    OVERESTIMATION of CV, the well-known real-world misalignment bias
    (Farina & Merletti); tilt_deg=0 reproduces make_pair() exactly for
    n_rows=2."""
    if n_rows is None:
        n_rows = CFG.n_rows
    n = np.arange(N)
    th = theta_fn(n) * np.cos(np.deg2rad(tilt_deg))
    s = source
    spline = CubicSpline(n, s, extrapolate=True)
    p_sig = np.mean(s ** 2)
    p_noise = p_sig / (10 ** (snr_db / 10))
    cols = np.empty((n_rows, N))
    for k in range(n_rows):
        cols[k] = spline(n - k * th) + rng.standard_normal(N) * np.sqrt(p_noise)
    return cols


def tvd_cost_mc(d, cols):
    """Multichannel generalization of tvd_cost(): each ADJACENT pair
    (k, k+1) is warped by the SAME shared theta(n;d) and its residual
    summed over the middle 90% of the window and over all K-1 pairs --
    matching CLAP's own adjacent-pair pooling strategy (see clap_theta_end)
    rather than warping a single reference channel by k*theta(n;d) for
    every k. The reference-channel version was tried first and is not used:
    for K=13 it needs warps as large as 12*theta(n) (up to ~60 samples),
    forcing the cubic spline for that comparison well outside the 600-
    sample window it was built from into unreliable extrapolation, and
    empirically that miscalibrates the fit badly (see PAPER_Optimizer_CRLB
    session notes). Adjacent-pair warps stay of order theta(n) (a few
    samples) regardless of K, avoiding the issue entirely."""
    K, Nsamp = cols.shape
    n = np.arange(Nsamp)
    theta = np.polyval(d[::-1], n)
    deb = int(round(0.05 * Nsamp))
    fin = int(round(0.95 * Nsamp))
    cost = 0.0
    for k in range(K - 1):
        spline = CubicSpline(n, cols[k], extrapolate=True)
        warped = spline(n - theta)
        cost += np.sum((cols[k + 1, deb:fin] - warped[deb:fin]) ** 2)
    return float(cost)


def refine_bfgs_mc(cols, d0):
    res = minimize(tvd_cost_mc, d0, args=(cols,), method="BFGS",
                    options={"maxiter": 100})
    return res.x


def sliding_window_cv_end_mc(cols):
    """Multichannel local order-1 polynomial fit (tvd_cost_mc + BFGS) per
    SEG_LEN segment, sliding across the full K-channel column; same
    'window-end' convention as sliding_window_cv_end()."""
    from src.mfcv import _gcc_delay
    d_last = None
    for start in _segment_starts():
        seg = cols[:, start:start + SEG_LEN]
        delay0, _ = _gcc_delay(seg[0], seg[1], max_lag=40)
        d0 = np.array([delay0, 0.0])
        d_last = refine_bfgs_mc(seg, d0)
    theta_end = np.polyval(d_last[::-1], SEG_LEN - 1)
    if abs(theta_end) < 1e-9:
        return np.nan
    return CFG.ied_m / (abs(theta_end) / CFG.fs)


def sliding_window_constmle_cv_end_mc(cols):
    """Multichannel local order-0 constant-delay fit: reuses coarse_grid +
    refine_brent UNCHANGED (they already operate on a (K, samples) column
    for the scalar case), just applied per SEG_LEN segment instead of once
    over the whole window."""
    theta_end = None
    for start in _segment_starts():
        seg = cols[:, start:start + SEG_LEN]
        X, freqs, t0, step = coarse_grid(seg)
        theta_end = refine_brent(X, freqs, t0, step)
    if theta_end is None or abs(theta_end) < 1e-9:
        return np.nan
    return CFG.ied_m / (abs(theta_end) / CFG.fs)


def gcc_cv_end_mc(cols):
    """Classic generalized cross-correlation (plain CC, Knapp & Carter 1976)
    local order-0 estimator: per SEG_LEN segment, average the per-adjacent-
    -channel-pair cross-correlation delay (src.mfcv._gcc_delay, the same
    routine already used elsewhere in this codebase only as an MLE
    initializer) over all K-1 pairs, matching CLAP's adjacent-pair pooling.
    Not a new estimator -- the textbook baseline every other local
    estimator in this paper is implicitly compared against -- but never
    previously benchmarked here in its own right for the multichannel,
    sliding-window case."""
    from src.mfcv import _gcc_delay
    theta_end = None
    for start in _segment_starts():
        seg = cols[:, start:start + SEG_LEN]
        K = seg.shape[0]
        delays = [_gcc_delay(seg[k], seg[k + 1], max_lag=40)[0] for k in range(K - 1)]
        theta_end = float(np.mean(delays))
    if theta_end is None or abs(theta_end) < 1e-9:
        return np.nan
    return CFG.ied_m / (abs(theta_end) / CFG.fs)


def clap_theta_end(cols, R=LAP_R, W=SEG_LEN):
    """Common Local All-Pass (CLAP) delay estimate: the actual multichannel
    extension proposed in Gilliam et al. (2018), Eq. (9) -- rather than
    fitting one all-pass filter per channel PAIR, a SINGLE shared filter
    (hence a single c1) is fit jointly across all K-1 adjacent-channel
    pairs, by summing each pair's (A, B) contributions (see lap_theta_end's
    derivation) before solving for c1. This is the CLAP extension that
    lap_theta_end's single-pair LAP explicitly omitted."""
    K, Nsamp = cols.shape
    k_filt = np.arange(-R, R + 1)
    sigma = R / 2.0 - 0.2
    p0 = np.exp(-(k_filt.astype(float) ** 2) / (2.0 * sigma ** 2))
    p1 = k_filt * p0
    xw = cols[:, -W:]
    if xw.shape[1] <= 2 * R:
        return np.nan
    A_sum = None
    B_sum = None
    for kk in range(K - 1):
        x_a = xw[kk].astype(float)
        x_b = xw[kk + 1].astype(float)
        c_p0_a = np.convolve(x_a, p0, mode="valid")
        c_p0_b = np.convolve(x_b, p0, mode="valid")
        c_p1_a = np.convolve(x_a, p1, mode="valid")
        c_p1_b = np.convolve(x_b, p1, mode="valid")
        A = c_p0_a - c_p0_b
        B = c_p1_a + c_p1_b
        A_sum = A if A_sum is None else A_sum + A
        B_sum = B if B_sum is None else B_sum + B
    denom = float(np.sum(B_sum * B_sum))
    if denom < 1e-12:
        return np.nan
    c1 = -float(np.sum(A_sum * B_sum)) / denom
    papp = p0 + c1 * p1
    denom2 = float(np.sum(papp))
    if abs(denom2) < 1e-9:
        return np.nan
    return 2.0 * float(np.sum(k_filt * papp)) / denom2


def clap_cv_end(cols):
    theta_end = clap_theta_end(cols, R=LAP_R, W=SEG_LEN)
    if not np.isfinite(theta_end) or abs(theta_end) < 1e-9:
        return np.nan
    return CFG.ied_m / (abs(theta_end) / CFG.fs)


def main_multichannel(tilt_degs=(0.0, 15.0), n_rows_list=(2, 13),
                        pdf_name="tvd_multichannel.pdf",
                        csv_name="tvd_multichannel_data.csv",
                        theta_fn=None, source_fixed=None, variant_label="baseline"):
    """Compares all three LOCAL sliding-window TVD estimators (poly,
    const-delay, CLAP) across (a) 2 vs. 13 (or however many) channels, on a
    configurable ground truth (default: the synthetic linear-ramp used
    throughout this script) and source (default: fresh Shwedyk-PSD colored
    noise per realization; pass source_fixed for a realistic waveform, e.g.
    one channel from the real motor-unit-population SEMG_simulator_v4
    simulator, reused across realizations the same way load_muap_source()
    is for the 2-channel case), and (b) an UNTILTED vs. a TILTED array
    (make_multichannel_pair's tilt_deg), modeling electrode misalignment
    relative to the muscle fiber direction. Uses BFGS for poly (matching
    sliding_window_cv_end's convention) and reports mean abs. %CV error
    and its SD at window end, by SNR."""
    if theta_fn is None:
        theta_fn = true_theta
    cv_ref = CFG.ied_m * CFG.fs / theta_fn(N - 1)
    reps = REPS_FAST
    methods = {
        "poly": sliding_window_cv_end_mc,
        "constmle": sliding_window_constmle_cv_end_mc,
        "clap": clap_cv_end,
        "gcc": gcc_cv_end_mc,
    }
    rows = []
    print(f"\n--- multichannel/tilt sweep [{variant_label}] "
          f"(cv_ref at window end = {cv_ref:.4f} m/s) ---")
    for n_rows in n_rows_list:
        for tilt in tilt_degs:
            for snr in SNR_VALUES:
                seed0 = int(snr * 137 + n_rows * 31 + tilt * 7)
                ests = {m: [] for m in methods}
                for r in range(reps):
                    rng = np.random.default_rng(seed0 + r)
                    src = source_fixed if source_fixed is not None else shwedyk_semg(N, CFG.fs, rng)
                    cols = make_multichannel_pair(rng, snr, src, theta_fn,
                                                   n_rows=n_rows, tilt_deg=tilt)
                    for m, fn in methods.items():
                        ests[m].append(fn(cols))
                for m in methods:
                    e = np.asarray(ests[m])
                    e = e[np.isfinite(e)]
                    pct_err = 100.0 * (e - cv_ref) / cv_ref
                    mean_abs = np.mean(np.abs(pct_err)) if len(pct_err) else np.nan
                    rmse = np.sqrt(np.mean(pct_err ** 2)) if len(pct_err) else np.nan
                    sd = np.std(pct_err) if len(pct_err) else np.nan
                    bias = np.mean(pct_err) if len(pct_err) else np.nan
                    rows.append(dict(n_rows=n_rows, tilt_deg=tilt, snr_db=snr, method=m,
                                      mean_abs_pct=mean_abs, rmse_pct=rmse, sd_pct=sd,
                                      bias_pct=bias, n_valid=len(pct_err)))
                print(f"K={n_rows:2d} tilt={tilt:4.1f} SNR={snr:4.1f}: " +
                      "  ".join(f"{m}={rows[-len(methods)+i]['rmse_pct']:.3f}%"
                                for i, m in enumerate(methods)))

    import csv
    csv_path = REPO / "journal_edit" / csv_name
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n_rows", "tilt_deg", "snr_db", "method",
                                           "mean_abs_pct", "rmse_pct", "sd_pct",
                                           "bias_pct", "n_valid"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {csv_path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    COLORS = {"poly": "#0B2A4A", "constmle": "#3C9D45", "clap": "#8E44AD", "gcc": "#E08A1E"}
    fig, axes = plt.subplots(len(tilt_degs), len(n_rows_list), figsize=(5.0 * len(n_rows_list), 4.0 * len(tilt_degs)),
                              squeeze=False)
    for i, tilt in enumerate(tilt_degs):
        for j, n_rows in enumerate(n_rows_list):
            ax = axes[i][j]
            sub = [r for r in rows if r["n_rows"] == n_rows and r["tilt_deg"] == tilt]
            for m in methods:
                ys = [r["rmse_pct"] for r in sub if r["method"] == m]
                ax.plot(SNR_VALUES, ys, "o-", color=COLORS[m], label=m, lw=1.6, ms=4)
            ax.set_yscale("log")
            ax.set_title(f"K={n_rows} channels, tilt={tilt:.0f}°", loc="left", fontsize=9)
            ax.grid(alpha=0.2, which="both")
            if i == len(tilt_degs) - 1:
                ax.set_xlabel("SNR [dB]")
            if j == 0:
                ax.set_ylabel("RMSE in CV [%]")
    axes[0][0].legend(fontsize=7, framealpha=0.9)
    fig.suptitle(f"Multichannel TVD estimators [{variant_label}]", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = REPO / "journal_edit" / pdf_name
    plt.savefig(fig_path)
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    variant_arg = "baseline"
    for a in sys.argv[1:]:
        if a.startswith("--variant"):
            if "=" in a:
                variant_arg = a.split("=", 1)[1]
            else:
                i = sys.argv.index(a)
                variant_arg = sys.argv[i + 1]
    variant_arg = variant_arg.strip().lower()

    if variant_arg == "baseline":
        main()
    elif variant_arg == "sinusoidal":
        main_sinusoidal()
    elif variant_arg == "muap":
        main_muap()
    elif variant_arg in ("sinusoidal_sliding", "sliding"):
        main_sinusoidal_sliding()
    elif variant_arg in ("variance", "variance_vs_time"):
        main_variance_vs_time()
    elif variant_arg in ("multichannel", "mc", "tilt"):
        main_multichannel()
    elif variant_arg == "all":
        main()
        main_sinusoidal()
        main_muap()
        main_sinusoidal_sliding()
        main_variance_vs_time()
    else:
        raise SystemExit(f"Unknown --variant '{variant_arg}' "
                          "(expected baseline, sinusoidal, muap, sliding, or all)")
