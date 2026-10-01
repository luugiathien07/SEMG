"""
GCC-pooled variant of real_data_validation.py.

The joint order-0 constant-delay MLE (estimate_delay_mle, K=2 vs K=11
double-differential channels) shows a strong real-data acceptance-rate gain
from pooling but a real-data temporal-precision gain that does NOT reproduce
under the current double-differential pipeline (see real_data_validation.py
and PAPER_Multichannel_TVD.tex, Section results-realdata).

PAPER_Multichannel_TVD.tex already defines a K-channel extension of GCC that
is benchmarked on synthetic data (Section sub:estimators, sub:results-pooling)
but was never run on real recordings: "the per-adjacent-pair cross-correlation
delay is computed independently for each of the K-1 pairs and averaged" --
as opposed to estimate_delay_mle's single joint fit over all K channels at
once. This script applies exactly that GCC-pooled estimator to the same real
recordings, same windows, same double-differential preprocessing, and the
same K=2-vs-K=11 comparison, to test whether pairwise-averaged pooling
recovers a real precision gain that joint pooling does not.

Per-pair delay comes from src.mfcv._gcc_delay (FFT cross-correlation +
parabolic sub-sample interpolation). A first version of this script reused
max_lag=40 samples from _gcc_delay's other call site (initializing the MLE,
where a wide search window is harmless since the MLE only uses it as a
starting point). Used here as the FINAL per-pair answer, max_lag=40 is far
wider than the physiologically plausible adjacent-channel delay range
(theta in [1.6,8] samples for CV in [2,10] m/s at Fs=2000Hz, IED=8mm): most
per-pair delays on the real double-differential data saturated at +-40
samples with inconsistent sign pair to pair, i.e. the peak search was
locking onto cross-correlation sidelobes/noise rather than the true delay.
max_lag is therefore restricted to 10 samples here (a small margin above
the gate's own 8-sample bound), matching the physiological range the same
way the MLE's own coarse grid already does.
"""
from __future__ import annotations

import sys
import glob
import csv
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.mfcv import (  # noqa: E402
    GridConfig, QualityGate, preprocess, serp_grid,
    single_differential, double_differential, _gcc_delay,
)

cfg = GridConfig()
gate = QualityGate()

WIN_S = 0.5
HOP_S = 0.25
FULL_COLS = [1, 2, 3, 4]
K_VALUES = {"K2": 2, "K11": 11}
MAX_LAG = 10  # physiological range: theta in [1.6, 8] samples for CV in [2,10] m/s

OUT_CSV = PROJECT_ROOT / "journal_edit" / "tvd_multichannel_realdata_gcc.csv"


def load_emg(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=";")
    return data[:, 1:65].T


def gcc_pooled_delay(x: np.ndarray) -> float:
    """Average of the K-1 adjacent-pair GCC delays (paper's GCC extension)."""
    K = x.shape[0]
    delays = [
        _gcc_delay(x[k], x[k + 1], max_lag=MAX_LAG)[0]
        for k in range(K - 1)
    ]
    return float(np.mean(delays))


def main() -> None:
    files = sorted(glob.glob(str(PROJECT_ROOT / "dataset" / "full_emg_data" / "Sujet_*" / "*_emg.csv")))
    rows = []
    for fpath in files:
        fpath = Path(fpath)
        name = fpath.stem
        parts = name.split("_")
        subject = f"{parts[0]}_{parts[1]}"
        condition = "_".join(parts[2:-1]) or "unk"

        x = load_emg(fpath)
        g = serp_grid(x)
        n_samples = g.shape[2]
        w = int(WIN_S * cfg.fs)
        h = int(HOP_S * cfg.fs)
        if n_samples < w:
            continue

        for col in FULL_COLS:
            col_sig = preprocess(g[:, col, :], cfg)
            sd_sig = single_differential(col_sig)
            dd_sig = double_differential(sd_sig)
            for start in range(0, n_samples - w + 1, h):
                seg = dd_sig[:, start:start + w]
                seg = seg - seg.mean(axis=1, keepdims=True)
                t_s = start / cfg.fs
                for k_label, K in K_VALUES.items():
                    theta = gcc_pooled_delay(seg[:K])
                    if abs(theta) < 1e-6:
                        continue
                    cv = cfg.ied_m / (abs(theta) / cfg.fs)
                    accepted = gate.cv_min <= cv <= gate.cv_max
                    rows.append(dict(
                        subject=subject, condition=condition, col=col, K=k_label,
                        t_s=round(t_s, 3), cv_ms=round(cv, 4), accepted=int(accepted),
                    ))
        print(f"done {fpath.name}: {n_samples/cfg.fs:.1f}s, {len(rows)} rows so far", file=sys.stderr)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["subject", "condition", "col", "K", "t_s", "cv_ms", "accepted"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
