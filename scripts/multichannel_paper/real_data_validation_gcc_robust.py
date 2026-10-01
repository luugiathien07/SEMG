"""
Robust-pooling variants of real_data_validation_gcc.py's GCC-pooled K-channel
extension, testing the fix proposed for its own real-data breakdown
(PAPER_Multichannel_TVD.tex, Section results-realdata: "averaging K-1
independent per-pair delay estimates has no defense against any single
noisy or weak-signal pair dragging the mean off").

Same windows, same double-differential preprocessing, same K=2-vs-K=11
comparison as real_data_validation_gcc.py, but THREE pooling rules computed
from the SAME per-pair (delay, peak-correlation) values in one pass:

  - mean:      the original rule (average of all K-1 per-pair delays).
  - median:    the sample median of the K-1 per-pair delays -- a single bad
               pair can no longer drag the estimate arbitrarily far, only
               shift which pair sits at the median.
  - threshold: the mean of only the per-pair delays whose peak correlation
               coefficient (already returned by _gcc_delay, unused for this
               purpose in the original script) falls in [LOW, HIGH] --
               excluding pairs whose GCC peak is too WEAK (LOW=0.1: the
               match is barely above the noise floor, so its delay is
               unreliable) as well as pairs whose peak is suspiciously too
               STRONG (HIGH=0.95: plausibly a near-duplicate/crosstalk pair
               or another degenerate match, not really independent
               information -- this bound rarely triggers on this specific
               dataset, whose per-pair peaks top out empirically around
               0.85, but is kept as a symmetric safety rule rather than a
               one-sided one calibrated only to this data). A window with
               zero pairs surviving the filter is marked not-computable
               (NaN), separate from acceptance-gate rejection.

LOW/HIGH were calibrated from this dataset's own per-pair peak-correlation
distribution (11-channel, all recordings): median 0.24, 90th percentile
0.51, max 0.85 -- see module history for the calibration run.
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
MAX_LAG = 10  # matches real_data_validation_gcc.py's own "final answer" convention
PEAK_LOW = 0.1
PEAK_HIGH = 0.95

OUT_CSV = PROJECT_ROOT / "journal_edit" / "tvd_multichannel_realdata_gcc_robust.csv"


def load_emg(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=";")
    return data[:, 1:65].T


def pooled_delays(delays: np.ndarray, peaks: np.ndarray) -> dict[str, float]:
    out = {}
    out["mean"] = float(np.mean(delays))
    out["median"] = float(np.median(delays))
    mask = (np.abs(peaks) >= PEAK_LOW) & (np.abs(peaks) <= PEAK_HIGH)
    out["threshold"] = float(np.mean(delays[mask])) if mask.any() else np.nan
    return out


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
                    pair_delays = []
                    pair_peaks = []
                    for k in range(K - 1):
                        d, p = _gcc_delay(seg[k], seg[k + 1], max_lag=MAX_LAG)
                        pair_delays.append(d)
                        pair_peaks.append(p)
                    pair_delays = np.asarray(pair_delays)
                    pair_peaks = np.asarray(pair_peaks)
                    pooled = pooled_delays(pair_delays, pair_peaks)
                    for rule, theta in pooled.items():
                        if not np.isfinite(theta) or abs(theta) < 1e-6:
                            continue
                        cv = cfg.ied_m / (abs(theta) / cfg.fs)
                        accepted = gate.cv_min <= cv <= gate.cv_max
                        rows.append(dict(
                            subject=subject, condition=condition, col=col, K=k_label, rule=rule,
                            t_s=round(t_s, 3), cv_ms=round(cv, 4), accepted=int(accepted),
                        ))
        print(f"done {fpath.name}: {n_samples/cfg.fs:.1f}s, {len(rows)} rows so far", file=sys.stderr)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["subject", "condition", "col", "K", "rule", "t_s", "cv_ms", "accepted"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
