"""
Real-data validation of the channel-pooling claim (Section~\\ref{sub:results-pooling}
of PAPER_Multichannel_TVD.tex) on the in-house 64-electrode (13x5) HD-sEMG grid
dataset already used for the "Independent validation on a 64-channel HD-sEMG grid
dataset" section of the companion Luu2026BSPC / PAPERV5-V7 manuscripts
(dataset/full_emg_data, biceps brachii, 13x5 monopolar grid, IED=8mm, Fs=2000Hz).

Unlike the synthetic benchmark, this real data has no known ground-truth CV, so
"validation" here means testing whether K=13 pooling improves ESTIMATOR PRECISION
(repeatability) the same way it improves accuracy against a known ground truth in
simulation -- two proxies, neither requiring ground truth:

  1. Temporal repeatability: SD of the CV estimate across successive 0.5s windows
     within one (subject, condition, column) recording, for K=2 vs K=13.
  2. Cross-column agreement: SD of the per-column mean CV across the 4 full-length
     (13-channel) columns of the same grid, for K=2 vs K=13. All 4 columns sit on
     the same muscle under the same contraction, so they should, in principle,
     estimate a similar CV if the estimator is not swamped by column-specific noise.

Both are estimated with the SAME constant-delay MLE model (x_k(n) = s(n-k*theta)
+w_k(n), estimate_delay_mle in src/mfcv.py) used throughout this manuscript
series, applied to double-differential (DD) channels rather than raw monopolar
ones: an earlier version of this script ran directly on monopolar channels and
found the multichannel MLE collapsing to the grid-search boundary (a strong
shared-reference common-mode component swamps the cost function). A first fix
(single-differencing only) helped but still leaves far-field/common low-
frequency contamination that a single subtraction does not fully cancel, so,
exactly as mfcv_window() (this module's own real-time pipeline) does by
default (use_double_diff=True), the signal is differenced twice: 13 monopolar
channels -> 12 single-differential (SD) channels -> 11 double-differential
(DD) channels, DD_k = SD_{k+1} - SD_k, before estimate_delay_mle is applied --
K=2 (first two DD channels) vs K=11 (all DD channels).
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
    GridConfig, QualityGate, preprocess, serp_grid, estimate_delay_mle,
    single_differential, double_differential,
)

cfg = GridConfig()          # fs=2000, ied=0.008, bandpass 20-400, notch 50+harmonics
gate = QualityGate()        # cv_min=2, cv_max=10 m/s

WIN_S = 0.5
HOP_S = 0.25
FULL_COLS = [1, 2, 3, 4]    # 13-channel columns (col 0 has 1 missing electrode, Sec. real_data note)
# NOTE: raw monopolar x_k(n)=s(n-k*theta)+w_k(n) (Eq. model) collapses to the
# grid-search boundary on this real data -- a strong shared-reference
# common-mode component swamps the multichannel MLE cost, exactly the reason
# mfcv_window() (this module's own real-time pipeline) always applies
# single_differential() THEN double_differential() by default. We do the
# same here: 13 monopolar channels -> 12 SD channels -> 11 DD channels
# (DD_k = SD_{k+1}-SD_k), which still obey the same delay model with s(n)
# replaced by the DD source, and use K=2 (first two DD channels) vs K=11
# (all DD channels) -- the differential-domain analogue of the paper's
# synthetic K=2-vs-K=13 comparison.
K_VALUES = {"K2": 2, "K11": 11}

OUT_CSV = PROJECT_ROOT / "journal_edit" / "tvd_multichannel_realdata.csv"


def load_emg(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=";")
    return data[:, 1:65].T  # (64, N) monopolar channels


def main() -> None:
    files = sorted(glob.glob(str(PROJECT_ROOT / "dataset" / "full_emg_data" / "Sujet_*" / "*_emg.csv")))
    rows = []
    for fpath in files:
        fpath = Path(fpath)
        name = fpath.stem  # e.g. Sujet_10_90_emg
        parts = name.split("_")
        subject = f"{parts[0]}_{parts[1]}"          # Sujet_10
        condition = "_".join(parts[2:-1]) or "unk"  # 90 / fatigue_70 / 10_ap_fatigue

        x = load_emg(fpath)
        g = serp_grid(x)  # (13, 5, N)
        n_samples = g.shape[2]
        w = int(WIN_S * cfg.fs)
        h = int(HOP_S * cfg.fs)
        if n_samples < w:
            continue

        for col in FULL_COLS:
            col_sig = preprocess(g[:, col, :], cfg)   # (13, N) zero-phase filtered
            sd_sig = single_differential(col_sig)     # (12, N) common-mode removed
            dd_sig = double_differential(sd_sig)      # (11, N) far-field removed too
            for start in range(0, n_samples - w + 1, h):
                seg = dd_sig[:, start:start + w]
                seg = seg - seg.mean(axis=1, keepdims=True)
                t_s = start / cfg.fs
                for k_label, K in K_VALUES.items():
                    theta = estimate_delay_mle(seg[:K], cfg, gate)
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
