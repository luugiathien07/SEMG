"""
Innervation-zone-aware GCC pooling: tests whether excluding channel pairs
that straddle the innervation zone (IZ) -- rather than median-pooling
(real_data_validation_gcc_robust.py) or plain mean-pooling
(real_data_validation_gcc.py) -- fixes GCC's real-data breakdown under
channel pooling (PAPER_Multichannel_TVD.tex, Section results-realdata).

At the IZ, action potentials propagate in OPPOSITE directions on either
side, so the per-adjacent-pair delay sign flips there (src/mfcv.py's
detect_innervation_zone, already used elsewhere in this codebase to locate
the IZ from a single-differential chain). A window's IZ was never
identified or excluded in the original GCC-pooled real-data script: the
mean (or median) was taken across ALL K-1 adjacent pairs regardless of
whether some of them straddled the IZ, where the "constant-delay-per-pair"
model the estimator assumes is not just noisy but qualitatively wrong (the
delay's own SIGN depends on which side of the IZ a pair sits on).

Per window: detect the IZ from the full K=11-channel double-differential
segment (src.mfcv.select_propagation_channels), keep only the longer
same-direction side (dropping the shorter side and the IZ-straddling pair
entirely), then pool with the median rule already found more robust to a
single bad pair. Windows where the surviving side has fewer than 2 channels
(no pair left to average) are marked not-computable.
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
    select_propagation_channels,
)

cfg = GridConfig()
gate = QualityGate()

WIN_S = 0.5
HOP_S = 0.25
FULL_COLS = [1, 2, 3, 4]
K_FULL = 11
MAX_LAG = 10  # matches real_data_validation_gcc.py's own "final answer" convention

OUT_CSV = PROJECT_ROOT / "journal_edit" / "tvd_multichannel_realdata_gcc_iz.csv"


def load_emg(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=";")
    return data[:, 1:65].T


def gcc_median_delay(x: np.ndarray) -> float:
    K = x.shape[0]
    delays = [_gcc_delay(x[k], x[k + 1], max_lag=MAX_LAG)[0] for k in range(K - 1)]
    return float(np.median(delays))


def main() -> None:
    files = sorted(glob.glob(str(PROJECT_ROOT / "dataset" / "full_emg_data" / "Sujet_*" / "*_emg.csv")))
    rows = []
    n_iz_found = 0
    n_iz_total = 0
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

                # K=2 baseline: unaffected by IZ selection (single pair)
                theta2 = _gcc_delay(seg[0], seg[1], max_lag=MAX_LAG)[0]
                if abs(theta2) >= 1e-6:
                    cv2 = cfg.ied_m / (abs(theta2) / cfg.fs)
                    rows.append(dict(subject=subject, condition=condition, col=col, K="K2",
                                      t_s=round(t_s, 3), cv_ms=round(cv2, 4),
                                      accepted=int(gate.cv_min <= cv2 <= gate.cv_max), iz_found=0))

                # K=11, IZ-aware: restrict to the longer same-direction side first
                n_iz_total += 1
                chosen, iz = select_propagation_channels(seg[:K_FULL], gate)
                if iz is not None:
                    n_iz_found += 1
                if chosen.shape[0] < 2:
                    continue
                theta11 = gcc_median_delay(chosen)
                if abs(theta11) < 1e-6:
                    continue
                cv11 = cfg.ied_m / (abs(theta11) / cfg.fs)
                rows.append(dict(subject=subject, condition=condition, col=col, K="K11_iz",
                                  t_s=round(t_s, 3), cv_ms=round(cv11, 4),
                                  accepted=int(gate.cv_min <= cv11 <= gate.cv_max),
                                  iz_found=int(iz is not None)))
        print(f"done {fpath.name}: {n_samples/cfg.fs:.1f}s, {len(rows)} rows so far, "
              f"IZ found in {n_iz_found}/{n_iz_total} windows so far", file=sys.stderr)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["subject", "condition", "col", "K", "t_s", "cv_ms", "accepted", "iz_found"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT_CSV}")
    print(f"IZ detected in {n_iz_found}/{n_iz_total} windows overall ({100*n_iz_found/max(n_iz_total,1):.1f}%)")


if __name__ == "__main__":
    main()
