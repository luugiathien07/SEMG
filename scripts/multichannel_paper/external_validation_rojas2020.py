"""
External validation of this project's GCC-pooled MFCV estimator on an
independent, publicly available HD-sEMG dataset: Rojas-Martinez, Serna
Higuita, Jordanic, Marateb, Merletti & Manas Villanueva, "A high-density
Surface EMG dataset of upper-limb muscles during isometric contractions in
healthy humans," Scientific Data 7:397 (2020), doi:10.1038/s41597-020-00717-6
(data on Figshare, DOIs 10.6084/m9.figshare.11860572 and .11860851, parts
1/3 and 2/3; part 3/3 is a reader tool only).

Dataset geometry (from the dataset's own readme.pdf): 12 subjects, biceps
brachii recorded with an 8-row x 15-column monopolar grid (IED=10mm in both
directions, columns aligned with the arm/fiber axis), Fs=2048 Hz, 4 isometric
tasks (elbow flexion/extension, forearm pronation/supination) at 10/30/50%
MVC, 10s per recording. This script uses only the biceps grid and the elbow
flexion/extension tasks (flexion: biceps is the prime mover; extension:
biceps is antagonist/stabilizer only).

This is a genuinely independent test of the estimator used throughout this
manuscript series (different lab, country, hardware -- EMG-USB 128-channel
amplifiers vs. this project's own acquisition system -- IED, and population)
from the Sujet_* real-data validation elsewhere in this paper (Section
results-realdata), which reuses this project's own recordings.

Per-column pipeline (same building blocks as real_data_validation_gcc.py):
single-differential along the column (7 bipolar channels from 8 monopolar
rows), GCC-pooled delay (median of the 7 adjacent-pair \_gcc_delay values --
median chosen over mean per this paper's own Section results-realdata
follow-up, which found median pooling more robust to a single noisy pair),
0.5s sliding window with 0.25s hop, repeated over all 15 columns and pooled
per recording.

Data must be downloaded and extracted separately (see the paper text for
DOIs); this script expects PROJECT_ROOT/scratch_external_dataset/extracted/
sN/biceps/sN_{e,f}{10,30,50}_bb.bin (raw float64 little-endian, channels x
samples, channel k (0-based) -> row=k%8, col=k//8), matching the archive's
own layout.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.mfcv import _gcc_delay, QualityGate  # noqa: E402

FS = 2048.0
IED_M = 0.010
N_ROWS, N_COLS = 8, 15
WIN_S = 0.5
HOP_S = 0.25
MAX_LAG = 15
GATE = QualityGate()

DATA_ROOT = PROJECT_ROOT / "scratch_external_dataset" / "extracted"
OUT_DIR = PROJECT_ROOT / "journal_edit"
SUBJECTS = [f"s{i}" for i in range(1, 13)]
TASKS = ["e10", "e30", "e50", "f10", "f30", "f50"]


def load_grid(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype="<f8")
    n = raw.size // (N_ROWS * N_COLS)
    mono = raw.reshape(N_ROWS * N_COLS, n, order="C")
    return mono.reshape(N_COLS, N_ROWS, n).transpose(1, 0, 2)  # (row, col, n)


def estimate_cv(path: Path) -> np.ndarray:
    grid = load_grid(path)
    n = grid.shape[2]
    w = int(WIN_S * FS)
    h = int(HOP_S * FS)
    cvs = []
    for col in range(N_COLS):
        sd = np.diff(grid[:, col, :], axis=0)
        for start in range(0, n - w + 1, h):
            seg = sd[:, start:start + w]
            seg = seg - seg.mean(axis=1, keepdims=True)
            delays = [_gcc_delay(seg[k], seg[k + 1], max_lag=MAX_LAG)[0] for k in range(seg.shape[0] - 1)]
            theta = float(np.median(delays))
            if abs(theta) < 1e-6:
                continue
            cv = IED_M / (abs(theta) / FS)
            if GATE.cv_min <= cv <= GATE.cv_max:
                cvs.append(cv)
    return np.array(cvs)


def main() -> None:
    rows = []
    for subj in SUBJECTS:
        for task in TASKS:
            path = DATA_ROOT / subj / "biceps" / f"{subj}_{task}_bb.bin"
            if not path.exists():
                continue
            cvs = estimate_cv(path)
            median_cv = float(np.median(cvs)) if len(cvs) else float("nan")
            rows.append(dict(subject=subj, task=task[0], mvc=int(task[1:]), n=len(cvs), median_cv=median_cv))
            print(f"{subj} {task}: n={len(cvs):4d} median_CV={median_cv:.3f}")

    csv_path = OUT_DIR / "tvd_external_validation_rojas2020_data.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subject", "task", "mvc", "n", "median_cv"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
