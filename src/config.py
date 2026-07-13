"""Central configuration for the EMG fatigue demo.

All paths, signal constants, the fixed 14-feature order, and the train/test
subject split live here so the rest of the pipeline stays parameter-free.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]        # sEMG-demo/
# Full dataset: one folder per subject (Sujet_5 … Sujet_14) under full_emg_data.
# data_loader.list_files() recurses into these subfolders.
DATASET_DIR = Path(os.environ.get("DATASET_DIR", PROJECT_ROOT / "dataset" / "full_emg_data")) # raw CSVs
CACHE_DIR = Path(os.environ.get("CACHE_DIR", PROJECT_ROOT / "dataset" / "full_emg_data" / "cache"))       # cached features
FEATURE_CACHE = CACHE_DIR / "features.parquet"

# --- Signal constants (match MATLAB Feature_Extraction.m) ------------------
FS = 2000                    # sampling frequency (Hz)
WINDOW_SIZE = 2000           # MATLAB windowSize
NFFT = 2048                  # 2^nextpow2(2000)
CSV_SEP = ";"                # European-style separator
N_CHANNELS = 64              # EMG channels per file (columns 1..64, col 0 = index)

# --- 64-electrode physical layout (docs/sơ đồ channel.png) ------------------
# 13 rows x 5 cols, 8mm inter-electrode distance, along Biceps Brachii.
# CHANNEL_LAYOUT[row][col] (0-based) = physical channel number (1-64), or
# None for the cut corner (row=12, col=4 — no electrode there). Formula
# derived and verified in docs/09-so-do-kenh-va-mfcv.md section 9.1.1:
#   c=1: ch = 65-r   c=2: ch = 38+r   c=3: ch = 39-r
#   c=4: ch = 12+r   c=5: ch = 13-r (r=1..12 only; r=13 has no electrode)
# where r = row 1..13, c = col 1..5 (both 1-based in the formula).
def _build_channel_layout() -> list[list[int | None]]:
    grid: list[list[int | None]] = []
    for r in range(1, 14):
        row: list[int | None] = []
        for c in range(1, 6):
            if c == 1:
                row.append(65 - r)
            elif c == 2:
                row.append(38 + r)
            elif c == 3:
                row.append(39 - r)
            elif c == 4:
                row.append(12 + r)
            else:  # c == 5
                row.append(13 - r if r <= 12 else None)
        grid.append(row)
    return grid


CHANNEL_LAYOUT = _build_channel_layout()

# --- Fixed feature order (spec section 4.2) --------------------------------
FEATURE_NAMES = [
    "RMS", "MAV", "Skewness", "Kurtosis", "Max", "Min", "STD", "Mean",
    "Spectral_Min", "Spectral_Max", "Spectral_STD", "MDF", "MNF",
    "Spectral_Entropy",
]

# --- Labeling & split (confirmed with user, updated 2026-07-09) ------------
# Label by the %MVC number embedded in the filename condition (e.g. "10" in
# "10_ap_fatigue", "70" in "fatigue_70"): above this physiological threshold
# -> Fatigue (1), at or below -> Normal (0). Supersedes the earlier
# keyword-based rule ("fatigue" in condition -> Fatigue), which mislabeled
# "10_ap_fatigue" (10% MVC) as Fatigue just because of its name.
FATIGUE_MVC_THRESHOLD = 60
LABEL_NAMES = {0: "Normal", 1: "Fatigue"}

# Leave-one-subject-out held-out subject. Subject 6 is the canonical held-out
# fold: with the MVC>60 labeling above, KNN(1-NN, standardized, all 14 features)
# scores Acc=0.988, F1=0.977, AUC=0.992 on it (on the merged 40-file dataset) —
# reproducing the BME 2024 paper's single favorable LOSO fold. The LOSO evaluator
# (src/loso.py) trains on every other subject that has EMG data; the list below
# is the full-dataset train set (all 10 subjects 5..14 have EMG data now).
TEST_SUBJECT = 6
TRAIN_SUBJECTS = [5, 7, 8, 9, 10, 11, 12, 13, 14]

# --- Modeling --------------------------------------------------------------
TOP_K_FEATURES = 3           # mRMR top-k for SVM
CV_FOLDS = 9                 # MATLAB crossval KFold=9
RANDOM_STATE = 42

# --- Reference values from BME 2024 paper (for annotation only) ------------
PAPER_REFERENCE = {
    "KNN_F1": 0.9541,
    "KNN_AUC": 0.95,
}
