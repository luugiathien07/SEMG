"""Configuration for the Usecase 2 (rehab recovery tracking) demo.

Deliberately independent from `src/config.py` (Usecase 1): the physiomio
dataset has its own sampling rate, channel naming and directory layout, and
none of it should leak into or be affected by the fatigue-classification
pipeline.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]     # sEMG-demo/
PHYSIOMIO_DIR = PROJECT_ROOT / "dataset" / "physiomio"

# --- Signal constants ------------------------------------------------------
# The files under dataset/physiomio/ follow the PhysioMio schema: `time`
# (resets to 0 at each movement segment, ~2048 Hz), `channel_01..channel_64`,
# `fma` (per-repetition index, NA during Rest) and `movement_type` (16 real
# hand-gesture labels, e.g. "MassFlexion", "HookGrasp" — NOT the placeholder
# "Gesture_NN" names used by usecase2_references/simulate_semg_rehab.py's
# simplified illustrative simulation). Movement labels are therefore read
# per-file rather than hardcoded here.
FS = 2048
N_CHANNELS = 64
CHANNEL_COLUMNS = [f"channel_{i:02d}" for i in range(1, N_CHANNELS + 1)]

ARMS = ["healthy_arm", "impaired_arm"]

DISCLAIMER = (
    "Dữ liệu mô phỏng cho demo — không phải bệnh nhân thật. "
    "Số thật cần thu tại Motion Lab."
)
