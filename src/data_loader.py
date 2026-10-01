"""Load raw sEMG CSVs and derive per-file metadata.

Filename convention: ``Sujet_{id}_{condition}_emg.csv`` where condition is one
of ``10/20/40/60/90`` (%MVC), ``fatigue_70`` or ``10_ap_fatigue``. The label is
derived from the %MVC number in the condition vs
``config.FATIGUE_MVC_THRESHOLD`` (see ``parse_filename``).

A "sample" for the classifier is a single EMG channel of a single file, so this
module exposes helpers to enumerate files, load a file's channel matrix, and
compute the per-file valid-channel mask (drop all-zero / dead channels).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from . import config

_NAME_RE = re.compile(r"^Sujet_(\d+)_(.+)_emg$", re.IGNORECASE)
_MVC_NUMBER_RE = re.compile(r"\d+")


@dataclass(frozen=True)
class FileInfo:
    path: Path
    subject: int
    condition: str          # e.g. "10", "fatigue_70", "10_ap_fatigue"
    label: int              # 0 = Normal, 1 = Fatigue

    @property
    def name(self) -> str:
        return self.path.stem


def parse_filename(path: Path) -> FileInfo | None:
    """Parse ``Sujet_X_cond_emg.csv`` -> FileInfo, or None if it doesn't match."""
    m = _NAME_RE.match(path.stem)
    if not m:
        return None
    subject = int(m.group(1))
    condition = m.group(2)
    # Label by the %MVC number embedded in the condition string (first match,
    # e.g. "10" in "10_ap_fatigue", "70" in "fatigue_70"): above the
    # physiological fatigue threshold -> Fatigue (1), else Normal (0).
    num_match = _MVC_NUMBER_RE.search(condition)
    mvc = int(num_match.group()) if num_match else 0
    label = 1 if mvc > config.FATIGUE_MVC_THRESHOLD else 0
    return FileInfo(path=path, subject=subject, condition=condition, label=label)


def list_files(dataset_dir: Path | None = None) -> list[FileInfo]:
    """All parseable CSVs in the dataset dir, sorted by (subject, condition)."""
    dataset_dir = dataset_dir or config.DATASET_DIR
    infos = []
    # os.walk with followlinks=True so symlinked subject folders are traversed.
    for root, _dirs, files in os.walk(dataset_dir, followlinks=True):
        for fname in sorted(files):
            if not fname.endswith(".csv"):
                continue
            info = parse_filename(Path(root) / fname)
            if info is not None:
                infos.append(info)
    infos.sort(key=lambda i: (i.subject, i.condition))
    return infos


def load_channels(path: Path) -> np.ndarray:
    """Return the (n_time, 64) EMG matrix from a CSV (drops the index column)."""
    df = pd.read_csv(path, sep=config.CSV_SEP, header=None, decimal=".")
    arr = df.to_numpy(dtype=float)
    # Column 0 is the time/index column; columns 1..64 are EMG channels.
    channels = arr[:, 1:1 + config.N_CHANNELS]
    return channels


def valid_channel_mask(channels: np.ndarray) -> np.ndarray:
    """Boolean mask of channels that are not entirely zero (dead channels)."""
    return ~np.all(channels == 0, axis=0)


def iter_files(subjects: list[int] | None = None) -> Iterator[FileInfo]:
    """Yield FileInfo optionally filtered to a set of subject ids."""
    for info in list_files():
        if subjects is None or info.subject in subjects:
            yield info
