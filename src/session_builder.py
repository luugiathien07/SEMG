"""Usecase-1 (v2) clinic demo: concatenate a subject's real EMG files, ordered
by ascending %MVC, into one continuous "session" signal for
app_realtime_session.py. Kept fully separate from the research pipeline in
pipeline.py / feature_extraction.py — this module only orchestrates them.
"""
from __future__ import annotations

import re

import numpy as np

from . import data_loader as dl

_MVC_RE = re.compile(r"\d+")


def parse_mvc(condition: str) -> int:
    """First integer found in a condition string, e.g. 70 in 'fatigue_70'."""
    m = _MVC_RE.search(condition)
    return int(m.group()) if m else 0


def list_ordered_segments(subject: int) -> list[dl.FileInfo]:
    """All files for `subject`, sorted by ascending %MVC (ties broken by
    condition string) — light -> heavy -> fatigued.
    """
    files = [f for f in dl.list_files() if f.subject == subject]
    files.sort(key=lambda f: (parse_mvc(f.condition), f.condition))
    return files


def common_valid_channels(subject: int) -> list[int]:
    """Channel indices that are non-dead in every file of `subject`."""
    files = list_ordered_segments(subject)
    masks = [dl.valid_channel_mask(dl.load_channels(f.path)) for f in files]
    common = np.logical_and.reduce(masks)
    return [i for i, valid in enumerate(common) if valid]
