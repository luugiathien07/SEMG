"""Usecase-1 (v2) clinic demo: concatenate a subject's real EMG files, ordered
by ascending %MVC, into one continuous "session" signal for
app_realtime_session.py. Kept fully separate from the research pipeline in
pipeline.py / feature_extraction.py — this module only orchestrates them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

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


@dataclass(frozen=True)
class SegmentInfo:
    file: dl.FileInfo
    mvc: int
    start_sample: int
    end_sample: int


def build_session_signal(
    subject: int, channel: int,
) -> tuple[np.ndarray, list[SegmentInfo]]:
    """Load `channel` from every file of `subject` (ascending %MVC order) and
    concatenate into one long array, with per-file segment boundaries.
    """
    files = list_ordered_segments(subject)
    if len(files) < 2:
        raise ValueError(
            f"Subject {subject} không đủ file để ghép buổi tập "
            f"(cần ít nhất 2, có {len(files)})."
        )

    chunks: list[np.ndarray] = []
    segments: list[SegmentInfo] = []
    offset = 0
    for f in files:
        x = dl.load_channels(f.path)[:, channel]
        chunks.append(x)
        segments.append(SegmentInfo(
            file=f, mvc=parse_mvc(f.condition),
            start_sample=offset, end_sample=offset + len(x),
        ))
        offset += len(x)

    signal = np.concatenate(chunks)
    return signal, segments
