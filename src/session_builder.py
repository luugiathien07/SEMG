"""Usecase-1 (v2) clinic demo: concatenate a subject's real EMG files, ordered
by ascending %MVC with any post-fatigue retest condition (e.g.
"10_ap_fatigue") placed last regardless of its %MVC number, into one
continuous "session" signal for app_realtime_session.py. Kept fully separate
from the research pipeline in pipeline.py / feature_extraction.py — this
module only orchestrates them.
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


def is_post_fatigue(condition: str) -> bool:
    """True for a post-fatigue retest condition (e.g. '10_ap_fatigue' —
    "après fatigue"), which is measured chronologically AFTER the fatiguing
    protocol despite carrying a low %MVC number. This must sort last
    regardless of `parse_mvc`, or the session would tell the story
    backwards (retesting fatigue effects before the fatiguing bout even
    happened).
    """
    return "ap_fatigue" in condition


def list_ordered_segments(subject: int) -> list[dl.FileInfo]:
    """All files for `subject`, chronologically ordered: ascending %MVC
    first (light -> heavy -> fatiguing), then any post-fatigue retest
    conditions last, regardless of their %MVC number (ties within each
    group broken by condition string).
    """
    files = [f for f in dl.list_files() if f.subject == subject]
    files.sort(key=lambda f: (
        is_post_fatigue(f.condition), parse_mvc(f.condition), f.condition,
    ))
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
