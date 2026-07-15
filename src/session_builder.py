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


def trim_bounds_for_display(
    signal: np.ndarray, segments: list[SegmentInfo], fs: int,
    chunk_sec: float = 0.25, lo_frac: float = 0.4, hi_frac: float = 2.0,
    min_run_sec: float = 2.0, edge_trim_sec: float = 1.0,
) -> list[tuple[int, int]]:
    """Per-segment `(start, end)` sample bounds (local to each segment's own
    slice of `signal`) that `trim_for_display` keeps — the steady-band
    onset/offset detection, exposed on its own so any other per-file signal
    sharing the same `segments` (e.g. the electrode column MFCV runs on,
    `session_builder.build_session_column`) can be trimmed identically.
    Reusing the exact same bounds — rather than re-detecting them on a
    different signal — is what keeps the CV trend on the same time axis as
    the RMS/MDF trend and waveform, which are computed on the signal this
    function was actually run on (see `realtime_session.compute_cv_series`).

    See `trim_for_display` for the detection method itself; this returns
    only the bounds, not the trimmed signal.
    """
    chunk = max(1, int(chunk_sec * fs))
    min_run = max(1, round(min_run_sec / chunk_sec))
    edge_trim = int(edge_trim_sec * fs)

    bounds: list[tuple[int, int]] = []
    for seg in segments:
        x = signal[seg.start_sample:seg.end_sample]
        env = np.array([
            np.sqrt(np.mean(x[i:i + chunk] ** 2))
            for i in range(0, max(1, len(x) - chunk + 1), chunk)
        ])
        start, end = 0, len(x)
        if len(env) >= min_run:
            mid = env[len(env) // 5 : max(len(env) // 5 + 1, 4 * len(env) // 5)]
            ref = np.median(mid) if len(mid) else np.median(env)
            in_band = (env >= lo_frac * ref) & (env <= hi_frac * ref)
            onset = 0
            for i in range(len(in_band) - min_run + 1):
                if in_band[i:i + min_run].all():
                    onset = i
                    break
            offset_i = len(in_band) - 1
            for i in range(len(in_band) - 1, min_run - 2, -1):
                if in_band[i - min_run + 1:i + 1].all():
                    offset_i = i
                    break
            start = onset * chunk + edge_trim
            end = (offset_i + 1) * chunk - edge_trim
            if end <= start:
                start, end = 0, len(x)
            else:
                start, end = max(0, start), min(len(x), end)
        bounds.append((start, end))
    return bounds


def trim_for_display(
    signal: np.ndarray, segments: list[SegmentInfo], fs: int,
    chunk_sec: float = 0.25, lo_frac: float = 0.4, hi_frac: float = 2.0,
    min_run_sec: float = 2.0, edge_trim_sec: float = 1.0,
) -> tuple[np.ndarray, list[SegmentInfo]]:
    """Trim each segment's non-steady-state edges before splicing segments
    back-to-back, so the concatenated waveform and RMS/MDF/CV trend read as
    one continuous session instead of swinging at every %MVC boundary.

    Two distinct edge artifacts show up in this dataset's raw files, both
    handled by the same "steady band" test (see `trim_bounds_for_display`):
    (1) the ordinary ramp-in/ramp-out as the subject reaches/releases the
    target force (low relative to the file's steady level), and (2) an
    occasional high-amplitude burst right at a file's start (looks like a
    leftover MVC-calibration contraction, not the submaximal task) — a
    plain "trim anything below X% of the file" threshold misses (2)
    entirely since it's abnormally *high*, not low. The band's reference
    level is the median of the file's own middle 60% (away from either
    edge), so it isn't itself skewed by the edge artifacts it's meant to
    detect.

    `edge_trim_sec` then cuts an extra buffer past the detected onset/offset
    on each side — the steady-band test finds where the signal *starts*
    settling, but the last stretch right at that boundary is still often
    visibly different from the true plateau, so a bit more is shaved off
    each cut edge rather than kept.

    Display only — classification (`realtime_session.assess_segments` /
    `assess_channel_grid`) must keep running on the untrimmed signal/segments
    from `build_session_signal_avg`: trimming before feature extraction was
    tested and flips some models' Fatigue/Normal calls (KNN especially)
    versus what they were trained on, since the pipeline trains on
    whole-file (untrimmed) features.
    """
    bounds = trim_bounds_for_display(
        signal, segments, fs, chunk_sec, lo_frac, hi_frac, min_run_sec, edge_trim_sec,
    )

    chunks_out: list[np.ndarray] = []
    new_segments: list[SegmentInfo] = []
    offset = 0
    for seg, (start, end) in zip(segments, bounds):
        x = signal[seg.start_sample:seg.end_sample]
        trimmed = x[start:end]
        chunks_out.append(trimmed)
        new_segments.append(SegmentInfo(
            file=seg.file, mvc=seg.mvc,
            start_sample=offset, end_sample=offset + len(trimmed),
        ))
        offset += len(trimmed)

    return np.concatenate(chunks_out), new_segments


def build_session_signal_avg(subject: int) -> tuple[np.ndarray, list[SegmentInfo]]:
    """Like `build_session_signal`, but each file's contribution is the
    per-sample mean across that subject's valid channels (`common_valid_channels`)
    instead of one selected channel — dead channels are excluded from the
    average so they don't pull it toward zero.
    """
    files = list_ordered_segments(subject)
    if len(files) < 2:
        raise ValueError(
            f"Subject {subject} không đủ file để ghép buổi tập "
            f"(cần ít nhất 2, có {len(files)})."
        )
    valid_channels = common_valid_channels(subject)

    chunks: list[np.ndarray] = []
    segments: list[SegmentInfo] = []
    offset = 0
    for f in files:
        channels = dl.load_channels(f.path)
        x = channels[:, valid_channels].mean(axis=1)
        chunks.append(x)
        segments.append(SegmentInfo(
            file=f, mvc=parse_mvc(f.condition),
            start_sample=offset, end_sample=offset + len(x),
        ))
        offset += len(x)

    signal = np.concatenate(chunks)
    return signal, segments


def build_session_column(
    subject: int, physical_channels: list[int],
) -> tuple[np.ndarray, list[SegmentInfo]]:
    """Like `build_session_signal`, but returns a `(len(physical_channels),
    n_time)` matrix of concatenated raw channel columns instead of one
    channel or an average. Used by MFCV (`src/mfcv.py`), which needs
    several physically-adjacent electrodes along the muscle fiber
    direction, not the averaged signal the rest of the realtime demo uses.

    `physical_channels` are 1-based physical channel numbers (as in
    `config.CHANNEL_LAYOUT`), converted to 0-based array indices internally.
    """
    files = list_ordered_segments(subject)
    if len(files) < 2:
        raise ValueError(
            f"Subject {subject} không đủ file để ghép buổi tập "
            f"(cần ít nhất 2, có {len(files)})."
        )
    idxs = [ch - 1 for ch in physical_channels]

    chunks: list[np.ndarray] = []
    segments: list[SegmentInfo] = []
    offset = 0
    for f in files:
        channels = dl.load_channels(f.path)
        x = channels[:, idxs].T  # (len(physical_channels), n_time)
        chunks.append(x)
        segments.append(SegmentInfo(
            file=f, mvc=parse_mvc(f.condition),
            start_sample=offset, end_sample=offset + x.shape[1],
        ))
        offset += x.shape[1]

    signal = np.concatenate(chunks, axis=1)
    return signal, segments
