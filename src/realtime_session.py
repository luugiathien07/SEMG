"""Usecase-1 (v2) clinic demo: near-real-time playback and per-segment
fatigue assessment over a concatenated session signal
(src.session_builder.build_session_signal). Kept separate from the research
pipeline — only orchestrates it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .session_builder import SegmentInfo


def rectify_envelope(x: np.ndarray, envelope_samples: int) -> np.ndarray:
    """Full-wave rectify + moving-average envelope, for display only."""
    envelope_samples = max(1, int(envelope_samples))
    kernel = np.ones(envelope_samples) / envelope_samples
    return np.convolve(np.abs(x), kernel, mode="same")


@dataclass(frozen=True)
class PlaybackStep:
    step_idx: int
    sample_pos: int
    segment_idx: int
    window_t: np.ndarray
    window_raw: np.ndarray
    window_processed: np.ndarray


def _segment_idx_for(sample_pos: int, segments: list[SegmentInfo]) -> int:
    for i, seg in enumerate(segments):
        if seg.start_sample <= sample_pos < seg.end_sample:
            return i
    return len(segments) - 1


def build_playback_steps(
    signal: np.ndarray,
    segments: list[SegmentInfo],
    n_steps: int,
    display_window_sec: float,
    envelope_window_sec: float,
    fs: int,
) -> list[PlaybackStep]:
    """Scrub through `signal` at `n_steps` evenly-spaced positions, each
    carrying a `display_window_sec`-long raw+processed window for the
    animated waveform chart. Does NOT compute fatigue status — that comes
    from `assess_segments`, using whole segments, not these short windows.
    """
    display_window_samples = int(display_window_sec * fs)
    envelope_samples = max(1, int(envelope_window_sec * fs))
    if len(signal) <= display_window_samples:
        raise ValueError(
            "Tín hiệu ghép quá ngắn so với display_window_sec đã chọn: cần "
            f"nhiều hơn {display_window_samples} mẫu, có {len(signal)}."
        )

    max_start = len(signal) - display_window_samples
    t_window = np.arange(display_window_samples) / fs

    steps: list[PlaybackStep] = []
    for i in range(n_steps):
        frac = i / (n_steps - 1) if n_steps > 1 else 0.0
        sample_pos = int(round(frac * max_start))
        window_raw = signal[sample_pos:sample_pos + display_window_samples]
        window_processed = rectify_envelope(window_raw, envelope_samples)
        steps.append(PlaybackStep(
            step_idx=i, sample_pos=sample_pos,
            segment_idx=_segment_idx_for(sample_pos, segments),
            window_t=t_window, window_raw=window_raw,
            window_processed=window_processed,
        ))
    return steps
