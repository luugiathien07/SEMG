"""Tests for src/realtime_session.py — Usecase 1 (v2) playback logic."""
from pathlib import Path

import numpy as np
import pytest

from src import data_loader as dl
from src.session_builder import SegmentInfo
from src.realtime_session import rectify_envelope, PlaybackStep, build_playback_steps


def _fake_segments() -> list[SegmentInfo]:
    f1 = dl.FileInfo(path=Path("Sujet_9_10_emg.csv"), subject=9, condition="10", label=0)
    f2 = dl.FileInfo(path=Path("Sujet_9_20_emg.csv"), subject=9, condition="20", label=0)
    return [
        SegmentInfo(file=f1, mvc=10, start_sample=0, end_sample=3000),
        SegmentInfo(file=f2, mvc=20, start_sample=3000, end_sample=6000),
    ]


class TestRectifyEnvelope:
    def test_output_is_nonnegative(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal(1000)
        result = rectify_envelope(x, envelope_samples=50)
        assert np.all(result >= 0)

    def test_output_same_length(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal(1000)
        result = rectify_envelope(x, envelope_samples=50)
        assert len(result) == len(x)


class TestBuildPlaybackSteps:
    def test_returns_n_steps(self):
        rng = np.random.default_rng(3)
        signal = rng.standard_normal(6000) * 0.1
        segments = _fake_segments()
        steps = build_playback_steps(
            signal, segments, n_steps=5,
            display_window_sec=0.5, envelope_window_sec=0.05, fs=2000)
        assert len(steps) == 5
        assert [s.step_idx for s in steps] == [0, 1, 2, 3, 4]

    def test_window_length_matches_display_window_sec(self):
        rng = np.random.default_rng(3)
        signal = rng.standard_normal(6000) * 0.1
        segments = _fake_segments()
        steps = build_playback_steps(
            signal, segments, n_steps=5,
            display_window_sec=0.5, envelope_window_sec=0.05, fs=2000)
        for s in steps:
            assert len(s.window_raw) == 1000
            assert len(s.window_processed) == 1000
            assert len(s.window_t) == 1000

    def test_segment_idx_nondecreasing_and_in_range(self):
        rng = np.random.default_rng(3)
        signal = rng.standard_normal(6000) * 0.1
        segments = _fake_segments()
        steps = build_playback_steps(
            signal, segments, n_steps=10,
            display_window_sec=0.5, envelope_window_sec=0.05, fs=2000)
        idxs = [s.segment_idx for s in steps]
        assert idxs == sorted(idxs)
        assert all(0 <= i < len(segments) for i in idxs)
        assert idxs[0] == 0
        assert idxs[-1] == 1

    def test_signal_too_short_raises(self):
        signal = np.zeros(100)
        segments = _fake_segments()
        with pytest.raises(ValueError):
            build_playback_steps(
                signal, segments, n_steps=5,
                display_window_sec=0.5, envelope_window_sec=0.05, fs=2000)
