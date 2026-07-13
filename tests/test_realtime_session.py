"""Tests for src/realtime_session.py — Usecase 1 (v2) playback logic."""
from pathlib import Path

import numpy as np
import pytest

from src import config as cfg
from src import data_loader as dl
from src.evaluate import ModelResult
from src.session_builder import SegmentInfo
from src.realtime_session import (
    rectify_envelope, PlaybackStep, build_playback_steps, SegmentAssessment,
    assess_segments, best_model_name, assess_channel_grid,
)


def _fake_segments() -> list[SegmentInfo]:
    f1 = dl.FileInfo(path=Path("Sujet_9_10_emg.csv"), subject=9, condition="10", label=0)
    f2 = dl.FileInfo(path=Path("Sujet_9_20_emg.csv"), subject=9, condition="20", label=0)
    return [
        SegmentInfo(file=f1, mvc=10, start_sample=0, end_sample=3000),
        SegmentInfo(file=f2, mvc=20, start_sample=3000, end_sample=6000),
    ]


def _fake_result(name, f1):
    return ModelResult(
        name=name, accuracy=0.9, precision=0.9, recall=0.9, f1=f1,
        cv_accuracy=0.9, auc=None, confusion=np.zeros((2, 2)),
        features_used=cfg.FEATURE_NAMES, fitted_estimator=None, threshold=0.5,
    )


class TestBestModelName:
    def test_picks_highest_f1(self):
        results = [_fake_result("SVM", 0.70), _fake_result("KNN", 0.85), _fake_result("LDA", 0.60)]
        assert best_model_name(results) == "KNN"

    def test_single_model(self):
        results = [_fake_result("DecisionTree", 0.5)]
        assert best_model_name(results) == "DecisionTree"

    def test_tie_breaks_to_first(self):
        results = [_fake_result("SVM", 0.8), _fake_result("KNN", 0.8)]
        assert best_model_name(results) == "SVM"


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

    def test_window_t_starts_at_zero_and_is_global_elapsed_time(self):
        """window_t must reflect real elapsed time *within the whole
        concatenated session* (starting at 0 for the very first step), not a
        fixed local 0..display_window_sec window reused every step — that's
        what lets the waveform chart's x-axis grow across the session like a
        real-time monitor instead of resetting every frame."""
        rng = np.random.default_rng(3)
        signal = rng.standard_normal(6000) * 0.1
        segments = _fake_segments()
        steps = build_playback_steps(
            signal, segments, n_steps=5,
            display_window_sec=0.5, envelope_window_sec=0.05, fs=2000)

        assert steps[0].window_t[0] == 0.0

        for s in steps:
            expected_start = s.sample_pos / 2000
            assert abs(s.window_t[0] - expected_start) < 1e-9
            assert abs(s.window_t[-1] - s.window_t[0] - (len(s.window_t) - 1) / 2000) < 1e-9

        starts = [s.window_t[0] for s in steps]
        assert starts == sorted(starts)

    def test_max_display_points_none_keeps_full_resolution(self):
        rng = np.random.default_rng(3)
        signal = rng.standard_normal(6000) * 0.1
        segments = _fake_segments()
        steps = build_playback_steps(
            signal, segments, n_steps=5,
            display_window_sec=0.5, envelope_window_sec=0.05, fs=2000,
            max_display_points=None)
        assert len(steps[0].window_raw) == 1000

    def test_max_display_points_decimates_window_arrays(self):
        rng = np.random.default_rng(3)
        signal = rng.standard_normal(6000) * 0.1
        segments = _fake_segments()
        steps = build_playback_steps(
            signal, segments, n_steps=5,
            display_window_sec=0.5, envelope_window_sec=0.05, fs=2000,
            max_display_points=100)
        for s in steps:
            assert len(s.window_raw) <= 100
            assert len(s.window_processed) == len(s.window_raw)
            assert len(s.window_t) == len(s.window_raw)
            # still monotonically increasing after decimation
            assert np.all(np.diff(s.window_t) > 0)

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


class _StubEstimator:
    """Minimal predict_proba-only estimator stub for testing assess_segments
    without training a real model."""
    def predict_proba(self, X):
        return np.array([[0.3, 0.7]])


class TestAssessSegments:
    def test_returns_one_assessment_per_segment(self):
        rng = np.random.default_rng(4)
        signal = rng.standard_normal(6000) * 0.1
        segments = _fake_segments()
        results = [ModelResult(
            name="KNN", accuracy=0.9, precision=0.9, recall=0.9, f1=0.9,
            cv_accuracy=0.9, auc=None, confusion=np.zeros((2, 2)),
            features_used=cfg.FEATURE_NAMES, fitted_estimator=_StubEstimator(),
            threshold=0.5,
        )]
        assessments = assess_segments(signal, segments, results)
        assert len(assessments) == 2
        assert all(isinstance(a, SegmentAssessment) for a in assessments)

    def test_assessment_uses_full_segment_and_stub_prediction(self):
        rng = np.random.default_rng(4)
        signal = rng.standard_normal(6000) * 0.1
        segments = _fake_segments()
        results = [ModelResult(
            name="KNN", accuracy=0.9, precision=0.9, recall=0.9, f1=0.9,
            cv_accuracy=0.9, auc=None, confusion=np.zeros((2, 2)),
            features_used=cfg.FEATURE_NAMES, fitted_estimator=_StubEstimator(),
            threshold=0.5,
        )]
        assessments = assess_segments(signal, segments, results)
        first = assessments[0]
        assert set(first.feats.keys()) == set(cfg.FEATURE_NAMES)
        assert first.rms == first.feats["RMS"]
        assert first.mdf == first.feats["MDF"]
        assert first.predictions[0]["model"] == "KNN"
        assert first.predictions[0]["pred"] == 1
        assert abs(first.predictions[0]["p_fatigue"] - 0.7) < 1e-9

    def test_rms_reflects_full_segment_not_a_subwindow(self):
        """Verify that assess_segments uses the FULL segment slice, not a
        short sub-window. This test independently computes the expected RMS
        from the known synthetic signal and asserts the result matches —
        directly catching any regression to sub-window slicing.

        A regression to signal[start:start+100] would produce wrong RMS,
        failing this test even though the old assertions would still pass.
        """
        rng = np.random.default_rng(4)
        signal = rng.standard_normal(6000) * 0.1
        segments = _fake_segments()
        results = [ModelResult(
            name="KNN", accuracy=0.9, precision=0.9, recall=0.9, f1=0.9,
            cv_accuracy=0.9, auc=None, confusion=np.zeros((2, 2)),
            features_used=cfg.FEATURE_NAMES, fitted_estimator=_StubEstimator(),
            threshold=0.5,
        )]
        assessments = assess_segments(signal, segments, results)

        # Ground-truth RMS computed directly from the full segment slice —
        # independent of assess_segments' own internals. A regression to a
        # short sub-window (e.g. signal[start:start+100]) would produce a
        # different RMS and fail this assertion.
        expected_rms_seg0 = float(np.sqrt(np.mean(signal[0:3000] ** 2)))
        expected_rms_seg1 = float(np.sqrt(np.mean(signal[3000:6000] ** 2)))
        assert abs(assessments[0].rms - expected_rms_seg0) < 1e-9
        assert abs(assessments[1].rms - expected_rms_seg1) < 1e-9

        # Also confirm it's NOT what a buggy 100-sample sub-window would give,
        # to make the regression this test guards against concrete.
        wrong_rms_seg0 = float(np.sqrt(np.mean(signal[0:100] ** 2)))
        assert abs(assessments[0].rms - wrong_rms_seg0) > 1e-6


class TestAssessChannelGrid:
    def test_returns_one_entry_per_segment_length_64(self, monkeypatch):
        segments = _fake_segments()
        # 3 channels total per file; channel idx 1 is invalid (not in
        # valid_channels) and must come back as None, not 0/1.
        channels_by_file = {
            segments[0].file.path: np.tile(
                np.array([0.1, 0.2, 0.3]), (100, 1)),
            segments[1].file.path: np.tile(
                np.array([0.1, 0.2, 0.3]), (100, 1)),
        }
        monkeypatch.setattr(dl, "load_channels", lambda path: channels_by_file[path])
        result_model = ModelResult(
            name="KNN", accuracy=0.9, precision=0.9, recall=0.9, f1=0.9,
            cv_accuracy=0.9, auc=None, confusion=np.zeros((2, 2)),
            features_used=cfg.FEATURE_NAMES, fitted_estimator=_StubEstimator(),
            threshold=0.5,
        )
        out = assess_channel_grid(segments, valid_channels=[0, 2], model_result=result_model)

        assert len(out) == 2
        for entry in out:
            assert len(entry) == 64
            # idx 1 (channel physical #2) is not in valid_channels -> None
            assert entry[1] is None
            # idx 0 and idx 2 are valid -> stub predicts Fatigue (1) always
            assert entry[0] == 1
            assert entry[2] == 1
            # every other index (not a real column in this 3-channel fake
            # file, and not in valid_channels) must also be None
            assert entry[3] is None
            assert entry[63] is None
