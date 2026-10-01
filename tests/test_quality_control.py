"""Tests for src/quality_control.py — clipping/dropout detectors."""
import numpy as np

from src import config as cfg
from src.quality_control import (
    _fraction_in_runs, clipped_fraction, dropout_fraction,
    QCResult, assess_signal_quality, assess_channel_matrix_quality,
)


class TestFractionInRuns:
    def test_no_runs_returns_zero(self):
        mask = np.array([True, False, True, False, True])
        assert _fraction_in_runs(mask, min_run=2) == 0.0

    def test_single_long_run_counted(self):
        mask = np.array([False, True, True, True, False])
        assert _fraction_in_runs(mask, min_run=3) == 3 / 5

    def test_run_shorter_than_min_run_not_counted(self):
        mask = np.array([False, True, True, False])
        assert _fraction_in_runs(mask, min_run=3) == 0.0

    def test_empty_mask_returns_zero(self):
        assert _fraction_in_runs(np.array([], dtype=bool), min_run=2) == 0.0


class TestClippedFraction:
    def test_clean_noise_has_near_zero_clip_fraction(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(4000) * 0.1
        assert clipped_fraction(x, min_run=cfg.QC_CLIP_MIN_RUN) < 0.005

    def test_sustained_pinned_run_detected(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal(4000) * 0.1
        peak = 5.0
        x[1000:1050] = peak  # 50-sample sustained saturation
        frac = clipped_fraction(x, min_run=cfg.QC_CLIP_MIN_RUN)
        assert frac >= 50 / 4000

    def test_single_sample_peak_not_flagged(self):
        rng = np.random.default_rng(3)
        x = rng.standard_normal(4000) * 0.1
        x[2000] = 5.0  # one isolated sample at the peak, not sustained
        frac = clipped_fraction(x, min_run=cfg.QC_CLIP_MIN_RUN)
        assert frac == 0.0


class TestDropoutFraction:
    def test_clean_noise_has_near_zero_dropout_fraction(self):
        rng = np.random.default_rng(4)
        x = rng.standard_normal(4000) * 0.1
        frac = dropout_fraction(
            x, window=cfg.QC_DROPOUT_WINDOW, std_frac=cfg.QC_DROPOUT_STD_FRAC,
            min_run=cfg.QC_DROPOUT_MIN_RUN,
        )
        assert frac < 0.01

    def test_sustained_flat_segment_detected(self):
        rng = np.random.default_rng(5)
        x = rng.standard_normal(4000) * 0.1
        x[1500:1800] = 0.0  # 300-sample flat segment mid-signal
        frac = dropout_fraction(
            x, window=cfg.QC_DROPOUT_WINDOW, std_frac=cfg.QC_DROPOUT_STD_FRAC,
            min_run=cfg.QC_DROPOUT_MIN_RUN,
        )
        assert frac > 0.03


class TestAssessSignalQuality:
    def test_clean_signal_passes(self):
        rng = np.random.default_rng(10)
        x = rng.standard_normal(4000) * 0.1
        result = assess_signal_quality(x, fs=2000)
        assert isinstance(result, QCResult)
        assert result.passed is True
        assert result.reasons == []

    def test_clipped_signal_fails_with_reason(self):
        rng = np.random.default_rng(11)
        x = rng.standard_normal(4000) * 0.1
        x[1000:1100] = 5.0  # sustained saturation, 100/4000 = 2.5% > 1% threshold
        result = assess_signal_quality(x, fs=2000)
        assert result.passed is False
        assert any("bao_hoa" in r for r in result.reasons)

    def test_dropout_signal_fails_with_reason(self):
        rng = np.random.default_rng(12)
        x = rng.standard_normal(4000) * 0.1
        x[500:1000] = 0.0  # 500/4000 = 12.5% > 5% threshold
        result = assess_signal_quality(x, fs=2000)
        assert result.passed is False
        assert any("mat_tin_hieu" in r for r in result.reasons)


class TestAssessChannelMatrixQuality:
    def test_dead_channel_flagged_without_running_detectors(self):
        rng = np.random.default_rng(13)
        channels = rng.standard_normal((4000, 3)) * 0.1
        channels[:, 1] = 0.0
        valid_mask = np.array([True, False, True])
        results = assess_channel_matrix_quality(channels, valid_mask, fs=2000)
        assert len(results) == 3
        assert results[1].passed is False
        assert results[1].reasons == ["kenh_chet"]

    def test_valid_clean_channels_pass(self):
        rng = np.random.default_rng(14)
        channels = rng.standard_normal((4000, 2)) * 0.1
        valid_mask = np.array([True, True])
        results = assess_channel_matrix_quality(channels, valid_mask, fs=2000)
        assert all(r.passed for r in results)
