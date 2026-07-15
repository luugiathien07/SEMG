"""Tests for src/realtime_html — client-side Canvas2D animation component."""
import numpy as np

from src.realtime_html import _CV_MAX_MS, _CV_MIN_MS, _compute_trend, build_realtime_html


def _fake_segments(fs=2000):
    return [
        {
            "start": 0.0, "end": 1.5, "label": "10%",
            "isPostFatigue": False, "rms": 0.05, "mdf": 120.0,
            "predictions": [{"model": "KNN", "pred": 0, "p_fatigue": 0.1}],
            "channelPreds": [0] * 32 + [None] * 32,
        },
        {
            "start": 1.5, "end": 3.0, "label": "20%",
            "isPostFatigue": False, "rms": 0.08, "mdf": 110.0,
            "predictions": [{"model": "KNN", "pred": 1, "p_fatigue": 0.8}],
            "channelPreds": [1] * 32 + [None] * 32,
        },
    ]


_FAKE_LAYOUT = [
    [64, 39, 38, 13, 12], [63, 40, 37, 14, 11], [62, 41, 36, 15, 10],
    [61, 42, 35, 16, 9], [60, 43, 34, 17, 8], [59, 44, 33, 18, 7],
    [58, 45, 32, 19, 6], [57, 46, 31, 20, 5], [56, 47, 30, 21, 4],
    [55, 48, 29, 22, 3], [54, 49, 28, 23, 2], [53, 50, 27, 24, 1],
    [52, 51, 26, 25, None],
]


def test_returns_valid_html():
    signal = np.random.default_rng(1).standard_normal(6000) * 0.1
    html = build_realtime_html(signal, _fake_segments(), fs=2000, channel_layout=_FAKE_LAYOUT)
    assert "<canvas" in html
    assert "requestAnimationFrame" in html
    assert "__DATA__" not in html


def test_data_embedded_as_json():
    signal = np.random.default_rng(1).standard_normal(6000) * 0.1
    html = build_realtime_html(signal, _fake_segments(), fs=2000, channel_layout=_FAKE_LAYOUT)
    assert '"raw"' in html
    assert '"proc"' in html
    assert '"segments"' in html


def test_display_window_in_output():
    signal = np.random.default_rng(1).standard_normal(6000) * 0.1
    html = build_realtime_html(
        signal, _fake_segments(), fs=2000, display_window_sec=5.0,
        channel_layout=_FAKE_LAYOUT)
    assert '"displayWindow": 5.0' in html or '"displayWindow":5.0' in html


def test_channel_layout_embedded_as_json():
    signal = np.random.default_rng(1).standard_normal(6000) * 0.1
    html = build_realtime_html(signal, _fake_segments(), fs=2000, channel_layout=_FAKE_LAYOUT)
    assert '"channelLayout"' in html
    assert '"channelPreds"' in html


def test_channel_map_canvas_and_draw_function_present():
    signal = np.random.default_rng(1).standard_normal(6000) * 0.1
    html = build_realtime_html(signal, _fake_segments(), fs=2000, channel_layout=_FAKE_LAYOUT)
    assert 'id="chmap"' in html
    assert "drawChannelMap" in html


def test_trend_cv_embedded_as_json():
    signal = np.random.default_rng(1).standard_normal(6000) * 0.1
    html = build_realtime_html(signal, _fake_segments(), fs=2000, channel_layout=_FAKE_LAYOUT)
    assert '"trendCv"' in html


def test_compute_trend_cv_within_physiological_range():
    signal = np.random.default_rng(2).standard_normal(20000) * 0.1
    times, rms_vals, mdf_vals, cv_vals = _compute_trend(signal, fs=2000)
    assert len(cv_vals) == len(times) == len(rms_vals) == len(mdf_vals)
    assert all(_CV_MIN_MS <= cv <= _CV_MAX_MS for cv in cv_vals)


def test_compute_trend_cv_constant_signal_no_crash():
    signal = np.zeros(6000)
    times, rms_vals, mdf_vals, cv_vals = _compute_trend(signal, fs=2000)
    assert len(cv_vals) == len(times)
    assert all(_CV_MIN_MS <= cv <= _CV_MAX_MS for cv in cv_vals)


def test_segment_bounds_prevent_smoothing_bleed_across_boundary():
    fs = 2000
    rng = np.random.default_rng(3)
    low = rng.standard_normal(20 * fs) * 0.05
    high = rng.standard_normal(20 * fs) * 1.0
    signal = np.concatenate([low, high])

    times, rms_unbounded, _, _ = _compute_trend(signal, fs)
    _, rms_bounded, _, _ = _compute_trend(signal, fs, segment_bounds_sec=[20.0])

    times = np.asarray(times)
    # First trend point comfortably inside the high segment: with global
    # smoothing its moving average still straddles the boundary and drags the
    # value down toward the low segment; per-segment smoothing should not.
    idx = int(np.searchsorted(times, 20.5))
    assert rms_bounded[idx] > rms_unbounded[idx]
    assert rms_bounded[idx] > 0.5  # close to the high segment's own RMS (~1.0)
