"""Tests for src/realtime_html — client-side Canvas2D animation component."""
import numpy as np

from src.realtime_html import _compute_trend, build_realtime_html


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


def test_compute_trend_returns_rms_and_mdf_only():
    signal = np.random.default_rng(2).standard_normal(20000) * 0.1
    times, rms_vals, mdf_vals = _compute_trend(signal, fs=2000)
    assert len(rms_vals) == len(times) == len(mdf_vals)


def test_cv_series_embedded_as_json():
    signal = np.random.default_rng(1).standard_normal(6000) * 0.1
    cv_series = [
        {"t_s": 0.5, "cv_ms": 4.2, "accepted": True},
        {"t_s": 1.0, "cv_ms": None, "accepted": False},
    ]
    html = build_realtime_html(
        signal, _fake_segments(), fs=2000, channel_layout=_FAKE_LAYOUT,
        cv_series=cv_series,
    )
    assert '"trendCvT"' in html
    assert '"trendCv"' in html
    assert '"trendCvAccepted"' in html
    assert "4.2" in html


def test_trend_chart_labels_the_velocity_line_mfcv_not_cv():
    """The trend chart's third line is muscle fiber conduction velocity —
    labeled 'MFCV', not the bare 'CV' (which reads as cross-validation
    elsewhere in this same page, e.g. the model metrics table's 'CV-Acc'
    column)."""
    signal = np.random.default_rng(1).standard_normal(6000) * 0.1
    html = build_realtime_html(signal, _fake_segments(), fs=2000, channel_layout=_FAKE_LAYOUT)
    assert "MFCV (m/s)" in html
    assert "Xu hướng RMS, MDF &amp; MFCV" in html


def test_cv_series_none_embeds_empty_arrays():
    signal = np.random.default_rng(1).standard_normal(6000) * 0.1
    html = build_realtime_html(
        signal, _fake_segments(), fs=2000, channel_layout=_FAKE_LAYOUT,
        cv_series=None,
    )
    assert '"trendCvT": []' in html or '"trendCvT":[]' in html


def test_segment_bounds_prevent_smoothing_bleed_across_boundary():
    fs = 2000
    rng = np.random.default_rng(3)
    low = rng.standard_normal(20 * fs) * 0.05
    high = rng.standard_normal(20 * fs) * 1.0
    signal = np.concatenate([low, high])

    times, rms_unbounded, _ = _compute_trend(signal, fs)
    _, rms_bounded, _ = _compute_trend(signal, fs, segment_bounds_sec=[20.0])

    times = np.asarray(times)
    # First trend point comfortably inside the high segment: with global
    # smoothing its moving average still straddles the boundary and drags the
    # value down toward the low segment; per-segment smoothing should not.
    idx = int(np.searchsorted(times, 20.5))
    assert rms_bounded[idx] > rms_unbounded[idx]
    assert rms_bounded[idx] > 0.5  # close to the high segment's own RMS (~1.0)


class TestTrendUsesFilteredSignal:
    def test_trend_mdf_reflects_filtered_not_raw_signal(self):
        import json

        from src.signal_processing import filter_signal

        fs = 2000
        n = 20000
        t = np.arange(n) / fs
        rng = np.random.default_rng(7)
        # Heavy 50Hz mains hum + weak in-band content: the notch filter
        # should remove most of the 50Hz energy, shifting the median
        # frequency well away from 50Hz.
        signal = (
            5.0 * np.sin(2 * np.pi * 50 * t)
            + 0.2 * rng.standard_normal(n)
            + 0.1 * np.sin(2 * np.pi * 120 * t)
        )
        segments = _fake_segments()
        html = build_realtime_html(signal, segments, fs=fs, channel_layout=_FAKE_LAYOUT)

        start = html.index("const D = ") + len("const D = ")
        end = html.index(";", start)
        data = json.loads(html[start:end])

        # Same segment_bounds_sec build_realtime_html derives internally
        # (segment end times except the last), so smoothing matches exactly.
        bounds = [s["end"] for s in segments[:-1]]
        _, _, mdf_raw = _compute_trend(signal, fs, segment_bounds_sec=bounds)
        filtered = filter_signal(signal, fs)
        _, _, mdf_filt = _compute_trend(filtered, fs, segment_bounds_sec=bounds)

        assert data["trendMdf"] == [round(v, 2) for v in mdf_filt]
        assert data["trendMdf"] != [round(v, 2) for v in mdf_raw]


class TestProcessedLineUsesRealFilter:
    def test_proc_differs_from_naive_envelope_of_raw(self):
        import json

        from src.realtime_session import rectify_envelope

        fs = 2000
        n = 6000
        t = np.arange(n) / fs
        signal = 3.0 + np.sin(2 * np.pi * 50 * t) + 0.1 * np.sin(2 * np.pi * 100 * t)
        html = build_realtime_html(signal, _fake_segments(), fs=fs, channel_layout=_FAKE_LAYOUT)

        # build_realtime_html embeds the JSON payload as `const D = {...};`
        # (src/realtime_html.py:334) — extract it back out of the HTML.
        start = html.index("const D = ") + len("const D = ")
        end = html.index(";", start)
        data = json.loads(html[start:end])

        naive = rectify_envelope(signal, max(1, int(0.05 * fs)))
        dec = max(1, fs // 250)
        naive_dec = [round(float(v), 3) for v in naive[::dec]]

        assert data["proc"] != naive_dec
