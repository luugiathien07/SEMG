"""Tests for src/realtime_html — client-side Canvas2D animation component."""
import numpy as np

from src.realtime_html import build_realtime_html


def _fake_segments(fs=2000):
    return [
        {
            "start": 0.0, "end": 1.5, "label": "10%",
            "isPostFatigue": False, "rms": 0.05, "mdf": 120.0,
            "predictions": [{"model": "KNN", "pred": 0, "p_fatigue": 0.1}],
        },
        {
            "start": 1.5, "end": 3.0, "label": "20%",
            "isPostFatigue": False, "rms": 0.08, "mdf": 110.0,
            "predictions": [{"model": "KNN", "pred": 1, "p_fatigue": 0.8}],
        },
    ]


def test_returns_valid_html():
    signal = np.random.default_rng(1).standard_normal(6000) * 0.1
    html = build_realtime_html(signal, _fake_segments(), fs=2000)
    assert "<canvas" in html
    assert "requestAnimationFrame" in html
    assert "__DATA__" not in html


def test_data_embedded_as_json():
    signal = np.random.default_rng(1).standard_normal(6000) * 0.1
    html = build_realtime_html(signal, _fake_segments(), fs=2000)
    assert '"raw"' in html
    assert '"proc"' in html
    assert '"segments"' in html


def test_display_window_in_output():
    signal = np.random.default_rng(1).standard_normal(6000) * 0.1
    html = build_realtime_html(signal, _fake_segments(), fs=2000, display_window_sec=5.0)
    assert '"displayWindow": 5.0' in html or '"displayWindow":5.0' in html
