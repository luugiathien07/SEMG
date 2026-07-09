"""Tests for src/visualization_realtime.py chart builders (realtime session demo)."""
import numpy as np
import plotly.graph_objects as go

from src.visualization_realtime import (
    build_waveform_figure, build_trend_figure, build_status_badge,
)


def _fake_window(n: int = 1000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    t = np.arange(n) / 2000.0
    raw = rng.standard_normal(n) * 0.2
    processed = np.abs(raw)
    return t, raw, processed


class TestBuildWaveformFigure:
    def test_returns_go_figure(self):
        t, raw, processed = _fake_window()
        fig = build_waveform_figure(t, raw, processed)
        assert isinstance(fig, go.Figure)

    def test_contains_raw_and_processed_traces(self):
        t, raw, processed = _fake_window()
        fig = build_waveform_figure(t, raw, processed)
        names = [tr.name for tr in fig.data]
        assert "Tín hiệu thô" in names
        assert "Sau khử nhiễu (rectify + envelope)" in names


class TestBuildTrendFigure:
    def test_returns_go_figure(self):
        fig = build_trend_figure([10, 20, 40], [0.1, 0.2, 0.3], [50.0, 45.0, 40.0])
        assert isinstance(fig, go.Figure)

    def test_contains_rms_and_mdf_traces(self):
        fig = build_trend_figure([10, 20, 40], [0.1, 0.2, 0.3], [50.0, 45.0, 40.0])
        names = [tr.name for tr in fig.data]
        assert "RMS" in names
        assert "MDF (Hz)" in names

    def test_mdf_trace_uses_secondary_axis(self):
        fig = build_trend_figure([10, 20, 40], [0.1, 0.2, 0.3], [50.0, 45.0, 40.0])
        mdf_trace = next(tr for tr in fig.data if tr.name == "MDF (Hz)")
        assert mdf_trace.yaxis == "y2"


class TestBuildStatusBadge:
    def test_returns_go_figure(self):
        fig = build_status_badge("Không mỏi", 0.2)
        assert isinstance(fig, go.Figure)

    def test_not_fatigued_shows_status_text_and_green(self):
        fig = build_status_badge("Không mỏi", 0.2)
        ind = fig.data[0]
        assert "Không mỏi" in ind.title.text
        assert ind.number.font.color == "#27AE60"

    def test_fatigued_shows_status_text_and_red(self):
        fig = build_status_badge("Mỏi", 0.85)
        ind = fig.data[0]
        assert "Mỏi" in ind.title.text
        assert ind.number.font.color == "#E74C3C"

    def test_handles_missing_p_fatigue(self):
        fig = build_status_badge("Mỏi", None)
        ind = fig.data[0]
        assert "Mỏi" in ind.title.text
