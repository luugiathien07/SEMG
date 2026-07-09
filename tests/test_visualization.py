"""Tests for src/visualization.py chart builders."""
import numpy as np
import plotly.graph_objects as go

from src.visualization import build_time_domain_figure, build_psd_figure


def _fake_signal(n: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic 1-D signal and its time axis at 2000 Hz."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(n) * 0.2
    t = np.arange(n) / 2000.0
    return t, x


def _fake_feats(x: np.ndarray) -> dict[str, float]:
    """Minimal feature dict with the keys the chart builders need."""
    return {
        "Mean": float(np.mean(x)),
        "STD": float(np.std(x, ddof=1)),
        "Max": float(np.max(x)),
        "Min": float(np.min(x)),
        "MDF": 42.0,
        "MNF": 55.0,
    }


class TestBuildTimeDomainFigure:
    def test_returns_go_figure(self):
        t, x = _fake_signal()
        fig = build_time_domain_figure(t, x, _fake_feats(x), channel=0)
        assert isinstance(fig, go.Figure)

    def test_contains_signal_trace(self):
        t, x = _fake_signal()
        fig = build_time_domain_figure(t, x, _fake_feats(x), channel=3)
        trace_names = [tr.name for tr in fig.data]
        assert "EMG signal" in trace_names

    def test_contains_max_min_markers(self):
        t, x = _fake_signal()
        fig = build_time_domain_figure(t, x, _fake_feats(x), channel=0)
        trace_names = [tr.name for tr in fig.data]
        assert "Max" in trace_names
        assert "Min" in trace_names

    def test_has_mean_hline(self):
        t, x = _fake_signal()
        feats = _fake_feats(x)
        fig = build_time_domain_figure(t, x, feats, channel=0)
        hlines = [s for s in fig.layout.shapes if s.type == "line" and s.y0 == s.y1]
        mean_lines = [s for s in hlines if abs(s.y0 - feats["Mean"]) < 1e-6]
        assert len(mean_lines) >= 1

    def test_has_std_band(self):
        t, x = _fake_signal()
        feats = _fake_feats(x)
        fig = build_time_domain_figure(t, x, feats, channel=0)
        rects = [s for s in fig.layout.shapes if s.type == "rect"]
        assert len(rects) >= 1


class TestBuildPsdFigure:
    def test_returns_go_figure(self):
        _, x = _fake_signal()
        f = np.linspace(0, 1000, 1025)
        pxx = np.abs(np.random.default_rng(0).standard_normal(1025))
        fig = build_psd_figure(f, pxx, _fake_feats(x))
        assert isinstance(fig, go.Figure)

    def test_contains_psd_trace(self):
        _, x = _fake_signal()
        f = np.linspace(0, 1000, 1025)
        pxx = np.abs(np.random.default_rng(0).standard_normal(1025))
        fig = build_psd_figure(f, pxx, _fake_feats(x))
        trace_names = [tr.name for tr in fig.data]
        assert "PSD" in trace_names

    def test_has_mdf_and_mnf_vlines(self):
        _, x = _fake_signal()
        feats = _fake_feats(x)
        f = np.linspace(0, 1000, 1025)
        pxx = np.abs(np.random.default_rng(0).standard_normal(1025))
        fig = build_psd_figure(f, pxx, feats)
        vlines = [s for s in fig.layout.shapes
                  if s.type == "line" and s.x0 == s.x1]
        vline_xs = {round(s.x0, 1) for s in vlines}
        assert feats["MDF"] in vline_xs
        assert feats["MNF"] in vline_xs
