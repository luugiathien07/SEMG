"""Plotly chart builders for the Usecase-1 (v2) realtime session demo
(app_realtime_session.py). Kept separate from src/visualization.py (used by
the technical Predict tab in app.py) so the two pages never share, and can't
accidentally break, the same chart code.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def build_waveform_figure(
    t: np.ndarray, raw_x: np.ndarray, processed_x: np.ndarray,
) -> go.Figure:
    """Faint raw scrub window vs. bold rectified+enveloped window."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=raw_x, mode="lines",
        line=dict(width=1, color="rgba(91, 155, 213, 0.4)"),
        name="Tín hiệu thô",
    ))
    fig.add_trace(go.Scatter(
        x=t, y=processed_x, mode="lines",
        line=dict(width=2.5, color="#F39C12"),
        name="Sau khử nhiễu (rectify + envelope)",
    ))
    fig.update_layout(
        xaxis_title="time (s)", yaxis_title="amplitude",
        title="① Tín hiệu EMG — vị trí hiện tại trong buổi tập",
        height=280,
        margin=dict(t=50, b=40, l=50, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    return fig


def build_trend_figure(
    x_labels: list[str], rms_values: list[float], mdf_values: list[float],
) -> go.Figure:
    """RMS (left axis) and MDF (right axis) across the segments visited so
    far, plotted in chronological session order (`x_labels`, e.g. "10%",
    "20%", ..., "10% (sau mỏi)").

    Uses a categorical x-axis (not raw %MVC numbers): a post-fatigue retest
    segment can share the same %MVC as an earlier segment (e.g. "10%" at the
    start vs. "10% (sau mỏi)" at the end), so plotting against the numeric
    %MVC value would snap the line backward instead of continuing forward.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_labels, y=rms_values, mode="lines+markers",
        line=dict(color="#E74C3C", width=2),
        name="RMS",
    ))
    fig.add_trace(go.Scatter(
        x=x_labels, y=mdf_values, mode="lines+markers",
        line=dict(color="#3498DB", width=2),
        name="MDF (Hz)", yaxis="y2",
    ))
    fig.update_layout(
        xaxis=dict(title="Giai đoạn", type="category"),
        yaxis=dict(title="RMS"),
        yaxis2=dict(title="MDF (Hz)", overlaying="y", side="right"),
        title="② RMS & Median Frequency theo giai đoạn",
        height=280,
        margin=dict(t=50, b=40, l=50, r=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    return fig


def build_status_badge(status: str, p_fatigue: float | None) -> go.Figure:
    """Large colored binary status indicator (green/red) for the segment
    currently being played, with the KNN model's P(Fatigue) shown in the
    title text when available.
    """
    is_fatigued = status == "Mỏi"
    color = "#E74C3C" if is_fatigued else "#27AE60"
    icon = "⚠️" if is_fatigued else "✅"
    pct_text = f" ({p_fatigue:.0%})" if p_fatigue is not None else ""
    fig = go.Figure(go.Indicator(
        mode="number",
        value=1,
        number={"font": {"size": 1, "color": color}},
        title={"text": f"{icon} {status}{pct_text}",
               "font": {"size": 22, "color": color}},
    ))
    fig.update_layout(height=130, margin=dict(t=10, b=10, l=10, r=10))
    return fig
