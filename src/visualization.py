"""Plotly chart builders for the signal-processing pipeline visualization.

Each function returns a go.Figure ready to pass to the app's chart() wrapper.
Computation stays in feature_extraction.py — these are presentation only.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def build_time_domain_figure(
    t: np.ndarray,
    x: np.ndarray,
    feats: dict[str, float],
    channel: int,
) -> go.Figure:
    """Waveform annotated with Mean, +/-1 STD band, Max and Min markers."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=t, y=x, mode="lines",
        line=dict(width=1, color="#5B9BD5"),
        name="EMG signal",
    ))

    mean_val = feats["Mean"]
    std_val = feats["STD"]

    fig.add_hrect(
        y0=mean_val - std_val, y1=mean_val + std_val,
        fillcolor="#FFD700", opacity=0.08, line_width=0,
        annotation_text=f"±1 STD ({std_val:.4g})",
        annotation_position="top left",
        annotation_font=dict(size=12, color="#DAA520"),
    )
    fig.add_hline(
        y=mean_val, line_dash="dash", line_color="#FFD700",
        annotation_text=f"Mean = {mean_val:.4g}",
        annotation_position="top right",
        annotation_font=dict(size=13, color="#FFD700"),
    )

    max_val = feats["Max"]
    max_idx = int(np.argmax(x))
    fig.add_trace(go.Scatter(
        x=[t[max_idx]], y=[max_val], mode="markers+text",
        marker=dict(size=10, color="#E74C3C", symbol="triangle-up"),
        text=[f"Max = {max_val:.4g}"], textposition="top center",
        textfont=dict(size=12, color="#E74C3C"),
        name="Max",
    ))

    min_val = feats["Min"]
    min_idx = int(np.argmin(x))
    fig.add_trace(go.Scatter(
        x=[t[min_idx]], y=[min_val], mode="markers+text",
        marker=dict(size=10, color="#2ECC71", symbol="triangle-down"),
        text=[f"Min = {min_val:.4g}"], textposition="bottom center",
        textfont=dict(size=12, color="#2ECC71"),
        name="Min",
    ))

    fig.update_layout(
        xaxis_title="time (s)", yaxis_title="amplitude",
        title=f"Tín hiệu EMG — channel {channel}",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    return fig


def build_psd_figure(
    f: np.ndarray,
    pxx: np.ndarray,
    feats: dict[str, float],
) -> go.Figure:
    """PSD plot annotated with MDF and MNF vertical lines."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=f, y=pxx, mode="lines",
        line=dict(width=1.5, color="#5B9BD5"),
        name="PSD",
    ))

    # MDF and MNF are often only a few Hz apart, so anchoring both labels to
    # their line position (the old approach) makes them collide. Instead we
    # draw the lines without inline annotations and stack two fixed-height
    # labels above the plot, each pointing down to its line with an arrow.
    mdf_val = feats["MDF"]
    fig.add_vline(x=mdf_val, line_dash="dash", line_color="#E74C3C", line_width=2)

    mnf_val = feats["MNF"]
    fig.add_vline(x=mnf_val, line_dash="dot", line_color="#F39C12", line_width=2)

    fig.add_annotation(
        x=mdf_val, y=1.0, xref="x", yref="paper",
        text=f"MDF = {mdf_val:.1f} Hz",
        showarrow=True, arrowhead=0, arrowcolor="#E74C3C",
        ax=0, ay=-34, yshift=0,
        font=dict(size=13, color="#E74C3C"),
        bgcolor="rgba(0,0,0,0.55)", borderpad=3,
    )
    fig.add_annotation(
        x=mnf_val, y=1.0, xref="x", yref="paper",
        text=f"MNF = {mnf_val:.1f} Hz",
        showarrow=True, arrowhead=0, arrowcolor="#F39C12",
        ax=0, ay=-10, yshift=0,
        font=dict(size=13, color="#F39C12"),
        bgcolor="rgba(0,0,0,0.55)", borderpad=3,
    )

    fig.update_layout(
        xaxis_title="frequency (Hz)", yaxis_title="PSD",
        title="Power Spectral Density",
        height=420,
        margin=dict(t=90),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    return fig
