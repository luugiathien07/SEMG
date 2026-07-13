"""Streamlit UI for Usecase 2 (rehab recovery tracking) — rendered as one tab
inside app_realtime_session.py's `st.tabs(...)`, not a standalone page.

Layout: 5 panels rendered sequentially (scroll-based, no nested tabs) to
suit a continuous video demo:
  Panel 0  Dashboard tổng quan (KPI cards + narrative + P(mỏi) explainer)
  Panel 1  Xác suất mỏi trong buổi tập (Chart 1 — in-session comparison)
  Panel 2  Sức bền cơ qua các buổi  (Chart 2 — endurance trend)
  Panel 3  Chỉ số phục hồi          (Chart 3 — symmetry + MDF)
  Panel 4  Khám phá tín hiệu        (signal viewer)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from . import config as rc
from . import data_loader as rdl
from . import endurance as end

# ── Color palette (matches charts from management) ──────────────────────
TEAL = "#1B7A7D"
TEAL_LIGHT = "rgba(27,122,125,0.12)"
CORAL = "#E8604C"
CORAL_LIGHT = "rgba(232,96,76,0.12)"
GREEN_BASELINE = "#27AE60"
GRAY_THRESHOLD = "#999999"
BG_CARD = "#F8FAFB"

HOVERLABEL = dict(
    font=dict(size=14, color="white", family="Inter, Arial, sans-serif"),
    bgcolor="#1E1E1E", bordercolor="#1E1E1E",
    namelength=-1,
)


def _chart(fig, height: int = 420, **kwargs):
    fig.update_layout(
        hoverlabel=HOVERLABEL, hovermode="closest",
        font=dict(family="Inter, Arial, sans-serif"),
        plot_bgcolor="white",
        height=height,
        margin=dict(l=60, r=30, t=50, b=50),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#F0F0F0", gridwidth=1)
    fig.update_yaxes(showgrid=True, gridcolor="#F0F0F0", gridwidth=1)
    kwargs.setdefault("width", "stretch")
    st.plotly_chart(fig, **kwargs)


# ═══════════════════════════════════════════════════════════════════════
#  PUBLIC ENTRY
# ═══════════════════════════════════════════════════════════════════════
def render() -> None:
    """Render the full Usecase 2 content into the current Streamlit container."""

    # ── Cached heavy computations ───────────────────────────────────
    @st.cache_data(show_spinner="Đang mô phỏng và huấn luyện KNN…")
    def _endurance_data():
        results, f1 = end.run_with_proba()
        return results, f1

    endurance_results, knn_f1 = _endurance_data()

    # ── Panel 0: Dashboard ──────────────────────────────────────────
    _render_dashboard(endurance_results, knn_f1)

    st.divider()

    # ── Panel 1: P(mỏi) in-session ─────────────────────────────────
    _render_fatigue_in_session(endurance_results, knn_f1)

    st.divider()

    # ── Panel 2: Endurance trend ────────────────────────────────────
    _render_endurance_trend(endurance_results)

    st.divider()

    # ── Panel 3: Recovery metrics (physiomio) ───────────────────────
    _render_recovery_metrics()

    st.divider()

    # ── Panel 4: Signal explorer ────────────────────────────────────
    _render_signal_explorer()


# ═══════════════════════════════════════════════════════════════════════
#  PANEL 0 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════
def _render_dashboard(
    results: list[end.SessionResultWithProba], f1: float,
) -> None:
    st.markdown("### 📊 Tổng quan phục hồi")
    st.caption(rc.DISCLAIMER)

    first, last = results[0], results[-1]
    n = len(results)
    improvement = round((last.endurance_sec / first.endurance_sec - 1) * 100)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số buổi tập", f"{n}")
    c2.metric("Sức bền ban đầu", f"{first.endurance_sec:.0f}s")
    c3.metric("Sức bền gần nhất", f"{last.endurance_sec:.0f}s",
              delta=f"+{improvement}%")
    c4.metric("F1 (KNN phân loại mỏi)", f"{f1:.3f}")

    st.info(
        f"Sau **{n} buổi** trị liệu, sức bền cơ tăng từ "
        f"**{first.endurance_sec:.0f}s → {last.endurance_sec:.0f}s** "
        f"(+{improvement}%). Cơ mỏi muộn hơn = phục hồi tốt hơn.",
        icon="💪",
    )

    with st.expander("💡 **P(mỏi) là gì?** — Giải thích cho người không chuyên"):
        st.markdown("""
1. Cắt tín hiệu mỗi buổi thành các **đoạn 1 giây**. Mỗi đoạn trích 7 đặc trưng
   (RMS, MAV, WL, ZC, SSC, MDF, MNF) → thành 1 điểm trong không gian đặc trưng.

2. Trong dữ liệu huấn luyện, mỗi đoạn đã có nhãn: **mỏi (1)** hoặc **chưa mỏi (0)**.

3. Với 1 đoạn mới cần đánh giá: model tìm **7 đoạn GIỐNG NHẤT** (khoảng cách gần nhất
   trong trường đặc trưng), rồi đếm bao nhiêu đoạn trong 7 cái đó là "mỏi".

> **P(mỏi) = số hàng xóm "mỏi" / 7.**
> Ví dụ 5/7 là mỏi → P = 0,71.

**Trực giác:** Đầu buổi cơ khoẻ (RMS thấp, MDF cao) → giống đám "chưa mỏi" → P thấp.
Cuối buổi cơ mỏi (RMS cao, MDF thấp) → giống đám "mỏi" → P cao. Đó là lý do đường P
leo dần từ 0 lên 1; **giây nó vượt 0,5 = lúc bắt đầu mỏi**.
""")


# ═══════════════════════════════════════════════════════════════════════
#  PANEL 1 — P(MỎI) IN-SESSION  (Chart 1)
# ═══════════════════════════════════════════════════════════════════════
def _render_fatigue_in_session(
    results: list[end.SessionResultWithProba], f1: float,
) -> None:
    st.markdown("### 📈 Xác suất mỏi (KNN) trong bài co cơ duy trì")
    st.caption(
        "So sánh đường P(mỏi) giữa 2 buổi. Buổi đầu mỏi sớm (đường lên nhanh), "
        "buổi sau mỏi muộn (đường lên chậm) → chứng tỏ phục hồi."
    )

    n_sessions = len(results)
    col_a, col_b = st.columns(2)
    idx_a = col_a.selectbox(
        "Buổi A (so sánh)",
        range(n_sessions),
        index=0,
        format_func=lambda i: f"Buổi {results[i].session}",
        key="uc2_proba_a",
    )
    idx_b = col_b.selectbox(
        "Buổi B (so sánh)",
        range(n_sessions),
        index=n_sessions - 1,
        format_func=lambda i: f"Buổi {results[i].session}",
        key="uc2_proba_b",
    )

    ra, rb = results[idx_a], results[idx_b]
    fig = go.Figure()

    # Buổi A — coral
    fig.add_trace(go.Scatter(
        x=ra.t_centers, y=ra.proba, mode="lines+markers",
        name=f"Buổi {ra.session} (mỏi sớm)",
        line=dict(color=CORAL, width=2.5),
        marker=dict(size=5, color=CORAL),
    ))
    # Buổi B — teal
    fig.add_trace(go.Scatter(
        x=rb.t_centers, y=rb.proba, mode="lines+markers",
        name=f"Buổi {rb.session} (mỏi muộn)",
        line=dict(color=TEAL, width=2.5),
        marker=dict(size=5, color=TEAL),
    ))

    # Ngưỡng P = 0.5
    fig.add_hline(y=0.5, line_dash="dash", line_color=GRAY_THRESHOLD, line_width=1,
                  annotation_text="Ngưỡng mỏi (P=0.5)",
                  annotation_position="top left",
                  annotation_font_color=GRAY_THRESHOLD)

    # Onset annotations
    fig.add_vline(x=ra.endurance_sec, line_dash="dot", line_color=CORAL, line_width=1,
                  annotation_text=f"{ra.endurance_sec:.1f}s",
                  annotation_position="bottom",
                  annotation_font_color=CORAL,
                  annotation_font_size=14)
    fig.add_vline(x=rb.endurance_sec, line_dash="dot", line_color=TEAL, line_width=1,
                  annotation_text=f"{rb.endurance_sec:.1f}s",
                  annotation_position="bottom",
                  annotation_font_color=TEAL,
                  annotation_font_size=14)

    fig.update_layout(
        title="Xác suất mỏi (KNN) trong bài co cơ duy trì",
        xaxis_title="Thời gian trong buổi (s)",
        yaxis_title="P(mỏi) — đầu ra KNN",
        yaxis=dict(range=[-0.05, 1.08]),
        legend=dict(x=0.55, y=0.35, bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#DDD", borderwidth=1),
    )
    _chart(fig, height=450)

    st.caption(
        f"KNN phân loại mỏi F1 = {f1:.3f} (mô phỏng). "
        "Điểm sức bền = thời điểm P(mỏi) vượt 0.5."
    )


# ═══════════════════════════════════════════════════════════════════════
#  PANEL 2 — ENDURANCE TREND  (Chart 2)
# ═══════════════════════════════════════════════════════════════════════
def _render_endurance_trend(results: list[end.SessionResultWithProba]) -> None:
    st.markdown("### 🏋️ Điểm sức bền cơ (thời điểm khởi phát mỏi) qua các buổi")
    st.caption(
        "Phục hồi tốt hơn → mỏi đến muộn hơn → điểm sức bền tăng. "
        "Đây là chỉ số cốt lõi của Usecase 2."
    )

    sessions = [r.session for r in results]
    endurance = [r.endurance_sec for r in results]

    fig = go.Figure()

    # Area fill
    fig.add_trace(go.Scatter(
        x=sessions, y=endurance,
        fill="tozeroy", fillcolor=TEAL_LIGHT,
        mode="lines+markers+text",
        text=[f"{e:.0f}s" for e in endurance],
        textposition="top center",
        textfont=dict(size=13, color=TEAL, family="Inter, Arial, sans-serif"),
        line=dict(color=TEAL, width=3),
        marker=dict(size=10, color="white", line=dict(color=TEAL, width=2.5)),
        name="Điểm sức bền (KNN)",
        hovertemplate="Buổi %{x}<br>Sức bền: %{y:.1f}s<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text="Điểm sức bền cơ (thời điểm khởi phát mỏi) qua các buổi",
            font=dict(size=16, color="#2C3E50"),
        ),
        xaxis_title="Buổi trị liệu",
        yaxis_title="Thời gian tới khi mỏi (s)",
        xaxis=dict(dtick=1),
        yaxis=dict(rangemode="tozero"),
        showlegend=False,
    )
    _chart(fig, height=420)

    # Data table
    display_df = pd.DataFrame([{
        "Buổi": r.session,
        "Điểm sức bền (s)": round(r.endurance_sec, 1),
        "% cửa sổ chưa mỏi": round(r.pct_nonfatigue, 1),
    } for r in results])
    st.dataframe(
        display_df.style.format({
            "Điểm sức bền (s)": "{:.1f}",
            "% cửa sổ chưa mỏi": "{:.1f}",
        }),
        width="stretch", hide_index=True,
    )


# ═══════════════════════════════════════════════════════════════════════
#  PANEL 3 — RECOVERY METRICS  (Chart 3)
# ═══════════════════════════════════════════════════════════════════════
def _render_recovery_metrics() -> None:
    st.markdown("### 📉 Chỉ số phục hồi (dữ liệu PhysioMio)")
    st.caption(
        "RMS tăng = cơ huy động tốt hơn. MDF dịch lên = dẫn truyền thần kinh "
        "cải thiện. Symmetry Index = RMS tay liệt / RMS tay lành × 100%."
    )

    patients = rdl.list_patients()
    if not patients:
        st.error(
            f"Không tìm thấy dữ liệu trong `{rc.PHYSIOMIO_DIR}`. Chạy "
            "`usecase2_references/simulate_semg_rehab.py` để sinh dữ liệu mô phỏng."
        )
        return

    patient = st.selectbox("Bệnh nhân", patients, key="uc2_recovery_patient")

    @st.cache_data(show_spinner="Đang tính RMS/MDF/Symmetry từng buổi…")
    def _get_trend(patient: str) -> pd.DataFrame:
        return rdl.compute_recovery_trend(patient)

    trend_df = _get_trend(patient)
    if trend_df.empty:
        st.info(f"Không có session impaired_arm cho {patient}.")
        return

    baseline_rms = trend_df["baseline_rms"].iloc[0]
    baseline_mdf = trend_df["baseline_mdf"].iloc[0]

    col_sym, col_mdf = st.columns(2)

    # ── Left: Symmetry Index ──
    with col_sym:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trend_df["Buổi"], y=trend_df["Symmetry (%)"],
            fill="tozeroy", fillcolor=TEAL_LIGHT,
            mode="lines+markers",
            line=dict(color=TEAL, width=3),
            marker=dict(size=9, color="white", line=dict(color=TEAL, width=2.5)),
            name="Symmetry (%)",
            hovertemplate="Buổi %{x}<br>Symmetry: %{y:.1f}%<extra></extra>",
        ))
        fig.add_hline(y=100, line_dash="dash", line_color=GRAY_THRESHOLD,
                      line_width=1, annotation_text="100% (tay lành)",
                      annotation_position="top left",
                      annotation_font_color=GRAY_THRESHOLD)
        fig.update_layout(
            title="Chỉ số phục hồi (đối xứng RMS lành-liệt, %)",
            xaxis_title="Buổi trị liệu",
            yaxis_title="Symmetry Index (%)",
            xaxis=dict(dtick=1),
            yaxis=dict(range=[0, 110]),
            showlegend=False,
        )
        _chart(fig, height=380)

    # ── Right: MDF trend ──
    with col_mdf:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trend_df["Buổi"], y=trend_df["MDF"],
            mode="lines+markers",
            line=dict(color=CORAL, width=3),
            marker=dict(size=9, symbol="square", color="white",
                        line=dict(color=CORAL, width=2.5)),
            name="MDF (Hz)",
            hovertemplate="Buổi %{x}<br>MDF: %{y:.1f} Hz<extra></extra>",
        ))
        if baseline_mdf is not None:
            fig.add_hline(y=baseline_mdf, line_dash="dash",
                          line_color=GREEN_BASELINE, line_width=1,
                          annotation_text="baseline tay lành",
                          annotation_position="top left",
                          annotation_font_color=GREEN_BASELINE)
        fig.update_layout(
            title="MDF (Hz) — dịch lên khi phục hồi",
            xaxis_title="Buổi trị liệu",
            yaxis_title="MDF (Hz)",
            xaxis=dict(dtick=1),
            showlegend=False,
        )
        _chart(fig, height=380)

    # Data table
    st.dataframe(
        trend_df[["Buổi", "RMS", "MDF", "Symmetry (%)"]].style.format({
            "RMS": "{:.2f}", "MDF": "{:.1f}", "Symmetry (%)": "{:.1f}",
        }),
        width="stretch", hide_index=True,
    )


# ═══════════════════════════════════════════════════════════════════════
#  PANEL 4 — SIGNAL EXPLORER
# ═══════════════════════════════════════════════════════════════════════
def _render_signal_explorer() -> None:
    st.markdown("### 🔬 Khám phá tín hiệu 1 buổi / 1 kênh")

    patients = rdl.list_patients()
    if not patients:
        return

    c1, c2 = st.columns(2)
    patient = c1.selectbox("Bệnh nhân", patients, key="uc2_sig_patient")
    arm = c2.selectbox("Tay", rc.ARMS, index=rc.ARMS.index("impaired_arm"),
                       key="uc2_sig_arm")

    sessions = rdl.list_sessions(patient, arm)
    if not sessions:
        st.info(f"Không có session nào cho {patient}/{arm}.")
        return

    sel_session = st.selectbox(
        "Buổi", sessions,
        format_func=lambda s: f"Buổi {s.session_no} ({s.path.name})",
        key="uc2_sig_session",
    )
    df = rdl.load_session(sel_session.path)
    movements = list(df["movement_type"].unique())
    sel_movement = st.selectbox("Cử chỉ", movements, key="uc2_sig_movement")
    sel_channel = st.selectbox("Kênh", rc.CHANNEL_COLUMNS, key="uc2_sig_channel")

    seg = df[df["movement_type"] == sel_movement]
    x = seg[sel_channel].to_numpy()
    t = np.arange(len(x)) / rc.FS

    fig = px.line(
        x=t, y=x,
        labels={"x": "Thời gian (s)", "y": "Biên độ (µV)"},
        title=f"Buổi {sel_session.session_no} · {sel_movement} · {sel_channel}",
    )
    fig.update_traces(line=dict(width=1, color=TEAL))
    _chart(fig, height=350)
