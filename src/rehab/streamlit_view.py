"""Streamlit UI for Usecase 2 (rehab recovery tracking) — rendered as one tab
inside app_realtime_session.py's `st.tabs(...)`, not a standalone page.

Kept as a plain render function (not a `pages/*.py` multipage entry) so it
can be embedded inside the existing realtime-monitoring app instead of
spawning a second sidebar page.
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

HOVERLABEL = dict(font=dict(size=16, color="white", family="Arial"),
                  bgcolor="#1E1E1E", bordercolor="#1E1E1E")


def _chart(fig, **kwargs):
    fig.update_layout(hoverlabel=HOVERLABEL, hovermode="closest")
    kwargs.setdefault("width", "stretch")
    st.plotly_chart(fig, **kwargs)


def render() -> None:
    """Render the full Usecase 2 tab content into the current Streamlit container."""
    st.warning(rc.DISCLAIMER)
    st.caption("Bài toán khác Usecase 1 (giám sát mỏi cơ real-time): đây là "
               "theo dõi xu hướng **phục hồi cơ qua nhiều buổi tập PHCN**, dùng "
               "dữ liệu sEMG theo schema PhysioMio "
               f"(`channel_01..channel_{rc.N_CHANNELS}` + `movement_type`, ~{rc.FS} Hz).")

    tab_recovery, tab_endurance = st.tabs(
        ["Theo dõi phục hồi (dữ liệu physiomio)",
         "Minh hoạ điểm sức bền (mô phỏng bổ sung)"]
    )

    with tab_recovery:
        _render_recovery_tab()
    with tab_endurance:
        _render_endurance_tab()


def _render_recovery_tab() -> None:
    patients = rdl.list_patients()
    if not patients:
        st.error(f"Không tìm thấy dữ liệu trong `{rc.PHYSIOMIO_DIR}`. Chạy "
                 "`usecase2_references/simulate_semg_rehab.py` để sinh dữ liệu mô phỏng.")
        return

    c1, c2 = st.columns(2)
    patient = c1.selectbox("Bệnh nhân", patients, key="rehab_patient")
    arm = c2.selectbox("Tay", rc.ARMS, index=rc.ARMS.index("impaired_arm"), key="rehab_arm")

    sessions = rdl.list_sessions(patient, arm)
    if not sessions:
        st.info(f"Không có session nào cho {patient}/{arm}.")
        return

    st.markdown(f"**{len(sessions)} buổi tập** cho `{patient}/{arm}`")
    session_table = pd.DataFrame([
        {"Buổi": s.session_no, "File": s.path.name} for s in sessions
    ])
    st.dataframe(session_table, width="stretch", hide_index=True)

    @st.cache_data(show_spinner="Đang tính RMS/MDF từng buổi…")
    def _trend(patient: str, arm: str) -> pd.DataFrame:
        rows = []
        for s in rdl.list_sessions(patient, arm):
            df = rdl.load_session(s.path)
            rms, mdf = rdl.session_rms_mdf(df)
            rows.append({"Buổi": s.session_no, "RMS": rms, "MDF": mdf})
        return pd.DataFrame(rows)

    trend_df = _trend(patient, arm)

    baseline_rows = []
    for s in rdl.list_sessions(patient, "healthy_arm"):
        df = rdl.load_session(s.path)
        baseline_rows.append(rdl.session_rms_mdf(df))
    baseline_rms = float(np.mean([r[0] for r in baseline_rows])) if baseline_rows else None
    baseline_mdf = float(np.mean([r[1] for r in baseline_rows])) if baseline_rows else None

    st.markdown("### Xu hướng phục hồi qua các buổi")
    st.caption("RMS tăng dần và MDF dịch lên qua các buổi phản ánh cơ "
               "huy động tốt hơn (RMS) và dẫn truyền thần kinh cải "
               "thiện (MDF) — hai dấu hiệu phục hồi kinh điển. Đường "
               "nét đứt là baseline tay lành cùng bệnh nhân.")
    cc1, cc2 = st.columns(2)
    with cc1:
        fig = px.line(trend_df, x="Buổi", y="RMS", markers=True,
                      title="RMS trung bình qua các buổi (µV)")
        if baseline_rms is not None:
            fig.add_hline(y=baseline_rms, line_dash="dash", line_color="#27AE60",
                          annotation_text="baseline tay lành")
        _chart(fig)
    with cc2:
        fig = px.line(trend_df, x="Buổi", y="MDF", markers=True,
                      title="MDF trung bình qua các buổi (Hz)")
        if baseline_mdf is not None:
            fig.add_hline(y=baseline_mdf, line_dash="dash", line_color="#27AE60",
                          annotation_text="baseline tay lành")
        _chart(fig)

    st.markdown("### Xem tín hiệu 1 buổi / 1 kênh")
    sel_session = st.selectbox(
        "Buổi", sessions, format_func=lambda s: f"Buổi {s.session_no} ({s.path.name})",
        key="rehab_session",
    )
    df = rdl.load_session(sel_session.path)
    movements = list(df["movement_type"].unique())
    sel_movement = st.selectbox("Cử chỉ", movements, key="rehab_movement")
    sel_channel = st.selectbox("Kênh", rc.CHANNEL_COLUMNS, key="rehab_channel")

    seg = df[df["movement_type"] == sel_movement]
    x = seg[sel_channel].to_numpy()
    t = np.arange(len(x)) / rc.FS
    fig = px.line(x=t, y=x, labels={"x": "thời gian (s)", "y": "biên độ"},
                  title=f"Buổi {sel_session.session_no} · {sel_movement} · {sel_channel}")
    fig.update_traces(line=dict(width=1))
    _chart(fig)


def _render_endurance_tab() -> None:
    st.caption("⚠️ Phần này **không đọc dữ liệu trong `dataset/physiomio/`** — "
               "đây là mô phỏng độc lập minh hoạ khái niệm 'điểm sức bền = "
               "thời điểm khởi phát mỏi', dùng tín hiệu co cơ duy trì (khác "
               "với tín hiệu theo cử chỉ ngắn ở tab bên cạnh).")

    @st.cache_data(show_spinner="Đang mô phỏng và huấn luyện KNN…")
    def _run_endurance():
        results, f1 = end.run()
        return pd.DataFrame([r.__dict__ for r in results]), f1

    endurance_df, f1 = _run_endurance()
    st.metric("F1 (KNN, tập test mô phỏng)", f"{f1:.3f}")

    display_df = endurance_df.rename(columns={
        "session": "Buổi", "t_half": "t_half (s, thật)",
        "endurance_sec": "Điểm sức bền (s, KNN dự đoán)",
        "pct_nonfatigue": "% cửa sổ chưa mỏi",
    })
    st.dataframe(
        display_df.style.format({
            "t_half (s, thật)": "{:.1f}",
            "Điểm sức bền (s, KNN dự đoán)": "{:.1f}",
            "% cửa sổ chưa mỏi": "{:.1f}",
        }),
        width="stretch", hide_index=True,
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=endurance_df["session"], y=endurance_df["endurance_sec"],
                             mode="lines+markers", name="Điểm sức bền (KNN)"))
    fig.add_trace(go.Scatter(x=endurance_df["session"], y=endurance_df["t_half"],
                             mode="lines+markers", name="t_half thật", line=dict(dash="dot")))
    fig.update_layout(xaxis_title="Buổi", yaxis_title="giây",
                      title="Điểm sức bền tăng dần qua các buổi → phục hồi", height=420)
    _chart(fig)
