# app_realtime_session.py
"""Standalone Streamlit demo: a PT session assembled from real per-subject
sEMG recordings (ascending %MVC), played back near-real-time with live
muscle-fatigue monitoring (Usecase 1 v2,
plans/sEMGxAI_DeXuatDemoUsecase.html).

Run from the project root:
    streamlit run app_realtime_session.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config                        # noqa: E402
from src import pipeline as pl                # noqa: E402
from src import realtime_session as rts       # noqa: E402
from src import session_builder as sb         # noqa: E402
from src import visualization_realtime as vc  # noqa: E402

st.set_page_config(page_title="Demo Giám sát Mỏi cơ Near-Real-Time — PHCN", layout="wide")

MUSCLE_OPTIONS = {
    "Cơ nhị đầu tay (Biceps brachii)": True,
    "Cơ tứ đầu đùi (Quadriceps)": False,
    "Cơ delta vai (Deltoid)": False,
}
SUBJECT = config.TEST_SUBJECT
STEP_INTERVAL_SEC = 1.0
DISPLAY_WINDOW_SEC = 1.5
ENVELOPE_WINDOW_SEC = 0.05
HOVERLABEL = dict(font=dict(size=20, color="white", family="Arial"),
                  bgcolor="#1E1E1E", bordercolor="#1E1E1E")


def chart(fig, **kwargs) -> None:
    fig.update_layout(hoverlabel=HOVERLABEL, hovermode="closest")
    kwargs.setdefault("width", "stretch")
    st.plotly_chart(fig, **kwargs)


@st.cache_data(show_spinner="Đang huấn luyện / tải mô hình…")
def get_results():
    return pl.run_pipeline(use_cache=True).results


@st.cache_data(show_spinner="Đang ghép dữ liệu buổi tập…")
def get_session(subject: int, channel: int):
    return sb.build_session_signal(subject, channel)


st.title("🏥 Giám sát Mỏi cơ Near-Real-Time — Buổi tập ghép từ dữ liệu thật")
st.caption(
    "Bệnh nhân phục hồi chức năng chi trên · buổi tập được ghép từ các bản ghi "
    "sEMG thật của cùng một đối tượng nghiên cứu ở nhiều mức %MVC (10→90%), "
    "phát lại nén thời gian để mô phỏng một buổi tập liên tục từ lúc bắt đầu "
    "đến khi cơ mỏi và sau mỏi."
)

col_sel1, col_sel2, col_sel3 = st.columns(3)
with col_sel1:
    muscle = st.selectbox(
        "Nhóm cơ", list(MUSCLE_OPTIONS.keys()),
        format_func=lambda m: m if MUSCLE_OPTIONS[m] else f"{m} (Sắp có)",
    )

if not MUSCLE_OPTIONS[muscle]:
    st.warning(
        f"**{muscle}** chưa có dữ liệu để mô phỏng — hiện chỉ hỗ trợ "
        "Cơ nhị đầu tay (Biceps brachii). Chọn nhóm cơ đó để chạy demo."
    )
    st.stop()

valid_channels = sb.common_valid_channels(SUBJECT)
with col_sel2:
    channel = st.selectbox("Kênh EMG", valid_channels, format_func=lambda i: f"Kênh {i}")
with col_sel3:
    n_steps = st.slider("Số bước phát", min_value=30, max_value=60, value=50)

col_btn1, col_btn2 = st.columns([1, 1])
start = col_btn1.button("▶ Bắt đầu mô phỏng buổi tập", type="primary")
reset = col_btn2.button("⟲ Reset")

if "session_started" not in st.session_state:
    st.session_state["session_started"] = False
if reset:
    st.session_state["session_started"] = False
    st.rerun()

signal, segments = get_session(SUBJECT, channel)
mvc_sequence = " → ".join(f"{s.mvc}%" for s in segments)
st.caption(f"Thứ tự giai đoạn (%MVC): {mvc_sequence}")

if not st.session_state["session_started"] and not start:
    st.caption("🔍 Đang hiển thị dữ liệu xem trước — nhấn "
               "\"Bắt đầu mô phỏng buổi tập\" để chạy trực tiếp theo thời gian thực")
    preview_len = int(DISPLAY_WINDOW_SEC * config.FS)
    preview_x = signal[:preview_len]
    preview_t = np.arange(len(preview_x)) / config.FS
    chart(vc.build_waveform_figure(preview_t, preview_x, preview_x))

if start:
    st.session_state["session_started"] = True
    results = get_results()
    assessments = rts.assess_segments(signal, segments, results)
    steps = rts.build_playback_steps(
        signal, segments, n_steps,
        display_window_sec=DISPLAY_WINDOW_SEC,
        envelope_window_sec=ENVELOPE_WINDOW_SEC,
        fs=config.FS,
    )

    ph_wave = st.empty()
    ph_trend = st.empty()
    col_g1, col_g2 = st.columns([1, 1])
    ph_phase = col_g1.empty()
    ph_badge = col_g2.empty()
    ph_status = st.empty()
    ph_table = st.empty()

    mvc_hist: list[int] = []
    rms_hist: list[float] = []
    mdf_hist: list[float] = []
    seen_segments: set[int] = set()

    for step in steps:
        with ph_wave.container():
            st.markdown("### ① Tín hiệu thô vs. sau khử nhiễu — vị trí hiện tại")
            chart(vc.build_waveform_figure(step.window_t, step.window_raw, step.window_processed))

        seg_idx = step.segment_idx
        if seg_idx not in seen_segments:
            seen_segments.add(seg_idx)
            newly_seen = assessments[seg_idx]
            mvc_hist.append(newly_seen.segment.mvc)
            rms_hist.append(newly_seen.rms)
            mdf_hist.append(newly_seen.mdf)

        with ph_trend.container():
            st.markdown("### ② Xu hướng RMS & MDF theo giai đoạn (%MVC)")
            chart(vc.build_trend_figure(mvc_hist, rms_hist, mdf_hist))

        current = assessments[seg_idx]
        ph_phase.metric("Giai đoạn hiện tại", f"{current.segment.mvc}% MVC",
                         help=f"File: {current.segment.file.condition}")

        knn_pred = next(p for p in current.predictions if p["model"] == "KNN")
        status = "Mỏi" if knn_pred["pred"] == 1 else "Không mỏi"
        with ph_badge.container():
            chart(vc.build_status_badge(status, knn_pred["p_fatigue"]))

        if status == "Không mỏi":
            ph_status.success(f"Trạng thái (model KNN): {status}")
        else:
            ph_status.error(
                f"Trạng thái (model KNN): {status} — ⚠ Khuyến nghị: giảm cường độ "
                "hoặc cho bệnh nhân nghỉ giữa hiệp.")

        with ph_table.container():
            st.markdown("#### P(Fatigue) theo từng model (giai đoạn hiện tại)")
            st.table([
                {
                    "Model": p["model"],
                    "Dự đoán": config.LABEL_NAMES[p["pred"]],
                    "P(Fatigue)": f"{p['p_fatigue']:.1%}" if p["p_fatigue"] is not None else "—",
                }
                for p in current.predictions
            ])

        time.sleep(STEP_INTERVAL_SEC)

st.caption(
    "Dữ liệu là các bản ghi sEMG thật của cùng một đối tượng nghiên cứu "
    "(subject 9) ở nhiều mức %MVC, ghép nối theo thứ tự tăng dần và phát lại "
    "nén thời gian (near-real-time) để minh hoạ một buổi tập liên tục — không "
    "phải đo trực tiếp liên tục trên bệnh nhân."
)
