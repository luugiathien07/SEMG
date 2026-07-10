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
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config                        # noqa: E402
from src import pipeline as pl                # noqa: E402
from src import realtime_html as rth          # noqa: E402
from src import realtime_session as rts       # noqa: E402
from src import session_builder as sb         # noqa: E402

st.set_page_config(page_title="Demo Giám sát Mỏi cơ Near-Real-Time — PHCN", layout="wide")

MUSCLE_OPTIONS = {
    "Cơ nhị đầu tay (Biceps brachii)": True,
    "Cơ tứ đầu đùi (Quadriceps)": False,
    "Cơ delta vai (Deltoid)": False,
}
SUBJECT = config.TEST_SUBJECT
DISPLAY_WINDOW_SEC = 3.0


@st.cache_data(show_spinner="Đang huấn luyện / tải mô hình…")
def get_results():
    return pl.run_pipeline(use_cache=True).results


@st.cache_data(show_spinner="Đang ghép dữ liệu buổi tập…")
def get_session(subject: int, channel: int):
    return sb.build_session_signal(subject, channel)


@st.cache_data(show_spinner="Đang tải danh sách kênh hợp lệ…")
def get_valid_channels(subject: int):
    return sb.common_valid_channels(subject)


st.title("Giám sát Mỏi cơ")

col_sel1, col_sel2, col_btn1, col_btn2 = st.columns(
    [2, 2, 1.4, 1], vertical_alignment="bottom",
)
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

valid_channels = get_valid_channels(SUBJECT)
with col_sel2:
    channel = st.selectbox("Kênh EMG", valid_channels, format_func=lambda i: f"Kênh {i}")
with col_btn1:
    start = st.button("Bắt đầu mô phỏng buổi tập", type="primary")
with col_btn2:
    reset = st.button("Đặt lại")

if "session_started" not in st.session_state:
    st.session_state["session_started"] = False
if reset:
    st.session_state["session_started"] = False
    st.session_state.pop("_rt_html", None)
    st.rerun()


def _segment_label(seg) -> str:
    suffix = " (sau mỏi)" if sb.is_post_fatigue(seg.file.condition) else ""
    return f"{seg.mvc}%{suffix}"


signal, segments = get_session(SUBJECT, channel)

if start:
    st.session_state["session_started"] = True
    st.session_state.pop("_rt_html", None)

if st.session_state["session_started"]:
    if "_rt_html" not in st.session_state:
        results = get_results()
        assessments = rts.assess_segments(signal, segments, results)
        segments_info = []
        for seg, assess in zip(segments, assessments):
            segments_info.append({
                "start": round(seg.start_sample / config.FS, 3),
                "end": round(seg.end_sample / config.FS, 3),
                "label": f"{seg.mvc}%",
                "isPostFatigue": sb.is_post_fatigue(seg.file.condition),
                "rms": round(float(assess.rms), 4),
                "mdf": round(float(assess.mdf), 2),
                "predictions": assess.predictions,
            })
        total_sec = len(signal) / config.FS
        st.session_state["_rt_html"] = rth.build_realtime_html(
            signal, segments_info, config.FS,
            display_window_sec=DISPLAY_WINDOW_SEC,
            playback_duration_sec=total_sec,
        )

    st.iframe(st.session_state["_rt_html"], height=680)
