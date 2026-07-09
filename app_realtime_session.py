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
STEP_INTERVAL_SEC = 0.3
DISPLAY_WINDOW_SEC = 1.5
ENVELOPE_WINDOW_SEC = 0.05
HOVERLABEL = dict(font=dict(size=20, color="white", family="Arial"),
                  bgcolor="#1E1E1E", bordercolor="#1E1E1E")


def chart(fig, target=st, key: str | None = None, **kwargs) -> None:
    """Render a figure via `target` (the `st` module, or an `st.empty()`
    placeholder — both expose `.plotly_chart`), applying shared hover styling.
    """
    fig.update_layout(hoverlabel=HOVERLABEL, hovermode="closest")
    kwargs.setdefault("width", "stretch")
    target.plotly_chart(fig, key=key, **kwargs)


@st.cache_data(show_spinner="Đang huấn luyện / tải mô hình…")
def get_results():
    return pl.run_pipeline(use_cache=True).results


@st.cache_data(show_spinner="Đang ghép dữ liệu buổi tập…")
def get_session(subject: int, channel: int):
    return sb.build_session_signal(subject, channel)


@st.cache_data(show_spinner="Đang tải danh sách kênh hợp lệ…")
def get_valid_channels(subject: int):
    return sb.common_valid_channels(subject)


st.title("🏥 Giám sát Mỏi cơ Near-Real-Time")
st.caption(
    "Buổi tập ghép từ các bản ghi sEMG thật của cùng một đối tượng nghiên cứu "
    "ở nhiều mức %MVC (10→90%), phát lại nén thời gian để mô phỏng một buổi "
    "tập liên tục từ lúc bắt đầu đến khi cơ mỏi và sau mỏi."
)

col_sel1, col_sel2, col_sel3, col_btn1, col_btn2 = st.columns([2, 2, 2, 1.4, 1])
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
with col_sel3:
    n_steps = st.slider(
        "Số bước phát", min_value=60, max_value=300, value=180,
        help="Càng nhiều bước, mỗi bước càng nhích ít trên tín hiệu gốc "
             "→ hình ảnh chạy mượt/liên tục hơn.",
    )
with col_btn1:
    st.write("")
    start = st.button("▶ Bắt đầu mô phỏng buổi tập", type="primary")
with col_btn2:
    st.write("")
    reset = st.button("⟲ Reset")

if "session_started" not in st.session_state:
    st.session_state["session_started"] = False
if reset:
    st.session_state["session_started"] = False
    st.rerun()

def segment_label(seg) -> str:
    suffix = " (sau mỏi)" if sb.is_post_fatigue(seg.file.condition) else ""
    return f"{seg.mvc}%{suffix}"


signal, segments = get_session(SUBJECT, channel)
mvc_sequence = " → ".join(segment_label(s) for s in segments)
st.caption(f"Thứ tự giai đoạn: {mvc_sequence}")

if not st.session_state["session_started"] and not start:
    st.caption("🔍 Đang hiển thị dữ liệu xem trước — nhấn "
               "\"Bắt đầu mô phỏng buổi tập\" để chạy trực tiếp theo thời gian thực")
    preview_len = int(DISPLAY_WINDOW_SEC * config.FS)
    preview_x = signal[:preview_len]
    preview_t = np.arange(len(preview_x)) / config.FS
    chart(vc.build_waveform_figure(preview_t, preview_x, preview_x), key="preview_waveform")

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

    col_chart1, col_chart2 = st.columns([1, 1])
    ph_wave = col_chart1.empty()
    ph_trend = col_chart2.empty()

    col_g1, col_g2, col_g3 = st.columns([1, 1, 2])
    ph_phase = col_g1.empty()
    ph_badge = col_g2.empty()
    ph_status = col_g3.empty()

    ph_table_toggle = st.expander("Chi tiết P(Fatigue) theo từng model (giai đoạn hiện tại)", expanded=False)
    ph_table = ph_table_toggle.empty()

    label_hist: list[str] = []
    rms_hist: list[float] = []
    mdf_hist: list[float] = []
    seen_segments: set[int] = set()

    for step in steps:
        chart(vc.build_waveform_figure(step.window_t, step.window_raw, step.window_processed),
              target=ph_wave, key=f"waveform_{step.step_idx}")

        seg_idx = step.segment_idx
        if seg_idx not in seen_segments:
            seen_segments.add(seg_idx)
            newly_seen = assessments[seg_idx]
            label_hist.append(segment_label(newly_seen.segment))
            rms_hist.append(newly_seen.rms)
            mdf_hist.append(newly_seen.mdf)

        chart(vc.build_trend_figure(label_hist, rms_hist, mdf_hist),
              target=ph_trend, key=f"trend_{step.step_idx}")

        current = assessments[seg_idx]
        ph_phase.metric("Giai đoạn hiện tại", segment_label(current.segment),
                         help=f"File: {current.segment.file.condition}")

        knn_pred = next(p for p in current.predictions if p["model"] == "KNN")
        status = "Mỏi" if knn_pred["pred"] == 1 else "Không mỏi"
        chart(vc.build_status_badge(status, knn_pred["p_fatigue"]),
              target=ph_badge, key=f"badge_{step.step_idx}")

        if status == "Không mỏi":
            ph_status.success(f"Trạng thái (model KNN): {status}")
        else:
            ph_status.error(
                f"Trạng thái (model KNN): {status} — ⚠ Khuyến nghị: giảm cường độ "
                "hoặc cho bệnh nhân nghỉ giữa hiệp.")

        ph_table.table([
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
