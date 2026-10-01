# app_realtime_session.py
"""Standalone Streamlit demo: a PT session assembled from real per-subject
sEMG recordings (ascending %MVC), played back near-real-time with live
muscle-fatigue monitoring (Usecase 1 v2,
plans/sEMGxAI_DeXuatDemoUsecase.html), plus a Usecase 2 tab (rehab recovery
tracking on the physiomio dataset, src/rehab/).

# Run from the project root:
#     streamlit run app_realtime_session.py
# (Trigger hot reload)
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

import importlib

from src import config                        # noqa: E402
from src import pipeline as pl                # noqa: E402
from src import realtime_html as rth          # noqa: E402
from src import realtime_session as rts       # noqa: E402
from src import session_builder as sb         # noqa: E402
from src.rehab import streamlit_view as rehab_view  # noqa: E402

importlib.reload(rth) # Force reload to apply recent UI changes

st.set_page_config(page_title="Demo Giám sát Mỏi cơ Near-Real-Time — PHCN", layout="wide")

MUSCLE_OPTIONS = {
    "Cơ nhị đầu tay (Biceps brachii)": True,
    "Cơ tứ đầu đùi (Quadriceps)": False,
    "Cơ delta vai (Deltoid)": False,
}
SUBJECT = config.TEST_SUBJECT
DISPLAY_WINDOW_SEC = 3.0
DEMO_MODEL = "LogisticRegression"  # Usecase 1 demo pins this model instead of
                     # auto-picking the highest-F1 one (see rts.best_model_name)
                     # — chosen after 4-subject LOSO check showed it's both the
                     # strongest AND most stable (F1 0.830±0.101, Acc 0.878±0.076
                     # vs SVM's 0.665±0.210 / 0.695±0.242). Falls
                     # back to auto-pick if this name isn't among results.


@st.cache_data(show_spinner="Đang huấn luyện / tải mô hình…")
def get_results():
    return pl.run_pipeline(use_cache=True).results


@st.cache_data(show_spinner="Đang ghép dữ liệu buổi tập…")
def get_session(subject: int):
    return sb.build_session_signal_avg(subject)


@st.cache_data(show_spinner="Đang tải danh sách kênh hợp lệ…")
def get_valid_channels(subject: int):
    return sb.common_valid_channels(subject)


@st.cache_data(show_spinner="Đang đánh giá mỏi cho từng kênh (64 kênh)…")
def get_channel_preds(subject: int, best_model: str):
    results = get_results()
    _, segments = get_session(subject)
    valid_channels = get_valid_channels(subject)
    model_result = next(r for r in results if r.name == best_model)
    return rts.assess_channel_grid(segments, valid_channels, model_result)


@st.cache_data(show_spinner="Đang ước lượng vận tốc dẫn truyền (CV) thật…")
def get_cv_series(subject: int, trim_bounds: tuple[tuple[int, int], ...]):
    return rts.compute_cv_series(subject, trim_bounds=list(trim_bounds))


def _segment_label(seg) -> str:
    suffix = " (sau mỏi)" if sb.is_post_fatigue(seg.file.condition) else ""
    return f"{seg.mvc}%{suffix}"


def _render_realtime_tab() -> None:
    col_sel1, col_btn1, col_btn2 = st.columns(
        [2, 1.4, 1], vertical_alignment="bottom",
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
        return

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

    signal, segments = get_session(SUBJECT)

    if start:
        st.session_state["session_started"] = True
        st.session_state.pop("_rt_html", None)

    if st.session_state["session_started"]:
        if "_rt_html" not in st.session_state:
            results = get_results()
            assessments = rts.assess_segments(signal, segments, results)
            best_model = (
                DEMO_MODEL if any(r.name == DEMO_MODEL for r in results)
                else rts.best_model_name(results)
            )
            channel_preds = get_channel_preds(SUBJECT, best_model)

            # Display-only: trims each file's low-amplitude contraction
            # ramp-in/ramp-out so the concatenated waveform/trend chart reads
            # as one continuous session instead of dipping at every %MVC
            # boundary. Classification above already ran on the untrimmed
            # `signal`/`segments`, so predictions are unaffected.
            trim_bounds = sb.trim_bounds_for_display(signal, segments, config.FS)
            display_signal, display_segments = sb.trim_for_display(
                signal, segments, config.FS,
            )

            # Real MFCV(t), trimmed with the exact same per-segment bounds
            # as the waveform/RMS/MDF above so the CV trend line lands on
            # the same time axis instead of drifting against it.
            cv_series = get_cv_series(SUBJECT, tuple(trim_bounds))

            segments_info = []
            for seg, disp_seg, assess, ch_preds in zip(
                segments, display_segments, assessments, channel_preds,
            ):
                segments_info.append({
                    "start": round(disp_seg.start_sample / config.FS, 3),
                    "end": round(disp_seg.end_sample / config.FS, 3),
                    "label": f"{seg.mvc}%",
                    "isPostFatigue": sb.is_post_fatigue(seg.file.condition),
                    "rms": round(float(assess.rms), 4),
                    "mdf": round(float(assess.mdf), 2),
                    "predictions": assess.predictions,
                    "channelPreds": ch_preds,
                })
            model_metrics = [
                {
                    "name": r.name,
                    "accuracy": round(float(r.accuracy), 4),
                    "precision": round(float(r.precision), 4),
                    "recall": round(float(r.recall), 4),
                    "f1": round(float(r.f1), 4),
                    "cvAccuracy": round(float(r.cv_accuracy), 4),
                    "auc": round(float(r.auc), 4) if r.auc is not None else None,
                    "confusion": r.confusion.tolist(),
                }
                for r in results
            ]

            total_sec = len(display_signal) / config.FS
            st.session_state["_rt_html"] = rth.build_realtime_html(
                display_signal, segments_info, config.FS,
                channel_layout=config.CHANNEL_LAYOUT,
                display_window_sec=DISPLAY_WINDOW_SEC,
                playback_duration_sec=total_sec,
                model_metrics=model_metrics,
                best_model=best_model,
                cv_series=cv_series,
            )

        st.iframe(st.session_state["_rt_html"], height="content")


st.title("Giám sát Mỏi cơ")

tab_realtime, tab_rehab = st.tabs(
    ["Giám sát Mỏi cơ (Real-time)", "Giám sát phục hồi cơ"]
)

with tab_realtime:
    _render_realtime_tab()

with tab_rehab:
    rehab_view.render()
