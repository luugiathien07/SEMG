"""POC landing page cho buổi gặp Vinmec — sEMG AI Platform.

Cấu trúc 3 trang (sidebar navigation):
  1. Giới thiệu   — bối cảnh, bài toán, pipeline tóm tắt
  2. Demo UC1     — giám sát mỏi cơ real-time (tái sử dụng app_realtime_session logic)
  3. Demo UC2     — theo dõi phục hồi cơ theo phiên (rehab)

Chạy:
    streamlit run poc_vinmec.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

import importlib

from src import config
from src import pipeline as pl
from src import realtime_html as rth
from src import realtime_session as rts
from src import session_builder as sb
from src.rehab import streamlit_view as rehab_view

importlib.reload(rth)

# ── Hằng số demo ────────────────────────────────────────────────────────────
SUBJECT = config.TEST_SUBJECT
DISPLAY_WINDOW_SEC = 3.0
DEMO_MODEL = "LogisticRegression"

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="sEMG AI — POC Vinmec",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS chung ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Sidebar nav buttons */
[data-testid="stSidebar"] .stButton button {
    width: 100%;
    text-align: left;
    background: transparent;
    border: none;
    padding: 0.55rem 0.75rem;
    border-radius: 8px;
    font-size: 0.95rem;
    font-weight: 500;
    color: #e2e8f0;
    transition: background 0.15s;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(255,255,255,0.08);
    color: #fff;
}
/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #0b3b3c 0%, #0f766e 55%, #0891b2 100%);
    border-radius: 16px;
    padding: 40px 48px 36px;
    color: #fff;
    margin-bottom: 28px;
}
.hero-banner h1 { font-size: 2rem; margin: 0 0 10px; line-height: 1.3; }
.hero-banner p  { font-size: 1rem; color: #ccf0ee; margin: 0 0 20px; max-width: 720px; }
.hero-meta { display:flex; gap:32px; flex-wrap:wrap; border-top:1px solid rgba(255,255,255,0.2); padding-top:16px; font-size:0.82rem; color:#a7deda; }
.hero-meta b { display:block; color:#fff; font-size:0.9rem; margin-bottom:2px; }
/* KPI cards */
.kpi-row { display:flex; gap:16px; flex-wrap:wrap; margin:24px 0; }
.kpi-card { flex:1 1 160px; background:#fff; border:1px solid #e2e8f0; border-radius:14px; padding:20px 22px; box-shadow:0 1px 3px rgba(0,0,0,0.05); }
.kpi-card .num { font-size:2rem; font-weight:800; color:#0f766e; line-height:1; }
.kpi-card .lbl { font-size:0.78rem; color:#64748b; margin-top:6px; }
/* Section headings */
.sec-title { font-size:1.2rem; font-weight:700; color:#0b3b3c; margin:0 0 4px; }
.sec-sub   { font-size:0.85rem; color:#64748b; margin:0 0 20px; }
/* Pipeline steps */
.pipe { display:flex; gap:0; overflow-x:auto; margin:20px 0; }
.pipe-step { flex:1 1 0; min-width:130px; background:#f8fafb; border:1px solid #e2e8f0; border-radius:12px; padding:16px 14px; text-align:center; }
.pipe-step .ico { font-size:1.6rem; }
.pipe-step .t   { font-weight:700; font-size:0.82rem; color:#0b3b3c; margin-top:8px; }
.pipe-step .d   { font-size:0.75rem; color:#64748b; margin-top:4px; }
.pipe-arrow { display:flex; align-items:center; justify-content:center; width:32px; color:#0f766e; font-size:1.2rem; flex-shrink:0; }
/* Badge */
.badge { display:inline-block; background:#0f766e; color:#fff; font-size:0.7rem; font-weight:700; padding:3px 10px; border-radius:20px; letter-spacing:.05em; vertical-align:middle; margin-left:8px; }
/* Highlight box */
.hbox { background:#f0fdfa; border-left:4px solid #0f766e; border-radius:0 10px 10px 0; padding:14px 18px; margin:16px 0; font-size:0.9rem; color:#134e4a; }
</style>
""", unsafe_allow_html=True)


# ── Cache helpers ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Đang huấn luyện / tải mô hình…")
def _get_results():
    return pl.run_pipeline(use_cache=True).results


@st.cache_data(show_spinner="Đang ghép dữ liệu buổi tập…")
def _get_session(subject: int):
    return sb.build_session_signal_avg(subject)


@st.cache_data(show_spinner="Đang tải danh sách kênh hợp lệ…")
def _get_valid_channels(subject: int):
    return sb.common_valid_channels(subject)


@st.cache_data(show_spinner="Đang đánh giá mỏi cho 64 kênh…")
def _get_channel_preds(subject: int, best_model: str):
    results = _get_results()
    _, segments = _get_session(subject)
    valid_channels = _get_valid_channels(subject)
    model_result = next(r for r in results if r.name == best_model)
    return rts.assess_channel_grid(segments, valid_channels, model_result)


@st.cache_data(show_spinner="Đang ước lượng vận tốc dẫn truyền (CV)…")
def _get_cv_series(subject: int, trim_bounds):
    return rts.compute_cv_series(subject, trim_bounds=list(trim_bounds))


# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:16px 0 8px;">
        <div style="font-size:2rem;">🧠</div>
        <div style="font-size:1rem;font-weight:700;color:#fff;margin-top:4px;">sEMG AI Platform</div>
        <div style="font-size:0.72rem;color:#94a3b8;margin-top:2px;">POC for Vinmec</div>
    </div>
    <hr style="border-color:rgba(255,255,255,0.1);margin:12px 0;">
    """, unsafe_allow_html=True)

    if "page" not in st.session_state:
        st.session_state["page"] = "intro"

    def _nav(label: str, key: str):
        active = st.session_state["page"] == key
        prefix = "▶ " if active else "   "
        if st.button(f"{prefix}{label}", key=f"nav_{key}"):
            st.session_state["page"] = key
            st.rerun()

    _nav("📋  Giới thiệu & Bài toán", "intro")
    _nav("⚡  Demo UC1 — Giám sát mỏi cơ", "uc1")
    _nav("📈  Demo UC2 — Theo dõi phục hồi", "uc2")

    st.markdown("""
    <hr style="border-color:rgba(255,255,255,0.1);margin:16px 0 8px;">
    <div style="font-size:0.72rem;color:#64748b;padding:0 4px;">
        Dữ liệu: 10 bệnh nhân · 40 file EMG<br>
        Cảm biến: 64-kênh · 2000 Hz<br>
        Mô hình: KNN · SVM · LDA · LR<br>
        <span style="color:#0f766e;font-weight:600;">F1 = 0.977 · AUC = 0.992</span>
    </div>
    """, unsafe_allow_html=True)


page = st.session_state["page"]

# ════════════════════════════════════════════════════════════════════════════
# TRANG 1 — GIỚI THIỆU
# ════════════════════════════════════════════════════════════════════════════
if page == "intro":
    st.markdown("""
    <div class="hero-banner">
        <div style="font-size:0.75rem;font-weight:700;letter-spacing:.08em;opacity:.7;margin-bottom:10px;">
            VINMEC × VINSMART FUTURE — PROOF OF CONCEPT
        </div>
        <h1>Giám sát Mỏi cơ Real-time bằng sEMG & AI</h1>
        <p>Phát hiện sớm mỏi cơ trong buổi tập phục hồi chức năng — theo dõi 64 điện cực,
           phân loại tự động, cảnh báo kịp thời trước khi xảy ra chấn thương.</p>
        <div class="hero-meta">
            <div><b>Đối tác</b>Vinmec International Hospital</div>
            <div><b>Công nghệ</b>Surface EMG · Machine Learning</div>
            <div><b>Phần cứng</b>Noraxon Core EMG — 64 kênh</div>
            <div><b>Trạng thái</b>Proof of Concept · 2026</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    st.markdown("""
    <div class="kpi-row">
        <div class="kpi-card"><div class="num">0.977</div><div class="lbl">F1-score (KNN · Subject 6)</div></div>
        <div class="kpi-card"><div class="num">0.992</div><div class="lbl">AUC — ROC</div></div>
        <div class="kpi-card"><div class="num">64</div><div class="lbl">Điện cực sEMG / buổi đo</div></div>
        <div class="kpi-card"><div class="num">10</div><div class="lbl">Bệnh nhân · 40 file EMG</div></div>
        <div class="kpi-card"><div class="num">2000 Hz</div><div class="lbl">Tần số lấy mẫu</div></div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        st.markdown('<div class="sec-title">Bài toán</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-sub">Tại sao phát hiện mỏi cơ sớm quan trọng trong PHCN?</div>', unsafe_allow_html=True)
        st.markdown("""
        **Mỏi cơ** (muscle fatigue) là sự suy giảm khả năng sinh lực khi hoạt động kéo dài.
        Trong phục hồi chức năng, nếu bệnh nhân tiếp tục tập khi cơ đã mỏi:

        - Bù trừ bằng nhóm cơ sai → lệch trục khớp, thoái hóa
        - Tăng nguy cơ chấn thương tái phát
        - Kỹ thuật viên không thể quan sát liên tục 64 điện cực bằng mắt thường

        **Giải pháp:** Hệ thống AI phân tích tín hiệu sEMG liên tục, tự động cảnh báo
        khi phát hiện dấu hiệu mỏi trên từng vùng cơ.
        """)

        st.markdown('<div class="hbox">🔬 <b>Dấu hiệu sinh lý của mỏi cơ trên sEMG:</b> phổ công suất dịch về tần số thấp (MDF ↓, MNF ↓), biên độ RMS tăng, tốc độ dẫn truyền MFCV giảm.</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="sec-title">Use Cases</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-sub">Hai kịch bản ứng dụng thực tế tại Vinmec</div>', unsafe_allow_html=True)

        st.markdown("""
        **⚡ UC1 — Giám sát real-time trong buổi tập**
        - Phát hiện mỏi từng vùng cơ liên tục
        - Sơ đồ 64 điện cực đổi màu theo trạng thái
        - Hiển thị RMS, MDF, MFCV(t) theo thời gian thực
        - Cổng abstention: im lặng thay vì đoán bừa

        ---

        **📈 UC2 — Theo dõi tiến trình phục hồi**
        - So sánh sức bền giữa các buổi tập (ngày/tuần)
        - Chỉ số đối xứng trái/phải
        - Phát hiện hồi phục chậm hoặc thoái lui
        - Dashboard cho bác sĩ theo dõi nhiều bệnh nhân
        """)

    st.divider()

    # Pipeline
    st.markdown('<div class="sec-title">Pipeline xử lý tín hiệu</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">4 bước từ dữ liệu thô đến kết quả phân loại</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="pipe">
        <div class="pipe-step">
            <div class="ico">📡</div>
            <div class="t">Thu tín hiệu</div>
            <div class="d">64 kênh · 2000 Hz<br>Noraxon Core EMG</div>
        </div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-step">
            <div class="ico">⚙️</div>
            <div class="t">Trích đặc trưng</div>
            <div class="d">14 đặc trưng<br>Thời gian + Tần số</div>
        </div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-step">
            <div class="ico">🎯</div>
            <div class="t">Chọn đặc trưng</div>
            <div class="d">mRMR Top-3<br>MAV · Spectral Max · Kurtosis</div>
        </div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-step">
            <div class="ico">🤖</div>
            <div class="t">Phân loại</div>
            <div class="d">KNN · SVM · LDA<br>Logistic Regression</div>
        </div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-step">
            <div class="ico">✅</div>
            <div class="t">Kết quả</div>
            <div class="d">Mỏi / Bình thường<br>Cảnh báo real-time</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Kết quả
    st.markdown('<div class="sec-title">Kết quả mô hình (LOSO — Leave-One-Subject-Out)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Đánh giá trên Subject 6 (256 mẫu · không dùng trong training)</div>', unsafe_allow_html=True)

    import pandas as pd
    df_results = pd.DataFrame([
        {"Mô hình": "KNN (1-NN) ⭐", "Accuracy": "98.8%", "F1": "0.977", "AUC": "0.992"},
        {"Mô hình": "KNN (5-NN)",    "Accuracy": "99.2%", "F1": "0.985", "AUC": "1.000"},
        {"Mô hình": "SVM (linear)",  "Accuracy": "81.2%", "F1": "0.727", "AUC": "0.994"},
        {"Mô hình": "LDA",           "Accuracy": "83.6%", "F1": "0.750", "AUC": "0.990"},
    ])
    st.dataframe(df_results, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="hbox">
        📄 <b>Tham chiếu:</b> BME 2024 — "Surface EMG-based muscle fatigue detection using machine learning" ·
        KNN F1 ≈ 0.9541, AUC ≈ 0.95 (single favorable LOSO fold). Kết quả POC này tái lập và vượt con số đó
        trên cùng điều kiện đánh giá.
    </div>
    """, unsafe_allow_html=True)

    st.info("👉 Chọn **Demo UC1** hoặc **Demo UC2** trên thanh bên trái để xem demo trực tiếp.")


# ════════════════════════════════════════════════════════════════════════════
# TRANG 2 — DEMO UC1
# ════════════════════════════════════════════════════════════════════════════
elif page == "uc1":
    import io
    import numpy as np
    import pandas as pd
    from src import feature_extraction as fe
    from src import inference as inf
    from src import data_loader as dl

    st.markdown("""
    <div style="background:linear-gradient(135deg,#0b3b3c,#0f766e);border-radius:14px;
                padding:28px 36px 24px;color:#fff;margin-bottom:24px;">
        <div style="font-size:0.72rem;font-weight:700;letter-spacing:.1em;opacity:.7;margin-bottom:8px;">
            USE CASE 1
        </div>
        <h2 style="margin:0 0 8px;font-size:1.6rem;">⚡ Giám sát Mỏi cơ Real-time</h2>
        <p style="margin:0;color:#ccf0ee;font-size:0.9rem;">
            Mô phỏng buổi tập PHCN với dữ liệu sEMG thật · 64 kênh · Cơ nhị đầu tay (Biceps Brachii)
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs: Demo mặc định | Upload file ────────────────────────────────────
    tab_demo, tab_upload = st.tabs(["▶  Demo mặc định (Subject 6)", "📂  Upload file của bạn"])

    # ── Tab 1: Demo mặc định ─────────────────────────────────────────────────
    with tab_demo:
        st.markdown("""
        **Kịch bản:** Bệnh nhân thực hiện co cơ từ 10% → 90% MVC, sau đó bài gây mỏi.
        Hệ thống phân loại trạng thái **Mỏi / Bình thường** liên tục trên 64 kênh điện cực,
        hiển thị RMS, MDF và MFCV(t) theo thời gian thực.
        """)

        col_btn1, col_btn2, _ = st.columns([1.4, 1, 4])
        with col_btn1:
            start = st.button("▶  Bắt đầu mô phỏng", type="primary", use_container_width=True)
        with col_btn2:
            reset = st.button("↺  Đặt lại", use_container_width=True)

        if "uc1_started" not in st.session_state:
            st.session_state["uc1_started"] = False

        if reset:
            st.session_state["uc1_started"] = False
            st.session_state.pop("_uc1_html", None)
            st.rerun()

        if start:
            st.session_state["uc1_started"] = True
            st.session_state.pop("_uc1_html", None)

        if not st.session_state["uc1_started"]:
            st.markdown("""
            <div style="background:#f8fafb;border:2px dashed #cbd5e1;border-radius:14px;
                        padding:40px;text-align:center;color:#64748b;margin-top:16px;">
                <div style="font-size:2.5rem;margin-bottom:12px;">📡</div>
                <div style="font-size:1rem;font-weight:600;margin-bottom:6px;">Sẵn sàng mô phỏng</div>
                <div style="font-size:0.85rem;">Nhấn <b>Bắt đầu mô phỏng</b> để khởi động demo real-time</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            if "_uc1_html" not in st.session_state:
                with st.spinner("Đang chuẩn bị dữ liệu buổi tập…"):
                    results = _get_results()
                    signal, segments = _get_session(SUBJECT)
                    assessments = rts.assess_segments(signal, segments, results)
                    best_model = (
                        DEMO_MODEL if any(r.name == DEMO_MODEL for r in results)
                        else rts.best_model_name(results)
                    )
                    channel_preds = _get_channel_preds(SUBJECT, best_model)

                    trim_bounds = sb.trim_bounds_for_display(signal, segments, config.FS)
                    display_signal, display_segments = sb.trim_for_display(
                        signal, segments, config.FS,
                    )
                    cv_series = _get_cv_series(SUBJECT, tuple(trim_bounds))

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
                    st.session_state["_uc1_html"] = rth.build_realtime_html(
                        display_signal, segments_info, config.FS,
                        channel_layout=config.CHANNEL_LAYOUT,
                        display_window_sec=DISPLAY_WINDOW_SEC,
                        playback_duration_sec=total_sec,
                        model_metrics=model_metrics,
                        best_model=best_model,
                        cv_series=cv_series,
                    )

            st.iframe(st.session_state["_uc1_html"], height="content")

        with st.expander("ℹ️  Đọc các chỉ số trên màn hình"):
            st.markdown("""
            | Chỉ số | Ý nghĩa | Xu hướng khi mỏi |
            |---|---|---|
            | **RMS** | Biên độ tín hiệu (cơ gồng mạnh cỡ nào) | ↑ Tăng |
            | **MDF** | Tần số trung vị (tín hiệu dao động nhanh/chậm) | ↓ Giảm |
            | **MFCV** | Tốc độ dẫn truyền dọc sợi cơ | ↓ Giảm |
            | **%MVC** | Cường độ co cơ / lực tối đa đo được | — Cài đặt bài tập |
            | **Abstention** (ô xám) | Hệ thống từ chối dự đoán khi tín hiệu không đủ tin cậy | — Tính năng an toàn |
            """)

    # ── Tab 2: Upload file ────────────────────────────────────────────────────
    with tab_upload:
        st.markdown("""
        Tải lên một file CSV sEMG để hệ thống phân tích và hiển thị kết quả tương tự demo.
        """)

        with st.expander("📋  Yêu cầu định dạng file CSV"):
            st.markdown("""
            | Mục | Giá trị |
            |---|---|
            | Dấu phân cách | `;` (semicolon) |
            | Header | Không có (no header row) |
            | Cột 0 | Index / timestamp (bỏ qua) |
            | Cột 1 – 64 | Tín hiệu EMG 64 kênh (µV) |
            | Tần số lấy mẫu | 2000 Hz |
            | Tên file (gợi ý) | `Sujet_{id}_{condition}_emg.csv` |

            **Ví dụ tên file để hệ thống tự gán nhãn:**
            - `Sujet_1_10_emg.csv` → Normal (10% MVC ≤ 60)
            - `Sujet_1_fatigue_70_emg.csv` → Fatigue (70% MVC > 60)
            - Tên file không đúng quy ước → nhãn sẽ được chọn thủ công bên dưới
            """)

        uploaded_file = st.file_uploader(
            "Chọn file CSV sEMG",
            type=["csv"],
            key="uc1_upload",
            help="File CSV, phân cách `;`, không có header, cột 0 là index, cột 1–64 là EMG",
        )

        if uploaded_file is not None:
            # Đọc và parse CSV
            try:
                raw_bytes = uploaded_file.read()
                df_up = pd.read_csv(io.BytesIO(raw_bytes), sep=";", header=None, decimal=".")
            except Exception:
                try:
                    df_up = pd.read_csv(io.BytesIO(raw_bytes), sep=",", header=None, decimal=".")
                except Exception as exc:
                    st.error(f"Không đọc được file CSV: {exc}")
                    df_up = None

            if df_up is not None:
                arr = df_up.to_numpy(dtype=float)
                # Bỏ cột 0 (index), lấy tối đa 64 kênh EMG
                emg_mat = arr[:, 1:min(arr.shape[1], 65)]
                n_samples, n_ch = emg_mat.shape

                st.success(
                    f"Đọc thành công: **{n_samples:,} mẫu** · **{n_ch} kênh** "
                    f"· {n_samples / config.FS:.1f} giây @ {config.FS} Hz"
                )

                # Gán nhãn: tự động từ tên file nếu đúng quy ước, ngược lại để user chọn
                auto_info = dl.parse_filename(Path(uploaded_file.name))
                if auto_info is not None:
                    auto_label = auto_info.label
                    label_name = config.LABEL_NAMES[auto_label]
                    st.info(f"Nhãn tự động từ tên file: **{label_name}** (MVC condition: `{auto_info.condition}`)")
                    chosen_label = auto_label
                else:
                    st.warning("Tên file không đúng quy ước `Sujet_{{id}}_{{condition}}_emg.csv`. Chọn nhãn thủ công:")
                    chosen_label = st.radio(
                        "Nhãn của file này",
                        options=[0, 1],
                        format_func=lambda x: config.LABEL_NAMES[x],
                        horizontal=True,
                        key="uc1_upload_label",
                    )

                run_btn = st.button("🔍  Phân tích file", type="primary", key="uc1_run_upload")

                if run_btn or st.session_state.get("_uc1_upload_done") == uploaded_file.name:
                    st.session_state["_uc1_upload_done"] = uploaded_file.name
                    with st.spinner("Đang trích đặc trưng và phân loại…"):
                        results = _get_results()
                        best_model_name = (
                            DEMO_MODEL if any(r.name == DEMO_MODEL for r in results)
                            else rts.best_model_name(results)
                        )

                        # Lấy các kênh hợp lệ (không toàn 0)
                        valid_mask = ~np.all(emg_mat == 0, axis=0)
                        valid_indices = [i for i, v in enumerate(valid_mask) if v]

                        # Tín hiệu trung bình qua các kênh hợp lệ để hiển thị waveform
                        if valid_indices:
                            avg_signal = emg_mat[:, valid_indices].mean(axis=1)
                        else:
                            avg_signal = emg_mat.mean(axis=1)

                        # Phân loại từng kênh
                        ch_grid: list[int | None] = [None] * 64
                        ch_feats: list[dict | None] = [None] * 64
                        for ch_idx in valid_indices:
                            x_ch = emg_mat[:, ch_idx]
                            preds = inf.predict_channel(x_ch, results)
                            ch_grid[ch_idx] = next(
                                (p["pred"] for p in preds if p["model"] == best_model_name),
                                preds[0]["pred"] if preds else None,
                            )
                            ch_feats[ch_idx] = fe.extract_channel_features(x_ch)

                        # Đặc trưng trung bình (cho metric tổng quan)
                        avg_feats = fe.extract_channel_features(avg_signal)
                        avg_preds = inf.predict_channel(avg_signal, results)

                    # ── Hiển thị kết quả ─────────────────────────────────────
                    st.divider()

                    # KPI row
                    n_fatigue = sum(1 for v in ch_grid if v == 1)
                    n_normal = sum(1 for v in ch_grid if v == 0)
                    n_dead = sum(1 for v in ch_grid if v is None)
                    overall_pred = next(
                        (p for p in avg_preds if p["model"] == best_model_name),
                        avg_preds[0] if avg_preds else None,
                    )
                    overall_label = config.LABEL_NAMES.get(overall_pred["pred"], "?") if overall_pred else "?"
                    p_fat_str = f'{overall_pred["p_fatigue"]:.1%}' if overall_pred and overall_pred["p_fatigue"] is not None else "—"

                    label_color = "#dc2626" if overall_label == "Fatigue" else "#16a34a"
                    st.markdown(f"""
                    <div class="kpi-row">
                        <div class="kpi-card">
                            <div class="num" style="color:{label_color};">{overall_label}</div>
                            <div class="lbl">Kết quả tổng thể ({best_model_name})</div>
                        </div>
                        <div class="kpi-card">
                            <div class="num" style="color:{label_color};">{p_fat_str}</div>
                            <div class="lbl">P(Fatigue) trung bình</div>
                        </div>
                        <div class="kpi-card">
                            <div class="num" style="color:#dc2626;">{n_fatigue}</div>
                            <div class="lbl">Kênh Fatigue</div>
                        </div>
                        <div class="kpi-card">
                            <div class="num" style="color:#16a34a;">{n_normal}</div>
                            <div class="lbl">Kênh Normal</div>
                        </div>
                        <div class="kpi-card">
                            <div class="num" style="color:#94a3b8;">{n_dead}</div>
                            <div class="lbl">Kênh chết / không có dữ liệu</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    col_grid, col_table = st.columns([1, 1], gap="large")

                    # ── Sơ đồ 64 kênh ────────────────────────────────────────
                    with col_grid:
                        st.markdown("**Sơ đồ 64 kênh — trạng thái phân loại**")
                        colors = {1: "#fca5a5", 0: "#bbf7d0", None: "#e2e8f0"}
                        labels_map = {1: "F", 0: "N", None: "—"}
                        rows_html = ""
                        for row in config.CHANNEL_LAYOUT:
                            row_cells = ""
                            for ch_num in row:
                                if ch_num is None:
                                    row_cells += '<td style="background:transparent;border:none;"></td>'
                                else:
                                    ch_idx = ch_num - 1
                                    pred_val = ch_grid[ch_idx] if ch_idx < len(ch_grid) else None
                                    bg = colors[pred_val]
                                    lbl = labels_map[pred_val]
                                    row_cells += (
                                        f'<td style="background:{bg};border-radius:4px;'
                                        f'padding:4px 6px;text-align:center;font-size:0.65rem;'
                                        f'font-weight:600;color:#1e293b;border:1px solid #cbd5e1;">'
                                        f'{ch_num}<br><span style="font-size:0.6rem;color:#475569;">{lbl}</span></td>'
                                    )
                            rows_html += f"<tr>{row_cells}</tr>"

                        st.markdown(f"""
                        <table style="border-collapse:separate;border-spacing:3px;font-family:monospace;">
                        {rows_html}
                        </table>
                        <div style="margin-top:8px;font-size:0.75rem;color:#64748b;">
                            <span style="background:#fca5a5;padding:2px 8px;border-radius:4px;margin-right:6px;">F = Fatigue</span>
                            <span style="background:#bbf7d0;padding:2px 8px;border-radius:4px;margin-right:6px;">N = Normal</span>
                            <span style="background:#e2e8f0;padding:2px 8px;border-radius:4px;">— = Chết/thiếu dữ liệu</span>
                        </div>
                        """, unsafe_allow_html=True)

                    # ── Bảng kết quả từng mô hình ─────────────────────────────
                    with col_table:
                        st.markdown("**Kết quả phân loại tín hiệu trung bình (avg kênh)**")
                        pred_rows = []
                        for p in avg_preds:
                            pred_rows.append({
                                "Mô hình": p["model"],
                                "Kết quả": config.LABEL_NAMES.get(p["pred"], "?"),
                                "P(Fatigue)": f'{p["p_fatigue"]:.3f}' if p["p_fatigue"] is not None else "—",
                            })
                        st.dataframe(
                            pd.DataFrame(pred_rows),
                            use_container_width=True,
                            hide_index=True,
                        )

                        st.markdown("**Đặc trưng tín hiệu trung bình**")
                        feat_rows = [
                            {"Đặc trưng": k, "Giá trị": f"{v:.4g}"}
                            for k, v in avg_feats.items()
                        ]
                        st.dataframe(
                            pd.DataFrame(feat_rows),
                            use_container_width=True,
                            hide_index=True,
                        )

        else:
            st.markdown("""
            <div style="background:#f8fafb;border:2px dashed #cbd5e1;border-radius:14px;
                        padding:40px;text-align:center;color:#64748b;margin-top:16px;">
                <div style="font-size:2.5rem;margin-bottom:12px;">📂</div>
                <div style="font-size:1rem;font-weight:600;margin-bottom:6px;">Chưa có file nào được tải lên</div>
                <div style="font-size:0.85rem;">Chọn file CSV sEMG ở trên để bắt đầu phân tích</div>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TRANG 3 — DEMO UC2
# ════════════════════════════════════════════════════════════════════════════
elif page == "uc2":
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1e3a5f,#1d4ed8 55%,#0891b2);border-radius:14px;
                padding:28px 36px 24px;color:#fff;margin-bottom:24px;">
        <div style="font-size:0.72rem;font-weight:700;letter-spacing:.1em;opacity:.7;margin-bottom:8px;">
            USE CASE 2
        </div>
        <h2 style="margin:0 0 8px;font-size:1.6rem;">📈 Theo dõi Tiến trình Phục hồi Cơ</h2>
        <p style="margin:0;color:#bfdbfe;font-size:0.9rem;">
            So sánh sức bền qua các buổi tập · Phát hiện thoái lui hoặc hồi phục chậm
        </p>
    </div>
    """, unsafe_allow_html=True)

    rehab_view.render()
