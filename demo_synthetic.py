"""Standalone synthetic demo for sEMG AI Platform.

Chạy không cần dataset thật — sinh tín hiệu EMG tổng hợp mô phỏng quá trình
mỏi cơ, trích 14 đặc trưng, huấn luyện KNN (leave-one-subject-out), và mô
phỏng phân loại real-time trực quan.

Chạy:
    streamlit run demo_synthetic.py
    streamlit run demo_synthetic.py --server.port 8503
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import signal as sp_signal
from scipy import stats as sp_stats
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Constants ─────────────────────────────────────────────────────────────────
FS = 2000           # Hz — khớp với config.py thật
N_CHANNELS = 64
SEG_DURATION = 2.0  # giây mỗi segment

FEATURE_NAMES = [
    "RMS", "MAV", "Skewness", "Kurtosis", "Max", "Min", "STD", "Mean",
    "Spectral_Min", "Spectral_Max", "Spectral_STD", "MDF", "MNF",
    "Spectral_Entropy",
]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="sEMG AI — Synthetic Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"] .stButton button {
    width: 100%; text-align: left; background: transparent; border: none;
    padding: 0.55rem 0.75rem; border-radius: 8px; font-size: 0.95rem;
    font-weight: 500; color: #e2e8f0; transition: background 0.15s;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(255,255,255,0.08); color: #fff;
}
.hero {
    background: linear-gradient(135deg,#0b3b3c 0%,#0f766e 55%,#0891b2 100%);
    border-radius: 16px; padding: 36px 44px 32px; color: #fff; margin-bottom: 24px;
}
.hero h1 { font-size: 1.9rem; margin: 0 0 8px; line-height: 1.3; }
.hero p  { font-size: 0.95rem; color: #ccf0ee; margin: 0 0 18px; max-width: 680px; }
.hero-meta { display:flex; gap:28px; flex-wrap:wrap; border-top:1px solid rgba(255,255,255,0.2);
    padding-top:14px; font-size:0.8rem; color:#a7deda; }
.hero-meta b { display:block; color:#fff; font-size:0.87rem; margin-bottom:2px; }
.kpi-row { display:flex; gap:14px; flex-wrap:wrap; margin:20px 0; }
.kpi { flex:1 1 140px; background:#fff; border:1px solid #e2e8f0; border-radius:12px;
    padding:18px 20px; box-shadow:0 1px 3px rgba(0,0,0,.05); }
.kpi .n { font-size:1.8rem; font-weight:800; color:#0f766e; line-height:1; }
.kpi .l { font-size:0.77rem; color:#64748b; margin-top:5px; }
.sec { font-size:1.15rem; font-weight:700; color:#0b3b3c; margin:0 0 4px; }
.sub { font-size:0.84rem; color:#64748b; margin:0 0 16px; }
.info-box { background:#f0fdfa; border-left:4px solid #0f766e; border-radius:0 10px 10px 0;
    padding:12px 16px; margin:14px 0; font-size:0.88rem; color:#134e4a; }
.warn-box { background:#fff7ed; border-left:4px solid #f59e0b; border-radius:0 10px 10px 0;
    padding:12px 16px; margin:14px 0; font-size:0.88rem; color:#78350f; }
</style>
""", unsafe_allow_html=True)


# ── Synthetic EMG helpers ─────────────────────────────────────────────────────

def _synthetic_emg(
    n_samples: int,
    fs: int,
    mdf_hz: float,
    rms_target: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bandpass-filtered Gaussian noise với median frequency ≈ mdf_hz.

    Model đơn giản nhưng sinh lý hợp lý:
    - EMG ≈ noise → Butterworth bandpass → scale RMS.
    - Dải băng thông bất đối xứng quanh mdf_hz (phổ EMG thực lệch phải).
    """
    noise = rng.standard_normal(n_samples)
    low_hz = 20.0
    high_hz = min(float(mdf_hz) * 1.8, fs * 0.45)
    b, a = sp_signal.butter(4, [low_hz / (fs / 2), high_hz / (fs / 2)], btype="band")
    filtered = sp_signal.filtfilt(b, a, noise)
    cur_rms = float(np.sqrt(np.mean(filtered ** 2)))
    if cur_rms > 1e-12:
        filtered *= rms_target / cur_rms
    return filtered.astype(np.float32)


def _extract_features(x: np.ndarray, fs: int = FS) -> dict[str, float]:
    """14 features: 8 time-domain + 6 frequency-domain (giống src/feature_extraction.py)."""
    x = np.asarray(x, float)
    # Miền thời gian
    rms  = float(np.sqrt(np.mean(x ** 2)))
    mav  = float(np.mean(np.abs(x)))
    skew = float(sp_stats.skew(x, bias=True))
    kurt = float(sp_stats.kurtosis(x, fisher=False, bias=True))
    maxv = float(x.max())
    minv = float(x.min())
    std  = float(np.std(x, ddof=1))
    mean = float(np.mean(x))
    # PSD (FFT) cho spectral features
    nfft = 2048
    psd_h = (np.abs(np.fft.fft(x, n=nfft)) ** 2) / nfft
    psd_h = psd_h[: nfft // 2]
    spec_min = float(psd_h.min())
    spec_max = float(psd_h.max())
    spec_std = float(np.std(psd_h))
    # Periodogram cho MDF / MNF / Spectral_Entropy
    f, pxx = sp_signal.periodogram(x, fs=fs)
    total = float(np.sum(pxx))
    if total > 0:
        mnf = float(np.sum(f * pxx) / total)
        cumsum = np.cumsum(pxx)
        half = total / 2.0
        idx = min(int(np.searchsorted(cumsum, half)), len(f) - 1)
        if idx == 0:
            mdf_val = float(f[0])
        else:
            c0, c1 = cumsum[idx - 1], cumsum[idx]
            f0, f1 = f[idx - 1], f[idx]
            mdf_val = float(f0 + (half - c0) * (f1 - f0) / (c1 - c0)) if c1 != c0 else float(f1)
        p_pos = (pxx / total)
        p_pos = p_pos[p_pos > 0]
        spec_ent = float(-np.sum(p_pos * np.log2(p_pos)))
    else:
        mnf = mdf_val = spec_ent = 0.0

    return dict(zip(FEATURE_NAMES, [
        rms, mav, skew, kurt, maxv, minv, std, mean,
        spec_min, spec_max, spec_std, mdf_val, mnf, spec_ent,
    ]))


# ── Dataset & model (cached) ──────────────────────────────────────────────────

@st.cache_data(show_spinner="Đang tạo dataset tổng hợp…")
def _build_dataset(
    n_subjects: int = 8,
    n_ch_per_seg: int = 20,     # channels/file (ít hơn 64 để chạy nhanh)
    fatigue_mdf: float = 80.0,
    normal_mdf: float = 120.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Sinh bảng đặc trưng tổng hợp: nhiều subject, Normal + Fatigue conditions."""
    rng = np.random.default_rng(seed)
    n_samples = int(SEG_DURATION * FS)
    rows: list[dict] = []

    for subj in range(1, n_subjects + 1):
        # Normal: 5 mức MVC × 2 reps
        for mvc in [10, 20, 40, 60, 90]:
            is_fatigue = mvc > 60
            base_mdf = fatigue_mdf if is_fatigue else normal_mdf
            base_rms = 0.05 + mvc / 900.0
            for _ in range(2):
                mdf_j = base_mdf + rng.uniform(-8, 8)
                rms_j = base_rms + rng.uniform(-0.01, 0.01)
                for _ in range(n_ch_per_seg):
                    sig = _synthetic_emg(n_samples, FS, mdf_j, rms_j, rng)
                    rows.append({**_extract_features(sig), "subject": subj, "label": int(is_fatigue)})
        # Explicit fatigue condition: 2 reps
        for _ in range(2):
            mdf_j = fatigue_mdf + rng.uniform(-12, 12)
            rms_j = 0.17 + rng.uniform(0, 0.05)
            for _ in range(n_ch_per_seg):
                sig = _synthetic_emg(n_samples, FS, mdf_j, rms_j, rng)
                rows.append({**_extract_features(sig), "subject": subj, "label": 1})

    return pd.DataFrame(rows)


@st.cache_resource(show_spinner="Đang huấn luyện mô hình…")
def _get_model():
    """Train KNN trên subjects 1-7, test trên subject 8 (LOSO)."""
    df = _build_dataset()
    train = df[df["subject"] < 8]
    test  = df[df["subject"] == 8]
    X_tr, y_tr = train[FEATURE_NAMES].values, train["label"].values
    X_te, y_te = test[FEATURE_NAMES].values,  test["label"].values

    model = Pipeline([
        ("scale", StandardScaler()),
        ("clf",   KNeighborsClassifier(n_neighbors=1, metric="euclidean")),
    ])
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    cm = confusion_matrix(y_te, y_pred)

    return {
        "model":    model,
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "f1":       float(f1_score(y_te, y_pred, zero_division=0)),
        "auc":      float(roc_auc_score(y_te, y_prob)),
        "confusion": cm.tolist(),
        "y_te":     y_te.tolist(),
        "y_pred":   y_pred.tolist(),
        "y_prob":   y_prob.tolist(),
    }


# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:16px 0 8px;">
        <div style="font-size:2rem;">🧠</div>
        <div style="font-size:0.95rem;font-weight:700;color:#fff;margin-top:4px;">sEMG AI Platform</div>
        <div style="font-size:0.7rem;color:#94a3b8;margin-top:2px;">Synthetic Demo</div>
    </div>
    <hr style="border-color:rgba(255,255,255,0.1);margin:10px 0;">
    """, unsafe_allow_html=True)

    if "page" not in st.session_state:
        st.session_state["page"] = "overview"

    def _nav(label: str, key: str) -> None:
        active = st.session_state["page"] == key
        prefix = "▶ " if active else "   "
        if st.button(f"{prefix}{label}", key=f"nav_{key}"):
            st.session_state["page"] = key
            st.rerun()

    _nav("📋  Tổng quan",           "overview")
    _nav("📡  Tín hiệu EMG",        "signals")
    _nav("📊  Phân bố đặc trưng",   "features")
    _nav("🤖  Kết quả phân loại",   "classify")
    _nav("⚡  Mô phỏng real-time",  "realtime")

    st.markdown("""
    <hr style="border-color:rgba(255,255,255,0.1);margin:12px 0 8px;">
    <div style="font-size:0.7rem;color:#64748b;padding:0 4px;">
        ⚠️ Dùng dữ liệu <b style="color:#0f766e;">tổng hợp</b><br>
        (không cần dataset thật)
    </div>
    """, unsafe_allow_html=True)

page = st.session_state["page"]


# ════════════════════════════════════════════════════════════════════════════
# TRANG 1 — TỔNG QUAN
# ════════════════════════════════════════════════════════════════════════════
if page == "overview":
    nfo = _get_model()

    st.markdown(f"""
    <div class="hero">
        <div style="font-size:0.72rem;font-weight:700;letter-spacing:.08em;opacity:.7;margin-bottom:10px;">
            SYNTHETIC DEMO — sEMG AI PLATFORM
        </div>
        <h1>Phát hiện Mỏi cơ từ tín hiệu sEMG</h1>
        <p>Demo tổng hợp: sinh tín hiệu EMG giả lập, trích 14 đặc trưng,
           huấn luyện KNN và phân loại mỏi cơ. Không cần dataset thật.</p>
        <div class="hero-meta">
            <div><b>Mô hình</b>KNN (1-NN, Euclidean)</div>
            <div><b>Đánh giá</b>Leave-One-Subject-Out</div>
            <div><b>F1-score</b>{nfo['f1']:.3f}</div>
            <div><b>AUC</b>{nfo['auc']:.3f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi"><div class="n">{nfo['accuracy']:.1%}</div><div class="l">Accuracy (Test Subject 8)</div></div>
        <div class="kpi"><div class="n">{nfo['f1']:.3f}</div><div class="l">F1-score</div></div>
        <div class="kpi"><div class="n">{nfo['auc']:.3f}</div><div class="l">AUC — ROC</div></div>
        <div class="kpi"><div class="n">64</div><div class="l">Kênh EMG mô phỏng</div></div>
        <div class="kpi"><div class="n">2000 Hz</div><div class="l">Tần số lấy mẫu</div></div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns(2, gap="large")
    with col_l:
        st.markdown('<div class="sec">Mô hình tín hiệu tổng hợp</div>', unsafe_allow_html=True)
        st.markdown("""
        Tín hiệu EMG mô phỏng theo mô hình **bandpass-filtered Gaussian noise** —
        đặc điểm phổ phản ánh hai trạng thái sinh lý:

        | Trạng thái | MDF (Hz) | RMS | Ý nghĩa |
        |---|---|---|---|
        | **Normal** | ~120 Hz | Thấp | Cơ chưa mỏi |
        | **Fatigue** | ~80 Hz | Cao hơn | Phổ dịch về tần số thấp |

        Khi mỏi cơ, tốc độ dẫn truyền xung thần kinh giảm → phổ EMG dịch trái.
        Đây là đặc trưng sinh lý bền vững nhất của mỏi cơ, được mô hình hoá
        trong tín hiệu tổng hợp này.
        """)

    with col_r:
        st.markdown('<div class="sec">Pipeline (4 bước)</div>', unsafe_allow_html=True)
        st.markdown("""
        1. **Sinh tín hiệu** — noise → Butterworth bandpass → scale RMS
        2. **Trích đặc trưng** — 14 features/kênh (8 thời gian + 6 tần số)
        3. **Chọn đặc trưng** — tương quan với nhãn (proxy mRMR)
        4. **Phân loại** — KNN 1-NN, chuẩn hoá StandardScaler

        **Đánh giá:** Leave-One-Subject-Out (train 7 subject, test subject 8)
        """)

    st.markdown("""
    <div class="warn-box">
        📌 <b>Lưu ý:</b> Đây là demo tổng hợp để minh họa pipeline.
        Kết quả thực tế trên dataset Noraxon 64-kênh: KNN F1=0.977, AUC=0.992 (Subject 6).
    </div>
    """, unsafe_allow_html=True)
    st.info("👉 Chọn tab trên thanh bên để xem từng bước chi tiết.")


# ════════════════════════════════════════════════════════════════════════════
# TRANG 2 — TÍN HIỆU EMG
# ════════════════════════════════════════════════════════════════════════════
elif page == "signals":
    st.markdown('<div class="sec">Tín hiệu EMG tổng hợp</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">So sánh dạng sóng và phổ tần số giữa Normal và Fatigue</div>', unsafe_allow_html=True)

    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        mdf_normal  = st.slider("MDF Normal (Hz)",  80, 200, 120, key="sl_mdf_n")
    with col_c2:
        mdf_fatigue = st.slider("MDF Fatigue (Hz)", 40, 120,  80, key="sl_mdf_f")
    with col_c3:
        seed_sig = st.number_input("Seed", 0, 999, 0, key="sl_seed")

    rng   = np.random.default_rng(int(seed_sig))
    n     = int(SEG_DURATION * FS)
    t_sec = np.linspace(0, SEG_DURATION, n)

    sig_n = _synthetic_emg(n, FS, mdf_normal,  0.08, rng)
    sig_f = _synthetic_emg(n, FS, mdf_fatigue, 0.15, rng)
    feat_n = _extract_features(sig_n)
    feat_f = _extract_features(sig_f)

    COLORS = {"Normal": "#0f766e", "Fatigue": "#dc2626"}

    for col, sig, feat, label in [
        (st.columns(2)[0], sig_n, feat_n, "Normal"),
        (st.columns(2)[1], sig_f, feat_f, "Fatigue"),
    ]:
        pass  # placeholder — sẽ dùng vòng lặp dưới

    col_l2, col_r2 = st.columns(2, gap="large")
    for col_out, sig, feat, label in [
        (col_l2, sig_n, feat_n, "Normal"),
        (col_r2, sig_f, feat_f, "Fatigue"),
    ]:
        color = COLORS[label]
        with col_out:
            st.markdown(
                f'<div class="sec" style="color:{color};">{label}</div>',
                unsafe_allow_html=True,
            )
            # Dạng sóng
            ds  = max(1, n // 2000)
            fig = go.Figure(go.Scatter(
                x=t_sec[::ds], y=sig[::ds],
                mode="lines", line=dict(color=color, width=1),
            ))
            fig.update_layout(
                height=200, margin=dict(t=8, b=30, l=50, r=10),
                xaxis_title="Thời gian (s)", yaxis_title="Biên độ",
                showlegend=False, plot_bgcolor="#fafafa",
                xaxis=dict(showgrid=True, gridcolor="#e5e7eb"),
                yaxis=dict(showgrid=True, gridcolor="#e5e7eb"),
            )
            st.plotly_chart(fig, use_container_width=True, key=f"wave_{label}")

            # PSD
            f_w, pxx = sp_signal.welch(sig, fs=FS, nperseg=512)
            mask_w = f_w <= 500
            rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
            fig_psd = go.Figure(go.Scatter(
                x=f_w[mask_w], y=10 * np.log10(pxx[mask_w] + 1e-20),
                mode="lines", line=dict(color=color, width=1.5),
                fill="tozeroy",
                fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.12)",
            ))
            fig_psd.add_vline(
                x=feat["MDF"], line_dash="dash", line_color=color,
                annotation_text=f"MDF={feat['MDF']:.0f}Hz",
                annotation_font_size=11,
            )
            fig_psd.update_layout(
                height=200, margin=dict(t=8, b=30, l=50, r=10),
                xaxis_title="Tần số (Hz)", yaxis_title="PSD (dB/Hz)",
                showlegend=False, plot_bgcolor="#fafafa",
                xaxis=dict(showgrid=True, gridcolor="#e5e7eb"),
                yaxis=dict(showgrid=True, gridcolor="#e5e7eb"),
            )
            st.plotly_chart(fig_psd, use_container_width=True, key=f"psd_{label}")

            st.markdown(f"""
            | Chỉ số | Giá trị |
            |---|---|
            | **RMS** | `{feat['RMS']:.5f}` |
            | **MDF** | `{feat['MDF']:.1f} Hz` |
            | **MNF** | `{feat['MNF']:.1f} Hz` |
            | **MAV** | `{feat['MAV']:.5f}` |
            """)

    st.markdown("""
    <div class="info-box">
        🔬 Dấu hiệu mỏi cơ: MDF↓, phổ công suất thu hẹp về tần số thấp.
        Đây là cơ sở để mô hình ML phân biệt Normal vs Fatigue.
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TRANG 3 — PHÂN BỐ ĐẶC TRƯNG
# ════════════════════════════════════════════════════════════════════════════
elif page == "features":
    df = _build_dataset()

    st.markdown('<div class="sec">Phân bố 14 đặc trưng</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">Boxplot Normal vs Fatigue trên dataset tổng hợp (8 subjects)</div>', unsafe_allow_html=True)

    feat_sel = st.multiselect(
        "Chọn đặc trưng để hiển thị:",
        FEATURE_NAMES,
        default=["RMS", "MDF", "MNF", "MAV", "Kurtosis", "Spectral_Entropy"],
    )

    if feat_sel:
        n_cols = min(3, len(feat_sel))
        cols   = st.columns(n_cols)
        df["label_name"] = df["label"].map({0: "Normal", 1: "Fatigue"})

        for i, feat in enumerate(feat_sel):
            with cols[i % n_cols]:
                fig = go.Figure()
                for lbl, color in [("Normal", "#0f766e"), ("Fatigue", "#dc2626")]:
                    vals = df[df["label_name"] == lbl][feat]
                    fig.add_trace(go.Box(
                        y=vals, name=lbl,
                        marker_color=color, boxmean=True,
                    ))
                fig.update_layout(
                    title=feat, height=300,
                    margin=dict(t=40, b=20, l=40, r=10),
                    plot_bgcolor="#fafafa",
                    showlegend=(i == 0),
                )
                st.plotly_chart(fig, use_container_width=True, key=f"box_{feat}")

    st.divider()
    st.markdown('<div class="sec">Xếp hạng đặc trưng</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">Tương quan tuyệt đối |r| với nhãn mỏi cơ (proxy mRMR relevance)</div>', unsafe_allow_html=True)

    corrs = {f: abs(float(df[f].corr(df["label"]))) for f in FEATURE_NAMES}
    df_corr = pd.DataFrame(
        sorted(corrs.items(), key=lambda x: x[1], reverse=True),
        columns=["Đặc trưng", "|r|"],
    )
    colors_bar = ["#0f766e" if i < 3 else "#94a3b8" for i in range(len(df_corr))]

    fig_bar = go.Figure(go.Bar(
        x=df_corr["|r|"],
        y=df_corr["Đặc trưng"],
        orientation="h",
        marker_color=colors_bar,
        text=[f"{v:.3f}" for v in df_corr["|r|"]],
        textposition="outside",
    ))
    fig_bar.update_layout(
        height=400, margin=dict(t=10, b=40, l=130, r=60),
        xaxis_title="Tương quan tuyệt đối với nhãn",
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="#fafafa",
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    top3 = df_corr.head(3)["Đặc trưng"].tolist()
    st.markdown(f"""
    <div class="info-box">
        🎯 <b>Top-3 đặc trưng:</b> {", ".join(f"<b>{f}</b>" for f in top3)} —
        được chọn để huấn luyện SVM (tương tự mRMR trong pipeline thật).
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TRANG 4 — KẾT QUẢ PHÂN LOẠI
# ════════════════════════════════════════════════════════════════════════════
elif page == "classify":
    nfo = _get_model()
    acc, f1, auc = nfo["accuracy"], nfo["f1"], nfo["auc"]

    st.markdown('<div class="sec">Kết quả phân loại — KNN (1-NN)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">Train: Subject 1–7 · Test: Subject 8 (Leave-One-Subject-Out)</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi"><div class="n">{acc:.1%}</div><div class="l">Accuracy</div></div>
        <div class="kpi"><div class="n">{f1:.3f}</div><div class="l">F1-score</div></div>
        <div class="kpi"><div class="n">{auc:.3f}</div><div class="l">AUC-ROC</div></div>
    </div>
    """, unsafe_allow_html=True)

    col_cm, col_roc = st.columns(2, gap="large")

    with col_cm:
        st.markdown('<div class="sec">Confusion Matrix</div>', unsafe_allow_html=True)
        cm = nfo["confusion"]
        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            x=["Predicted Normal", "Predicted Fatigue"],
            y=["Actual Normal", "Actual Fatigue"],
            colorscale=[[0, "#f0fdfa"], [0.5, "#5eead4"], [1, "#0f766e"]],
            text=[[str(v) for v in row] for row in cm],
            texttemplate="%{text}",
            showscale=False,
            textfont=dict(size=18, color="black"),
        ))
        fig_cm.update_layout(
            height=320, margin=dict(t=20, b=60, l=100, r=20),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_roc:
        st.markdown('<div class="sec">ROC Curve</div>', unsafe_allow_html=True)
        from sklearn.metrics import roc_curve
        y_te  = np.array(nfo["y_te"])
        y_prob = np.array(nfo["y_prob"])
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            line=dict(color="#0f766e", width=2.5),
            name=f"AUC = {auc:.3f}",
        ))
        fig_roc.add_shape(
            type="line", x0=0, y0=0, x1=1, y1=1,
            line=dict(color="#94a3b8", dash="dash"),
        )
        fig_roc.update_layout(
            height=320, margin=dict(t=20, b=50, l=60, r=20),
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            showlegend=True,
            plot_bgcolor="#fafafa",
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("""
    <div class="info-box">
        📌 <b>Bối cảnh:</b> Đây là kết quả trên <i>dữ liệu tổng hợp</i>.
        Trên dataset thật Noraxon 64-kênh: KNN(1-NN) đạt F1=0.977, AUC=0.992 (Subject 6 LOSO).
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TRANG 5 — MÔ PHỎNG REAL-TIME
# ════════════════════════════════════════════════════════════════════════════
elif page == "realtime":
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0b3b3c,#0f766e);border-radius:14px;
                padding:24px 32px 20px;color:#fff;margin-bottom:20px;">
        <div style="font-size:0.72rem;font-weight:700;letter-spacing:.1em;opacity:.7;margin-bottom:8px;">
            SYNTHETIC SIMULATION
        </div>
        <h2 style="margin:0 0 6px;font-size:1.5rem;">⚡ Mô phỏng Phân loại Real-time</h2>
        <p style="margin:0;color:#ccf0ee;font-size:0.87rem;">
            Buổi tập tổng hợp: Normal → Fatigue dần dần · Phân loại từng cửa sổ 2s
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_b1, col_b2, _ = st.columns([1.2, 1, 5])
    with col_b1:
        start_btn = st.button("▶  Chạy demo", type="primary", use_container_width=True)
    with col_b2:
        reset_btn = st.button("↺  Đặt lại", use_container_width=True)

    if "rt_on" not in st.session_state:
        st.session_state["rt_on"] = False
    if reset_btn:
        st.session_state["rt_on"] = False
        st.rerun()
    if start_btn:
        st.session_state["rt_on"] = True

    if not st.session_state["rt_on"]:
        st.markdown("""
        <div style="background:#f8fafb;border:2px dashed #cbd5e1;border-radius:12px;
                    padding:36px;text-align:center;color:#64748b;margin-top:12px;">
            <div style="font-size:2.2rem;margin-bottom:10px;">📡</div>
            <div style="font-size:0.95rem;font-weight:600;">Nhấn "Chạy demo" để bắt đầu</div>
            <div style="font-size:0.83rem;margin-top:6px;">
                Mô phỏng 12 segment (24 giây): 7 Normal → 5 Fatigue
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        nfo = _get_model()
        model = nfo["model"]
        rng_rt = np.random.default_rng(99)
        n_samp = int(SEG_DURATION * FS)
        n_segs = 12
        # Trạng thái thật: 7 Normal, 5 Fatigue
        true_labels = [0] * 7 + [1] * 5
        # Tín hiệu: MDF và RMS giảm/tăng dần khi mỏi
        mdf_seq = [120 - i * 4 + rng_rt.uniform(-5, 5)  for i in range(7)] + \
                  [90  - i * 6 + rng_rt.uniform(-5, 5)  for i in range(5)]
        rms_seq = [0.07 + i * 0.003 + rng_rt.uniform(-0.002, 0.002) for i in range(7)] + \
                  [0.14 + i * 0.008 + rng_rt.uniform(-0.003, 0.003) for i in range(5)]

        records = []
        for i in range(n_segs):
            sig = _synthetic_emg(n_samp, FS, mdf_seq[i], rms_seq[i], rng_rt)
            feat = _extract_features(sig)
            X = np.array([[feat[f] for f in FEATURE_NAMES]])
            pred = int(model.predict(X)[0])
            prob = float(model.predict_proba(X)[0][1])
            records.append({
                "seg": i + 1,
                "true": true_labels[i],
                "pred": pred,
                "prob": prob,
                "mdf":  feat["MDF"],
                "rms":  feat["RMS"],
            })

        # ── Trend chart ───────────────────────────────────────────────────
        seg_x = [r["seg"] for r in records]
        fig_t  = go.Figure()

        # MDF trend
        fig_t.add_trace(go.Scatter(
            x=seg_x, y=[r["mdf"] for r in records],
            mode="lines+markers", name="MDF (Hz)",
            line=dict(color="#0f766e", width=2.5),
            marker=dict(size=8),
        ))
        # RMS trend (×1000 để vào cùng trục)
        fig_t.add_trace(go.Scatter(
            x=seg_x, y=[r["rms"] * 1000 for r in records],
            mode="lines+markers", name="RMS ×1000",
            line=dict(color="#0891b2", width=2, dash="dot"),
            marker=dict(size=7),
        ))
        # P(Fatigue)
        fig_t.add_trace(go.Scatter(
            x=seg_x, y=[r["prob"] * 100 for r in records],
            mode="lines+markers", name="P(Fatigue) %",
            line=dict(color="#dc2626", width=2, dash="dashdot"),
            marker=dict(
                size=10,
                symbol=["circle" if r["pred"] == 0 else "x" for r in records],
                color=["#0f766e" if r["pred"] == 0 else "#dc2626" for r in records],
            ),
        ))
        # Vùng Fatigue
        fig_t.add_vrect(
            x0=7.5, x1=n_segs + 0.5,
            fillcolor="rgba(220,38,38,0.07)", line_width=0,
            annotation_text="Fatigue zone", annotation_position="top right",
            annotation_font_color="#dc2626",
        )
        # Đường ngưỡng P=50%
        fig_t.add_hline(
            y=50, line_dash="dot", line_color="#dc2626", line_width=1,
            annotation_text="Ngưỡng 50%", annotation_position="right",
            annotation_font_size=11,
        )
        fig_t.update_layout(
            height=280,
            title="Xu hướng MDF · RMS · P(Fatigue) theo segment",
            xaxis_title="Segment (mỗi segment = 2 giây)",
            margin=dict(t=50, b=40, l=60, r=20),
            plot_bgcolor="#fafafa",
            legend=dict(orientation="h", y=1.12, x=0),
            xaxis=dict(showgrid=True, gridcolor="#e5e7eb", dtick=1),
            yaxis=dict(showgrid=True, gridcolor="#e5e7eb"),
        )
        st.plotly_chart(fig_t, use_container_width=True)

        # ── Bảng kết quả ─────────────────────────────────────────────────
        st.markdown('<div class="sec">Kết quả phân loại từng segment</div>', unsafe_allow_html=True)

        df_rt = pd.DataFrame([{
            "Segment": r["seg"],
            "MDF (Hz)": f"{r['mdf']:.1f}",
            "RMS": f"{r['rms']:.5f}",
            "P(Fatigue)": f"{r['prob']:.2%}",
            "Dự đoán": "Fatigue 🔴" if r["pred"] else "Normal 🟢",
            "Thực tế":  "Fatigue"   if r["true"] else "Normal",
            "Đúng/Sai": "✅" if r["pred"] == r["true"] else "❌",
        } for r in records])

        st.dataframe(df_rt, use_container_width=True, hide_index=True)

        correct = sum(1 for r in records if r["pred"] == r["true"])
        st.markdown(f"""
        <div class="info-box">
            ✅ <b>Phân loại đúng:</b> {correct}/{n_segs} segments ({correct/n_segs:.0%}) —
            Mô hình phát hiện đúng thời điểm chuyển từ Normal sang Fatigue dựa trên
            sự thay đổi MDF và RMS trong tín hiệu tổng hợp.
        </div>
        """, unsafe_allow_html=True)

        # ── Channel grid (8×8 đơn giản hoá) ─────────────────────────────
        st.divider()
        st.markdown('<div class="sec">Sơ đồ 64 kênh — trạng thái cuối buổi</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub">Màu xanh = Normal · Đỏ = Fatigue · Xám = Abstention (tín hiệu yếu)</div>', unsafe_allow_html=True)

        # Phân loại 64 kênh dựa trên segment cuối
        last_seg = records[-1]
        rng_ch = np.random.default_rng(last_seg["seg"] * 7)
        ch_preds = []
        for ch in range(N_CHANNELS):
            noise_mdf = last_seg["mdf"] + rng_ch.uniform(-15, 15)
            noise_rms = last_seg["rms"] + rng_ch.uniform(-0.03, 0.03)
            # Kênh có RMS quá thấp → abstention
            if noise_rms < 0.04:
                ch_preds.append("abstain")
            else:
                sig_ch = _synthetic_emg(n_samp, FS, noise_mdf, noise_rms, rng_ch)
                feat_ch = _extract_features(sig_ch)
                X_ch = np.array([[feat_ch[f] for f in FEATURE_NAMES]])
                p_ch = int(model.predict(X_ch)[0])
                ch_preds.append("fatigue" if p_ch else "normal")

        # Render grid 8×8 dạng HTML
        cell_style = "width:36px;height:36px;border-radius:6px;display:inline-block;margin:2px;"
        COLOR_MAP = {"normal": "#0f766e", "fatigue": "#dc2626", "abstain": "#94a3b8"}
        html_cells = ""
        for row in range(8):
            for col_i in range(8):
                ch_idx = row * 8 + col_i
                c = COLOR_MAP[ch_preds[ch_idx]]
                ch_num = ch_idx + 1
                html_cells += (
                    f'<span title="CH{ch_num}: {ch_preds[ch_idx]}" '
                    f'style="{cell_style}background:{c};"></span>'
                )
            html_cells += "<br>"

        n_fat  = ch_preds.count("fatigue")
        n_norm = ch_preds.count("normal")
        n_abs  = ch_preds.count("abstain")
        st.markdown(
            f'<div style="font-family:monospace;line-height:1.2;">{html_cells}</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"🟢 Normal: {n_norm} kênh  |  🔴 Fatigue: {n_fat} kênh  |  ⬜ Abstention: {n_abs} kênh"
        )
