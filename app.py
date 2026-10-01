"""Streamlit demo for EMG muscle-fatigue classification (spec section 6.3).

Run from the project root:
    streamlit run emg_fatigue_demo/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Allow "streamlit run" to find the src package at the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config          # noqa: E402
from src import data_loader as dl  # noqa: E402
from src import feature_extraction as fe  # noqa: E402
from src import inference as inf  # noqa: E402
from src import pipeline as pl   # noqa: E402
from src import visualization as viz  # noqa: E402

st.set_page_config(page_title="EMG Fatigue Demo", layout="wide")

# Thicker, clearer borders + dark card background — scoped to the Features
# boxplot cards only (they carry a `featcard*` container key).
st.markdown(
    """
    <style>
    [class*="st-key-featcard"] {
        border: 3px solid #5A5A5A !important;
        border-radius: 12px !important;
        background: #111111 !important;
        padding: 8px !important;
    }
    [class*="st-key-predtable"] table {
        font-size: 1.1rem !important;
    }
    [class*="st-key-predtable"] table th,
    [class*="st-key-predtable"] table td {
        padding: 0.5rem 0.75rem !important;
    }
    .feat-tooltip-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 10px;
        font-family: 'Inter', 'Segoe UI', sans-serif;
        font-size: 0.95rem;
        border: 1px solid #333;
    }
    .feat-tooltip-table thead th {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #e0e0e0;
        padding: 12px 16px;
        text-align: left;
        font-weight: 600;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border-bottom: 2px solid #0f3460;
    }
    .feat-tooltip-table tbody tr {
        transition: background 0.2s ease;
    }
    .feat-tooltip-table tbody tr:nth-child(odd)  { background: #0d1117; }
    .feat-tooltip-table tbody tr:nth-child(even) { background: #161b22; }
    .feat-tooltip-table tbody tr:hover { background: #1a2332; }
    .feat-tooltip-table td {
        padding: 10px 16px;
        border-bottom: 1px solid #21262d;
        color: #c9d1d9;
    }
    .feat-tooltip-table td:last-child {
        font-variant-numeric: tabular-nums;
        text-align: right;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        color: #79c0ff;
    }
    .feat-name-wrap {
        position: relative;
        display: inline-block;
        cursor: help;
        border-bottom: 1px dashed #58a6ff;
        padding-bottom: 1px;
        color: #58a6ff;
        font-weight: 500;
        transition: color 0.2s ease;
    }
    .feat-name-wrap:hover { color: #79c0ff; }
    .feat-name-wrap .feat-tip {
        visibility: hidden;
        opacity: 0;
        position: absolute;
        z-index: 9999;
        left: 0;
        bottom: calc(100% + 10px);
        width: 340px;
        padding: 14px 16px;
        border-radius: 10px;
        background: linear-gradient(145deg, #1e2a3a 0%, #172030 100%);
        border: 1px solid #30465e;
        box-shadow: 0 8px 32px rgba(0,0,0,0.45), 0 0 0 1px rgba(88,166,255,0.1);
        color: #e6edf3;
        font-size: 0.82rem;
        font-weight: 400;
        line-height: 1.55;
        letter-spacing: 0.01em;
        transition: opacity 0.25s ease, visibility 0.25s ease, transform 0.25s ease;
        transform: translateY(4px);
        pointer-events: none;
    }
    .feat-tooltip-table tbody tr:nth-child(-n+2) .feat-name-wrap .feat-tip {
        bottom: auto;
        top: calc(100% + 10px);
        transform: translateY(-4px);
    }
    .feat-tooltip-table tbody tr:nth-child(-n+2) .feat-name-wrap:hover .feat-tip {
        transform: translateY(0);
    }
    .feat-name-wrap .feat-tip::after {
        content: '';
        position: absolute;
        top: 100%;
        left: 24px;
        border: 7px solid transparent;
        border-top-color: #30465e;
    }
    .feat-tooltip-table tbody tr:nth-child(-n+2) .feat-name-wrap .feat-tip::after {
        top: auto;
        bottom: 100%;
        border-top-color: transparent;
        border-bottom-color: #30465e;
    }
    .feat-name-wrap .feat-tip .tip-title {
        display: block;
        font-weight: 700;
        color: #58a6ff;
        margin-bottom: 6px;
        font-size: 0.85rem;
    }
    .feat-name-wrap .feat-tip .tip-meaning {
        display: block;
        margin-bottom: 5px;
    }
    .feat-name-wrap .feat-tip .tip-role {
        display: block;
        color: #8b949e;
        font-size: 0.78rem;
        padding-top: 5px;
        border-top: 1px solid rgba(48,70,94,0.6);
    }
    .feat-name-wrap:hover .feat-tip {
        visibility: visible;
        opacity: 1;
        transform: translateY(0);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

FEATURE_TOOLTIPS: dict[str, dict[str, str]] = {
    "RMS": {
        "meaning": "Root Mean Square — căn bậc hai trung bình bình phương biên độ tín hiệu.",
        "role": "Đại diện cho công suất tín hiệu EMG. Khi cơ mỏi, RMS thường tăng do cần huy động thêm đơn vị vận động.",
    },
    "MAV": {
        "meaning": "Mean Absolute Value — trung bình giá trị tuyệt đối của tín hiệu.",
        "role": "Phản ánh mức co cơ trung bình. Tăng khi cơ mỏi do biên độ tín hiệu tăng. Phổ biến trong phân tích sEMG lâm sàng.",
    },
    "Skewness": {
        "meaning": "Độ lệch (Skewness) — đo mức bất đối xứng của phân bố biên độ tín hiệu.",
        "role": "Tín hiệu mỏi thường có phân bố lệch hơn so với trạng thái bình thường, phản ánh sự thay đổi trong mẫu kích hoạt cơ.",
    },
    "Kurtosis": {
        "meaning": "Độ nhọn (Kurtosis) — đo mức tập trung đỉnh của phân bố biên độ so với phân bố chuẩn.",
        "role": "Giá trị cao cho thấy tín hiệu có nhiều xung đột ngột. Thay đổi khi cơ mỏi do biến đổi trong tốc độ phóng thích đơn vị vận động.",
    },
    "Max": {
        "meaning": "Giá trị cực đại của biên độ tín hiệu EMG trong cửa sổ phân tích.",
        "role": "Phản ánh đỉnh co cơ tối đa. Khi mỏi, biên độ đỉnh có thể tăng do huy động thêm sợi cơ để bù trừ lực giảm.",
    },
    "Min": {
        "meaning": "Giá trị cực tiểu của biên độ tín hiệu EMG trong cửa sổ phân tích.",
        "role": "Kết hợp với Max để đánh giá biên độ dao động tổng thể của tín hiệu sEMG.",
    },
    "STD": {
        "meaning": "Standard Deviation — độ lệch chuẩn, đo mức biến thiên của biên độ tín hiệu.",
        "role": "Tương quan chặt với RMS, phản ánh công suất tín hiệu. Tăng khi cơ mỏi do biên độ dao động lớn hơn.",
    },
    "Mean": {
        "meaning": "Giá trị trung bình cộng của biên độ tín hiệu EMG.",
        "role": "Thường gần 0 với tín hiệu EMG đã lọc. Sai lệch khỏi 0 có thể chỉ ra vấn đề về baseline hoặc nhiễu DC.",
    },
    "Spectral_Min": {
        "meaning": "Giá trị nhỏ nhất trong phổ công suất (PSD) của tín hiệu.",
        "role": "Cho biết mức năng lượng tối thiểu trong miền tần số. Hữu ích để đánh giá mức nền nhiễu phổ.",
    },
    "Spectral_Max": {
        "meaning": "Giá trị lớn nhất trong phổ công suất (PSD) của tín hiệu.",
        "role": "Cho biết đỉnh năng lượng trong miền tần số. Khi mỏi, đỉnh phổ thường dịch về tần số thấp hơn.",
    },
    "Spectral_STD": {
        "meaning": "Độ lệch chuẩn của phổ công suất — đo mức phân tán năng lượng trên các tần số.",
        "role": "Đặc trưng quan trọng trong mRMR top-3. Phản ánh sự tập trung hay phân tán năng lượng phổ khi cơ chuyển từ bình thường sang mỏi.",
    },
    "MDF": {
        "meaning": "Median Frequency — tần số trung vị chia đôi tổng công suất phổ.",
        "role": "Chỉ số vàng trong nghiên cứu mỏi cơ sEMG. MDF giảm khi cơ mỏi do dẫn truyền thần kinh chậm lại, là dấu hiệu sinh lý cổ điển.",
    },
    "MNF": {
        "meaning": "Mean Frequency — tần số trung bình (trọng tâm phổ công suất).",
        "role": "Tương tự MDF nhưng nhạy hơn với nhiễu. MNF giảm khi mỏi, phản ánh sự dịch chuyển phổ về tần số thấp.",
    },
    "Spectral_Entropy": {
        "meaning": "Shannon Entropy của phổ công suất chuẩn hóa — đo mức hỗn loạn/phân tán năng lượng phổ.",
        "role": "Đặc trưng mạnh nhất theo mRMR. Giá trị cao = năng lượng phân tán đều; thấp = tập trung vào vài tần số. Phân biệt rõ trạng thái bình thường vs mỏi.",
    },
}


def _build_feature_tooltip_table(feats: dict[str, float]) -> str:
    rows = []
    for name in config.FEATURE_NAMES:
        tip = FEATURE_TOOLTIPS.get(name, {})
        meaning = tip.get("meaning", "")
        role = tip.get("role", "")
        val = feats.get(name, 0.0)
        rows.append(
            f'<tr>'
            f'<td>'
            f'<span class="feat-name-wrap">{name}'
            f'<span class="feat-tip">'
            f'<span class="tip-title">{name}</span>'
            f'<span class="tip-meaning">{meaning}</span>'
            f'<span class="tip-role">⚡ {role}</span>'
            f'</span></span></td>'
            f'<td>{val:.4g}</td>'
            f'</tr>'
        )
    return (
        '<table class="feat-tooltip-table">'
        '<thead><tr><th>Đặc trưng</th><th>Giá trị</th></tr></thead>'
        '<tbody>' + ''.join(rows) + '</tbody></table>'
    )


LABEL_COLORS = {"Normal": "#2E86DE", "Fatigue": "#E74C3C"}


@st.cache_data(show_spinner="Extracting features / running pipeline…")
def get_pipeline():
    return pl.run_pipeline(use_cache=True)


@st.cache_data(show_spinner="Loading signal…")
def load_signal(path_str: str):
    return dl.load_channels(Path(path_str))


def label_name(v: int) -> str:
    return config.LABEL_NAMES[int(v)]


# Larger, high-contrast hover tooltip applied to every chart:
# dark background with white text.
HOVERLABEL = dict(font=dict(size=20, color="white", family="Arial"),
                  bgcolor="#1E1E1E", bordercolor="#1E1E1E")


def chart(fig, **kwargs):
    """Apply the shared hover styling, then render the Plotly figure."""
    fig.update_layout(hoverlabel=HOVERLABEL, hovermode="closest")
    kwargs.setdefault("width", "stretch")
    st.plotly_chart(fig, **kwargs)


out = get_pipeline()
df = out.feature_table
df = df.assign(Class=df["label"].map(config.LABEL_NAMES))
files = dl.list_files()

st.title("EMG Muscle-Fatigue Classification — Demo")
st.caption("sEMG (64-channel, 2000 Hz) · 14 features/channel · mRMR + SVM/KNN/LDA/DecisionTree · "
           "leave-one-subject-out (test = subject "
           f"{config.TEST_SUBJECT})")

(tab_overview, tab_signal, tab_features, tab_select, tab_classify,
 tab_predict) = st.tabs(
    ["Overview", "Signal & PSD", "Features", "Feature Selection",
     "Classification", "Predict / Inference"]
)

# --------------------------------------------------------------------------
# Tab 1 — Data overview
# --------------------------------------------------------------------------
with tab_overview:
    st.subheader("Dataset overview")
    total_ch = config.N_CHANNELS * len(files)
    valid_ch = len(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Subjects", df["subject"].nunique())
    c2.metric("Files", df["file"].nunique())
    c3.metric("Valid channel-samples", valid_ch)
    c4.metric("Dropped (dead) channels", total_ch - valid_ch)

    c5, c6 = st.columns(2)
    with c5:
        class_counts = df["Class"].value_counts().reindex(["Normal", "Fatigue"]).fillna(0)
        fig = px.bar(class_counts, color=class_counts.index,
                     color_discrete_map=LABEL_COLORS,
                     labels={"value": "samples", "index": "class"},
                     title="Class distribution (channel-samples)")
        fig.update_layout(showlegend=False)
        chart(fig)
    with c6:
        by_subj = df.groupby(["subject", "Class"]).size().reset_index(name="count")
        fig = px.bar(by_subj, x="subject", y="count", color="Class",
                     color_discrete_map=LABEL_COLORS, barmode="group",
                     title="Samples per subject")
        chart(fig)

    st.markdown("**Per-file summary** (label from filename: %MVC number > "
                f"{config.FATIGUE_MVC_THRESHOLD} → Fatigue)")
    per_file = (df.groupby(["file", "subject", "condition", "Class"])
                  .size().reset_index(name="valid_channels")
                  .sort_values(["subject", "condition"]))
    st.dataframe(per_file, width="stretch", hide_index=True)

# --------------------------------------------------------------------------
# Tab 2 — Signal & PSD
# --------------------------------------------------------------------------
with tab_signal:
    st.subheader("Signal & power spectral density")
    file_names = [f.name for f in files]
    sel_file = st.selectbox("File", file_names)
    info = next(f for f in files if f.name == sel_file)

    channels = load_signal(str(info.path))
    mask = dl.valid_channel_mask(channels)
    valid_idx = list(np.where(mask)[0])
    st.caption(f"Subject {info.subject} · condition `{info.condition}` · "
               f"label **{label_name(info.label)}** · "
               f"{len(valid_idx)}/{config.N_CHANNELS} valid channels")

    sel_ch = st.selectbox("Channel", valid_idx,
                          format_func=lambda i: f"Channel {i}")
    x = channels[:, sel_ch]
    t = np.arange(len(x)) / config.FS

    max_pts = 20000  # decimate long signals for a responsive plot
    step = max(1, len(x) // max_pts)
    fig_t = px.line(x=t[::step], y=x[::step],
                    labels={"x": "time (s)", "y": "amplitude"},
                    title=f"Time-domain signal — channel {sel_ch}")
    fig_t.update_traces(line=dict(width=1))
    chart(fig_t)

    f, pxx = fe.channel_psd(x)
    fig_f = px.line(x=f, y=pxx, labels={"x": "frequency (Hz)", "y": "PSD"},
                    title="Power spectral density")
    chart(fig_f)

    st.markdown("**14 features for this channel**")
    feats = fe.extract_channel_features(x)
    st.markdown(
        _build_feature_tooltip_table(feats),
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------
# Tab 3 — Feature distributions
# --------------------------------------------------------------------------
with tab_features:
    st.subheader("Feature distributions: Normal vs Fatigue")
    st.caption("Boxplots show which features separate the two classes.")

    st.markdown("### MNF và MDF")
    st.caption("MDF là tần số chia đôi tổng công suất phổ; MNF là trọng tâm phổ công suất. "
               "Khi cơ mỏi, cả hai thường giảm vì năng lượng phổ dịch về vùng tần số thấp.")
    chart(viz.build_mnf_mdf_comparison_figure(df))

    st.markdown("### 14 đặc trưng")
    select_all = st.checkbox("Select all features", value=False)
    # Changing the key when the checkbox toggles forces the multiselect to
    # re-seed its default; the user can still fine-tune the selection after.
    default_feats = config.FEATURE_NAMES if select_all else out.top_k
    sel_feats = st.multiselect("Features to show", config.FEATURE_NAMES,
                               default=default_feats,
                               key=f"feat_select_{select_all}")
    if sel_feats:
        n_cols = min(3, len(sel_feats))
        cols = st.columns(n_cols, gap="medium")
        for i, feat in enumerate(sel_feats):
            with cols[i % n_cols].container(border=False, key=f"featcard{i}"):
                fig = px.box(df, x="Class", y=feat, color="Class",
                             color_discrete_map=LABEL_COLORS,
                             category_orders={"Class": ["Normal", "Fatigue"]},
                             points=False, title=feat)
                fig.update_layout(
                    showlegend=False, height=340,
                    title=dict(font=dict(size=18, color="#F0F0F0"),
                               x=0.5, xanchor="center"),
                    margin=dict(l=70, r=20, t=50, b=45),
                    font=dict(color="#E8E8E8"),
                    plot_bgcolor="#111111", paper_bgcolor="#111111",
                )
                fig.update_xaxes(showgrid=False, linecolor="#666",
                                 title_text="", tickfont=dict(size=13),
                                 automargin=True)
                fig.update_yaxes(gridcolor="#333", zeroline=False,
                                 tickfont=dict(size=13), automargin=True,
                                 ticklabelstandoff=6)
                chart(fig)
    else:
        st.info("Select at least one feature.")

# --------------------------------------------------------------------------
# Tab 4 — Feature selection (mRMR)
# --------------------------------------------------------------------------
with tab_select:
    st.subheader("mRMR feature ranking")
    st.caption("Ranking computed on the training subjects only. "
               f"Top {config.TOP_K_FEATURES} are used by the SVM classifier.")
    rank_df = pd.DataFrame({
        "rank": range(1, len(out.ranked_features) + 1),
        "feature": out.ranked_features,
        "relevance (MI)": [out.relevance[f] for f in out.ranked_features],
        "selected": [f in out.top_k for f in out.ranked_features],
    })
    fig = px.bar(rank_df, x="feature", y="relevance (MI)", color="selected",
                 color_discrete_map={True: "#27AE60", False: "#B0BEC5"},
                 title="Mutual-information relevance (bars) · mRMR order (left→right)")
    fig.update_xaxes(categoryorder="array", categoryarray=out.ranked_features)
    chart(fig)
    st.dataframe(rank_df, width="stretch", hide_index=True)

# --------------------------------------------------------------------------
# Tab 5 — Classification
# --------------------------------------------------------------------------
with tab_classify:
    st.subheader("Classification results (held-out subject "
                 f"{config.TEST_SUBJECT})")
    chosen = st.multiselect("Models", [r.name for r in out.results],
                            default=[r.name for r in out.results])
    results = [r for r in out.results if r.name in chosen]

    if results:
        table = pd.DataFrame([{
            "Model": r.name,
            "Accuracy": r.accuracy,
            "Precision": r.precision,
            "Recall": r.recall,
            "F1": r.f1,
            "CV-Acc (train)": r.cv_accuracy,
            "AUC": r.auc,
            "Features": "top-3" if len(r.features_used) == config.TOP_K_FEATURES else "all 14",
        } for r in results])
        st.dataframe(
            table.style.format({c: "{:.3f}" for c in
                                ["Accuracy", "Precision", "Recall", "F1",
                                 "CV-Acc (train)", "AUC"]}, na_rep="—")
                 .background_gradient(subset=["F1"], cmap="Greens"),
            width="stretch", hide_index=True)
        st.caption("Tham chiếu bài báo (BME 2024, 10 đối tượng): "
                   f"KNN F1 ≈ {config.PAPER_REFERENCE['KNN_F1']}, "
                   f"AUC ≈ {config.PAPER_REFERENCE['KNN_AUC']} — là **một fold "
                   "LOSO thuận lợi** (test trên 1 đối tượng), không phải trung "
                   f"bình gộp. Demo này giữ riêng **subject {config.TEST_SUBJECT}** "
                   "làm tập test và tái lập kết quả đó: KNN(1NN) Acc=0.988, "
                   "F1=0.977, AUC=0.992.")

        st.markdown("### Phương pháp huấn luyện (bám sát bài báo)")
        st.caption("Đúng như code MATLAB gốc: mỗi mô hình **chuẩn hoá đặc trưng "
                   "(StandardScaler)** rồi phân loại trên **cả 14 đặc trưng** — "
                   "KNN dùng **1 láng giềng gần nhất (Euclidean)**. Huấn luyện "
                   f"theo **leave-one-subject-out**: train trên các subject "
                   f"{config.TRAIN_SUBJECTS}, test trên subject "
                   f"{config.TEST_SUBJECT} (mô hình chưa từng thấy). **Không** "
                   "dùng SMOTE hay tinh chỉnh ngưỡng — giữ nguyên ngưỡng quyết "
                   "định mặc định 0.5 để số liệu so sánh trực tiếp với bài báo.")
        tune_table = pd.DataFrame([{
            "Model": r.name,
            "Đặc trưng": "14 (đầy đủ)",
            "Siêu tham số": ", ".join(f"{k}={v}" for k, v in r.best_params.items())
                            or "(mặc định)",
            "Ngưỡng P(Fatigue)": r.threshold,
        } for r in results])
        st.dataframe(
            tune_table.style.format({"Ngưỡng P(Fatigue)": "{:.3f}"}),
            width="stretch", hide_index=True)

        st.markdown("### Confusion matrices")
        n_cm_cols = min(len(results), 2)   # fewer columns -> larger tiles
        cols = st.columns(n_cm_cols, gap="large")
        for i, r in enumerate(results):
            with cols[i % n_cm_cols]:
                z = r.confusion
                fig = px.imshow(z, text_auto=True, color_continuous_scale="Blues",
                                x=["Pred N", "Pred F"], y=["True N", "True F"],
                                title=r.name)
                fig.update_layout(
                    coloraxis_showscale=False, height=460,
                    title=dict(font=dict(size=24), x=0.5, xanchor="center"),
                    margin=dict(l=20, r=20, t=60, b=20),
                )
                fig.update_traces(textfont_size=40)          # cell numbers
                fig.update_xaxes(tickfont=dict(size=18), side="bottom")
                fig.update_yaxes(tickfont=dict(size=18))
                chart(fig)

        st.markdown("### ROC curves")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                 line=dict(dash="dash", color="grey"),
                                 name="chance", showlegend=True))
        for r in results:
            if r.roc is not None:
                fpr, tpr = r.roc
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                         name=f"{r.name} (AUC={r.auc:.3f})"))
        fig.update_layout(xaxis_title="False positive rate",
                          yaxis_title="True positive rate", height=450)
        chart(fig)
    else:
        st.info("Select at least one model.")

# --------------------------------------------------------------------------
# Tab 6 — Predict / Inference (signal in -> fatigue or not)
# --------------------------------------------------------------------------
with tab_predict:
    st.subheader("Đưa tín hiệu vào → có mỏi hay không")
    st.caption("Chọn một file + một kênh EMG **của subject "
               f"{config.TEST_SUBJECT} (tập test, mô hình chưa từng thấy)**, "
               "bấm Dự đoán: mỗi mô hình đã huấn luyện sẽ trả lời Normal / "
               "Fatigue kèm xác suất P(Fatigue). Nhãn thật (suy từ tên file) "
               "hiển thị để đối chiếu. Chỉ dùng dữ liệu subject test vì đây "
               "mới là phép thử công bằng cho khả năng tổng quát hoá — dữ "
               "liệu subject train mô hình đã thấy trong lúc huấn luyện.")

    p_files = [f for f in dl.list_files() if f.subject == config.TEST_SUBJECT]
    p_names = [f.name for f in p_files]
    p_sel_file = st.selectbox("File (subject "
                              f"{config.TEST_SUBJECT})", p_names, key="pred_file")
    p_info = next(f for f in p_files if f.name == p_sel_file)

    p_channels = load_signal(str(p_info.path))
    p_mask = dl.valid_channel_mask(p_channels)
    p_valid_idx = list(np.where(p_mask)[0])

    st.caption(f"Nhãn thật: **{label_name(p_info.label)}** · condition "
               f"`{p_info.condition}` · subject {p_info.subject} = **tập test "
               "(chưa từng thấy)** → dự đoán đáng tin cậy")

    p_sel_ch = st.selectbox("Channel", p_valid_idx,
                            format_func=lambda i: f"Channel {i}",
                            key="pred_channel")

    if st.button("Dự đoán", type="primary"):
        x = p_channels[:, p_sel_ch]
        feats = fe.extract_channel_features(x)
        t = np.arange(len(x)) / config.FS
        step = max(1, len(x) // 20000)
        t_dec, x_dec = t[::step], x[::step]

        # ── ① Tín hiệu EMG & đặc trưng miền thời gian ──────────
        st.markdown("---")
        st.markdown("### ① Tín hiệu EMG · Đặc trưng miền thời gian")
        st.caption("Tín hiệu thô của kênh EMG đã chọn. Các đường chú thích "
                   "cho thấy vị trí của Mean, dải ±1 STD, và điểm Max / Min "
                   "— 4 trong 8 đặc trưng miền thời gian.")
        chart(viz.build_time_domain_figure(t_dec, x_dec, feats, p_sel_ch))

        # ── ② Phổ công suất & đặc trưng miền tần số ─────────────
        st.markdown("### ② Phổ công suất (PSD) · Đặc trưng miền tần số")
        st.caption("Phổ công suất (periodogram) cho thấy năng lượng tín hiệu "
                   "phân bố theo tần số. MDF (tần số trung vị) và MNF (tần số "
                   "trung bình) là hai chỉ dấu kinh điển — cả hai đều **giảm** "
                   "khi cơ mỏi.")
        f_psd, pxx = fe.channel_psd(x)
        chart(viz.build_psd_figure(f_psd, pxx, feats))

        # ── ③ Vector đặc trưng → Dự đoán ────────────────────────
        st.markdown("### ③ Vector 14 đặc trưng → Kết quả dự đoán")
        st.caption("14 đặc trưng (8 miền thời gian + 6 miền tần số) được "
                   "đưa vào các mô hình đã huấn luyện để phân loại "
                   "Normal / Fatigue.")

        preds = inf.predict_channel(x, out.results)
        truth = int(p_info.label)
        rows = []
        for p in preds:
            p_fat = p["p_fatigue"]
            rows.append({
                "Model": p["model"],
                "Dự đoán": p["pred_name"],
                "P(Fatigue)": p_fat,
                "Đúng?": "✅" if p["pred"] == truth else "❌",
            })
        pred_df = pd.DataFrame(rows)

        def _color_pred(v):
            if v == "Fatigue":
                return f"color: {LABEL_COLORS['Fatigue']}; font-weight: bold"
            if v == "Normal":
                return f"color: {LABEL_COLORS['Normal']}; font-weight: bold"
            return ""

        col_feat, col_pred = st.columns(2, gap="large")

        with col_feat:
            st.markdown("#### 14 đặc trưng của kênh này")
            with st.container(key="predtable_features"):
                st.markdown(
                    _build_feature_tooltip_table(feats),
                    unsafe_allow_html=True,
                )

        with col_pred:
            st.markdown("#### Kết quả dự đoán của 4 mô hình")
            with st.container(key="predtable_verdict"):
                st.table(
                    pred_df.style
                           .format({"P(Fatigue)": "{:.1%}"}, na_rep="—")
                           .map(_color_pred, subset=pd.IndexSlice[:, ["Dự đoán"]])
                           .hide(axis="index"))

            n_agree = sum(1 for p in preds if p["pred"] == truth)
            st.caption(f"{n_agree}/{len(preds)} mô hình dự đoán khớp nhãn thật "
                       f"(**{label_name(truth)}**).")
    else:
        st.info("Chọn file + kênh rồi bấm **Dự đoán** để xem kết quả.")

    # ── ④ Thống kê dự đoán trên toàn bộ 64 kênh của file ────────
    st.markdown("---")
    st.markdown("### ④ Thống kê dự đoán 64 kênh (toàn bộ file)")
    st.caption("Chạy từng mô hình trên **tất cả kênh hợp lệ** của file đang "
               "chọn ở trên, rồi đếm số kênh được dự đoán Mỏi / Không mỏi. "
               "Cho thấy các mô hình có nhất quán trên toàn bộ file hay không, "
               "thay vì chỉ nhìn 1 kênh đơn lẻ.")

    if st.button("Thống kê 64 kênh", key="stat_button"):
        truth = int(p_info.label)
        per_model: dict[str, list[int]] = {r.name: [] for r in out.results
                                            if r.fitted_estimator is not None}
        for ch in p_valid_idx:
            x_ch = p_channels[:, ch]
            for p in inf.predict_channel(x_ch, out.results):
                per_model[p["model"]].append(p["pred"])

        stat_rows = []
        for model, preds_ch in per_model.items():
            n_total = len(preds_ch)
            n_fatigue = sum(preds_ch)
            n_normal = n_total - n_fatigue
            n_match = sum(1 for pr in preds_ch if pr == truth)
            stat_rows.append({
                "Model": model,
                "Số kênh Mỏi": n_fatigue,
                "Số kênh Không mỏi": n_normal,
                "% Mỏi": n_fatigue / n_total if n_total else 0.0,
                "% khớp nhãn thật": n_match / n_total if n_total else 0.0,
            })
        stat_df = pd.DataFrame(stat_rows)

        st.caption(f"File **{p_info.name}** · nhãn thật **{label_name(truth)}** · "
                   f"{len(p_valid_idx)}/{config.N_CHANNELS} kênh hợp lệ đã dùng.")
        with st.container(key="predtable_stats"):
            st.table(
                stat_df.style
                       .format({"% Mỏi": "{:.1%}", "% khớp nhãn thật": "{:.1%}"})
                       .background_gradient(subset=["% Mỏi"], cmap="Reds")
                       .background_gradient(subset=["% khớp nhãn thật"], cmap="Greens")
                       .hide(axis="index"))
    else:
        st.info("Bấm **Thống kê 64 kênh** để xem tổng hợp dự đoán trên toàn bộ "
                "kênh hợp lệ của file.")
