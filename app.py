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
    </style>
    """,
    unsafe_allow_html=True,
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
    kwargs.setdefault("use_container_width", True)
    st.plotly_chart(fig, **kwargs)


out = get_pipeline()
df = out.feature_table
df = df.assign(Class=df["label"].map(config.LABEL_NAMES))
files = dl.list_files()

st.title("🦾 EMG Muscle-Fatigue Classification — Demo")
st.caption("sEMG (64-channel, 2000 Hz) · 14 features/channel · mRMR + SVM/KNN/LDA/DecisionTree · "
           "leave-one-subject-out (test = subject "
           f"{config.TEST_SUBJECT})")

(tab_overview, tab_signal, tab_features, tab_select, tab_classify,
 tab_predict) = st.tabs(
    ["📊 Overview", "📈 Signal & PSD", "🎯 Features", "🏅 Feature Selection",
     "🤖 Classification", "🔮 Predict / Inference"]
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

    st.markdown("**Per-file summary** (label from filename: contains "
                f"`{config.FATIGUE_KEYWORD}` → Fatigue)")
    per_file = (df.groupby(["file", "subject", "condition", "Class"])
                  .size().reset_index(name="valid_channels")
                  .sort_values(["subject", "condition"]))
    st.dataframe(per_file, use_container_width=True, hide_index=True)

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
    fdf = pd.DataFrame({"feature": config.FEATURE_NAMES,
                        "value": [feats[n] for n in config.FEATURE_NAMES]})
    st.dataframe(fdf, use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------
# Tab 3 — Feature distributions
# --------------------------------------------------------------------------
with tab_features:
    st.subheader("Feature distributions: Normal vs Fatigue")
    st.caption("Boxplots show which features separate the two classes.")
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
    st.dataframe(rank_df, use_container_width=True, hide_index=True)

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
            use_container_width=True, hide_index=True)
        st.caption("Paper reference (BME 2024, 10 subjects): "
                   f"KNN F1 ≈ {config.PAPER_REFERENCE['KNN_F1']}, "
                   f"AUC ≈ {config.PAPER_REFERENCE['KNN_AUC']}. "
                   "This demo uses 5 subjects, so absolute values differ.")

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
    st.caption("Chọn một file + một kênh EMG, bấm Dự đoán: mỗi mô hình đã huấn "
               "luyện sẽ trả lời Normal / Fatigue kèm xác suất P(Fatigue). "
               "Nhãn thật (suy từ tên file) hiển thị để đối chiếu.")

    p_files = dl.list_files()
    p_names = [f.name for f in p_files]
    p_sel_file = st.selectbox("File", p_names, key="pred_file")
    p_info = next(f for f in p_files if f.name == p_sel_file)

    p_channels = load_signal(str(p_info.path))
    p_mask = dl.valid_channel_mask(p_channels)
    p_valid_idx = list(np.where(p_mask)[0])

    # Honest note: was this subject in the training set or held out?
    if p_info.subject == config.TEST_SUBJECT:
        seen = (f"subject {p_info.subject} = **tập test (chưa từng thấy)** "
                "→ dự đoán đáng tin hơn")
    elif p_info.subject in config.TRAIN_SUBJECTS:
        seen = (f"subject {p_info.subject} thuộc **tập train (mô hình đã thấy "
                "dữ liệu tương tự)**")
    else:
        seen = f"subject {p_info.subject} không thuộc train/test"
    st.caption(f"Nhãn thật: **{label_name(p_info.label)}** · condition "
               f"`{p_info.condition}` · {seen}")

    p_sel_ch = st.selectbox("Channel", p_valid_idx,
                            format_func=lambda i: f"Channel {i}",
                            key="pred_channel")

    if st.button("🔮 Dự đoán", type="primary"):
        x = p_channels[:, p_sel_ch]

        # Context waveform for the chosen channel.
        t = np.arange(len(x)) / config.FS
        step = max(1, len(x) // 20000)
        fig_p = px.line(x=t[::step], y=x[::step],
                        labels={"x": "time (s)", "y": "amplitude"},
                        title=f"Tín hiệu vào — channel {p_sel_ch}")
        fig_p.update_traces(line=dict(width=1))
        chart(fig_p)

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

        st.markdown("### Kết quả dự đoán của 4 mô hình")
        st.dataframe(
            pred_df.style
                   .format({"P(Fatigue)": "{:.1%}"}, na_rep="—")
                   .map(_color_pred, subset=pd.IndexSlice[:, ["Dự đoán"]]),
            use_container_width=True, hide_index=True)

        n_agree = sum(1 for p in preds if p["pred"] == truth)
        st.caption(f"{n_agree}/{len(preds)} mô hình dự đoán khớp nhãn thật "
                   f"(**{label_name(truth)}**).")

        st.markdown("**14 đặc trưng của kênh này (đầu vào cho mô hình)**")
        feats = fe.extract_channel_features(x)
        fdf = pd.DataFrame({"feature": config.FEATURE_NAMES,
                            "value": [feats[n] for n in config.FEATURE_NAMES]})
        st.dataframe(fdf, use_container_width=True, hide_index=True)
    else:
        st.info("Chọn file + kênh rồi bấm **🔮 Dự đoán** để xem kết quả.")
