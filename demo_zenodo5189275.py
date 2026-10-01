"""
Streamlit demo — Zenodo 5189275 (EMG Fatigue Dataset)
15 subjects, 8 channels, 200 Hz, 120-second isometric hold
Run: streamlit run demo_zenodo5189275.py --server.port 8504 --server.headless true
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from scipy import stats as sp_stats, signal as sp_signal
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FS = 200
N_CH = 8
WIN_SEC = 2.0
STEP_SEC = 1.0
NORMAL_END = 0.40
FATIGUE_START = 0.60

DATA_DIR = Path("/home/vsf-thienlg-u/Downloads/zenodo_5189275/data/Dataset EMG Fatigue/Data as txt Files")

st.set_page_config(
    page_title="sEMG Fatigue Demo — Zenodo 5189275",
    page_icon="💪",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_features(x):
    rms  = float(np.sqrt(np.mean(x**2)))
    mav  = float(np.mean(np.abs(x)))
    skew = float(sp_stats.skew(x, bias=True))
    kurt = float(sp_stats.kurtosis(x, fisher=False, bias=True))
    std  = float(np.std(x, ddof=1))
    f, pxx = sp_signal.periodogram(x, fs=FS)
    total = np.sum(pxx)
    if total > 0:
        mnf = float(np.sum(f * pxx) / total)
        cum = np.cumsum(pxx)
        idx = min(int(np.searchsorted(cum, total / 2)), len(f) - 1)
        mdf = float(f[idx])
        p = pxx / total
        p = p[p > 0]
        ent = float(-np.sum(p * np.log2(p)))
    else:
        mnf = mdf = ent = 0.0
    return [rms, mav, skew, kurt, std, mnf, mdf, ent]

FEAT_NAMES_BASE = ["RMS", "MAV", "Skew", "Kurt", "STD", "MNF", "MDF", "Ent"]
FEAT_NAMES = [f"{n}_ch{c}" for c in range(N_CH) for n in FEAT_NAMES_BASE]


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Đang tải và trích đặc trưng…")
def load_dataset():
    rows = []
    win  = int(WIN_SEC * FS)
    step = int(STEP_SEC * FS)
    raw_signals = {}

    for txt in sorted(DATA_DIR.glob("sub*.txt")):
        subj = int(txt.stem.replace("sub", ""))
        data = np.loadtxt(txt)
        raw_signals[subj] = data
        n = len(data)
        ne = int(n * NORMAL_END)
        fs2 = int(n * FATIGUE_START)

        for pos in range(0, n - win + 1, step):
            mid = pos + win // 2
            if mid < ne:
                label = 0
            elif mid >= fs2:
                label = 1
            else:
                continue
            feats_all = []
            valid = True
            for ch in range(N_CH):
                seg = data[pos : pos + win, ch]
                if np.std(seg) < 1e-6:
                    valid = False
                    break
                feats_all.extend(extract_features(seg))
            if valid:
                rows.append(feats_all + [subj, label, pos / FS])

    df = pd.DataFrame(rows, columns=FEAT_NAMES + ["subject", "label", "time_start"])
    return df, raw_signals


@st.cache_data(show_spinner="Đang chạy LOSO…")
def run_loso(_df):
    models = {
        "KNN(1-NN)":   Pipeline([("sc", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=1, metric="euclidean"))]),
        "KNN(5-NN)":   Pipeline([("sc", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5, metric="euclidean"))]),
        "SVM(linear)": Pipeline([("sc", StandardScaler()), ("clf", CalibratedClassifierCV(SVC(kernel="linear", random_state=42), ensemble=False))]),
        "LDA":         Pipeline([("sc", StandardScaler()), ("clf", LinearDiscriminantAnalysis())]),
    }
    results = []
    for test_s in sorted(_df.subject.unique()):
        tr = _df[_df.subject != test_s]
        te = _df[_df.subject == test_s]
        X_tr, y_tr = tr[FEAT_NAMES].values, tr.label.values
        X_te, y_te = te[FEAT_NAMES].values, te.label.values
        for name, m in models.items():
            from sklearn.base import clone
            est = clone(m)
            est.fit(X_tr, y_tr)
            yp = est.predict(X_te)
            yprob = est.predict_proba(X_te)[:, 1]
            results.append({
                "model": name,
                "subject": int(test_s),
                "acc": accuracy_score(y_te, yp),
                "f1":  f1_score(y_te, yp, zero_division=0),
                "auc": roc_auc_score(y_te, yprob),
            })
    return pd.DataFrame(results), models


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
st.title("💪 sEMG Fatigue Detection — Zenodo 5189275")
st.markdown(
    """
**Dataset:** [Zenodo 5189275](https://zenodo.org/records/5189275) · 15 subjects · 8 kênh · 200 Hz · 120 giây/người
**Giao thức:** Giữ isometric elbow flexion (tạ 6 kg, góc 90°) cho đến khi mỏi
**Nhãn (time-based):** 0–40% thời gian → Normal | 60–100% → Fatigue | 40–60% bỏ qua (transition zone)
"""
)

df, raw_signals = load_dataset()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Tổng quan",
    "📈 Tín hiệu EMG",
    "🔬 Phân tích đặc trưng",
    "🏆 Kết quả LOSO",
    "🎯 Demo real-time",
])

# ------------------------------------------------------------------
# Tab 1 — Overview
# ------------------------------------------------------------------
with tab1:
    st.header("Tổng quan dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Subjects", df.subject.nunique())
    c2.metric("Samples (windows)", len(df))
    c3.metric("Normal", int((df.label == 0).sum()))
    c4.metric("Fatigue", int((df.label == 1).sum()))

    st.subheader("Phân bố samples theo subject")
    counts = df.groupby(["subject", "label"]).size().reset_index(name="count")
    counts["label_str"] = counts.label.map({0: "Normal", 1: "Fatigue"})
    fig = px.bar(
        counts, x="subject", y="count", color="label_str",
        barmode="group",
        labels={"subject": "Subject", "count": "Số windows", "label_str": "Nhãn"},
        color_discrete_map={"Normal": "#4CAF50", "Fatigue": "#F44336"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Thông tin dataset")
    st.markdown("""
| Thông số | Giá trị |
|---|---|
| Tần số lấy mẫu | 200 Hz |
| Số kênh | 8 |
| Thời gian/subject | 120 giây |
| Cửa sổ phân tích | 2 giây |
| Bước nhảy | 1 giây (50% overlap) |
| Số đặc trưng | 8 × 8 kênh = 64 |
| Phân loại | Normal vs Fatigue |
| Nguồn gốc | Isometric elbow flexion, biceps brachii |
    """)

# ------------------------------------------------------------------
# Tab 2 — EMG signals
# ------------------------------------------------------------------
with tab2:
    st.header("Tín hiệu EMG theo thời gian")
    col_s, col_c = st.columns(2)
    sel_subj = col_s.selectbox("Subject", sorted(raw_signals.keys()), key="sig_subj")
    sel_ch   = col_c.selectbox("Kênh", list(range(N_CH)), format_func=lambda c: f"CH {c+1}", key="sig_ch")

    sig = raw_signals[sel_subj][:, sel_ch]
    t   = np.arange(len(sig)) / FS
    n   = len(sig)
    ne  = int(n * NORMAL_END)
    fs2 = int(n * FATIGUE_START)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t[:ne], y=sig[:ne], mode="lines", name="Normal",
                             line=dict(color="#4CAF50", width=0.8)))
    fig.add_trace(go.Scatter(x=t[ne:fs2], y=sig[ne:fs2], mode="lines", name="Transition",
                             line=dict(color="#FF9800", width=0.8)))
    fig.add_trace(go.Scatter(x=t[fs2:], y=sig[fs2:], mode="lines", name="Fatigue",
                             line=dict(color="#F44336", width=0.8)))
    fig.add_vline(x=t[ne],  line_dash="dash", line_color="#FF9800", annotation_text="40%")
    fig.add_vline(x=t[fs2], line_dash="dash", line_color="#F44336",  annotation_text="60%")
    fig.update_layout(
        xaxis_title="Thời gian (s)", yaxis_title="Biên độ (a.u.)",
        title=f"Subject {sel_subj} — CH {sel_ch+1}",
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Phổ tần số (MNF/MDF theo thời gian)")
    win = int(WIN_SEC * FS)
    step = int(STEP_SEC * FS)
    times_c, mnf_list, mdf_list = [], [], []
    for pos in range(0, n - win + 1, step):
        seg = sig[pos:pos+win]
        tc  = (pos + win / 2) / FS
        f_arr, pxx = sp_signal.periodogram(seg, fs=FS)
        total = np.sum(pxx)
        if total > 0:
            mnf_list.append(np.sum(f_arr * pxx) / total)
            cum = np.cumsum(pxx)
            idx = min(int(np.searchsorted(cum, total/2)), len(f_arr)-1)
            mdf_list.append(f_arr[idx])
        else:
            mnf_list.append(0)
            mdf_list.append(0)
        times_c.append(tc)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=times_c, y=mnf_list, mode="lines+markers",
                              name="MNF", line=dict(color="royalblue")))
    fig2.add_trace(go.Scatter(x=times_c, y=mdf_list, mode="lines+markers",
                              name="MDF", line=dict(color="tomato")))
    fig2.add_vline(x=t[ne],  line_dash="dash", line_color="#FF9800")
    fig2.add_vline(x=t[fs2], line_dash="dash", line_color="#F44336")
    fig2.update_layout(
        xaxis_title="Thời gian (s)", yaxis_title="Tần số (Hz)",
        title="MNF & MDF theo thời gian (xu hướng giảm khi mỏi)", height=300,
    )
    st.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------------------------------
# Tab 3 — Feature analysis
# ------------------------------------------------------------------
with tab3:
    st.header("Phân bố đặc trưng")
    feat_base = st.selectbox("Đặc trưng", FEAT_NAMES_BASE, key="feat_base")
    ch_sel    = st.selectbox("Kênh", list(range(N_CH)), format_func=lambda c: f"CH {c+1}", key="feat_ch")
    feat_col  = f"{feat_base}_ch{ch_sel}"

    df_plot = df[["label", "subject", feat_col]].copy()
    df_plot["label_str"] = df_plot.label.map({0: "Normal", 1: "Fatigue"})

    fig = px.violin(
        df_plot, x="label_str", y=feat_col, color="label_str",
        box=True, points="outliers",
        color_discrete_map={"Normal": "#4CAF50", "Fatigue": "#F44336"},
        labels={"label_str": "", feat_col: feat_col},
        title=f"Phân bố {feat_col}",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Tương quan đặc trưng (CH 0, Normal vs Fatigue)")
    avg_normal  = df[df.label==0][[f"{n}_ch0" for n in FEAT_NAMES_BASE]].mean()
    avg_fatigue = df[df.label==1][[f"{n}_ch0" for n in FEAT_NAMES_BASE]].mean()
    comp = pd.DataFrame({"Normal": avg_normal.values, "Fatigue": avg_fatigue.values},
                         index=FEAT_NAMES_BASE)
    fig3 = px.bar(comp.reset_index(), x="index", y=["Normal","Fatigue"],
                  barmode="group",
                  labels={"index":"Đặc trưng","value":"Giá trị trung bình"},
                  color_discrete_map={"Normal":"#4CAF50","Fatigue":"#F44336"},
                  title="Giá trị trung bình các đặc trưng — CH 0")
    st.plotly_chart(fig3, use_container_width=True)

# ------------------------------------------------------------------
# Tab 4 — LOSO results
# ------------------------------------------------------------------
with tab4:
    st.header("Kết quả Leave-One-Subject-Out (LOSO)")
    with st.spinner("Đang chạy 15-fold LOSO cho 4 mô hình…"):
        res_df, _ = run_loso(df)

    summary = res_df.groupby("model")[["acc","f1","auc"]].mean().reset_index()
    summary.columns = ["Mô hình","Accuracy","F1-Score","AUC"]
    summary = summary.sort_values("Accuracy", ascending=False)

    best_model = summary.iloc[0]["Mô hình"]
    best_acc   = summary.iloc[0]["Accuracy"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Best model", best_model)
    c2.metric("Best Accuracy", f"{best_acc:.1%}")
    c3.metric("Best AUC", f"{summary.iloc[0]['AUC']:.3f}")

    st.subheader("Kết quả trung bình 15 subjects")
    st.dataframe(
        summary.style.format({"Accuracy": "{:.3f}", "F1-Score": "{:.3f}", "AUC": "{:.3f}"}),
        use_container_width=True, hide_index=True,
    )

    fig = px.bar(
        summary, x="Mô hình", y=["Accuracy","F1-Score","AUC"],
        barmode="group", title="So sánh các mô hình",
        labels={"value":"Điểm","variable":"Chỉ số"},
        color_discrete_sequence=["#2196F3","#4CAF50","#FF9800"],
    )
    fig.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Accuracy theo từng subject")
    fig2 = px.line(
        res_df[res_df.model == best_model], x="subject", y="acc",
        markers=True, title=f"{best_model} — Accuracy theo subject",
        labels={"subject":"Subject","acc":"Accuracy"},
    )
    fig2.add_hline(y=res_df[res_df.model==best_model].acc.mean(),
                   line_dash="dash", annotation_text="Trung bình")
    fig2.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------------------------------
# Tab 5 — Real-time demo
# ------------------------------------------------------------------
with tab5:
    st.header("Demo phân loại real-time")
    st.markdown(
        "Chọn một subject, một thời điểm trong phiên 120 giây — hệ thống sẽ trích đặc trưng và phân loại bằng mô hình đã train LOSO."
    )

    col1, col2, col3 = st.columns(3)
    rt_subj  = col1.selectbox("Subject test", sorted(raw_signals.keys()), key="rt_subj")
    rt_model = col2.selectbox("Mô hình", ["KNN(1-NN)","KNN(5-NN)","SVM(linear)","LDA"], key="rt_model")
    max_t    = len(raw_signals[rt_subj]) / FS - WIN_SEC
    rt_time  = col3.slider("Thời điểm bắt đầu (s)", 0.0, float(max_t), float(max_t * 0.7), step=0.5)

    if st.button("▶ Phân loại", type="primary"):
        sig_all = raw_signals[rt_subj]
        pos     = int(rt_time * FS)
        win_s   = int(WIN_SEC * FS)

        feat_vec = []
        for ch in range(N_CH):
            seg = sig_all[pos : pos + win_s, ch]
            feat_vec.extend(extract_features(seg))
        X_test = np.array(feat_vec).reshape(1, -1)

        train_df = df[df.subject != rt_subj]
        X_tr = train_df[FEAT_NAMES].values
        y_tr = train_df.label.values

        mdl_map = {
            "KNN(1-NN)":   Pipeline([("sc", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=1, metric="euclidean"))]),
            "KNN(5-NN)":   Pipeline([("sc", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5, metric="euclidean"))]),
            "SVM(linear)": Pipeline([("sc", StandardScaler()), ("clf", CalibratedClassifierCV(SVC(kernel="linear", random_state=42), ensemble=False))]),
            "LDA":         Pipeline([("sc", StandardScaler()), ("clf", LinearDiscriminantAnalysis())]),
        }
        est = mdl_map[rt_model]
        est.fit(X_tr, y_tr)
        pred  = est.predict(X_test)[0]
        prob  = est.predict_proba(X_test)[0]
        label_str = "🔴 FATIGUE" if pred == 1 else "🟢 NORMAL"

        n_total    = len(sig_all)
        true_label = 1 if rt_time / (n_total / FS) >= FATIGUE_START else (
                     0 if rt_time / (n_total / FS) <= NORMAL_END else -1)
        true_str   = "Fatigue" if true_label == 1 else ("Normal" if true_label == 0 else "Transition")
        correct    = "✅" if true_label == pred else ("⚠️" if true_label == -1 else "❌")

        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Dự đoán", label_str)
        rc2.metric("Xác suất Fatigue", f"{prob[1]:.1%}")
        rc3.metric(f"Nhãn thực ({correct})", true_str)

        fig_prob = go.Figure(go.Bar(
            x=["Normal", "Fatigue"], y=prob,
            marker_color=["#4CAF50","#F44336"],
            text=[f"{v:.1%}" for v in prob], textposition="outside",
        ))
        fig_prob.update_layout(yaxis_range=[0,1.1], title="Xác suất phân loại", height=300)
        st.plotly_chart(fig_prob, use_container_width=True)

        st.subheader(f"Đoạn tín hiệu [{rt_time:.1f}s – {rt_time+WIN_SEC:.1f}s]")
        t_seg = np.linspace(rt_time, rt_time + WIN_SEC, win_s)
        fig_seg = go.Figure()
        for ch in range(N_CH):
            fig_seg.add_trace(go.Scatter(
                x=t_seg, y=sig_all[pos:pos+win_s, ch],
                mode="lines", name=f"CH {ch+1}", line=dict(width=0.8),
            ))
        fig_seg.update_layout(xaxis_title="Thời gian (s)", yaxis_title="Biên độ",
                               height=300, showlegend=True)
        st.plotly_chart(fig_seg, use_container_width=True)

st.sidebar.markdown("""
---
**Dataset:** Zenodo 5189275
**Fs:** 200 Hz | **CH:** 8 | **Subjects:** 15
**Window:** 2s | **Step:** 1s
**Features:** 64 (8 × 8 kênh)
**Models:** KNN, SVM, LDA
**Validation:** LOSO 15-fold
""")
