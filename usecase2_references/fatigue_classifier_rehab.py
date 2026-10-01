"""
fatigue_classifier_rehab.py
---------------------------
Ghép Usecase 1 (phân loại mỏi cơ, KNN) làm ĐỘNG CƠ tính điểm cho Usecase 2
(theo dõi sức bền qua các buổi PHCN).

Ý tưởng:
  1) Mỗi buổi: mô phỏng 1 bài CO CƠ DUY TRÌ với động học mỏi thật
       - trong bài: mỏi tăng dần  ->  RMS TĂNG, MDF GIẢM  (mỏi ngoại biên kinh điển)
       - buổi phục hồi tốt hơn: mỏi đến MUỘN hơn (độ dốc mỏi thoải hơn)
  2) Cắt cửa sổ, trích đặc trưng họ time/freq (RMS, MAV, WL, ZC, SSC, MDF, MNF)
  3) KNN phân loại từng cửa sổ: MỎI / CHƯA MỎI
  4) Điểm sức bền của buổi = THỜI ĐIỂM KHỞI PHÁT MỎI (giây) do KNN xác định
       -> không phụ thuộc biên độ tuyệt đối (giải quyết nhập nhằng "tăng/giảm")
  5) Theo dõi điểm sức bền TĂNG DẦN qua các buổi -> đầu ra Usecase 2

Dữ liệu mô phỏng cho demo — KHÔNG phải bệnh nhân thật.
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

FS = 2048
WIN_SEC, OVERLAP = 1.0, 0.5
RMS0, MDF0 = 20.0, 105.0          # mức nền lúc chưa mỏi

# ---- điểm sức bền mỗi buổi: thời gian (giây) để mỏi đạt ngưỡng (f=0.5) ----
# buổi đầu mỏi nhanh (~6s), buổi cuối mỏi chậm (~24s) -> phục hồi
def session_t_half(n_sessions=8):
    rng = np.random.default_rng(11)
    return np.linspace(6, 24, n_sessions) + rng.normal(0, 1.0, n_sessions)


def _bp_noise(n, center, rms):
    lo, hi = 20.0, float(np.clip(2 * center + 40, 120, 480))
    x = rng.standard_normal(n)
    sos = butter(4, [lo, hi], btype="band", fs=FS, output="sos")
    x = sosfiltfilt(sos, x)
    return (x * rms / (np.sqrt(np.mean(x ** 2)) + 1e-9)).astype(np.float32)


def simulate_sustained(t_half, dur=30.0):
    """Sinh tín hiệu co cơ duy trì; trả về (signal, latent_fatigue f(t))."""
    n = int(dur * FS)
    t = np.arange(n) / FS
    f = 1 - np.exp(-t / (t_half / np.log(2)))       # f=0.5 tại t=t_half
    step = int(WIN_SEC * FS)
    sig = np.zeros(n, dtype=np.float32)
    for a in range(0, n - step, step // 2):          # dựng theo block chuẩn tĩnh
        fc = f[a + step // 2]
        rms = RMS0 * (1 + 0.6 * fc)                   # mỏi -> biên độ TĂNG
        cen = MDF0 * (1 - 0.35 * fc)                  # mỏi -> MDF GIẢM
        sig[a:a + step] = _bp_noise(step, cen, rms)
    return sig, f


# ---------- đặc trưng sEMG (đúng họ feature của anh) ----------
def features(w):
    diff = np.diff(w)
    zc = np.sum((w[:-1] * w[1:] < 0) & (np.abs(diff) > 1e-3))
    ssc = np.sum((diff[:-1] * diff[1:] < 0))
    fr, p = welch(w, fs=FS, nperseg=512)
    cs = np.cumsum(p)
    mdf = fr[np.searchsorted(cs, cs[-1] / 2)]
    mnf = np.sum(fr * p) / (np.sum(p) + 1e-12)
    return [np.sqrt(np.mean(w ** 2)), np.mean(np.abs(w)),
            np.sum(np.abs(diff)), zc, ssc, mdf, mnf]

FEAT_NAMES = ["RMS", "MAV", "WL", "ZC", "SSC", "MDF", "MNF"]


def windowize(sig, f):
    step = int(WIN_SEC * FS)
    hop = int(step * (1 - OVERLAP))
    X, y, tc = [], [], []
    for a in range(0, len(sig) - step, hop):
        w = sig[a:a + step]
        X.append(features(w))
        y.append(int(f[a + step // 2] > 0.5))         # nhãn thật: mỏi nếu f>0.5
        tc.append((a + step / 2) / FS)
    return np.array(X), np.array(y), np.array(tc)


rng = np.random.default_rng(11)

def run():
    t_halves = session_t_half()
    sessions = []
    Xall, yall = [], []
    for s, th in enumerate(t_halves, start=1):
        sig, f = simulate_sustained(th)
        X, y, tc = windowize(sig, f)
        sessions.append({"session": s, "X": X, "y": y, "tc": tc, "t_half": th})
        Xall.append(X)
        yall.append(y)
    Xall = np.vstack(Xall)
    yall = np.concatenate(yall)

    # ---- huấn luyện KNN phân loại mỏi ----
    Xtr, Xte, ytr, yte = train_test_split(Xall, yall, test_size=0.3,
                                          random_state=0, stratify=yall)
    sc = StandardScaler().fit(Xtr)
    knn = KNeighborsClassifier(n_neighbors=7).fit(sc.transform(Xtr), ytr)
    f1 = f1_score(yte, knn.predict(sc.transform(Xte)))

    # ---- áp KNN lên từng buổi -> điểm sức bền = thời điểm khởi phát mỏi ----
    for ss in sessions:
        proba = knn.predict_proba(sc.transform(ss["X"]))[:, 1]
        ss["proba"] = proba
        onset_idx = np.argmax(proba >= 0.5) if np.any(proba >= 0.5) else len(proba)
        ss["endurance_sec"] = float(ss["tc"][min(onset_idx, len(ss["tc"]) - 1)]) \
            if np.any(proba >= 0.5) else float(ss["tc"][-1])
        ss["pct_nonfatigue"] = float(np.mean(proba < 0.5) * 100)
    return sessions, f1, knn


if __name__ == "__main__":
    sessions, f1, _ = run()
    print(f"KNN fatigue classifier — F1 (test, mô phỏng) = {f1:.3f}\n")
    print(f"{'Buổi':>4} {'t_half(s)':>9} {'Điểm sức bền (s)':>16} {'% chưa mỏi':>11}")
    for ss in sessions:
        print(f"{ss['session']:>4} {ss['t_half']:>9.1f} "
              f"{ss['endurance_sec']:>16.1f} {ss['pct_nonfatigue']:>11.1f}")
