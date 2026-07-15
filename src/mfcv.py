"""
mfcv.py — Ước lượng Muscle Fiber Conduction Velocity (MFCV) từ HD-sEMG.

Phương pháp: Maximum Likelihood Estimation đa kênh (Farina & Merletti; Luu et al.)
    x_k(n) = s(n - (k-1)*theta) + w_k(n)
    theta_hat = argmin_theta  sum_k || x_k - s_hat(n - (k-1)*theta) ||^2
Trễ phân số thực hiện bằng dịch pha trong miền tần số (tương đương nội suy sinc).

    MFCV = IED / (theta / Fs)      [m/s]

Cổng chất lượng (abstention) theo Luu, Ravier & Buttelli, IEEE ATC 2015:
    - hệ số tương quan giữa 2 kênh liền kề > 0.75
    - CV nằm trong dải sinh lý (2–10 m/s)
    - hướng lan truyền nhất quán, kênh không nằm ở vùng innervation zone (IZ)

Tham chiếu:
    Ravier, Luu, Jabloun, Buttelli — ISABEL 2011 (GCC)
    Luu, Ravier, Buttelli — IJABE 6(1):6-11, 2013 (MLE)
    Luu, Ravier, Buttelli — IEEE ATC 145-148, 2015 (validation dữ liệu thật)
    Luu, Boualem, Duy, Ravier, Buttelli — Fluct. Noise Lett. 17(2):1850015, 2018
    Nguyen, Luu et al. — APSIPA ASC 2023 (CV trên lưới 64 điện cực)
"""

from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


# ----------------------------------------------------------------------------
# Cấu hình
# ----------------------------------------------------------------------------

@dataclass
class GridConfig:
    """Cấu hình lưới điện cực. Mặc định theo APSIPA 2023: 13 hàng x 5 cột, IED 8 mm."""
    fs: float = 2000.0          # Hz  (demo dùng 2000; các bài báo dùng 2048)
    ied_m: float = 0.008        # m   khoảng cách liên điện cực dọc theo sợi cơ
    n_rows: int = 13            # số điện cực dọc theo hướng sợi cơ
    n_cols: int = 5
    bp_low: float = 20.0        # Hz  bandpass theo APSIPA 2023
    bp_high: float = 400.0      # Hz
    notch_hz: Optional[float] = 50.0


@dataclass
class QualityGate:
    """Ngưỡng của cổng abstention."""
    min_corr: float = 0.75      # Luu et al. 2015 / APSIPA 2023
    cv_min: float = 2.0         # m/s  dải sinh lý (mô hình TVD: 2–8 m/s)
    cv_max: float = 10.0        # m/s  Luu et al. 2015: CV phải < 10 m/s
    min_channels: int = 3       # tối thiểu 3 kênh vi sai đơn


@dataclass
class MFCVResult:
    """Kết quả một cửa sổ. accepted=False => hệ thống TỪ CHỐI kết luận."""
    t_s: float
    cv_ms: Optional[float]
    delay_samples: Optional[float]
    corr: Optional[float]
    accepted: bool
    reason: str = "ok"

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------------
# Tiền xử lý
# ----------------------------------------------------------------------------

def preprocess(x: np.ndarray, cfg: GridConfig) -> np.ndarray:
    """Lọc bandpass 20–400 Hz + notch. x: (n_ch, n_samples)."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    nyq = cfg.fs / 2.0
    hi = min(cfg.bp_high, 0.99 * nyq)
    b, a = butter(4, [cfg.bp_low / nyq, hi / nyq], btype="band")
    y = filtfilt(b, a, x, axis=-1)
    if cfg.notch_hz:
        bn, an = iirnotch(cfg.notch_hz / nyq, Q=30.0)
        y = filtfilt(bn, an, y, axis=-1)
    return y


def single_differential(x: np.ndarray) -> np.ndarray:
    """Vi sai đơn dọc theo cột: SD_k = x_{k+1} - x_k."""
    return np.diff(np.asarray(x, dtype=float), axis=0)


def double_differential(sd: np.ndarray) -> np.ndarray:
    """Vi sai kép từ tín hiệu vi sai đơn."""
    return np.diff(np.asarray(sd, dtype=float), axis=0)


# ----------------------------------------------------------------------------
# Innervation zone
# ----------------------------------------------------------------------------

def detect_innervation_zone(sd: np.ndarray) -> Optional[int]:
    """
    Dò vùng IZ trên tín hiệu vi sai đơn (APSIPA 2023, Fig. 2).

    Tại IZ, điện thế lan về hai phía => trễ giữa các cặp kênh liền kề ĐỔI DẤU.
    Trả về chỉ số kênh nơi xảy ra đảo chiều, hoặc None nếu lan truyền một chiều.
    """
    delays = []
    for k in range(sd.shape[0] - 1):
        d, _ = _gcc_delay(sd[k], sd[k + 1])
        delays.append(d)
    delays = np.asarray(delays)

    signs = np.sign(delays)
    valid = signs != 0
    if valid.sum() < 2:
        return None
    idx = np.flatnonzero(valid)
    s = signs[idx]
    flips = np.flatnonzero(np.diff(s) != 0)
    if flips.size == 0:
        return None
    return int(idx[flips[0]] + 1)


def select_propagation_channels(sd: np.ndarray, gate: QualityGate):
    """
    Chọn dải kênh nằm cùng một phía của IZ (lan truyền một chiều).
    Trả về (mảng kênh đã chọn, chỉ số IZ hoặc None).
    """
    iz = detect_innervation_zone(sd)
    if iz is None:
        return sd, None
    below, above = sd[:iz], sd[iz:]
    chosen = below if below.shape[0] >= above.shape[0] else above
    return chosen, iz


# ----------------------------------------------------------------------------
# Ước lượng trễ
# ----------------------------------------------------------------------------

def _gcc_delay(x1: np.ndarray, x2: np.ndarray, max_lag: int = 40):
    """
    Cross-correlation + nội suy parabol (Ravier, Luu et al., ISABEL 2011).
    Dùng để khởi tạo cho MLE và để dò IZ.
    Trả về (trễ [mẫu], hệ số tương quan đỉnh).
    """
    x1 = x1 - x1.mean()
    x2 = x2 - x2.mean()
    n = len(x1)
    denom = np.sqrt(np.sum(x1 ** 2) * np.sum(x2 ** 2))
    if denom == 0:
        return 0.0, 0.0

    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    R = np.fft.irfft(np.fft.rfft(x1, nfft) * np.conj(np.fft.rfft(x2, nfft)), nfft)
    R = np.concatenate((R[-max_lag:], R[: max_lag + 1])) / denom
    lags = np.arange(-max_lag, max_lag + 1)

    i = int(np.argmax(R))
    peak = float(R[i])
    # nội suy parabol cho phần thập phân
    if 0 < i < len(R) - 1:
        y0, y1, y2 = R[i - 1], R[i], R[i + 1]
        den = (y0 - 2 * y1 + y2)
        frac = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    else:
        frac = 0.0
    # Quy ước: trả về trễ CỦA x2 SO VỚI x1 (dương = x2 đến sau).
    # irfft(X1 * conj(X2)) cho đỉnh tại lag = -d, nên đảo dấu.
    return -float(lags[i] + frac), peak


def _shift_fd(X: np.ndarray, freqs: np.ndarray, tau: float) -> np.ndarray:
    """Dịch trễ phân số trong miền tần số (tương đương nội suy sinc)."""
    return X * np.exp(-2j * np.pi * freqs * tau)


def _mle_cost(theta: float, X: np.ndarray, freqs: np.ndarray) -> float:
    """
    Hàm giá MLE đa kênh: căn chỉnh mọi kênh về kênh tham chiếu rồi đo
    độ lệch so với ước lượng nguồn s_hat (trung bình các kênh đã căn chỉnh).
    """
    K = X.shape[0]
    aligned = np.stack([_shift_fd(X[k], freqs, -k * theta) for k in range(K)])
    s_hat = aligned.mean(axis=0)
    return float(np.sum(np.abs(aligned - s_hat) ** 2))


def estimate_delay_mle(x: np.ndarray, cfg: "GridConfig", gate: "QualityGate",
                       tol: float = 1e-4) -> float:
    """
    MLE đa kênh cho trễ theo từng bước điện cực (đơn vị: mẫu).

    Không dùng khởi tạo từ GCC (dễ bão hòa khi tổng trễ vượt max_lag).
    Thay vào đó quét thẳng DẢI TRỄ SINH LÝ suy từ [cv_min, cv_max]:
        theta = IED / CV * Fs     =>   theta in [IED/cv_max*Fs, IED/cv_min*Fs]
    Quét cả hai dấu vì hướng lan truyền phụ thuộc thứ tự đánh số kênh.
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean(axis=1, keepdims=True)
    K, N = x.shape

    X = np.fft.rfft(x, axis=-1)
    freqs = np.fft.rfftfreq(N, d=1.0)  # chu kỳ/mẫu

    th_lo = cfg.ied_m / gate.cv_max * cfg.fs   # trễ nhỏ nhất (CV nhanh nhất)
    th_hi = cfg.ied_m / gate.cv_min * cfg.fs   # trễ lớn nhất (CV chậm nhất)

    pos = np.linspace(th_lo, th_hi, 80)
    grid = np.concatenate([-pos[::-1], pos])
    costs = np.array([_mle_cost(t, X, freqs) for t in grid])
    t0 = float(grid[int(np.argmin(costs))])

    # tinh chỉnh golden-section quanh cực tiểu thô
    step = pos[1] - pos[0]
    a, b = t0 - step, t0 + step
    gr = (np.sqrt(5) - 1) / 2
    c, d_ = b - gr * (b - a), a + gr * (b - a)
    fc, fd = _mle_cost(c, X, freqs), _mle_cost(d_, X, freqs)
    while abs(b - a) > tol:
        if fc < fd:
            b, d_, fd = d_, c, fc
            c = b - gr * (b - a)
            fc = _mle_cost(c, X, freqs)
        else:
            a, c, fc = c, d_, fd
            d_ = a + gr * (b - a)
            fd = _mle_cost(d_, X, freqs)
    return float((a + b) / 2)


def mean_adjacent_corr(x: np.ndarray) -> float:
    """Hệ số tương quan trung bình giữa các kênh liền kề (sau khi bù trễ)."""
    cs = []
    for k in range(x.shape[0] - 1):
        _, r = _gcc_delay(x[k], x[k + 1])
        cs.append(abs(r))
    return float(np.mean(cs)) if cs else 0.0


# ----------------------------------------------------------------------------
# Ước lượng MFCV một cửa sổ + cổng abstention
# ----------------------------------------------------------------------------

def mfcv_window(x_col: np.ndarray, cfg: GridConfig, gate: QualityGate,
                t_s: float = 0.0, use_double_diff: bool = True) -> MFCVResult:
    """
    Ước lượng MFCV cho MỘT cửa sổ, MỘT cột điện cực dọc theo sợi cơ.

    x_col : (n_rows, n_samples) tín hiệu THÔ (monopolar) đã lọc.
    Trả về MFCVResult; accepted=False nghĩa là cổng abstention từ chối.
    """
    sd = single_differential(x_col)

    sel, iz = select_propagation_channels(sd, gate)
    if sel.shape[0] < gate.min_channels:
        return MFCVResult(t_s, None, None, None, False,
                          f"khong_du_kenh_sau_IZ (IZ@{iz})")

    sig = double_differential(sel) if (use_double_diff and sel.shape[0] >= 3) else sel
    if sig.shape[0] < 2:
        return MFCVResult(t_s, None, None, None, False, "khong_du_kenh")

    corr = mean_adjacent_corr(sig)
    if corr < gate.min_corr:
        return MFCVResult(t_s, None, None, round(corr, 3), False,
                          f"tuong_quan_thap ({corr:.2f} < {gate.min_corr})")

    theta = estimate_delay_mle(sig, cfg, gate)
    if abs(theta) < 1e-6:
        return MFCVResult(t_s, None, None, round(corr, 3), False, "tre_bang_khong")

    cv = cfg.ied_m / (abs(theta) / cfg.fs)

    if not (gate.cv_min <= cv <= gate.cv_max):
        return MFCVResult(t_s, None, round(theta, 4), round(corr, 3), False,
                          f"CV_ngoai_dai_sinh_ly ({cv:.1f} m/s)")

    return MFCVResult(t_s, round(cv, 3), round(theta, 4), round(corr, 3), True, "ok")


def mfcv_timeseries(x_col: np.ndarray, cfg: GridConfig,
                    gate: Optional[QualityGate] = None,
                    win_ms: float = 500.0, hop_ms: float = 250.0) -> List[MFCVResult]:
    """
    Đường MFCV theo thời gian — dùng cho biểu đồ trong demo.

    Cửa sổ 500 ms theo APSIPA 2023 (đoạn được coi là dừng).
    hop 250 ms cho đường mượt; đặt hop=1000 để trùng đúng cấu hình bài báo.
    """
    gate = gate or QualityGate()
    x = preprocess(x_col, cfg)
    w = int(win_ms * cfg.fs / 1000)
    h = int(hop_ms * cfg.fs / 1000)

    out = []
    for start in range(0, x.shape[1] - w + 1, h):
        seg = x[:, start:start + w]
        t = (start + w / 2) / cfg.fs
        out.append(mfcv_window(seg, cfg, gate, t_s=round(t, 3)))
    return out


def rms_timeseries(x_col: np.ndarray, cfg: GridConfig,
                   win_ms: float = 500.0, hop_ms: float = 250.0) -> List[Dict[str, float]]:
    """RMS trên cùng lưới thời gian với MFCV — để vẽ chung một biểu đồ."""
    x = preprocess(x_col, cfg)
    sd = single_differential(x)
    w = int(win_ms * cfg.fs / 1000)
    h = int(hop_ms * cfg.fs / 1000)
    out = []
    for start in range(0, sd.shape[1] - w + 1, h):
        seg = sd[:, start:start + w]
        out.append({"t_s": round((start + w / 2) / cfg.fs, 3),
                    "rms_uv": round(float(np.sqrt(np.mean(seg ** 2))), 3)})
    return out


def fit_slope(results: List[MFCVResult]) -> Optional[float]:
    """
    Dốc CV theo thời gian (m/s mỗi giây) — chỉ dùng các cửa sổ được chấp nhận.
    Đối chiếu APSIPA 2023 Bảng I: dốc CV -0.001 (10%MVC) -> -0.054 (70%MVC mỏi).
    """
    pts = [(r.t_s, r.cv_ms) for r in results if r.accepted and r.cv_ms is not None]
    if len(pts) < 3:
        return None
    t, cv = np.array(pts).T
    return float(np.polyfit(t, cv, 1)[0])


def export_for_demo(x_col: np.ndarray, cfg: GridConfig,
                    gate: Optional[QualityGate] = None) -> Dict[str, Any]:
    """
    Tiền tính toán offline -> JSON cho frontend replay.
    Không cần port MLE vào app; app chỉ vẽ lại.
    """
    res = mfcv_timeseries(x_col, cfg, gate)
    acc = [r for r in res if r.accepted]
    return {
        "mfcv": [r.as_dict() for r in res],
        "rms": rms_timeseries(x_col, cfg),
        "summary": {
            "cv_mean_ms": round(float(np.mean([r.cv_ms for r in acc])), 3) if acc else None,
            "cv_slope_ms_per_s": round(fit_slope(res), 5) if fit_slope(res) else None,
            "n_windows": len(res),
            "n_accepted": len(acc),
            "n_abstained": len(res) - len(acc),
            "abstain_reasons": sorted({r.reason for r in res if not r.accepted}),
        },
    }
