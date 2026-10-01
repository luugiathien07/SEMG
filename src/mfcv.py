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

Phương pháp bộ-bốn-kênh (mfcv_timeseries_quad, dùng cho demo — xem phần cuối
file): theo đúng mô tả APSIPA 2023 ("three single differential sEMG signals
were selected... calculates two double-differential sEMG signals and tries to
maximize the likelihood delay function between the two signals"). Mỗi bộ 4
kênh liền kề được chấm điểm RIÊNG, nên một bộ dính innervation zone chỉ bị
loại chính nó thay vì kéo tụt cả cột. `mfcv_timeseries` (đường-cả-cột, dò IZ
một lần cho toàn cột) vẫn giữ lại trong module nhưng KHÔNG dùng cho demo nữa —
xem ghi chú tại hàm đó.

Tham chiếu:
    Ravier, Luu, Jabloun, Buttelli — ISABEL 2011 (GCC)
    Luu, Ravier, Buttelli — IJABE 6(1):6-11, 2013 (MLE)
    Luu, Ravier, Buttelli — IEEE ATC 145-148, 2015 (validation dữ liệu thật)
    Luu, Boualem, Duy, Ravier, Buttelli — Fluct. Noise Lett. 17(2):1850015, 2018
    Nguyen, Luu et al. — APSIPA ASC 2023 (CV trên lưới 64 điện cực, Fig. 1B)
"""

from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from scipy.optimize import minimize_scalar


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
    notch_harmonics: bool = True    # notch cả sóng hài 100/150/200... Hz
    notch_q: float = 35.0


@dataclass
class QualityGate:
    """Ngưỡng của cổng abstention. KHÔNG nới lỏng cho demo — đây là điểm khác
    biệt an toàn của sản phẩm (từ chối kết luận khi không đủ tin cậy, thay vì
    luôn ép ra một con số)."""
    min_corr: float = 0.75      # Luu et al. 2015 / APSIPA 2023
    cv_min: float = 2.0         # m/s  dải sinh lý (mô hình TVD: 2–8 m/s)
    cv_max: float = 10.0        # m/s  Luu et al. 2015: CV phải < 10 m/s
    min_channels: int = 3       # tối thiểu 3 kênh vi sai đơn
    min_activity: float = 3.0   # RMS cửa sổ phải > 3x nền nghỉ;
                                 # không báo cáo CV khi cơ đang KHÔNG co


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
    """
    Lọc nhiễu cho sEMG monopolar:
      1. Bandpass 20-400 Hz (APSIPA 2023)
      2. Notch điện lưới 50 Hz VÀ CÁC SÓNG HÀI (100/150/200/... Hz)
      3. Tất cả dùng filtfilt => ZERO-PHASE, không làm lệch trễ
         (cực kỳ quan trọng: MFCV đo bằng TRỄ, bộ lọc lệch pha sẽ phá hỏng CV)
    x: (n_ch, n_samples)
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    nyq = cfg.fs / 2.0
    hi = min(cfg.bp_high, 0.99 * nyq)
    b, a = butter(4, [cfg.bp_low / nyq, hi / nyq], btype="band")
    y = filtfilt(b, a, x, axis=-1)

    if cfg.notch_hz:
        f0 = cfg.notch_hz
        k = 1
        while f0 * k < hi:                      # 50, 100, 150, ... trong dải
            fk = f0 * k
            if fk / nyq < 0.99:
                bn, an = iirnotch(fk / nyq, Q=cfg.notch_q)
                y = filtfilt(bn, an, y, axis=-1)
            k += 1
            if not cfg.notch_harmonics:
                break
    return y


def find_bad_channels(x: np.ndarray, z_thresh: float = 3.5):
    """
    Dò kênh hỏng: phẳng (đứt dây) hoặc biên độ bất thường (nhiễu/bão hoà).
    Trả về mảng bool: True = kênh XẤU.
    """
    x = np.asarray(x, dtype=float)
    rms = np.sqrt(np.nanmean(x ** 2, axis=1))
    bad = np.zeros(len(rms), dtype=bool)
    bad |= ~np.isfinite(rms)
    good = rms[np.isfinite(rms) & (rms > 0)]
    if len(good) < 3:
        return bad
    med = np.median(good)
    mad = np.median(np.abs(good - med)) + 1e-12
    z = np.abs(rms - med) / (1.4826 * mad)
    bad |= (z > z_thresh)                       # biên độ lệch quá xa
    bad |= (rms < 0.05 * med)                   # gần như phẳng
    return bad


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

    Tinh chỉnh quanh cực tiểu thô bằng Brent's method (scipy, bounded) thay
    vì golden-section tự cài — cùng độ chính xác (đã kiểm chứng bằng benchmark
    Monte Carlo trên dữ liệu tổng hợp có CV biết trước, xem lịch sử review)
    nhưng nhanh hơn ~8 lần nhờ kết hợp nội suy parabol với golden-section
    thay vì chỉ golden-section thuần.
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

    step = pos[1] - pos[0]
    res = minimize_scalar(_mle_cost, bounds=(t0 - step, t0 + step),
                          args=(X, freqs), method="bounded",
                          options={"xatol": tol})
    return float(res.x)


def mean_adjacent_corr(x: np.ndarray) -> float:
    """Hệ số tương quan trung bình giữa các kênh liền kề (sau khi bù trễ)."""
    cs = []
    for k in range(x.shape[0] - 1):
        _, r = _gcc_delay(x[k], x[k + 1])
        cs.append(abs(r))
    return float(np.mean(cs)) if cs else 0.0


# ----------------------------------------------------------------------------
# Ước lượng MFCV một cửa sổ + cổng abstention (đường cả-cột, dò IZ một lần)
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
                    win_ms: float = 500.0, hop_ms: float = 250.0,
                    use_double_diff: bool = True) -> List[MFCVResult]:
    """
    Đường MFCV theo thời gian, phương pháp cả-cột (dò 1 innervation zone cho
    toàn bộ cột mỗi cửa sổ).

    KHÔNG dùng nhánh này cho demo nữa — xem `mfcv_timeseries_quad` ở cuối
    file. Một cột dính innervation zone bị dò sai (hoặc IZ trôi theo thời
    gian khi cơ mỏi) sẽ kéo tụt độ chính xác toàn bộ cột, thay vì chỉ loại
    đúng bộ 4 kênh chứa IZ như phương pháp bộ-bốn-kênh. Giữ lại hàm này vì
    `tests/test_mfcv.py` vẫn kiểm chứng thuật toán MLE/GCC nền tảng qua nó,
    và các nơi khác có thể còn tham chiếu.

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
        out.append(mfcv_window(seg, cfg, gate, t_s=round(t, 3),
                               use_double_diff=use_double_diff))
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


# ----------------------------------------------------------------------------
# Sơ đồ điện cực — ánh xạ kiểu rắn bò (serpentine), Fig. 1B APSIPA 2023
# ----------------------------------------------------------------------------

def serp_grid(x: np.ndarray, missing: int = 64) -> np.ndarray:
    """
    Ánh xạ ma trận kênh thô (n_channels, n_samples) lên lưới 13x5 kiểu
    "rắn bò" (serpentine) theo đúng Fig. 1B của APSIPA 2023 — thứ tự kênh
    ngoằn ngoèo lên/xuống giữa các cột, khác với ánh xạ tuyến tính đơn giản.

    `missing` là chỉ số kênh (0-based, trong không gian 65 vị trí lưới,
    13*5=65) không có điện cực thật (góc bị cắt của mảng 5x13) — mặc định
    65 (vị trí cuối cùng). x's hàng được gán tuần tự vào các vị trí lưới,
    bỏ qua đúng vị trí `missing`.

    Trả về mảng (13, 5, n_samples); các ô không có điện cực = NaN.
    """
    x = np.asarray(x, dtype=float)
    n, N = x.shape
    ch = np.full((65, N), np.nan)
    s = 0
    for c in range(65):
        if c == missing:
            continue
        if s < n:
            ch[c] = x[s]
            s += 1
    g = np.full((13, 5, N), np.nan)
    for r in range(13):
        g[r, 0] = ch[64 - r]
        g[r, 1] = ch[39 + r]
        g[r, 2] = ch[38 - r]
        g[r, 3] = ch[13 + r]
        g[r, 4] = ch[12 - r]
    return g


# ----------------------------------------------------------------------------
# Phương pháp BỘ-BỐN-KÊNH — dùng cho demo, theo APSIPA 2023
# ----------------------------------------------------------------------------
# Bài báo: "three single differential sEMG signals were selected... the
# algorithm calculates two double-differential sEMG signals and tries to
# maximize the likelihood delay function between the two signals"
#
#   4 kênh monopolar liên tiếp -> 3 SD -> 2 DD -> MLE giữa ĐÚNG 2 tín hiệu đó.
#   Khoảng cách không gian giữa 2 DD = 1 IED  =>  CV = IED / (theta/Fs)
#
# Khác biệt then chốt so với đường cả-cột: mỗi bộ bốn kênh được CHẤM ĐIỂM
# riêng, nên bộ nào dính innervation zone sẽ bị loại, thay vì kéo tụt cả cột.

def mfcv_quadruple_scan(x_col, cfg, gate):
    """
    Quét mọi bộ 4 kênh liên tiếp trong một cột.
    Trả về list dict: {i, cv_ms, theta, corr, ok}
    """
    x_col = np.asarray(x_col, dtype=float)
    out = []
    for i in range(x_col.shape[0] - 3):
        quad = x_col[i:i + 4]
        if np.any(np.all(np.isnan(quad), axis=1)):
            continue
        sd = np.diff(quad, axis=0)          # 3 tín hiệu vi sai đơn
        dd = np.diff(sd, axis=0)            # 2 tín hiệu vi sai kép
        theta = estimate_delay_mle(dd, cfg, gate)
        _, r = _gcc_delay(dd[0], dd[1])
        r = abs(r)
        cv = cfg.ied_m / (abs(theta) / cfg.fs) if abs(theta) > 1e-9 else np.inf
        ok = (r >= gate.min_corr) and (gate.cv_min <= cv <= gate.cv_max)
        out.append({"i": i, "cv_ms": cv, "theta": theta, "corr": r, "ok": ok})
    return out


def mfcv_window_quad(x_col, cfg, gate, t_s=0.0):
    """
    Ước lượng MFCV một cửa sổ bằng phương pháp bộ-bốn-kênh.
    Lấy TRUNG VỊ của các bộ đạt tiêu chuẩn (chống nhiễu tốt hơn trung bình).
    """
    scan = mfcv_quadruple_scan(x_col, cfg, gate)
    good = [s for s in scan if s["ok"]]
    if not good:
        best_r = max((s["corr"] for s in scan), default=0.0)
        return MFCVResult(t_s, None, None, round(best_r, 3), False,
                          f"khong_bo_4_kenh_nao_dat (r_max={best_r:.2f})")
    cv = float(np.median([s["cv_ms"] for s in good]))
    th = float(np.median([s["theta"] for s in good]))
    r = float(np.median([s["corr"] for s in good]))
    return MFCVResult(t_s, round(cv, 3), round(th, 4), round(r, 3), True,
                      f"ok ({len(good)}/{len(scan)} bo 4 kenh)")


def mfcv_timeseries_quad(x_col, cfg, gate=None, win_ms=500.0, hop_ms=250.0,
                         activity_gate=True):
    """
    Đường MFCV theo thời gian, phương pháp bộ-bốn-kênh — DÙNG CHO DEMO
    (`realtime_session.compute_cv_series`), thay cho `mfcv_timeseries`.

    CỔNG HOẠT ĐỘNG (activity_gate=True, mặc định): cửa sổ có RMS thấp (co
    dạng NGHỈ) bị từ chối TRƯỚC khi tính CV. Lúc nghỉ không có tín hiệu lan
    truyền -> mọi CV đều là nhiễu. Nền nghỉ ước lượng bằng phân vị 5 của RMS
    toàn bản ghi — giả định bản ghi có một đoạn nghỉ thật để lấy mốc so
    sánh. Đây KHÔNG phải tiêu chí trích từ Luu et al. 2015/APSIPA 2023 (đó
    là min_corr/cv_min/cv_max, không đổi bởi tham số này) — đây là bộ lọc
    kỹ thuật thêm vào để tránh tính CV lúc cơ thật sự nghỉ.

    Đặt `activity_gate=False` để bỏ qua bước lọc này — dùng cho các bản ghi
    KHÔNG có đoạn nghỉ thật (vd. giữ lực liên tục ở một mức %MVC cố định):
    khi đó phân vị 5 chỉ phản ánh đáy dao động tự nhiên của tín hiệu đang
    hoạt động (không phải nghỉ thật), khiến cổng hoạt động từ chối sai phần
    lớn cửa sổ. min_corr/cv_min/cv_max vẫn được áp dụng nguyên vẹn — chỉ
    bước "cơ có đang co không" bị bỏ.
    """
    gate = gate or QualityGate()
    x = preprocess(x_col, cfg)
    w = int(win_ms * cfg.fs / 1000)
    h = int(hop_ms * cfg.fs / 1000)

    starts = list(range(0, x.shape[1] - w + 1, h))
    rms = thr = None
    if activity_gate:
        # QUAN TRỌNG: đo hoạt động trên VI SAI ĐƠN, không phải monopolar.
        # Monopolar bị common-mode áp đảo (common-mode vẫn còn khi cơ NGHỈ),
        # nên RMS monopolar gần như không đổi giữa nghỉ và co -> vô dụng.
        sd_all = single_differential(x)
        rms = np.array([np.sqrt(np.mean(sd_all[:, s:s + w] ** 2)) for s in starts])
        baseline = np.percentile(rms, 5)
        thr = baseline * gate.min_activity

    out = []
    for idx, s in enumerate(starts):
        t = round((s + w / 2) / cfg.fs, 3)
        if thr is not None and rms[idx] < thr:
            out.append(MFCVResult(t, None, None, None, False,
                                  f"co_dang_nghi (RMS {rms[idx]:.4f} < {thr:.4f})"))
            continue
        out.append(mfcv_window_quad(x[:, s:s + w], cfg, gate, t))
    return out
