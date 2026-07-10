"""
simulate_semg_rehab.py
----------------------
Mô phỏng tín hiệu sEMG THÔ qua các buổi tập PHCN để minh hoạ đánh giá mức độ
phục hồi cơ (Usecase 2). Đầu ra đúng schema PhysioMio (channel_01..64 +
movement_type, 2048 Hz), nên loader load_physiomio.py chạy được trực tiếp.

MÔ HÌNH (đơn giản, minh bạch — KHÔNG phải dữ liệu bệnh nhân thật):
  - sEMG = nhiễu Gaussian băng thông (Butterworth 20-450 Hz) nhân với bao hình
    kích hoạt (burst khi thực hiện cử chỉ, nền thấp khi Rest).
  - Phục hồi theo buổi được mã hoá bằng 2 marker sinh lý kinh điển:
      * biên độ RMS tăng dần  (huy động cơ tốt hơn)  -> tiến về baseline tay lành
      * MDF (median freq) dịch LÊN nhẹ (MFCV cải thiện) -> phổ "khoẻ" hơn
  - Tay lành: baseline cao, ổn định, làm mốc chuẩn hoá đối xứng.

Cảnh báo: đây là dữ liệu tổng hợp cho demo. Số thật cần thu tại Motion Lab.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch

FS = 2048
GESTURES = ["Rest"] + [f"Gesture_{i:02d}" for i in range(1, 16)]
CH = [f"channel_{i:02d}" for i in range(1, 65)]
SEG_SEC = 0.75                      # thời lượng mỗi cử chỉ (demo)
rng = np.random.default_rng(2026)


def _bandpass_noise(n, fs, lo, hi, rms):
    """Nhiễu Gaussian lọc băng thông -> RMS mong muốn (đơn vị ~ µV)."""
    x = rng.standard_normal(n)
    sos = butter(4, [lo, hi], btype="band", fs=fs, output="sos")
    x = sosfiltfilt(sos, x)
    x *= rms / (np.sqrt(np.mean(x ** 2)) + 1e-9)
    return x.astype(np.float32)


def _burst_envelope(n):
    """Bao hình co cơ: ramp lên - giữ - ramp xuống (hình thang) + rung nhẹ."""
    e = np.ones(n)
    r = int(0.15 * n)
    e[:r] = np.linspace(0, 1, r)
    e[-r:] = np.linspace(1, 0, r)
    e *= 1 + 0.08 * rng.standard_normal(n)     # co giật nhỏ, tự nhiên
    return np.clip(e, 0, None).astype(np.float32)


def simulate_session(activation_uV, mdf_center_hz, rest_uV=5.0):
    """Sinh 1 buổi: 16 cử chỉ nối tiếp, mỗi cử chỉ 1 đoạn burst; Rest = nền.
    mdf_center_hz điều khiển tâm băng thông -> chi phối median frequency."""
    lo, hi = 20.0, min(2.0 * mdf_center_hz + 40, 450.0)   # dịch band -> dịch MDF
    n = int(SEG_SEC * FS)
    frames = []
    for mv in GESTURES:
        seg = {}
        is_rest = (mv == "Rest")
        env = np.ones(n) * 0.2 if is_rest else _burst_envelope(n)
        base = rest_uV if is_rest else activation_uV
        for j, c in enumerate(CH):
            chan_amp = base * (0.85 + 0.3 * rng.random())    # biến thiên không gian
            sig = _bandpass_noise(n, FS, lo, hi, chan_amp) * env
            sig = sig + rng.normal(0, 0.6, n).astype(np.float32)   # nhiễu nền
            seg[c] = sig
        d = pd.DataFrame(seg)
        d["movement_type"] = mv
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def session_mdf(df):
    """MDF trung bình qua kênh, chỉ trên đoạn cử chỉ (loại Rest)."""
    act = df[df["movement_type"] != "Rest"]
    mdfs = []
    for c in CH:
        f, p = welch(act[c].values, fs=FS, nperseg=1024)
        csum = np.cumsum(p)
        mdfs.append(f[np.searchsorted(csum, csum[-1] / 2)])
    return float(np.mean(mdfs))


def build_dataset(root="physiomio_sim/patientSIM", n_sessions=8):
    root = Path(root)
    (root / "healthy_arm").mkdir(parents=True, exist_ok=True)
    (root / "impaired_arm").mkdir(parents=True, exist_ok=True)

    # Tay lành: baseline khoẻ, ổn định
    simulate_session(activation_uV=42.0, mdf_center_hz=110).to_parquet(
        root / "healthy_arm/01.parquet")

    # Tay liệt: quỹ đạo phục hồi — RMS 12->35 µV, MDF 70->102 Hz
    amps = np.linspace(12, 35, n_sessions) + rng.normal(0, 1.0, n_sessions)
    mdfs = np.linspace(70, 102, n_sessions) + rng.normal(0, 2.0, n_sessions)
    for i in range(n_sessions):
        simulate_session(float(amps[i]), float(mdfs[i])).to_parquet(
            root / f"impaired_arm/{i+1:02d}.parquet")
    return root


if __name__ == "__main__":
    build_dataset()
    print("Đã sinh dataset mô phỏng đúng schema PhysioMio -> physiomio_sim/patientSIM")
