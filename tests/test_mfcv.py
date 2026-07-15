"""Tests for src/mfcv.py — MFCV estimation + abstention gate, verified on
synthetic sEMG generated from the Shwedyk (1977) PSD model, matching the
data-generation method in Luu et al. 2013/2018.

    PSD(f) = k * fh^4 * f^2 / [(f^2 + fl^2) * (f^2 + fh^2)^2]     fl=60Hz, fh=120Hz
"""
import numpy as np
import pytest

from src.mfcv import (
    GridConfig, QualityGate, mfcv_timeseries, mfcv_window, preprocess,
    fit_slope, export_for_demo, single_differential, detect_innervation_zone,
)


def shwedyk_semg(n, fs, rng, fl=60.0, fh=120.0):
    """Sinh 1 kênh sEMG tổng hợp: lọc nhiễu trắng bằng đáp ứng theo PSD Shwedyk."""
    nfft = 1 << (int(np.ceil(np.log2(n))) + 1)
    f = np.fft.rfftfreq(nfft, d=1 / fs)
    psd = (fh ** 4) * (f ** 2) / (((f ** 2 + fl ** 2)) * ((f ** 2 + fh ** 2) ** 2))
    psd[0] = 0
    H = np.sqrt(psd)
    w = rng.standard_normal(nfft)
    s = np.fft.irfft(np.fft.rfft(w) * H, nfft)[:n]
    return s / (s.std() + 1e-12)


def delay_signal(s, tau):
    """Trễ phân số bằng dịch pha miền tần số (tương đương nội suy sinc)."""
    n = len(s)
    S = np.fft.rfft(s)
    f = np.fft.rfftfreq(n, d=1.0)
    return np.fft.irfft(S * np.exp(-2j * np.pi * f * tau), n)


def make_column(cv_true, cfg, rng, dur_s=2.0, snr_db=20.0, n_rows=None, iz_row=None):
    """Sinh 1 cột điện cực monopolar lan truyền với CV cho trước.
    iz_row: nếu đặt, mô phỏng innervation zone -> lan truyền hai chiều từ hàng đó."""
    n_rows = n_rows or cfg.n_rows
    n = int(dur_s * cfg.fs)
    s = shwedyk_semg(n, cfg.fs, rng)
    theta = cfg.ied_m / cv_true * cfg.fs

    chans = []
    for k in range(n_rows):
        tau = k * theta if iz_row is None else abs(k - iz_row) * theta
        chans.append(delay_signal(s, tau))
    x = np.stack(chans)

    p_sig = np.mean(x ** 2)
    p_noise = p_sig / (10 ** (snr_db / 10))
    x = x + rng.standard_normal(x.shape) * np.sqrt(p_noise)
    return x


CFG = GridConfig(fs=2000.0, ied_m=0.008, n_rows=13)
GATE = QualityGate()


class TestMfcvWindowRecoversKnownCV:
    @pytest.mark.parametrize("cv_true", [3.0, 4.0, 4.5, 5.0, 6.0])
    def test_mean_estimate_within_15_percent(self, cv_true):
        rng = np.random.default_rng(int(cv_true * 100))
        ests = []
        for _ in range(20):
            x = make_column(cv_true, CFG, rng, dur_s=0.5, snr_db=20)
            r = mfcv_window(preprocess(x, CFG), CFG, GATE)
            if r.accepted:
                ests.append(r.cv_ms)
        assert len(ests) >= 10, f"cổng abstention từ chối quá nhiều ở CV={cv_true}"
        mean_est = np.mean(ests)
        assert abs(mean_est - cv_true) / cv_true < 0.15


class TestAbstentionGate:
    def test_white_noise_rejected(self):
        rng = np.random.default_rng(1)
        x_junk = rng.standard_normal((13, 1000))
        r = mfcv_window(preprocess(x_junk, CFG), CFG, GATE)
        assert r.accepted is False

    def test_nonphysiological_cv_rejected(self):
        rng = np.random.default_rng(2)
        x_fast = make_column(20.0, CFG, rng, dur_s=0.5, snr_db=25)
        r = mfcv_window(preprocess(x_fast, CFG), CFG, GATE)
        assert r.accepted is False

    def test_clean_physiological_signal_accepted(self):
        rng = np.random.default_rng(3)
        x_ok = make_column(4.5, CFG, rng, dur_s=0.5, snr_db=25)
        r = mfcv_window(preprocess(x_ok, CFG), CFG, GATE)
        assert r.accepted is True
        assert abs(r.cv_ms - 4.5) / 4.5 < 0.15

    def test_innervation_zone_detected_and_still_estimates(self):
        rng = np.random.default_rng(4)
        x_iz = make_column(4.5, CFG, rng, dur_s=0.5, snr_db=25, iz_row=6)
        iz = detect_innervation_zone(single_differential(preprocess(x_iz, CFG)))
        assert iz is not None
        r = mfcv_window(preprocess(x_iz, CFG), CFG, GATE)
        assert r.accepted is True


class TestMfcvTimeseriesTracksFatigueTrend:
    def test_slope_is_negative_and_close_to_true_rate(self):
        rng = np.random.default_rng(5)
        segs = [
            make_column(5.0 - 1.0 * (i / 39), CFG, rng, dur_s=0.5, snr_db=22)
            for i in range(40)
        ]
        x_fatigue = np.concatenate(segs, axis=1)
        res = mfcv_timeseries(x_fatigue, CFG, GATE, win_ms=500, hop_ms=250)
        acc = [r for r in res if r.accepted]
        assert len(acc) >= 20
        slope = fit_slope(res)
        true_slope = -1.0 / 20
        assert slope is not None
        assert slope < 0
        assert abs(slope - true_slope) < 0.03


class TestExportForDemo:
    def test_returns_expected_keys(self):
        rng = np.random.default_rng(6)
        x = make_column(4.5, CFG, rng, dur_s=2.0, snr_db=22)
        out = export_for_demo(x, CFG, GATE)
        assert set(out.keys()) == {"mfcv", "rms", "summary"}
        assert set(out["summary"].keys()) == {
            "cv_mean_ms", "cv_slope_ms_per_s", "n_windows", "n_accepted",
            "n_abstained", "abstain_reasons",
        }
        assert out["summary"]["n_windows"] > 0
