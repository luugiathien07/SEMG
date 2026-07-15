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
    serp_grid, mfcv_timeseries_quad,
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


class TestSerpGrid:
    def test_shape_and_default_missing_slot_is_nan(self):
        x = np.arange(64 * 100).reshape(64, 100).astype(float)
        g = serp_grid(x)
        assert g.shape == (13, 5, 100)
        # default missing=64 -> grid position (row=0, col=0) has no electrode
        assert np.all(np.isnan(g[0, 0]))

    def test_column_values_map_to_expected_channel_rows(self):
        x = np.arange(64 * 10).reshape(64, 10).astype(float)
        g = serp_grid(x)
        # col 1 (index 0), row 2 (index 1) -> ch[64-1]=ch[63] -> x row 63
        np.testing.assert_array_equal(g[1, 0], x[63])
        # col 2 (index 1), row 1 (index 0) -> ch[39+0]=ch[39] -> x row 39
        np.testing.assert_array_equal(g[0, 1], x[39])
        # col 5 (index 4), row 1 (index 0) -> ch[12-0]=ch[12] -> x row 12
        np.testing.assert_array_equal(g[0, 4], x[12])

    def test_no_nan_columns_when_missing_out_of_range(self):
        # 65 rows so all 65 grid slots get real data once none is skipped.
        x = np.arange(65 * 5).reshape(65, 5).astype(float)
        g = serp_grid(x, missing=-1)
        assert not np.any(np.isnan(g))


class TestMfcvTimeseriesQuad:
    """Covers mfcv_timeseries_quad — the quadruple-channel method used by
    the demo (realtime_session.compute_cv_series), instead of
    mfcv_timeseries (whole-column, dead code for the demo).

    Rest periods below are 1200 samples (> the 1000-sample analysis window
    at win_ms=500/fs=2000) so at least one window sits fully inside the
    rest period — a rest period shorter than the window would mix rest and
    active samples in every window, defeating the activity gate by
    construction, not exercising it."""

    def test_recovers_known_cv_with_rest_then_active_structure(self):
        rng = np.random.default_rng(50)
        cv_true = 4.5
        rest = rng.standard_normal((13, 1200)) * 0.01
        active = make_column(cv_true, CFG, rng, dur_s=1.5, snr_db=20)
        x = np.concatenate([rest, active, rest], axis=1)

        res = mfcv_timeseries_quad(x, CFG, GATE, win_ms=500, hop_ms=250)
        acc = [r for r in res if r.accepted]
        assert len(acc) >= 2
        mean_cv = float(np.mean([r.cv_ms for r in acc]))
        assert abs(mean_cv - cv_true) / cv_true < 0.25

    def test_windows_fully_inside_rest_period_are_rejected_as_resting(self):
        rng = np.random.default_rng(51)
        cv_true = 4.5
        rest = rng.standard_normal((13, 2000)) * 0.01
        active = make_column(cv_true, CFG, rng, dur_s=1.0, snr_db=20)
        x = np.concatenate([rest, active], axis=1)

        res = mfcv_timeseries_quad(x, CFG, GATE, win_ms=500, hop_ms=250)
        # win=1000 samples: keep a 300-sample margin before the rest/active
        # boundary (2000) — preprocess()'s zero-phase filtfilt bleeds a
        # little energy backward across a sharp synthetic transition like
        # this one (real physiological onsets ramp up, they don't step),
        # so windows immediately adjacent to the boundary aren't a fair
        # test of the activity gate itself.
        margin = 300
        rest_windows = [r for r in res if r.t_s * CFG.fs + 500 <= 2000 - margin]
        assert len(rest_windows) >= 2
        assert all(not r.accepted for r in rest_windows)
        assert all(r.reason.startswith("co_dang_nghi") for r in rest_windows)

    def test_single_noisy_quad_does_not_block_the_others(self):
        """A quad spanning an innervation zone (or otherwise bad) must only
        drop itself, not the whole column — the key difference from
        mfcv_timeseries's whole-column IZ detection."""
        rng = np.random.default_rng(52)
        cv_true = 4.5
        active = make_column(cv_true, CFG, rng, dur_s=1.0, snr_db=22, iz_row=6)
        rest = rng.standard_normal((13, 1200)) * 0.01
        x = np.concatenate([rest, active], axis=1)

        res = mfcv_timeseries_quad(x, CFG, GATE, win_ms=500, hop_ms=250)
        acc = [r for r in res if r.accepted]
        assert len(acc) >= 1
