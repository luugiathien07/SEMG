"""Tests for src/signal_processing.py — reusable raw-signal processing
pipeline (bandpass+notch filter, composed with rectify+envelope) used by
the Usecase 1 realtime demo's 'raw vs processed' waveform display."""
import numpy as np

from src.signal_processing import filter_signal, process_for_display


def _band_power(x: np.ndarray, fs: float, f_lo: float, f_hi: float) -> float:
    """Power of x in [f_lo, f_hi) via FFT periodogram — good enough for a
    coarse before/after comparison, no need for scipy.welch precision."""
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    psd = np.abs(np.fft.rfft(x)) ** 2
    mask = (freqs >= f_lo) & (freqs < f_hi)
    return float(np.sum(psd[mask]))


class TestFilterSignal:
    def test_removes_dc_offset(self):
        fs = 2000.0
        n = 4000
        t = np.arange(n) / fs
        rng = np.random.default_rng(1)
        x = 5.0 + 0.05 * rng.standard_normal(n) + 0.1 * np.sin(2 * np.pi * 100 * t)
        y = filter_signal(x, fs)
        assert abs(float(np.mean(y))) < abs(float(np.mean(x))) * 0.1

    def test_attenuates_50hz_notch_band(self):
        fs = 2000.0
        n = 4000
        t = np.arange(n) / fs
        rng = np.random.default_rng(2)
        x = (0.05 * rng.standard_normal(n)
             + 1.0 * np.sin(2 * np.pi * 50 * t)
             + 0.1 * np.sin(2 * np.pi * 100 * t))
        y = filter_signal(x, fs)
        power_before = _band_power(x, fs, 48, 52)
        power_after = _band_power(y, fs, 48, 52)
        assert power_after < power_before * 0.1

    def test_keeps_passband_energy(self):
        fs = 2000.0
        n = 4000
        t = np.arange(n) / fs
        # 130Hz: inside the 20-400Hz passband and clear of the 50Hz mains
        # notch and its harmonics (100/150/200/...), which filter_signal
        # (via mfcv.preprocess) removes by design.
        x = np.sin(2 * np.pi * 130 * t)
        y = filter_signal(x, fs)
        power_before = _band_power(x, fs, 120, 140)
        power_after = _band_power(y, fs, 120, 140)
        assert power_after > power_before * 0.5

    def test_output_same_length_as_input(self):
        fs = 2000.0
        rng = np.random.default_rng(3)
        x = rng.standard_normal(1234)
        y = filter_signal(x, fs)
        assert y.shape == x.shape


class TestProcessForDisplay:
    def test_output_is_nonnegative(self):
        fs = 2000.0
        rng = np.random.default_rng(4)
        x = rng.standard_normal(4000) * 0.1
        y = process_for_display(x, fs, envelope_samples=100)
        assert np.all(y >= 0)

    def test_output_same_length_as_input(self):
        fs = 2000.0
        rng = np.random.default_rng(5)
        x = rng.standard_normal(3000) * 0.1
        y = process_for_display(x, fs, envelope_samples=50)
        assert y.shape == x.shape

    def test_differs_from_naive_envelope_of_raw_signal(self):
        """process_for_display filters first — its output must differ from
        rectify_envelope(x) applied directly to the unfiltered signal,
        proving the filter step actually runs before the envelope."""
        from src.realtime_session import rectify_envelope
        fs = 2000.0
        n = 4000
        t = np.arange(n) / fs
        x = 3.0 + np.sin(2 * np.pi * 50 * t) + 0.1 * np.sin(2 * np.pi * 100 * t)
        processed = process_for_display(x, fs, envelope_samples=100)
        naive = rectify_envelope(x, 100)
        assert not np.allclose(processed, naive, atol=1e-6)
