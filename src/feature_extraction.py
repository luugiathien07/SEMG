"""14-feature extraction per EMG channel (spec section 4.2).

Conventions chosen to mirror MATLAB:
- ``skewness``  : bias=True  (normalize by N)               -> scipy.stats.skew(bias=True)
- ``kurtosis``  : non-Fisher (normal == 3), bias=True       -> scipy.stats.kurtosis(fisher=False, bias=True)
- ``std``/``mean`` for MDF/MNF match MATLAB's periodogram-based ``medfreq``/``meanfreq``.
- ``std`` feature uses ddof=1 (MATLAB std normalizes by N-1).
- PSD for the spectral-min/max/std features: ``|fft(x, nfft)|**2 / nfft`` (the ``^4``
  typo in the MATLAB Test/Fatigue branch is fixed to ``^2`` for consistency).
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal
from scipy import stats as sp_stats

from . import config


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))


def _periodogram(x: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """One-sided periodogram PSD, matching MATLAB medfreq/meanfreq defaults."""
    f, pxx = sp_signal.periodogram(x, fs=fs)
    return f, pxx


def _mean_freq(x: np.ndarray, fs: int) -> float:
    f, pxx = _periodogram(x, fs)
    total = np.sum(pxx)
    if total <= 0:
        return 0.0
    return float(np.sum(f * pxx) / total)


def _median_freq(x: np.ndarray, fs: int) -> float:
    """Frequency dividing total spectral power in half (linear interpolation)."""
    f, pxx = _periodogram(x, fs)
    total = np.sum(pxx)
    if total <= 0:
        return 0.0
    cumulative = np.cumsum(pxx)
    half = total / 2.0
    idx = int(np.searchsorted(cumulative, half))
    if idx == 0:
        return float(f[0])
    if idx >= len(f):
        return float(f[-1])
    # Linear interpolation between the two bracketing bins.
    c0, c1 = cumulative[idx - 1], cumulative[idx]
    f0, f1 = f[idx - 1], f[idx]
    if c1 == c0:
        return float(f1)
    return float(f0 + (half - c0) * (f1 - f0) / (c1 - c0))


def _spectral_entropy(x: np.ndarray, fs: int) -> float:
    """Shannon entropy (bits) of the normalized power spectrum P(m)=S(m)/sum(S)."""
    _, pxx = _periodogram(x, fs)
    total = np.sum(pxx)
    if total <= 0:
        return 0.0
    p = pxx / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def extract_channel_features(x: np.ndarray) -> dict[str, float]:
    """Compute the fixed 14 features for a single 1-D channel signal."""
    x = np.asarray(x, dtype=float)

    # Frequency-domain PSD used for the spectral min/max/std features.
    fft_vals = np.fft.fft(x, n=config.NFFT)
    psd = (np.abs(fft_vals) ** 2) / config.NFFT

    return {
        "RMS": _rms(x),
        "MAV": float(np.mean(np.abs(x))),
        "Skewness": float(sp_stats.skew(x, bias=True)),
        "Kurtosis": float(sp_stats.kurtosis(x, fisher=False, bias=True)),
        "Max": float(np.max(x)),
        "Min": float(np.min(x)),
        "STD": float(np.std(x, ddof=1)),
        "Mean": float(np.mean(x)),
        "Spectral_Min": float(np.min(psd)),
        "Spectral_Max": float(np.max(psd)),
        "Spectral_STD": float(np.std(psd, ddof=1)),
        "MDF": _median_freq(x, config.FS),
        "MNF": _mean_freq(x, config.FS),
        "Spectral_Entropy": _spectral_entropy(x, config.FS),
    }


def channel_psd(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One-sided PSD (f, Pxx) for plotting in the Streamlit signal viewer."""
    return _periodogram(np.asarray(x, dtype=float), config.FS)
