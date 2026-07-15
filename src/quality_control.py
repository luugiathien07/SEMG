"""Signal-quality gate for the Usecase 1 realtime demo (bước 3: kiểm tra
chất lượng / cổng abstention). Pure functions, no Streamlit/HTML
dependency — every detector here is independently unit-tested against
synthetic clipping/dropout, then composed in ``assess_signal_quality`` /
``assess_channel_matrix_quality`` for use by ``realtime_session.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config


def _fraction_in_runs(mask: np.ndarray, min_run: int) -> float:
    """Fraction of ``mask``'s length covered by contiguous True-runs of at
    least ``min_run`` samples. Runs shorter than ``min_run`` don't count —
    a single stray sample at the signal's extreme is normal, a sustained
    pinned run is not."""
    if mask.size == 0:
        return 0.0
    padded = np.concatenate(([False], mask, [False]))
    edges = np.diff(padded.astype(int))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    lengths = ends - starts
    long_enough = lengths >= min_run
    return float(np.sum(lengths[long_enough])) / mask.size


def clipped_fraction(x: np.ndarray, min_run: int) -> float:
    """Fraction of samples in sustained runs pinned near the signal's own
    peak magnitude — a relative threshold, since the true ADC/sensor
    saturation ceiling isn't known from the CSV alone."""
    x = np.asarray(x, dtype=float)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak == 0.0:
        return 0.0
    pinned = np.abs(x) >= 0.999 * peak
    return _fraction_in_runs(pinned, min_run)


def dropout_fraction(x: np.ndarray, window: int, std_frac: float, min_run: int) -> float:
    """Fraction of samples in sustained runs where the local (rolling)
    standard deviation collapses well below the whole signal's standard
    deviation — a flat segment mid-recording, distinct from a channel that
    is entirely zero (handled separately as "dead channel")."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return 0.0
    overall_std = float(np.std(x))
    if overall_std == 0.0:
        return 1.0
    w = min(window, n)
    kernel = np.ones(w) / w
    mean_local = np.convolve(x, kernel, mode="same")
    sq_local = np.convolve(x ** 2, kernel, mode="same")
    var_local = np.maximum(sq_local - mean_local ** 2, 0.0)
    std_local = np.sqrt(var_local)
    flat = std_local < std_frac * overall_std
    return _fraction_in_runs(flat, min_run)


@dataclass(frozen=True)
class QCResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    clip_frac: float = 0.0
    dropout_frac: float = 0.0


def assess_signal_quality(x: np.ndarray, fs: int) -> QCResult:
    """QC verdict for one 1-D signal (a segment's averaged channel, or a
    single electrode's recording). ``fs`` is currently unused by the
    detectors (they work in sample counts) but kept in the signature so
    future frequency-domain checks can use it without changing callers."""
    x = np.asarray(x, dtype=float)
    clip_frac = clipped_fraction(x, config.QC_CLIP_MIN_RUN)
    dropout_frac = dropout_fraction(
        x, config.QC_DROPOUT_WINDOW, config.QC_DROPOUT_STD_FRAC,
        config.QC_DROPOUT_MIN_RUN,
    )
    reasons: list[str] = []
    if clip_frac > config.QC_CLIP_MAX_FRAC:
        reasons.append(f"bao_hoa_tin_hieu ({clip_frac:.1%} mau)")
    if dropout_frac > config.QC_DROPOUT_MAX_FRAC:
        reasons.append(f"mat_tin_hieu ({dropout_frac:.1%} mau)")
    return QCResult(
        passed=len(reasons) == 0, reasons=reasons,
        clip_frac=round(clip_frac, 4), dropout_frac=round(dropout_frac, 4),
    )


def assess_channel_matrix_quality(
    channels: np.ndarray, valid_mask: np.ndarray, fs: int,
) -> list[QCResult]:
    """QC verdict per column of a (n_time, n_channels) matrix. Channels
    already known dead (``valid_mask[i]`` is False) skip the detectors —
    they're already disqualified, no need to run clipping/dropout on an
    all-zero column."""
    out: list[QCResult] = []
    for ch in range(channels.shape[1]):
        if not valid_mask[ch]:
            out.append(QCResult(passed=False, reasons=["kenh_chet"],
                                clip_frac=0.0, dropout_frac=1.0))
        else:
            out.append(assess_signal_quality(channels[:, ch], fs))
    return out
