"""Reusable raw-signal processing pipeline for the sEMG demo — composes the
already-implemented bandpass+notch filter (mfcv.preprocess) with the
already-implemented rectify+envelope (realtime_session.rectify_envelope)
into one 'raw -> filtered -> envelope' pipeline. No new filtering algorithm
is written here; this module only wires existing pieces together so a
'raw vs processed' display (or any future caller) doesn't have to know
the internals of either.

Not used by classification: assess_segments/assess_channel_grid and the
feature-extraction pipeline keep running on the raw, unfiltered signal.
"""
from __future__ import annotations

import numpy as np

from . import mfcv as mfcv_mod
from .realtime_session import rectify_envelope


def filter_signal(x: np.ndarray, fs: float) -> np.ndarray:
    """Bandpass 20-400Hz + notch 50Hz for a 1-D signal, reusing
    mfcv.preprocess() (no new filter design — same coefficients used by
    the MFCV estimator)."""
    cfg = mfcv_mod.GridConfig(fs=fs)
    y = mfcv_mod.preprocess(x, cfg)
    return y[0] if y.ndim == 2 else y


def process_for_display(x: np.ndarray, fs: float, envelope_samples: int) -> np.ndarray:
    """Full display pipeline: filter (bandpass+notch) then rectify+envelope.
    Used for the 'Tín hiệu đã xử lý' line in the realtime demo; not used
    for classification, which stays on the raw signal."""
    filtered = filter_signal(x, fs)
    return rectify_envelope(filtered, envelope_samples)
