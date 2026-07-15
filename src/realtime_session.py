"""Usecase-1 (v2) clinic demo: near-real-time playback and per-segment
fatigue assessment over a concatenated session signal
(src.session_builder.build_session_signal). Kept separate from the research
pipeline — only orchestrates it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config
from . import data_loader as dl
from . import feature_extraction as fe
from . import inference as inf
from . import mfcv as mfcv_mod
from . import session_builder as sb
from .evaluate import ModelResult
from .session_builder import SegmentInfo


def best_model_name(results: list[ModelResult]) -> str:
    """Name of the model with the highest F1 on the held-out subject — F1
    balances Precision/Recall, so this is what the demo status banner and
    trend shading use instead of a hardcoded model name, letting the "best"
    model change automatically whenever the pipeline is retrained.
    """
    return max(results, key=lambda r: r.f1).name


def rectify_envelope(x: np.ndarray, envelope_samples: int) -> np.ndarray:
    """Full-wave rectify + moving-average envelope, for display only."""
    envelope_samples = max(1, int(envelope_samples))
    kernel = np.ones(envelope_samples) / envelope_samples
    return np.convolve(np.abs(x), kernel, mode="same")


@dataclass(frozen=True)
class PlaybackStep:
    step_idx: int
    sample_pos: int
    segment_idx: int
    window_t: np.ndarray
    window_raw: np.ndarray
    window_processed: np.ndarray


def _segment_idx_for(sample_pos: int, segments: list[SegmentInfo]) -> int:
    for i, seg in enumerate(segments):
        if seg.start_sample <= sample_pos < seg.end_sample:
            return i
    return len(segments) - 1


def build_playback_steps(
    signal: np.ndarray,
    segments: list[SegmentInfo],
    n_steps: int,
    display_window_sec: float,
    envelope_window_sec: float,
    fs: int,
    max_display_points: int | None = None,
) -> list[PlaybackStep]:
    """Scrub through `signal` at `n_steps` evenly-spaced positions, each
    carrying a `display_window_sec`-long raw+processed window for the
    animated waveform chart. Does NOT compute fatigue status — that comes
    from `assess_segments`, using whole segments, not these short windows.

    `max_display_points`, when given, decimates each window's arrays down to
    roughly that many points (same `[::step]` pattern as app.py's Signal &
    PSD tab) — a wide display window at full sample rate is a lot of points
    to re-serialize every animation frame, and that overhead is itself a
    source of choppiness independent of how much consecutive windows overlap.
    """
    display_window_samples = int(display_window_sec * fs)
    envelope_samples = max(1, int(envelope_window_sec * fs))
    if len(signal) <= display_window_samples:
        raise ValueError(
            "Tín hiệu ghép quá ngắn so với display_window_sec đã chọn: cần "
            f"nhiều hơn {display_window_samples} mẫu, có {len(signal)}."
        )

    max_start = len(signal) - display_window_samples
    local_offsets = np.arange(display_window_samples) / fs

    steps: list[PlaybackStep] = []
    for i in range(n_steps):
        frac = i / (n_steps - 1) if n_steps > 1 else 0.0
        sample_pos = int(round(frac * max_start))
        window_raw = signal[sample_pos:sample_pos + display_window_samples]
        window_processed = rectify_envelope(window_raw, envelope_samples)
        # Global elapsed time within the whole session (starts at 0 for the
        # first step, then grows monotonically) instead of a fixed local
        # window reused every step, so the waveform chart's x-axis reads
        # like a real-time monitor's growing time axis rather than
        # resetting every frame.
        window_t = sample_pos / fs + local_offsets
        if max_display_points is not None and len(window_t) > max_display_points:
            dec_step = max(1, len(window_t) // max_display_points)
            window_t = window_t[::dec_step]
            window_raw = window_raw[::dec_step]
            window_processed = window_processed[::dec_step]
        steps.append(PlaybackStep(
            step_idx=i, sample_pos=sample_pos,
            segment_idx=_segment_idx_for(sample_pos, segments),
            window_t=window_t, window_raw=window_raw,
            window_processed=window_processed,
        ))
    return steps


@dataclass
class SegmentAssessment:
    segment: SegmentInfo
    feats: dict[str, float]
    rms: float
    mdf: float
    predictions: list[dict]


def assess_segments(
    signal: np.ndarray,
    segments: list[SegmentInfo],
    results: list[ModelResult],
) -> list[SegmentAssessment]:
    """Extract features and run every trained model on each *whole* segment
    (not the short playback windows) — this matches how the models were
    trained (one feature vector per whole file) and avoids the
    train/inference distribution mismatch that broke the earlier
    crossfade-based design.
    """
    out: list[SegmentAssessment] = []
    for seg in segments:
        x = signal[seg.start_sample:seg.end_sample]
        feats = fe.extract_channel_features(x)
        predictions = inf.predict_channel(x, results)
        out.append(SegmentAssessment(
            segment=seg, feats=feats, rms=feats["RMS"], mdf=feats["MDF"],
            predictions=predictions,
        ))
    return out


def assess_channel_grid(
    segments: list[SegmentInfo],
    valid_channels: list[int],
    model_result: ModelResult,
) -> list[list[int | None]]:
    """Per-segment, per-channel Normal/Fatigue prediction for the 64-channel
    diagram. Each segment corresponds to exactly one file (session_builder
    invariant), so the file's own channel matrix is loaded directly instead
    of slicing the concatenated averaged signal. Only `model_result` (the
    demo's pinned/best model) is run — not the full model list — since the
    diagram only needs one verdict per channel, not a full per-model table.

    Returns one 64-length list per segment, indexed by `physical_channel - 1`
    (0-based): 0/1 for channels in `valid_channels`, None otherwise (dead or
    invalid for this subject).
    """
    valid_set = set(valid_channels)
    out: list[list[int | None]] = []
    for seg in segments:
        channels = dl.load_channels(seg.file.path)
        grid: list[int | None] = [None] * 64
        for ch in valid_channels:
            if ch not in valid_set or ch >= channels.shape[1]:
                continue
            x = channels[:, ch]
            preds = inf.predict_channel(x, [model_result])
            grid[ch] = preds[0]["pred"] if preds else None
        out.append(grid)
    return out


def _mfcv_file_series(
    channels: np.ndarray, cfg: "mfcv_mod.GridConfig", gate: "mfcv_mod.QualityGate",
    win_ms: float, hop_ms: float, min_col_accept: float,
) -> list[dict]:
    """MFCV(t) for one file's full 64-channel matrix, averaged across the
    electrode columns of the serpentine grid (`mfcv.serp_grid`) whose
    window-acceptance ratio reaches `min_col_accept` — same method as the
    verified reference script (serp.py): a column that mostly abstains is
    dropped entirely rather than diluting the average with noise.

    Returns one entry per window position on this file's own local time
    grid: `{"t_s", "cv_ms", "accepted", "sample_pos"}`. `sample_pos` (the
    window's start sample within this file) is used by the caller to
    remap onto the session/trimmed timeline; it is not part of the
    public JSON contract. Windows no qualifying column accepted get
    `cv_ms=None, accepted=False` — a uniform per-window grid, even though
    the quadruple-channel method's own acceptance is column-by-column, so
    the frontend can still draw gaps.
    """
    x = channels.T  # (64, n_time)
    grid = mfcv_mod.serp_grid(x)  # (13, 5, n_time)

    w = int(win_ms * cfg.fs / 1000)
    h = int(hop_ms * cfg.fs / 1000)
    n_time = x.shape[1]
    starts = list(range(0, n_time - w + 1, h))
    grid_ts = [round((s + w / 2) / cfg.fs, 3) for s in starts]

    per_t: dict[float, list[float]] = {}
    for c in range(grid.shape[1]):
        col = grid[:, c, :]
        col = col[~np.all(np.isnan(col), axis=1)]
        if col.shape[0] < 4:
            continue
        res = mfcv_mod.mfcv_timeseries_quad(col, cfg, gate, win_ms=win_ms, hop_ms=hop_ms)
        if not res:
            continue
        acc = [r for r in res if r.accepted]
        if len(acc) / len(res) < min_col_accept:
            continue
        for r in acc:
            per_t.setdefault(r.t_s, []).append(r.cv_ms)

    out = []
    for t, s in zip(grid_ts, starts):
        if t in per_t:
            out.append({"t_s": t, "cv_ms": float(np.mean(per_t[t])),
                       "accepted": True, "sample_pos": s})
        else:
            out.append({"t_s": t, "cv_ms": None, "accepted": False, "sample_pos": s})
    return out


def compute_cv_series(
    subject: int, trim_bounds: list[tuple[int, int]] | None = None,
    win_ms: float = 500.0, hop_ms: float = 250.0, min_col_accept: float = 0.5,
) -> list[dict]:
    """Real MFCV(t) series for the realtime trend chart — quadruple-channel
    method (`mfcv.mfcv_timeseries_quad`), averaged across the electrode
    columns of the serpentine 13x5 grid (`mfcv.serp_grid`, Fig. 1B APSIPA
    2023) that individually pass a >=50% window-acceptance ratio. This is
    the method verified against real data (Sujet_7, fatigue_70 file:
    CV=4.94 m/s, matches the published 4.4-5.1 m/s range).
    `mfcv.mfcv_timeseries` (the whole-column, single-IZ-detection method)
    is dead code for this demo — see that function's own docstring.

    Runs per file (`session_builder.list_ordered_segments`), on that
    file's own full 64-channel matrix, then concatenates. Per-file
    processing (rather than one big concatenated column) matters for two
    reasons: `serp_grid` needs all 64 channels, and the rest-activity
    gate built into `mfcv_timeseries_quad` needs each file's own natural
    rest/contraction structure to set its baseline — concatenating first
    would blend different %MVC segments' baselines together.

    Uses `QualityGate()`'s own defaults (min_corr=0.75 etc., Luu et al.
    2015/APSIPA 2023) unmodified — NOT relaxed for the demo. The
    abstention gate is a deliberate product differentiator, not a knob to
    tune for coverage.

    `trim_bounds`, if given, must be the same per-segment `(start, end)`
    bounds `session_builder.trim_bounds_for_display()` produced for this
    subject's averaged session signal (same file order). A window is kept
    only if its start sample falls inside that file's trim bounds — it is
    then remapped onto the trimmed timeline so this series lines up with
    the RMS/MDF trend and waveform; windows outside are dropped (they
    don't exist on the trimmed display timeline). Without `trim_bounds`,
    the raw untrimmed per-file timeline is used.
    """
    files = sb.list_ordered_segments(subject)
    if len(files) < 2:
        raise ValueError(
            f"Subject {subject} không đủ file để ghép buổi tập "
            f"(cần ít nhất 2, có {len(files)})."
        )

    cfg = mfcv_mod.GridConfig(fs=float(config.FS))
    gate = mfcv_mod.QualityGate()

    out: list[dict] = []
    untrimmed_offset = 0
    trimmed_offset = 0
    for i, f in enumerate(files):
        channels = dl.load_channels(f.path)  # (n_time, 64)
        local_entries = _mfcv_file_series(
            channels, cfg, gate, win_ms=win_ms, hop_ms=hop_ms,
            min_col_accept=min_col_accept,
        )

        if trim_bounds is not None:
            start, end = trim_bounds[i]
            for e in local_entries:
                if not (start <= e["sample_pos"] < end):
                    continue
                global_t = round((trimmed_offset + (e["sample_pos"] - start)) / cfg.fs, 3)
                out.append({"t_s": global_t, "cv_ms": e["cv_ms"], "accepted": e["accepted"]})
            trimmed_offset += end - start
        else:
            for e in local_entries:
                global_t = round((untrimmed_offset + e["sample_pos"]) / cfg.fs, 3)
                out.append({"t_s": global_t, "cv_ms": e["cv_ms"], "accepted": e["accepted"]})

        untrimmed_offset += channels.shape[0]

    return out
