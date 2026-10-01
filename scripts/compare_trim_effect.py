"""Experiment: does removing session_builder.trim_for_display's "smoothing
cut" (removes each file's ramp-in/ramp-out before splicing files together)
change the RMS/MDF/MFCV trend charts on Usecase 1?

Does NOT modify any existing code — app_realtime_session.py,
realtime_html.py, realtime_session.py, session_builder.py all stay exactly
as they are. This script just calls the same functions twice: once with
trimming (current demo behavior) and once without (raw concatenation), and
plots both.

    python -m scripts.compare_trim_effect [subject]

Writes compare_trim_effect.png (RMS+MDF) and compare_trim_effect_mfcv.png
to the project root, plus a printed report on the 70%->90% file boundary
specifically (checking whether the abstention gate produces a gap there
once trimming no longer hides it).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import config, realtime_html as rth, realtime_session as rts  # noqa: E402
from src import session_builder as sb, signal_processing as sp  # noqa: E402

NAVY = "#0B2A4A"
TEAL = "#028090"
AMBER = "#E08A1E"
GRAY = "#64748B"


def _rms_mdf_trend(signal: np.ndarray, fs: int, segment_bounds_sec: list[float]):
    filtered = sp.filter_signal(signal, fs)
    return rth._compute_trend(filtered, fs, segment_bounds_sec=segment_bounds_sec)


def main() -> None:
    subject = int(sys.argv[1]) if len(sys.argv) > 1 else config.TEST_SUBJECT
    print(f"Subject {subject}")

    signal, segments = sb.build_session_signal_avg(subject)

    # --- Untrimmed: raw concatenation, no ramp-in/out cut -------------------
    seg_bounds_untrimmed_sec = [s.end_sample / config.FS for s in segments[:-1]]
    t_u, rms_u, mdf_u = _rms_mdf_trend(signal, config.FS, seg_bounds_untrimmed_sec)
    cv_u = rts.compute_cv_series(subject, trim_bounds=None)

    # --- Trimmed: current demo behavior (unchanged) -------------------------
    trim_bounds = sb.trim_bounds_for_display(signal, segments, config.FS)
    display_signal, display_segments = sb.trim_for_display(signal, segments, config.FS)
    seg_bounds_trimmed_sec = [s.end_sample / config.FS for s in display_segments[:-1]]
    t_t, rms_t, mdf_t = _rms_mdf_trend(display_signal, config.FS, seg_bounds_trimmed_sec)
    cv_t = rts.compute_cv_series(subject, trim_bounds=trim_bounds)

    print(f"\nTotal duration: untrimmed={len(signal)/config.FS:.1f}s  "
          f"trimmed={len(display_signal)/config.FS:.1f}s")

    def _accept_rate(cv):
        acc = sum(1 for e in cv if e["accepted"])
        return acc, len(cv), (acc / len(cv) * 100 if cv else 0)

    a_u, n_u, r_u = _accept_rate(cv_u)
    a_t, n_t, r_t = _accept_rate(cv_t)
    print(f"MFCV acceptance: untrimmed={a_u}/{n_u} ({r_u:.1f}%)  "
          f"trimmed={a_t}/{n_t} ({r_t:.1f}%)")

    # --- Segment-by-segment MFCV acceptance (untrimmed) ---------------------
    print("\nUntrimmed, per segment:")
    for seg in segments:
        lo, hi = seg.start_sample / config.FS, seg.end_sample / config.FS
        in_range = [e for e in cv_u if lo <= e["t_s"] < hi]
        acc = [e for e in in_range if e["accepted"]]
        print(f"  mvc={seg.mvc:3d} [{lo:6.1f},{hi:6.1f}) "
              f"windows={len(in_range):4d} accepted={len(acc):4d} "
              f"({(len(acc)/len(in_range)*100 if in_range else 0):5.1f}%)")

    # --- 70% -> 90% boundary specifically -----------------------------------
    seg_70 = next((s for s in segments if s.mvc == 70), None)
    seg_90 = next((s for s in segments if s.mvc == 90), None)
    if seg_70 and seg_90:
        boundary_t = seg_70.end_sample / config.FS  # == seg_90.start_sample/FS
        window_s = 3.0  # look 3s each side of the file seam
        near = [e for e in cv_u if boundary_t - window_s <= e["t_s"] <= boundary_t + window_s]
        print(f"\n70%->90% file boundary at t={boundary_t:.2f}s (untrimmed), "
              f"+/-{window_s:.0f}s window:")
        for e in sorted(near, key=lambda e: e["t_s"]):
            side = "70%" if e["t_s"] < boundary_t else "90%"
            mark = "<-- boundary" if abs(e["t_s"] - boundary_t) < 0.3 else ""
            cv_str = f"{e['cv_ms']:.2f}" if e["cv_ms"] is not None else "  -  "
            print(f"  t={e['t_s']:7.2f}s ({side}) accepted={e['accepted']!s:5} cv={cv_str} {mark}")
        n_rejected_near = sum(1 for e in near if not e["accepted"])
        print(f"  -> {n_rejected_near}/{len(near)} windows near the seam rejected "
              f"(gate is expected to produce a gap here, since neither the tail of "
              f"70% nor the head of 90% is steady-state)")
    else:
        print("\n(70% and/or 90% segment not found for this subject — skipping boundary check)")

    # --- Plots ---------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
    ax1.plot(t_u, rms_u, color=AMBER, lw=1.5, label="RMS (chưa cắt)")
    ax1.set_title("RMS — chưa cắt (raw concat)", fontsize=11, color=NAVY, loc="left")
    ax1.set_xlabel("Thời gian [s]")
    ax1.set_ylabel("RMS")
    for b in seg_bounds_untrimmed_sec:
        ax1.axvline(b, color=GRAY, alpha=0.3, lw=1, ls="--")
    ax1.grid(alpha=0.15)

    ax2.plot(t_t, rms_t, color=TEAL, lw=1.5, label="RMS (đã cắt)")
    ax2.set_title("RMS — đã cắt ramp-in/out (hiện tại)", fontsize=11, color=NAVY, loc="left")
    ax2.set_xlabel("Thời gian [s]")
    ax2.set_ylabel("RMS")
    for b in seg_bounds_trimmed_sec:
        ax2.axvline(b, color=GRAY, alpha=0.3, lw=1, ls="--")
    ax2.grid(alpha=0.15)

    plt.tight_layout()
    out1 = PROJECT_ROOT / "compare_trim_effect.png"
    plt.savefig(out1, dpi=150)
    print(f"\nĐã ghi: {out1}")

    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
    acc_u = [(e["t_s"], e["cv_ms"]) for e in cv_u if e["accepted"]]
    acc_t = [(e["t_s"], e["cv_ms"]) for e in cv_t if e["accepted"]]
    if acc_u:
        tu, cu = np.array(acc_u).T
        ax3.plot(tu, cu, "o-", color=AMBER, ms=3, lw=1.2, label="MFCV (chưa cắt)")
    ax3.set_title(f"MFCV — chưa cắt ({a_u}/{n_u} chấp nhận, {r_u:.0f}%)",
                 fontsize=11, color=NAVY, loc="left")
    ax3.set_xlabel("Thời gian [s]")
    ax3.set_ylabel("MFCV [m/s]")
    for b in seg_bounds_untrimmed_sec:
        ax3.axvline(b, color=GRAY, alpha=0.3, lw=1, ls="--")
    ax3.grid(alpha=0.15)

    if acc_t:
        tt, ct = np.array(acc_t).T
        ax4.plot(tt, ct, "o-", color=TEAL, ms=3, lw=1.2, label="MFCV (đã cắt)")
    ax4.set_title(f"MFCV — đã cắt ({a_t}/{n_t} chấp nhận, {r_t:.0f}%)",
                 fontsize=11, color=NAVY, loc="left")
    ax4.set_xlabel("Thời gian [s]")
    ax4.set_ylabel("MFCV [m/s]")
    for b in seg_bounds_trimmed_sec:
        ax4.axvline(b, color=GRAY, alpha=0.3, lw=1, ls="--")
    ax4.grid(alpha=0.15)

    plt.tight_layout()
    out2 = PROJECT_ROOT / "compare_trim_effect_mfcv.png"
    plt.savefig(out2, dpi=150)
    print(f"Đã ghi: {out2}")


if __name__ == "__main__":
    main()
