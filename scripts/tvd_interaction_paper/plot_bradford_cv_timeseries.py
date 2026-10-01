"""
Regenerates tvd_bradford_cv_timeseries.pdf (PAPER_TVD_Interaction.tex, Section
results-realchecks, "Does CV track contraction level and speed?" paragraph /
Figure fig:bradford-timeseries) from journal_edit/tvd_bradford_realdata.csv.

Two panels of GCC-estimated CV(t) for one representative subject (default
S01, which has all five recordings):
  left  -- isotonic contraction, 25% vs. 50% MVIC
  right -- isokinetic contraction, 30 / 90 / 300 deg/s

Only estimates inside the physiological gate [2, 10] m/s are plotted (the
same acceptance range used throughout this manuscript's real-data checks);
gaps in a trace are windows the gate rejected, not missing data.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "journal_edit"
CV_MIN, CV_MAX = 2.0, 10.0
GAP_FACTOR = 3.0  # break the line where a gap exceeds this multiple of the median hop

DEFAULT_SUBJECT = "S01"
ISOTONIC = [("ISOT25_1", "25% MVIC", "#4C72B0"), ("ISOT50_1", "50% MVIC", "#DD8452")]
ISOKINETIC = [("ISOK30", "$30^\\circ$/s", "#4C72B0"), ("ISOK90", "$90^\\circ$/s", "#55A868"),
              ("ISOK300", "$300^\\circ$/s", "#C44E52")]


def trace(df: pd.DataFrame, subject: str, cond: str) -> pd.DataFrame:
    sel = df[df.file == f"{subject}_{cond}"]
    med = sel.groupby("t_s").cv_gcc.median().reset_index().sort_values("t_s")
    med = med[med.cv_gcc.between(CV_MIN, CV_MAX)]
    return med


def with_gap_breaks(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(t) < 3:
        return t, y
    hop = np.median(np.diff(t))
    gap = np.diff(t) > GAP_FACTOR * hop
    t_out, y_out = [t[0]], [y[0]]
    for i, is_gap in enumerate(gap, start=1):
        if is_gap:
            t_out.append(np.nan)
            y_out.append(np.nan)
        t_out.append(t[i])
        y_out.append(y[i])
    return np.array(t_out), np.array(y_out)


def panel(ax, df, subject, conds, title):
    for cond, label, color in conds:
        tr = trace(df, subject, cond)
        if tr.empty:
            continue
        t, y = with_gap_breaks(tr.t_s.to_numpy(), tr.cv_gcc.to_numpy())
        ax.plot(t, y, "o-", color=color, ms=3, lw=1.0, label=label)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Window-median MFCV, GCC [m/s]")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, framealpha=0.9)
    ax.grid(alpha=0.2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default=DEFAULT_SUBJECT)
    args = ap.parse_args()

    df = pd.read_csv(OUT_DIR / "tvd_bradford_realdata.csv")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    panel(axes[0], df, args.subject, ISOTONIC, f"Isotonic -- subject {args.subject}")
    panel(axes[1], df, args.subject, ISOKINETIC, f"Isokinetic -- subject {args.subject}")
    fig.tight_layout()

    out = OUT_DIR / "tvd_bradford_cv_timeseries.pdf"
    fig.savefig(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
