"""
Plots estimated CV vs. time for one real recording, K=2 vs K=11
double-differential channels, using the K-channel GCC-pooled estimator
(tvd_multichannel_realdata_gcc.csv, produced by real_data_validation_gcc.py's
sliding-window pass -- WIN_S=0.5s, HOP_S=0.25s per window).

Complements Fig. 4 (paired precision boxplots) with the actual per-window
CV(t) trace for a single (subject, condition, col), so the pooling effect
on GCC can be seen directly rather than only summarized as an SD.

Default recording is chosen as the one with the highest combined K=2/K=11
acceptance rate in the GCC CSV (Sujet_12, condition 20, col 2); pass
--subject/--condition/--col to pick another.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "journal_edit"

DEFAULT_SUBJECT = "Sujet_12"
DEFAULT_CONDITION = "20"
DEFAULT_COL = 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default=DEFAULT_SUBJECT)
    ap.add_argument("--condition", default=DEFAULT_CONDITION)
    ap.add_argument("--col", type=int, default=DEFAULT_COL)
    args = ap.parse_args()

    df = pd.read_csv(OUT_DIR / "tvd_multichannel_realdata_gcc.csv")
    df["condition"] = df["condition"].astype(str)
    sel = df[(df.subject == args.subject) & (df.condition == args.condition) & (df.col == args.col)]
    if sel.empty:
        raise SystemExit(f"no rows for subject={args.subject} condition={args.condition} col={args.col}")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = {"K2": "#4C72B0", "K11": "#DD8452"}
    for k_label, color in colors.items():
        sub = sel[sel.K == k_label].sort_values("t_s")
        acc = sub[sub.accepted == 1]
        rej = sub[sub.accepted == 0]
        ax.plot(acc.t_s, acc.cv_ms, "o-", color=color, ms=3, lw=1.0, label=f"{k_label} (accepted)")
        if not rej.empty:
            ax.plot(rej.t_s, rej.cv_ms, "x", color=color, ms=4, alpha=0.5,
                     label=f"{k_label} (rejected)")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("CV [m/s]")
    ax.set_title(f"GCC-pooled CV(t), K=2 vs K=11 -- {args.subject}, condition {args.condition}, col {args.col}",
                 fontsize=9)
    ax.legend(fontsize=7, framealpha=0.9)
    ax.grid(alpha=0.2)
    fig.tight_layout()

    out = OUT_DIR / "tvd_multichannel_realdata_gcc_timeseries.pdf"
    fig.savefig(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
