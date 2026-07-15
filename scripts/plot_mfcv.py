#!/usr/bin/env python3
"""
plot_mfcv.py — Ve duong MFCV va RMS theo thoi gian tu JSON cua run_mfcv.py.

Day la HINH cho slide: RMS di LEN, CV di XUONG = chu ky thi giac cua moi co.

Cach dung:
    python plot_mfcv.py mfcv.json --out mfcv.png
    python plot_mfcv.py mfcv.json --title "70% MVC - moi" --out fig.png
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY = "#0B2A4A"
TEAL = "#028090"
AMBER = "#E08A1E"
GRAY = "#64748B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_file")
    ap.add_argument("--out", default="mfcv.png")
    ap.add_argument("--title", default=None)
    ap.add_argument("--dpi", type=int, default=160)
    a = ap.parse_args()

    with open(a.json_file, encoding="utf-8") as f:
        d = json.load(f)

    mf = d["mfcv"]
    rms = d.get("rms", [])
    s = d["summary"]

    acc = [(m["t_s"], m["cv_ms"]) for m in mf if m["accepted"]]
    rej = [m["t_s"] for m in mf if not m["accepted"]]

    if not acc:
        print("! Khong co cua so nao duoc chap nhan — khong ve duoc.")
        print(f"  {s['n_abstained']}/{s['n_windows']} cua so bi tu choi.")
        print(f"  Ly do: {s['abstain_reasons']}")
        return

    t_cv, cv = np.array(acc).T
    fig, ax1 = plt.subplots(figsize=(9, 4.2))

    # RMS (truc trai)
    if rms:
        t_r = np.array([r["t_s"] for r in rms])
        v_r = np.array([r["rms_uv"] for r in rms])
        ax1.plot(t_r, v_r, color=AMBER, lw=1.8, label="RMS (bien do)")
        ax1.set_ylabel("RMS  [don vi tin hieu]", color=AMBER, fontsize=10)
        ax1.tick_params(axis="y", labelcolor=AMBER)
    ax1.set_xlabel("Thoi gian co co  [s]", fontsize=10)

    # CV (truc phai)
    ax2 = ax1.twinx()
    ax2.plot(t_cv, cv, "o-", color=TEAL, ms=3, lw=1.8, label="MFCV")
    ax2.set_ylabel("MFCV  [m/s]", color=TEAL, fontsize=10)
    ax2.tick_params(axis="y", labelcolor=TEAL)

    # duong hoi quy CV
    if len(t_cv) >= 3:
        k, b = np.polyfit(t_cv, cv, 1)
        ax2.plot(t_cv, k * t_cv + b, "--", color=NAVY, lw=1.4,
                 label=f"doc CV = {k:+.4f} (m/s)/s")

    # danh dau cua so bi tu choi (cong abstention)
    for t in rej:
        ax1.axvline(t, color=GRAY, alpha=0.12, lw=2)
    if rej:
        ax1.plot([], [], color=GRAY, alpha=0.4, lw=6,
                 label=f"cong abstention tu choi ({len(rej)})")

    title = a.title or "MFCV va RMS theo thoi gian"
    cfg = d.get("config", {})
    sub = (f"cot {cfg.get('column_used','?')} · IED {cfg.get('ied_mm','?')}mm · "
           f"{cfg.get('fs','?')}Hz · {s['n_accepted']}/{s['n_windows']} cua so dat")
    ax1.set_title(f"{title}\n{sub}", fontsize=11, color=NAVY, loc="left")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best", fontsize=8, framealpha=0.9)
    ax1.grid(alpha=0.15)
    for sp in ("top",):
        ax1.spines[sp].set_visible(False)
        ax2.spines[sp].set_visible(False)

    plt.tight_layout()
    plt.savefig(a.out, dpi=a.dpi)
    print(f"Da ghi: {a.out}")
    print(f"  CV trung binh = {s['cv_mean_ms']} m/s")
    print(f"  doc CV        = {s['cv_slope_ms_per_s']} (m/s)/s")
    print(f"  cua so dat    = {s['n_accepted']}/{s['n_windows']}")


if __name__ == "__main__":
    main()
