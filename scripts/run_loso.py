"""Run full Leave-One-Subject-Out evaluation and print/save the report.

    python -m scripts.run_loso            # build features (cached) + report both labelings
    python -m scripts.run_loso --rebuild  # force feature re-extraction

Writes the same report to ``loso_results.txt`` at the project root.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import config, loso  # noqa: E402


LABEL_TITLES = {
    "mvc": "MVC>60 THRESHOLD  (project labeling: 90%->Fatigue, 10_ap_fatigue->Normal)",
    "paper": "PAPER DESIGN  (BME 2024: 90%->Normal, 70%/fatigue/10_ap_fatigue->Fatigue)",
}


def _fmt(v: float) -> str:
    return "  nan " if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:6.3f}"


def report(df, out_lines: list[str]) -> None:
    def emit(s: str = "") -> None:
        print(s)
        out_lines.append(s)

    for scheme in ("mvc", "paper"):
        col = loso.LABEL_COLS[scheme]
        counts = df[col].value_counts().to_dict()
        res = loso.run_loso(df, scheme)

        emit("=" * 82)
        emit(f"LABELING: {LABEL_TITLES[scheme]}")
        emit("=" * 82)
        emit(f"Samples (channel rows): {len(df)}   "
             f"Normal={counts.get(0, 0)}  Fatigue={counts.get(1, 0)}")
        emit(f"Subjects (with EMG data): {res.subjects}")
        emit(f"mRMR top-{config.TOP_K_FEATURES} (reported; classifiers use all 14): "
             f"{res.selected_features}")
        emit("")

        # ---- Pooled metrics, both positive-class conventions ----
        for pos, tag in ((1, "positive = Fatigue(1)  [clinically correct — headline]"),
                         (0, "positive = Normal(0)   [paper's MATLAB convention]")):
            emit(f"POOLED LOSO — {tag}")
            emit(f"  {'Model':<12}{'Acc':>7}{'Prec':>7}{'Rec':>7}{'F1':>7}{'AUC':>7}")
            for name in res.models:
                m = res.pooled[name][pos]
                emit(f"  {name:<12}{_fmt(m['acc'])}{_fmt(m['prec'])}{_fmt(m['rec'])}"
                     f"{_fmt(m['f1'])}{_fmt(m['auc'])}")
            emit("")

        # ---- Per-subject (per-fold) F1 for every model, Fatigue-positive ----
        emit("PER-SUBJECT (per-fold) metrics — positive = Fatigue(1)")
        for name in res.models:
            emit(f"  [{name}]")
            emit(f"    {'subj':>5}{'n':>6}{'nFat':>6}{'Acc':>7}{'Prec':>7}"
                 f"{'Rec':>7}{'F1':>7}{'AUC':>7}")
            for s in res.subjects:
                m = res.per_fold[name].get(s)
                if m is None:
                    emit(f"    {s:>5}{'':>6}{'':>6}   skipped (train set single-class)")
                    continue
                flag = "  <-- single-class test" if m["n_pos"] in (0, m["n"]) else ""
                emit(f"    {s:>5}{m['n']:>6}{m['n_pos']:>6}{_fmt(m['acc'])}"
                     f"{_fmt(m['prec'])}{_fmt(m['rec'])}{_fmt(m['f1'])}"
                     f"{_fmt(m['auc'])}{flag}")
            # folds reaching the paper's target
            reached = [s for s in res.subjects
                       if res.per_fold[name].get(s)
                       and res.per_fold[name][s]["f1"] >= 0.95]
            emit(f"    folds with F1>=0.95 (Fatigue): {reached if reached else 'none'}")
            emit("")

        emit(f"Paper reference (BME 2024, single favorable LOSO fold): "
             f"KNN F1={config.PAPER_REFERENCE['KNN_F1']}, AUC~{config.PAPER_REFERENCE['KNN_AUC']}")
        emit("")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true",
                    help="force feature re-extraction (ignore the parquet cache)")
    args = ap.parse_args()

    print("Building LOSO feature table (per channel, all subjects)...")
    df = loso.build_feature_table(use_cache=not args.rebuild)

    out_lines: list[str] = []
    report(df, out_lines)

    out_path = PROJECT_ROOT / "loso_results.txt"
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"\nSaved report to {out_path}")


if __name__ == "__main__":
    main()
