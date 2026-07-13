"""Evaluate the canonical held-out subject (config.TEST_SUBJECT, default 6).

Trains on every other subject with EMG data and tests on the held-out one,
reproducing the BME 2024 paper's single-fold LOSO protocol. Uses the MVC>60
labeling (the scheme that yields the headline numbers).

    python -m scripts.eval_held_out_subject            # subject config.TEST_SUBJECT (6)
    python -m scripts.eval_held_out_subject --subject 12
    python -m scripts.eval_held_out_subject --label paper

Writes the report to ``held_out_subject_results.txt``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import config, loso  # noqa: E402


def _fmt(v) -> str:
    return "  nan " if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:6.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=config.TEST_SUBJECT)
    ap.add_argument("--label", choices=list(loso.LABELERS), default="mvc")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    df = loso.build_feature_table(use_cache=not args.rebuild)
    results = loso.evaluate_subject(df, args.subject, label_scheme=args.label)

    label_col = loso.LABEL_COLS[args.label]
    train = df[df["subject"] != args.subject]
    test = df[df["subject"] == args.subject]

    lines: list[str] = []
    def emit(s: str = "") -> None:
        print(s); lines.append(s)

    emit("=" * 70)
    emit("LEAVE-ONE-SUBJECT-OUT — single held-out fold (BME 2024 protocol)")
    emit("=" * 70)
    emit(f"Held-out TEST subject : {args.subject}")
    emit(f"TRAIN subjects        : {sorted(int(s) for s in train['subject'].unique())}")
    emit(f"Labeling scheme       : {args.label}  "
         f"({'MVC>60 -> Fatigue' if args.label == 'mvc' else 'paper design'})")
    emit(f"Train samples         : {len(train)}  "
         f"(Normal={int((train[label_col]==0).sum())}, "
         f"Fatigue={int((train[label_col]==1).sum())})")
    emit(f"Test samples (subj {args.subject}) : {len(test)}  "
         f"(Normal={int((test[label_col]==0).sum())}, "
         f"Fatigue={int((test[label_col]==1).sum())})")
    emit("-" * 70)
    emit(f"{'Model':<12}{'Acc':>8}{'Prec':>8}{'Rec':>8}{'F1':>8}{'AUC':>8}   (positive=Fatigue)")
    for name, m in results.items():
        emit(f"{name:<12}{_fmt(m['acc'])}{_fmt(m['prec'])}{_fmt(m['rec'])}"
             f"{_fmt(m['f1'])}{_fmt(m['auc'])}")
    emit("=" * 70)
    emit("Reference (paper KNN, single favorable fold): "
         f"F1={config.PAPER_REFERENCE['KNN_F1']}, AUC~{config.PAPER_REFERENCE['KNN_AUC']}")

    out = PROJECT_ROOT / "held_out_subject_results.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved report to {out}")


if __name__ == "__main__":
    main()
