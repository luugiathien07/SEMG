#!/usr/bin/env python3
"""
rerun_scalar_M500.py -- reruns the scalar six-optimizer benchmark
(crlb_optimizer_benchmark.main: tab:optim_benchmark, crlb_vs_snr.pdf) and its
MUAP-source counterpart (crlb_muap_benchmark.main) at
REPS_FAST=500 / REPS_SLOW=250 instead of 15 / 6, matching the TVD cells
(rerun_all_M500.py). REPS are monkey-patched on the module. Overwrites
journal_edit/optim_benchmark_data.csv, crlb_vs_snr.pdf,
optim_benchmark_muap_data.csv and crlb_vs_snr_muap.pdf.

Usage: rerun_scalar_M500.py [colored|muap]   (default: both)
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import scripts.optimizer_crlb_paper.crlb_optimizer_benchmark as optmod  # noqa: E402
import scripts.optimizer_crlb_paper.crlb_muap_benchmark as muapmod  # noqa: E402

# crlb_muap_benchmark imports REPS_* by value, so patch both modules.
for mod in (optmod, muapmod):
    mod.REPS_FAST = 500   # was 15
    mod.REPS_SLOW = 250   # was 6

if __name__ == "__main__":
    which = sys.argv[1:] or ["colored", "muap"]
    if "colored" in which:
        optmod.main()
    if "muap" in which:
        muapmod.main()
