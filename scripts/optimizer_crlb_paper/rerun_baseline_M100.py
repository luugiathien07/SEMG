#!/usr/bin/env python3
"""
rerun_baseline_M100.py -- reruns the four Monte Carlo cells that
PAPER_Optimizer_CRLB.tex's own tab:tvd_benchmark/tab:tvd_sinusoidal/
tab:tvd_muap/tab:tvd_sliding are built from, and that PAPER_TVD_Interaction.tex
borrows unchanged into its own tab:global/tab:local/tab:multipop, at
REPS_FAST=100/REPS_SLOW=50 instead of the original REPS_FAST=12/REPS_SLOW=6.

REPS_FAST/REPS_SLOW are monkey-patched on the crlb_tvd_benchmark module
itself (matching run_muap_sinusoidal.py's established pattern) rather than
edited in the module source, so this script alone controls the M used here
and main_multichannel() (a different paper, out of scope for this rerun) is
unaffected when called from its own entry point.

Writes to the SAME canonical filenames as the original M=12 run (baseline,
main_sinusoidal, main_muap, main_sinusoidal_sliding), overwriting them:
  tvd_benchmark_data.csv / tvd_vs_snr.pdf
  tvd_benchmark_sinusoidal_data.csv / tvd_vs_snr_sinusoidal.pdf
  tvd_benchmark_muap_data.csv / tvd_vs_snr_muap.pdf
  tvd_benchmark_sinusoidal_sliding_data.csv / tvd_vs_snr_sinusoidal_sliding.pdf

Explicitly out of scope (different REPS convention or not Monte Carlo, and
not cited by PAPER_TVD_Interaction.tex): tab:optim_benchmark and
tab:placement_sweep (crlb_optimizer_benchmark.py, REPS_FAST=15), the
tvd_sliding_gcc_fine_* figures, and the closed-form CRLB curves
(crlb_vs_snr.pdf, crlb_shwedyk_vs_muap.pdf -- analytical, no M).
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import scripts.optimizer_crlb_paper.crlb_tvd_benchmark as tvdmod  # noqa: E402

M_FAST = 100   # was 12
M_SLOW = 50    # was 6
tvdmod.REPS_FAST = M_FAST
tvdmod.REPS_SLOW = M_SLOW

if __name__ == "__main__":
    print(f"=== rerunning baseline cells at REPS_FAST={M_FAST}, REPS_SLOW={M_SLOW} ===\n")
    tvdmod.main()
    tvdmod.main_sinusoidal()
    tvdmod.main_muap()
    tvdmod.main_sinusoidal_sliding()
    print("\nDone.")
