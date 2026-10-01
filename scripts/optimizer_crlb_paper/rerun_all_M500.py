#!/usr/bin/env python3
"""
rerun_all_M500.py -- reruns every Monte Carlo TVD cell cited by
PAPER_Optimizer_CRLB.tex and PAPER_TVD_Interaction.tex at REPS_FAST=500 /
REPS_SLOW=250 (previously 100/50, see rerun_baseline_M100.py).

Cells: baseline, sinusoidal (global), MUAP (linear), sinusoidal sliding
(crlb_tvd_benchmark), plus MUAP x sinusoidal and multi-MU x sinusoidal
(scripts/tvd_interaction_paper). M is monkey-patched on each module so the
module sources keep their own defaults. Writes to the same canonical
journal_edit/ filenames; the M=100 outputs are kept in journal_edit/_backup_M100/.
"""
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import scripts.optimizer_crlb_paper.crlb_tvd_benchmark as tvdmod  # noqa: E402
import scripts.tvd_interaction_paper.run_muap_sinusoidal as muapsin  # noqa: E402
import scripts.tvd_interaction_paper.run_multipop_sinusoidal as multipop  # noqa: E402

M_FAST = 500   # was 100
M_SLOW = 250   # was 50

for mod in (muapsin, multipop):
    mod.M_FAST = M_FAST
    mod.M_SLOW = M_SLOW
tvdmod.REPS_FAST = M_FAST
tvdmod.REPS_SLOW = M_SLOW

if __name__ == "__main__":
    steps = [
        ("baseline", tvdmod.main),
        ("sinusoidal", tvdmod.main_sinusoidal),
        ("muap", tvdmod.main_muap),
        ("sinusoidal_sliding", tvdmod.main_sinusoidal_sliding),
        ("muap_sinusoidal_global", muapsin.run_global),
        ("muap_sinusoidal_sliding", muapsin.run_sliding),
        ("multipop_global", multipop.run_global),
        ("multipop_sliding", multipop.run_sliding),
    ]
    only = set(sys.argv[1:])
    for name, fn in steps:
        if only and name not in only:
            continue
        t0 = time.time()
        print(f"\n=== {name} at REPS_FAST={M_FAST}, REPS_SLOW={M_SLOW} ===", flush=True)
        fn()
        print(f"=== {name} done in {time.time() - t0:.0f} s ===", flush=True)
    print("\nDone.")
