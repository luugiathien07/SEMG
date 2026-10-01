"""
ied_sweep.py -- follow-up to PAPER_Multichannel_TVD.tex's own "Directions for
future work" item: every result in that paper uses a single fixed IED
(8 mm); real arrays span roughly 4-12 mm, and a larger IED trades a bigger,
easier-to-resolve inter-channel delay against a bigger channel-to-channel
MUAP waveform-shape difference (the same reshaping mechanism the companion
optimizer/CRLB manuscript identifies as its own accuracy-floor cause).
This script repeats the K=2-vs-K=9 MU-population-source channel-pooling
comparison (Section sub:results-realistic / Table tab:realistic) at four
additional IED values (4, 6, 10, 12 mm), to locate where -- if anywhere --
the pooling benefit degrades as IED grows.

IMPORTANT implementation note: GridConfig.ied_m must be patched in BOTH
crlb_optimizer_benchmark's own CFG (which coarse_grid()/GATE-bounds use to
size the search grid) and crlb_tvd_benchmark's separate CFG instance (which
theta_fn/CRLB conversions use) -- they are two independent GridConfig()
instances, not a shared singleton. The sinusoidal ground truth's derived
module-level constant _SIN_K is also computed once at import time from the
DEFAULT ied_m and must be recomputed by hand after patching, or every IED
value would silently reuse the 8mm ground truth.

Each IED value runs in its own subprocess (not just here in a loop) so
mutated module state from one run can never leak into the next.

Run from repo root: `python3 scripts/multichannel_paper/ied_sweep.py`
Output per IED: journal_edit/tvd_multichannel_realistic_ied<N>mm_data.csv
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IED_VALUES_MM = [4, 6, 10, 12]  # 8mm is the paper's existing baseline

WORKER_SNIPPET = """
import sys
sys.path.insert(0, "{project_root}")
sys.path.insert(0, "{project_root}/scripts/optimizer_crlb_paper")

ied_mm = {ied_mm}
ied_m = ied_mm / 1000.0

import crlb_optimizer_benchmark as scalar_mod
scalar_mod.CFG.ied_m = ied_m

import crlb_tvd_benchmark as tvd_mod
tvd_mod.CFG.ied_m = ied_m
# _SIN_K depends on ied_m (see module docstring); recompute by hand since it
# was baked in at import time from the module's own default ied_m=0.008.
tvd_mod._SIN_K = tvd_mod.CFG.ied_m * tvd_mod.CFG.fs * (tvd_mod._SIN_B - tvd_mod._SIN_C) / tvd_mod._SIN_CV_LOW

import scripts.multichannel_paper.realistic_source_multichannel as realistic_mod
source = realistic_mod.load_source()

tvd_mod.main_multichannel(
    tilt_degs=(0.0,),
    n_rows_list=(2, 9),
    pdf_name=f"tvd_multichannel_realistic_ied{{ied_mm}}mm.pdf",
    csv_name=f"tvd_multichannel_realistic_ied{{ied_mm}}mm_data.csv",
    theta_fn=tvd_mod.true_theta_sinusoidal,
    source_fixed=source,
    variant_label=f"realistic MU-population source, IED={{ied_mm}}mm",
)
print(f"IED={{ied_mm}}mm: cv_ref check = {{tvd_mod.CFG.ied_m * tvd_mod.CFG.fs / tvd_mod.true_theta_sinusoidal(tvd_mod.N - 1):.4f}} m/s")
"""


def main() -> None:
    for ied_mm in IED_VALUES_MM:
        print(f"\\n=== Running IED={ied_mm}mm ===", flush=True)
        snippet = WORKER_SNIPPET.format(project_root=PROJECT_ROOT, ied_mm=ied_mm)
        result = subprocess.run(
            [sys.executable, "-c", snippet],
            cwd=str(PROJECT_ROOT),
            capture_output=True, text=True,
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"FAILED for IED={ied_mm}mm:\\n{result.stderr}", file=sys.stderr)


if __name__ == "__main__":
    main()
