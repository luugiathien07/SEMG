"""
Regenerates PAPER_Multichannel_TVD.tex's Table tab:realistic (K=2 vs K=9,
untilted, realistic multi-motor-unit source) and Table tab:tilt-realistic
(K in {2,9}, tilt in {0,15}), using the SAME already-verified
make_multichannel_pair()/main_multichannel() pipeline from
crlb_tvd_benchmark.py that already reproduces this paper's colored-noise
tables (Tables tab:pooling, tab:tilt) exactly.

The realistic source is a single channel (index 4, the middle of the
9-channel array) extracted from scripts/multichannel_paper/muap_octave's
headless-patched SEMG_simulator_v4 run (see that directory's own docstrings
for the compatibility fixes applied) -- reused across all noise
realizations exactly the way load_muap_source() reuses a single Farina-
Merletti channel for the scalar-case realistic-source check elsewhere in
this manuscript series. Only ONE channel is used: this sidesteps a real
open question about the legacy simulator's own multichannel delay
calibration (its DeltaE argument only corrects the library's built-in
delay for CV != 4 m/s -- it does not appear to rescale the library's own,
undocumented native electrode spacing, and no library-generation code
survives in this repository to check), since the K-channel array itself is
built by this paper's own already-verified delay-injection model
(Eq. mc-model / make_multichannel_pair), not by the simulator's internal
geometry.

Unlike the earlier (predates-GCC) version of Table tilt-realistic, GCC is
included here automatically, since main_multichannel()'s methods dict
already contains it unconditionally -- this also resolves the paper's own
"left open, for a follow-up run" note in Section sub:results-tilt-realistic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "optimizer_crlb_paper"))

from crlb_tvd_benchmark import main_multichannel, true_theta_sinusoidal  # noqa: E402

SOURCE_CHANNEL = 4  # middle of the 9-channel array, matching MUAP_CHANNEL's
                     # "middle channel" convention in crlb_tvd_benchmark.py

MAT_PATH = Path(__file__).resolve().parent / "muap_octave" / "realistic_source_ch4.mat"


def load_source() -> np.ndarray:
    d = sio.loadmat(MAT_PATH, squeeze_me=True, struct_as_record=False)
    mp = np.asarray(d["MP"], dtype=float)
    return mp[SOURCE_CHANNEL]


def main() -> None:
    source = load_source()
    assert source.shape == (600,), source.shape

    # Table tab:realistic: untilted only, K in {2,9}, sinusoidal ground truth.
    main_multichannel(
        tilt_degs=(0.0,),
        n_rows_list=(2, 9),
        pdf_name="tvd_multichannel_realistic_sinusoidal.pdf",
        csv_name="tvd_multichannel_realistic_data.csv",
        theta_fn=true_theta_sinusoidal,
        source_fixed=source,
        variant_label="realistic MU-population source (untilted)",
    )

    # Table tab:tilt-realistic: tilt in {0,15}, K in {2,9}, GCC now included.
    main_multichannel(
        tilt_degs=(0.0, 15.0),
        n_rows_list=(2, 9),
        pdf_name="tvd_multichannel_realistic_tilt.pdf",
        csv_name="tvd_multichannel_realistic_tilt_data.csv",
        theta_fn=true_theta_sinusoidal,
        source_fixed=source,
        variant_label="realistic MU-population source (tilt sweep)",
    )


if __name__ == "__main__":
    main()
