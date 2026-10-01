"""
Regenerates the two (Shwedyk colored-noise, single tiled Farina-Merletti MUAP)
rows of PAPER_Multichannel_TVD.tex's Table tab:source-comparison, at the
inverse-sinusoidal TVD ground truth, K=2 vs. K=13, untilted -- using the same
main_multichannel() pipeline already used for tab:pooling/tab:tilt (colored
noise, linear ramp) and tab:realistic/tab:tilt-realistic (MU-population
source, via realistic_source_multichannel.py). The MU-population rows of
tab:source-comparison are NOT regenerated here: they are identical to
tab:realistic's own untilted (K=2,9) rows, already produced by
realistic_source_multichannel.py.

Outputs:
  - journal_edit/tvd_multichannel_shwedyk_sinusoidal_data.csv / .pdf
  - journal_edit/tvd_multichannel_emgmodel_sinusoidal_data.csv / .pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "optimizer_crlb_paper"))

from crlb_tvd_benchmark import (  # noqa: E402
    main_multichannel, true_theta_sinusoidal, load_muap_source,
)


def main() -> None:
    # Shwedyk colored-noise source, sinusoidal ground truth, K=2 vs 13, untilted.
    main_multichannel(
        tilt_degs=(0.0,),
        n_rows_list=(2, 13),
        pdf_name="tvd_multichannel_shwedyk_sinusoidal.pdf",
        csv_name="tvd_multichannel_shwedyk_sinusoidal_data.csv",
        theta_fn=true_theta_sinusoidal,
        source_fixed=None,
        variant_label="Shwedyk colored noise (sinusoidal ground truth)",
    )

    # Single tiled Farina-Merletti MUAP source, sinusoidal ground truth, K=2 vs 13, untilted.
    source = load_muap_source()
    main_multichannel(
        tilt_degs=(0.0,),
        n_rows_list=(2, 13),
        pdf_name="tvd_multichannel_emgmodel_sinusoidal.pdf",
        csv_name="tvd_multichannel_emgmodel_sinusoidal_data.csv",
        theta_fn=true_theta_sinusoidal,
        source_fixed=source,
        variant_label="single tiled Farina-Merletti MUAP (sinusoidal ground truth)",
    )


if __name__ == "__main__":
    main()
