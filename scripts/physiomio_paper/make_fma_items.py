#!/usr/bin/env python3
"""
make_fma_items.py -- tabulate the FMA item score of every PhysioMio recording
and gesture from the raw files' `fma` column (constant within a gesture, empty
for Rest).

Output: scripts/physiomio_paper/fma_items.csv
Run from repo root: python scripts/physiomio_paper/make_fma_items.py
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "dataset" / "physiomio" / "data"


def one(arg):
    rec, path = arg
    t = pq.read_table(DATA / path, columns=["fma", "movement_type"]).to_pandas()
    out = []
    for g, s in t.groupby("movement_type", sort=False):
        v = s.fma.dropna().unique()
        assert len(v) <= 1, (path, g, v)
        out.append((rec, g, len(s), len(v), v[0] if len(v) else np.nan, s.fma.isna().mean()))
    return out


def main():
    meta = pd.read_csv(DATA / "metadata.csv")
    with ProcessPoolExecutor(16) as ex:
        rows = [r for rs in ex.map(one, enumerate(meta.file_path)) for r in rs]
    T = pd.DataFrame(rows, columns=["rec", "gesture", "n", "nuniq", "fma", "na_frac"])
    T = T.merge(meta[["patient", "arm_type", "recording_index", "days_after_stroke"]],
                left_on="rec", right_index=True)
    T.to_csv(HERE / "fma_items.csv", index=False)
    print(f"wrote {len(T)} rows")


if __name__ == "__main__":
    main()
