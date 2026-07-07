"""MRMR feature ranking (replacement for MATLAB ``fscmrmr``).

Implements MRMR-FCQ: at each step pick the feature maximizing
``relevance(f) - mean(redundancy(f, already_selected))`` where relevance is
mutual information with the label and redundancy is mean absolute Pearson
correlation with previously selected features. Fit on the training set only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from . import config


def mrmr_rank(
    X: pd.DataFrame,
    y: np.ndarray,
    k: int | None = None,
    random_state: int = config.RANDOM_STATE,
) -> tuple[list[str], dict[str, float]]:
    """Return (ranked_feature_names, relevance_scores).

    ``ranked_feature_names`` is the full ordering (best first); take the first
    ``k`` for the top-k selection. ``relevance_scores`` maps every feature to
    its raw mutual-information relevance (for display).
    """
    feature_names = list(X.columns)
    k = k or len(feature_names)
    k = min(k, len(feature_names))

    Xv = X.to_numpy(dtype=float)
    relevance = mutual_info_classif(Xv, y, random_state=random_state)
    relevance = np.asarray(relevance, dtype=float)
    rel_by_name = {name: float(r) for name, r in zip(feature_names, relevance)}

    # Absolute correlation matrix for redundancy (nan -> 0 for constant cols).
    corr = np.corrcoef(Xv, rowvar=False)
    corr = np.nan_to_num(np.abs(corr), nan=0.0)

    remaining = list(range(len(feature_names)))
    selected: list[int] = []

    # First pick = highest relevance.
    first = int(np.argmax(relevance))
    selected.append(first)
    remaining.remove(first)

    while remaining and len(selected) < len(feature_names):
        best_idx, best_score = None, -np.inf
        for idx in remaining:
            redundancy = np.mean([corr[idx, s] for s in selected])
            score = relevance[idx] - redundancy
            if score > best_score:
                best_score, best_idx = score, idx
        selected.append(best_idx)
        remaining.remove(best_idx)

    ranked_names = [feature_names[i] for i in selected]
    return ranked_names, rel_by_name
