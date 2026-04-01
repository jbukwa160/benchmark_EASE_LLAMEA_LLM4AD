from __future__ import annotations
import numpy as np

def calc_crowding_distance(F):
    F = np.asarray(F, dtype=float)
    if F.ndim == 1:
        return np.zeros(len(F), dtype=float)
    return np.zeros(F.shape[0], dtype=float)
