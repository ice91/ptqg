"""
model_compare.py

C4: AIC/BIC helpers with explicit parameter counting.
"""

from __future__ import annotations

from typing import Tuple


def aic_bic(logL_max: float, k: int, Ntot: int) -> Tuple[float, float]:
    """
    AIC = -2 logL_max + 2 k
    BIC = -2 logL_max + k ln N
    """
    import math
    AIC = -2.0 * float(logL_max) + 2.0 * float(k)
    BIC = -2.0 * float(logL_max) + float(k) * math.log(max(Ntot, 1))
    return AIC, BIC
