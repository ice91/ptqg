"""
perturb_linear.py

C7: Minimal linear-perturbation posture for the tensor sector.

We expose a function that estimates (c_T^2 - 1) to O(alpha^2),
where alpha = (eps H0 r / c). For small alpha (outer disks, r ≲ 100 kpc),
the correction is expected to be negligible in this effective setup.

Here we return a conservative bound:
  |c_T^2 - 1| ≲ C_alpha * <alpha^2>
with C_alpha ~ O(1). Users can plug in astrophysical radii to get numeric
upper limits consistent with GW170817-scale constraints (order 1e-15).

Note: This is *not* a full derivation; it matches the paper's "posture"
and is sufficient for a first-line check.
"""

from __future__ import annotations

from typing import Union
import numpy as np
from .math_core import c_light


def tensor_speed_bound(
    eps: float,
    H0: float,
    r_m: Union[float, np.ndarray],
    C_alpha: float = 1.0,
) -> float:
    """
    Upper bound estimate for |c_T^2 - 1| using <alpha^2> with alpha = eps H0 r / c.

    Args:
      eps, H0: parameters
      r_m: radii at which the spatial imaginary block is relevant
      C_alpha: dimensionless O(1) factor

    Returns:
      Float upper bound (dimensionless)
    """
    r = np.asarray(r_m, dtype=float)
    alpha = (float(eps) * float(H0) / c_light) * r
    alpha2_mean = float(np.mean(alpha * alpha))
    return float(abs(C_alpha) * alpha2_mean)
