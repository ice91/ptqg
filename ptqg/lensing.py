"""
lensing.py

C6: Weak-lensing prediction scaffold for the outer-disk slope implied by
a constant acceleration a0 = eps c H0.

Caveat:
  Mapping a *non-Newtonian effective acceleration* to lensing shear rigorously
  depends on how light deflection couples to the underlying metric. Here we
  provide a pragmatic "effective-density" toy mapping to get stackable trends
  (∝ H(z)) for data-driven comparison, with clear disclaimers.

We model an effective projected "surface density" that yields a tangential
shear roughly scaling as  ~ a0 / (c^2)  up to order-unity geometry factors:

  gamma_t(R, z) ≈ A_geom(z) * [ a0(z) / c^2 ] * (R / R0)^p

with default p = 0 (flat plateau) or p = +1 (linearly rising) to bracket
reasonable outer-disk behaviors after baryon masking.

Functions:
- a0_of_z(eps, H_of_z): a0(z)
- gamma_t_linear(R, eps, H_of_z, A_geom=1.0, p=0)
"""

from __future__ import annotations

from typing import Callable, Union
import numpy as np
from .math_core import c_light


def a0_of_z(eps: float, H_of_z: Callable[[float], float], z: float) -> float:
    return float(eps) * c_light * float(H_of_z(z))


def gamma_t_linear(
    R_m: Union[float, np.ndarray],
    eps: float,
    H_of_z: Callable[[float], float],
    z: float,
    A_geom: float = 1.0,
    p: float = 0.0,
    R0_m: float = 10.0 * 3.085677581e19,  # 10 kpc in meters
) -> np.ndarray:
    """
    Toy gamma_t profile:
      gamma_t(R; z) = A_geom * (a0(z)/c^2) * (R / R0)^p

    Args:
      R_m: projected radius [m]
      eps: model parameter
      H_of_z: callable returning H(z) [s^-1]
      z: redshift
      A_geom: order-unity lensing geometry factor
      p: slope index (0: plateau; 1: rising)
      R0_m: reference radius

    Returns:
      gamma_t as a dimensionless array
    """
    R = np.asarray(R_m, dtype=float)
    a0z = a0_of_z(eps, H_of_z, z)
    base = (a0z / (c_light * c_light))
    return float(A_geom) * base * np.power(np.clip(R / float(R0_m), 1e-12, None), p)
