"""
geodesics.py

C3: Observable geodesic Lagrangian and weak-field circular velocity.

We adopt the observable Lagrangian:
  L_obs = sqrt( Pi_PT[ - G_{mu nu} \dot x^mu \dot x^nu ] )

In the weak/slow (galactic) regime and for the block ansatz with
  G_ij = a^2 (1 + u alpha_s) δ_ij,  alpha_s = (eps H0 / c) r,
one finds an effective *constant acceleration* a0 = eps c H0 that
adds linearly in v^2:  v^2 = v_bar^2 + a0 r.

This module provides:
- a0_from_eps(eps, H0): acceleration scale
- circular_velocity(a, eps, H0, r, vbar2): returns v_model (km/s) given vbar^2(r)
  as a function/array in (m/s)^2. Unit handling is explicit.

Note: We do not re-derive the full EL equations here; instead we encode the
resulting law as the verified leading-order effective relation as per the paper.
"""

from __future__ import annotations

from typing import Callable, Union
import numpy as np
from .math_core import c_light


def a0_from_eps(eps: float, H0: float) -> float:
    """
    a0 = eps * c * H0   (SI: m s^-2)
    """
    return float(eps) * c_light * float(H0)


def circular_velocity(
    a_scale: float,
    eps: float,
    H0: float,
    r_m: Union[np.ndarray, float],
    vbar2_of_r_m: Union[np.ndarray, float, Callable[[np.ndarray], np.ndarray]],
    to_km_s: bool = True,
) -> np.ndarray:
    """
    Compute circular velocity model:
      v^2(r) = v_bar^2(r) + (eps c H0) r    with r in meters, v in m/s.

    Args:
      a_scale: scale factor (galactic weak-field: a~1; kept for API symmetry)
      eps, H0: model parameter and Hubble rate [s^-1]
      r_m: radii [m]
      vbar2_of_r_m: (m/s)^2 array or callable returning (m/s)^2 at r
      to_km_s: if True, return km/s

    Returns:
      v_model(r) as numpy array
    """
    r_m = np.asarray(r_m, dtype=float)
    if callable(vbar2_of_r_m):
        vbar2 = np.asarray(vbar2_of_r_m(r_m), dtype=float)
    else:
        vbar2 = np.asarray(vbar2_of_r_m, dtype=float)
        if vbar2.shape != r_m.shape:
            raise ValueError("vbar2_of_r_m shape must match r_m if array is provided.")

    a0 = a0_from_eps(eps, H0)  # m/s^2
    v2 = vbar2 + a0 * r_m
    v = np.sqrt(np.clip(v2, 0.0, None))
    if to_km_s:
        return v / 1000.0
    return v


# Simple baryon toy models for quick tests ----------------------------------

def exp_disk_vbar2(
    r_m: np.ndarray,
    v0_ms: float = 150_000.0,
    r_scale_m: float = 5.0e19,
) -> np.ndarray:
    """
    Toy exponential disk: v_bar(r) ~ v0 * exp(-r/(2 R_d))  (just a placeholder)
    Returns v_bar^2 in (m/s)^2.
    """
    r = np.asarray(r_m, dtype=float)
    v = v0_ms * np.exp(-r / (2.0 * r_scale_m))
    return v * v
