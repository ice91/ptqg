"""
class_patch/params.py

C8: Export a CLASS-compatible .ini snippet for background-only replacement
via Omega_Lambda(eps) = eps^2 / (1 + eps^2).

We keep the file minimal; users can merge it with their main CLASS configs.
"""

from __future__ import annotations

from typing import Optional
from ..cosmology import omega_lambda_from_eps


def class_ini_from_eps(
    eps: float,
    h: float = 0.7,
    Omega_b: float = 0.048,
    Omega_cdm: float = 0.26,
    k_eq: Optional[float] = None,
) -> str:
    """
    Return a string with CLASS parameters reflecting Omega_Lambda(eps).
    """
    Omega_L = omega_lambda_from_eps(eps)
    # Renormalize Omega_cdm if necessary to keep Omega_tot ~ 1
    Omega_m = Omega_b + Omega_cdm
    if (Omega_m + Omega_L) != 1.0:
        # simple rescale of CDM to keep flatness
        Omega_cdm = max(0.0, 1.0 - Omega_b - Omega_L)

    lines = [
        "# CLASS ini auto-generated from ptqg",
        f"h = {h}",
        f"Omega_b = {Omega_b}",
        f"Omega_cdm = {Omega_cdm}",
        f"Omega_Lambda = {Omega_L}",
        "output = tCl,lCl,pCl,mPk",
        "lensing = yes",
        "non linear = halofit",
    ]
    if k_eq is not None:
        lines.append(f"k_eq = {k_eq}")
    return "\n".join(lines) + "\n"
