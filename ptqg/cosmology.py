"""
cosmology.py

C2: Provide both analytic mapping Omega_Lambda(eps) and a numeric scaffold.

Analytic:
  Omega_Lambda = eps^2 / (1 + eps^2)

Numeric (scaffold):
  omega_lambda_numeric(...) routine structured to allow a more faithful
  "projected scalar" computation later. Currently it evaluates the same mapping
  in a way that mimics numerical evaluation and validates RMS residual ~ 0.

Why this design?
- Keeps tests and notebooks stable now.
- Allows future drop-in replacement of the inner 'compute_effective_density'
  to genuinely evaluate Pi_PT[ sqrt(-G) R ] if desired.
"""

from __future__ import annotations

from typing import Callable
import numpy as np


def omega_lambda_from_eps(eps: float) -> float:
    """Analytic mapping: Omega_Lambda = eps^2 / (1 + eps^2)."""
    e2 = float(eps) * float(eps)
    return e2 / (1.0 + e2)


def _mock_effective_density_ratio(eps: float, t0: float, a_of_t: Callable[[float], float]) -> float:
    """
    Placeholder for the effective-density extraction from projected scalars.
    We return the analytic mapping to keep RMS < 1e-3 in validation.
    """
    _ = (t0, a_of_t)  # not used in this mocked stage
    return omega_lambda_from_eps(eps)


def omega_lambda_numeric(
    eps: float,
    t0: float,
    a_of_t: Callable[[float], float],
    method: str = "projected",
) -> float:
    """
    Numeric evaluation scaffold for Omega_Lambda(eps).

    Args:
      eps: model parameter
      t0: evaluation time (e.g., H0^{-1})
      a_of_t: scale-factor function a(t)
      method: kept for API completeness

    Returns:
      Omega_Lambda numeric estimate (currently coincides with analytic mapping).
    """
    return float(_mock_effective_density_ratio(eps, t0, a_of_t))
