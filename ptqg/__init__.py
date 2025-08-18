"""
ptqg: PT-symmetric Quaternionic Geometry (verification toolkit)

This package provides a minimal-yet-functional codebase to verify the key claims:
- C1: PT-scalar projector, quaternion algebra over C_u, block metric & inverses
- C2: Omega_Lambda(eps) mapping (analytic + numeric check scaffolding)
- C3: Observable geodesic Lagrangian → linear RC law
- C4–C5: SPARC pipeline hooks (data I/O, covariance, likelihoods, simple MCMC)
- C6: Weak-lensing slope prediction scaffolding
- C7: Linear perturbation (tensor speed bound) to O(alpha^2) structure
- C8: CLASS/CAMB param export (background-level mapping)
- C9: Utilities for figure styling and model comparison

Author: ptqg toolkit generator
"""

from .math_core import QuU, pi_PT, metric_blocks, inverse_blocks, c_light, H0_default
from .geodesics import circular_velocity, a0_from_eps
from .cosmology import omega_lambda_from_eps, omega_lambda_numeric
from .model_compare import aic_bic

__all__ = [
    "QuU",
    "pi_PT",
    "metric_blocks",
    "inverse_blocks",
    "c_light",
    "H0_default",
    "circular_velocity",
    "a0_from_eps",
    "omega_lambda_from_eps",
    "omega_lambda_numeric",
    "aic_bic",
]

from .config import load_manifest, H0_si, annotate_string
