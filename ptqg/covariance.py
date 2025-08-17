"""
covariance.py

C5: Build per-galaxy covariance including measurement noise + propagated distance
and inclination uncertainties, plus a small velocity floor.

Implements:
- build_covariance(gal, sigma_sys_kms=3.)
- sensitivity_jacobians(...): J^(D), J^(i) approximate derivatives
"""

from __future__ import annotations

from typing import Tuple
import numpy as np
from .geodesics import a0_from_eps
from .math_core import c_light


def sensitivity_jacobians(
    r_m: np.ndarray,
    v_model_kms: np.ndarray,
    theta_rad: np.ndarray,
    eps: float,
    H0: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute sensitivities wrt distance D and inclination i (approximate).

    Using r = D * theta, so dr/dD = theta. For the linear term v_lin^2 = a0 r,
    d v_lin / dD = (1/2) * (a0 * theta) / v_model.

    For inclination, when comparing deprojected velocities one may back-propagate
    uncertainty; here we model the standard dependence v_obs = v_true * sin i,
    so dv/di ≈ v_model * cot(i). (Radians)
    """
    r_m = np.asarray(r_m)
    v_model = np.asarray(v_model_kms) * 1000.0
    theta = np.asarray(theta_rad)

    a0 = a0_from_eps(eps, H0)
    dv_dD = 0.5 * (a0 * theta) / np.maximum(v_model, 1e-6)

    # inclination derivative (radians)
    dv_di = v_model * (1.0 / np.tan(np.clip(theta*0 + np.pi/4, 1e-3, np.pi/2-1e-3)))  # placeholder cot(i)
    # NOTE: In a full treatment you'd use the actual galaxy's inclination here.
    # We keep the interface simple; the pipeline can pass a per-galaxy i if desired.

    return dv_dD / 1000.0, dv_di / 1000.0  # return in km/s per unit of D or i(rad)


def build_covariance(
    v_err_kms: np.ndarray,
    r_m: np.ndarray,
    theta_rad: np.ndarray,
    v_model_kms: np.ndarray,
    eps: float,
    H0: float,
    sigma_D_m: float,
    sigma_inc_deg: float,
    sigma_sys_kms: float = 3.0,
) -> np.ndarray:
    """
    C_g = diag( sigma_meas^2 ) + sigma_D^2 J^D J^{D,T} + sigma_i^2 J^i J^{i,T} + sigma_sys^2 I
    """
    n = r_m.size
    C = np.diag(v_err_kms**2 + sigma_sys_kms**2)

    Jd, Ji = sensitivity_jacobians(r_m, v_model_kms, theta_rad, eps, H0)
    C += (sigma_D_m**2) * np.outer(Jd, Jd)

    sigma_i_rad = np.deg2rad(sigma_inc_deg)
    C += (sigma_i_rad**2) * np.outer(Ji, Ji)
    return C
