"""
likelihood.py

C4/C5: Gaussian and Student-t likelihoods for stacked galaxy data with a
shared epsilon and per-galaxy mass-to-light (single scalar) parameter.

We implement a minimal interface:
- galaxy_model_vbar2(): combine v_disk, v_bulge, v_gas with Upsilon_* (scalar)
- loglike_gaussian(...) and loglike_student_t(...)
"""

from __future__ import annotations

from typing import Dict, Tuple, List
import numpy as np
from scipy.stats import t as student_t

from .data_sparc import Galaxy
from .geodesics import circular_velocity, a0_from_eps
from .covariance import build_covariance


def galaxy_model_vbar2(
    gal: Galaxy,
    upsilon_star: float,
) -> np.ndarray:
    """
    v_bar^2 = (Upsilon_* v_* )^2 + (v_gas)^2, where v_*^2 = v_disk^2 + v_bulge^2
    (All velocities inside in m/s)
    Return v_bar^2 in (m/s)^2
    """
    v_star_ms = np.sqrt(gal.v_disk_ms**2 + gal.v_bulge_ms**2)
    v_bar_ms = np.sqrt((upsilon_star * v_star_ms) ** 2 + gal.v_gas_ms**2)
    return v_bar_ms**2


def _theta_from_r_D(r_m: np.ndarray, D_m: float) -> np.ndarray:
    return np.asarray(r_m) / float(D_m)


def model_velocity_kms(
    gal: Galaxy,
    eps: float,
    H0: float,
    upsilon_star: float,
) -> np.ndarray:
    """
    Returns model velocity in km/s for one galaxy.
    """
    vbar2 = galaxy_model_vbar2(gal, upsilon_star)
    v_model = circular_velocity(1.0, eps, H0, gal.r_m, vbar2, to_km_s=True)
    return v_model


def loglike_gaussian(
    galaxies: List[Galaxy],
    eps: float,
    H0: float,
    upsilons: Dict[str, float],
    sigma_sys_kms: float = 3.0,
) -> Tuple[float, Dict[str, float]]:
    """
    Sum of -0.5 * [ (v_obs - v_mod)^T C^{-1} (v_obs - v_mod) + ln det C + n ln 2pi ] over all galaxies.
    Returns (logL, per_gal_chi2_nu).
    """
    logL = 0.0
    per_chi2_nu: Dict[str, float] = {}
    for gal in galaxies:
        U = float(upsilons.get(gal.name, 0.5))
        v_mod = model_velocity_kms(gal, eps, H0, U)
        theta = _theta_from_r_D(gal.r_m, gal.D_m)
        C = build_covariance(
            gal.v_err_kms, gal.r_m, theta, v_mod, eps, H0, gal.sigma_D_m, gal.sigma_inc_deg, sigma_sys_kms
        )
        resid = gal.v_obs_kms - v_mod
        try:
            Ci = np.linalg.inv(C)
            chi2 = float(resid.T @ Ci @ resid)
            logdet = float(np.linalg.slogdet(C)[1])
        except np.linalg.LinAlgError:
            # regularize
            C += np.eye(C.shape[0]) * 1e-6
            Ci = np.linalg.inv(C)
            chi2 = float(resid.T @ Ci @ resid)
            logdet = float(np.linalg.slogdet(C)[1])

        n = gal.r_m.size
        logL -= 0.5 * (chi2 + logdet + n * np.log(2.0 * np.pi))
        # reduced chi2 using dof ~ n - 1 (one parameter U per galaxy)
        nu = max(n - 1, 1)
        per_chi2_nu[gal.name] = chi2 / nu
    return logL, per_chi2_nu


def loglike_student_t(
    galaxies: List[Galaxy],
    eps: float,
    H0: float,
    upsilons: Dict[str, float],
    nu: float = 6.0,
    sigma_sys_kms: float = 3.0,
) -> Tuple[float, Dict[str, float]]:
    """
    Robust Student-t log-likelihood with scale set by the diagonal of C (for simplicity).
    """
    logL = 0.0
    per_scale: Dict[str, float] = {}
    for gal in galaxies:
        U = float(upsilons.get(gal.name, 0.5))
        v_mod = model_velocity_kms(gal, eps, H0, U)
        theta = _theta_from_r_D(gal.r_m, gal.D_m)
        C = build_covariance(
            gal.v_err_kms, gal.r_m, theta, v_mod, eps, H0, gal.sigma_D_m, gal.sigma_inc_deg, sigma_sys_kms
        )
        resid = gal.v_obs_kms - v_mod
        scale = np.sqrt(np.clip(np.diag(C), 1e-12, None))
        per_scale[gal.name] = float(np.median(scale))

        # Student-t logpdf per-point
        y = resid / scale
        logpdf = student_t.logpdf(y, df=nu) - np.log(scale)
        logL += float(np.sum(logpdf))
    return logL, per_scale

