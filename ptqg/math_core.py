"""
math_core.py

C1: Quaternion algebra over C_u, PT-scalar projector, metric blocks G_{00}, G_{ij}
    and their exact inverses as per the paper's Eq. (2.3) style identities.

We model the commutative subalgebra C_u via a lightweight class QuU(a + b u),
with u^2 = -1 and (a, b) real (can be numpy arrays).

Also provides:
- pi_PT(x): PT-scalar projector → returns the real part for QuU
- metric_blocks(a, eps, H0, r, t): constructs G_{00} and G_{ij} blocks
- inverse_blocks(...): constructs the corresponding inverse blocks
- sanity helpers to check block products equal identity within tolerance
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Any
import numpy as np

Number = Union[float, int, np.ndarray]

# Physical constants (SI)
c_light: float = 299_792_458.0  # m/s
# Default H0 used across the code (change at call sites as needed)
# 70 km/s/Mpc -> convert to s^-1
H0_default: float = 70_000.0 / 3.085677581e22  # s^-1


@dataclass
class QuU:
    """
    Quaternion element restricted to C_u: x = a + b u, with u^2 = -1.
    Here (a, b) are real scalars or numpy arrays (broadcastable).
    """
    a: Number
    b: Number

    def __post_init__(self) -> None:
        self.a = np.asarray(self.a)
        self.b = np.asarray(self.b)

    def conj(self) -> "QuU":
        """Quaternionic conjugation over C_u: (a + b u)^* = a - b u."""
        return QuU(self.a, -self.b)

    # algebra: (a+bu) + (c+du) = (a+c) + (b+d) u
    def __add__(self, other: Union["QuU", Number]) -> "QuU":
        if isinstance(other, QuU):
            return QuU(self.a + other.a, self.b + other.b)
        return QuU(self.a + other, self.b)

    __radd__ = __add__

    def __sub__(self, other: Union["QuU", Number]) -> "QuU":
        if isinstance(other, QuU):
            return QuU(self.a - other.a, self.b - other.b)
        return QuU(self.a - other, self.b)

    def __rsub__(self, other: Union["QuU", Number]) -> "QuU":
        if isinstance(other, QuU):
            return QuU(other.a - self.a, other.b - self.b)
        return QuU(other - self.a, -self.b)

    # multiplication: (a+bu)(c+du) = (ac - bd) + (ad+bc) u
    def __mul__(self, other: Union["QuU", Number]) -> "QuU":
        if isinstance(other, QuU):
            a = self.a * other.a - self.b * other.b
            b = self.a * other.b + self.b * other.a
            return QuU(a, b)
        else:
            return QuU(self.a * other, self.b * other)

    __rmul__ = __mul__

    # inverse: (a + b u)^{-1} = (a - b u) / (a^2 + b^2)
    def inv(self) -> "QuU":
        denom = self.a * self.a + self.b * self.b
        return QuU(self.a / denom, -self.b / denom)

    def __truediv__(self, other: Union["QuU", Number]) -> "QuU":
        if isinstance(other, QuU):
            return self * other.inv()
        else:
            return QuU(self.a / other, self.b / other)

    def real(self) -> np.ndarray:
        return self.a

    def imagU(self) -> np.ndarray:
        return self.b

    def to_float_if_real(self) -> Union[float, np.ndarray, "QuU"]:
        if np.allclose(self.b, 0.0):
            return self.a
        return self

    def __repr__(self) -> str:
        return f"QuU(a={self.a!r}, b={self.b!r})"


def pi_PT(x: Union[QuU, Number]) -> Union[float, np.ndarray]:
    """
    PT-scalar projector on C_u: Pi_PT[a + b u] = a (real part).
    If x is already real/ndarray, returns x.
    """
    if isinstance(x, QuU):
        return x.real()
    return np.asarray(x)


def metric_blocks(
    a_scale: float,
    eps: float,
    H0: float,
    r: Number,
    t: float,
) -> Dict[str, Any]:
    """
    Construct block-diagonal metric components in C_u:
      G_00 = -1 + u * (eps H0 t) = QuU(-1, alpha0)
      G_ij = a^2 * (1 + u * alpha_s) δ_ij = QuU(a^2, a^2 * alpha_s) δ_ij
    where alpha_s = (eps H0 / c) r.

    Returns a dict with QuU objects:
      {
        "G00": QuU,
        "Gspatial": QuU,  # the scalar factor multiplying δ_ij
        "alpha0": float,
        "alphas": ndarray
      }
    """
    alpha0 = float(eps * H0 * t)
    alphas = np.asarray(eps * H0 / c_light * np.asarray(r))
    G00 = QuU(-1.0, alpha0)
    Gsp = QuU(a_scale * a_scale, a_scale * a_scale * alphas)
    return {"G00": G00, "Gspatial": Gsp, "alpha0": alpha0, "alphas": alphas}


def inverse_blocks(a_scale: float, alpha0: float, alphas: Number) -> Dict[str, Any]:
    """
    Exact inverses (still in C_u) for the blocks:

      G^{00} = (-1 - u alpha0) / (1 + alpha0^2)

      G^{ij} = (a^2 - u alpha_s) / (a^4 + alpha_s^2) δ^{ij}
             = QuU( a^2, -alpha_s ) / (a^4 + alpha_s^2)

    Returns:
      {
        "G00_inv": QuU,
        "Gspatial_inv": QuU  # the scalar factor multiplying δ^{ij}
      }
    """
    denom0 = 1.0 + alpha0 * alpha0
    G00_inv = QuU(-1.0 / denom0, -alpha0 / denom0)

    a2 = a_scale * a_scale
    a4 = a2 * a2
    alphas = np.asarray(alphas)
    denomS = a4 + alphas * alphas
    Gsp_inv = QuU(a2 / denomS, (-alphas) / denomS)
    return {"G00_inv": G00_inv, "Gspatial_inv": Gsp_inv}


def check_block_identity(
    a_scale: float, eps: float, H0: float, r: Number, t: float, atol: float = 1e-12
) -> Tuple[float, float]:
    """
    Returns the max absolute deviation from identity for temporal and spatial blocks
    when composing G and G^{-1}.

    For temporal block:
      G00 * G00_inv = QuU(1, 0)
    For spatial block (scalar factor) similarly equals identity.

    Outputs:
      (max_dev_temporal, max_dev_spatial)
    """
    blocks = metric_blocks(a_scale, eps, H0, r, t)
    invs = inverse_blocks(a_scale, blocks["alpha0"], blocks["alphas"])
    # temporal
    I0 = blocks["G00"] * invs["G00_inv"]
    dev0 = np.max(np.abs(I0.a - 1.0)) + np.max(np.abs(I0.b - 0.0))
    # spatial (per-element scalar factor)
    Is = blocks["Gspatial"] * invs["Gspatial_inv"]
    devs = np.max(np.abs(Is.a - 1.0)) + np.max(np.abs(Is.b - 0.0))
    return float(dev0), float(devs)
