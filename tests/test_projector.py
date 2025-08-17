# -*- coding: utf-8 -*-
import numpy as np
import pytest

mc = pytest.importorskip("ptqg.math_core", reason="ptqg.math_core not available")
from ptqg.math_core import QuU, pi_PT

def test_pi_PT_returns_real_for_QuU(rng):
    for _ in range(200):
        a = rng.normal()
        b = rng.normal()
        q = QuU(a, b)
        val = pi_PT(q)
        assert isinstance(val, float)
        assert np.isfinite(val)
        assert abs(val - a) < 1e-12  # 投影後應取實部 a

def test_multiplicative_conjugation_scalar_is_real(rng):
    # q * q^* = a^2 + b^2 (實數)；pi_PT 應回傳相同
    for _ in range(100):
        a = rng.normal()
        b = rng.normal()
        q = QuU(a, b)
        s = q * q.conj()
        val = pi_PT(s)
        assert abs(val - (a*a + b*b)) < 1e-12

def test_inverse_of_one_plus_alpha_u(rng):
    # (1 + α u)^{-1} = (1 - α u) / (1+α^2)
    for _ in range(100):
        alpha = rng.normal()
        denom = 1.0 + alpha*alpha
        q = QuU(1.0, alpha)
        inv_expected = QuU(1.0, -alpha) / denom
        prod = q * inv_expected
        # 應當等於 1（實部=1，虛部=0）
        assert abs(pi_PT(prod) - 1.0) < 1e-12
        assert abs((prod - QuU(1.0, 0.0)).b) < 1e-12  # imaginary part ≈ 0
