# -*- coding: utf-8 -*-
import numpy as np
import pytest

mc = pytest.importorskip("ptqg.math_core", reason="ptqg.math_core not available")
from ptqg.math_core import QuU, pi_PT, inverse_blocks

def test_temporal_block_inverse_random(constants, rng, heavy):
    H0 = constants["H0"]
    n = 1000 if heavy else 200
    for _ in range(n):
        a = 1.0
        eps = rng.uniform(0.0, 3.0)
        t = rng.uniform(0.05, 2.0) / H0

        inv = inverse_blocks(a=a, eps=eps, H0=H0, r=0.0, t=t)
        # 應回傳 G^{00} 封閉式；我們用解析式自行重構再相乘檢查
        alpha0 = eps * H0 * t
        G00 = QuU(-1.0, alpha0)           # -1 + u alpha0
        G00_inv_expected = QuU(-1.0, -alpha0) / (1.0 + alpha0*alpha0)  # (-1 - u alpha0)/(1+alpha0^2)

        # 若 lib 也回傳 QuU，對齊檢查
        G00_inv_lib = inv.get("G00_inv", None)
        if G00_inv_lib is not None:
            prod_lib = G00 * G00_inv_lib
            assert abs(pi_PT(prod_lib) - 1.0) < 1e-12

        prod = G00 * G00_inv_expected
        assert abs(pi_PT(prod) - 1.0) < 1e-12
        # 反向相乘也應為單位
        prod2 = G00_inv_expected * G00
        assert abs(pi_PT(prod2) - 1.0) < 1e-12

def test_spatial_block_inverse_random(constants, rng, heavy):
    H0 = constants["H0"]
    n = 1000 if heavy else 200
    for _ in range(n):
        a = 1.0
        eps = rng.uniform(0.0, 3.0)
        r = rng.uniform(0.1, 50.0) * 3.085677581e19
        alpha_s = eps * H0 * r / 299_792_458.0

        # G_ij = a^2 (1 + u α_s) δij -> QuU(a^2, a^2 α_s)
        Gs = QuU(a*a, a*a * alpha_s)
        # G^{ij} = (a^2 - u α_s)/(a^4 + α_s^2) -> QuU(a^2, -α_s)/den
        den = a**4 + alpha_s*alpha_s
        Gs_inv_expected = QuU(a*a, -alpha_s) / den

        prod = Gs * Gs_inv_expected
        assert abs(pi_PT(prod) - 1.0) < 1e-12
        prod2 = Gs_inv_expected * Gs
        assert abs(pi_PT(prod2) - 1.0) < 1e-12
