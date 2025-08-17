# -*- coding: utf-8 -*-
import numpy as np
import pytest

# geodesics 模組若缺，退而以 likelihood 的 model_velocity_kms 做核對（仍驗證線性係數）
has_geo = True
try:
    from ptqg import geodesics as _geo
except Exception:
    has_geo = False

ds = pytest.importorskip("ptqg.data_sparc", reason="ptqg.data_sparc not available")
lk = pytest.importorskip("ptqg.likelihood", reason="ptqg.likelihood not available")
from ptqg.data_sparc import toy_sample
from ptqg.likelihood import model_velocity_kms

def _fit_slope_r(y, r):
    A = np.vstack([r, np.ones_like(r)]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, b

def test_outer_linear_term_slope(constants):
    c, H0 = constants["c"], constants["H0"]
    gals = toy_sample(n_gal=1, seed=777)
    gal = gals[0]
    eps_true = 1.4
    ups = 0.5

    if has_geo and hasattr(_geo, "circular_velocity"):
        v = _geo.circular_velocity(a=1.0, eps=eps_true, H0=H0, r=gal.r_m,
                                   baryon_model=lambda r: (gal.v_disk_ms*ups + gal.v_bulge_ms*ups + gal.v_gas_ms))
        vmod = v / 1000.0  # 假設 geodesics 回 SI，轉 km/s
    else:
        # 後備：用 likelihood 的模型（已實作 v^2 = v_bar^2 + eps c H0 r）
        vmod = model_velocity_kms(gal, eps_true, H0, ups)

    vb2 = (gal.v_disk_ms*ups + gal.v_bulge_ms*ups + gal.v_gas_ms)**2
    vm2 = (vmod*1000.0)**2
    y = vm2 - vb2  # 應為 (eps c H0) r
    m, _ = _fit_slope_r(y, gal.r_m)
    target = eps_true * c * H0
    rel = abs(m - target) / target
    assert rel < 0.05  # 5% 內（玩具資料，含數值噪音）
