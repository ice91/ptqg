# -*- coding: utf-8 -*-
import numpy as np
import pytest

ds = pytest.importorskip("ptqg.data_sparc", reason="ptqg.data_sparc not available")
cv = pytest.importorskip("ptqg.covariance", reason="ptqg.covariance not available")
from ptqg.data_sparc import toy_sample
from ptqg.covariance import build_covariance

def is_spd(M):
    try:
        np.linalg.cholesky(M)
        return True
    except np.linalg.LinAlgError:
        return False

def test_covariance_spd_and_scaling():
    gals = toy_sample(n_gal=1, seed=999)
    g = gals[0]

    C1 = build_covariance(g, sigma_sys=3.0, include_distance=True, include_incl=True)
    C2 = build_covariance(g, sigma_sys=10.0, include_distance=True, include_incl=True)

    assert C1.shape[0] == C1.shape[1] == g.r_m.size
    assert is_spd(C1)
    assert is_spd(C2)

    # 增大速度地板，對角線應上升
    d1 = np.diag(C1).mean()
    d2 = np.diag(C2).mean()
    assert d2 > d1
