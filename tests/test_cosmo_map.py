# -*- coding: utf-8 -*-
import numpy as np
import pytest

cm = pytest.importorskip("ptqg.cosmology", reason="ptqg.cosmology not available")
from ptqg.cosmology import omega_lambda_from_eps, omega_lambda_numeric

def test_omega_lambda_numeric_matches_analytic(constants, heavy):
    H0 = constants["H0"]
    def a_of_t(t): return 1.0
    t0 = 1.0 / H0

    n = 601 if heavy else 241
    eps_grid = np.linspace(0.0, 3.0, n)
    ol_an = np.array([omega_lambda_from_eps(e) for e in eps_grid])
    ol_num = np.array([omega_lambda_numeric(e, t0=t0, a_of_t=a_of_t) for e in eps_grid])
    resid = ol_num - ol_an
    rms = float(np.sqrt(np.mean(resid**2)))
    assert rms < 1e-3
