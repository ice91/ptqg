# -*- coding: utf-8 -*-
import numpy as np
import pytest

ds = pytest.importorskip("ptqg.data_sparc", reason="ptqg.data_sparc not available")
lk = pytest.importorskip("ptqg.likelihood", reason="ptqg.likelihood not available")
from ptqg.data_sparc import toy_sample
from ptqg.likelihood import model_velocity_kms, loglike_gaussian, loglike_student_t

def test_gaussian_and_student_t_finite(constants):
    H0 = constants["H0"]
    gals = toy_sample(n_gal=1, seed=1234)
    gal = gals[0]
    eps = 1.5
    ups_map = {gal.name: 0.5}

    lg, _ = loglike_gaussian([gal], eps, H0, ups_map)
    lt = loglike_student_t([gal], eps, H0, ups_map, nu=6)
    assert np.isfinite(lg)
    assert np.isfinite(lt)

def test_student_t_robust_to_outlier(constants):
    # 人為製造一個極端離群點，Student-t 的 logL 下降應小於 Gaussian
    H0 = constants["H0"]
    gals = toy_sample(n_gal=1, seed=4321)
    gal = gals[0]
    eps = 1.5
    ups = 0.5
    v_model = model_velocity_kms(gal, eps, H0, ups)

    # 取末端一點升高 8σ，形成 outlier
    k = -1
    gal_out = gal._replace(
        v_obs_kms = gal.v_obs_kms.copy(),
        v_err_kms = gal.v_err_kms.copy()
    )
    gal_out.v_obs_kms[k] = v_model[k] + 8.0 * gal.v_err_kms[k]

    lg0, _ = loglike_gaussian([gal], eps, H0, {gal.name: ups})
    lt0 = loglike_student_t([gal], eps, H0, {gal.name: ups}, nu=6)

    lg1, _ = loglike_gaussian([gal_out], eps, H0, {gal_out.name: ups})
    lt1 = loglike_student_t([gal_out], eps, H0, {gal_out.name: ups}, nu=6)

    dG = lg1 - lg0
    dT = lt1 - lt0
    # 加入同一離群點，Gaussian 應更受影響（logL 降得更多 => 差值更負）
    assert dG < dT
