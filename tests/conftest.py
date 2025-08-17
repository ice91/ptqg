# -*- coding: utf-8 -*-
import os
import numpy as np
import pytest

# 全域常數與公差
C_LIGHT = 299_792_458.0  # m/s
H0_SI = 2.2683e-18       # 70 km/s/Mpc in s^-1
ATOL = 1e-12
RTOL = 1e-10

@pytest.fixture(scope="session", autouse=True)
def _set_seed():
    np.random.seed(20250817)

@pytest.fixture(scope="session")
def constants():
    return dict(c=C_LIGHT, H0=H0_SI, atol=ATOL, rtol=RTOL)

@pytest.fixture
def rng():
    return np.random.default_rng(20250817)

def pytest_addoption(parser):
    parser.addoption(
        "--heavy", action="store_true", default=False,
        help="run heavier/slow tests (e.g., many random draws, denser grids)"
    )

@pytest.fixture
def heavy(request):
    return request.config.getoption("--heavy")
