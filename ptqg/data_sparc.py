"""
data_sparc.py

C4/C5: Utilities to load SPARC-like data or generate a synthetic toy sample.

We define a Galaxy data container with typical fields expected by the pipeline.
For users without the real SPARC tables at hand, use `toy_sample()` to generate
a small bundle of galaxies sufficient to exercise the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np


@dataclass
class Galaxy:
    name: str
    r_m: np.ndarray               # radii [m]
    v_obs_kms: np.ndarray         # observed rotation speed [km/s]
    v_err_kms: np.ndarray         # measurement error [km/s]
    v_disk_ms: np.ndarray         # Newtonian unit-M/L disk contrib [m/s]
    v_bulge_ms: np.ndarray        # Newtonian unit-M/L bulge contrib [m/s]
    v_gas_ms: np.ndarray          # gas contrib (already massed) [m/s]
    D_m: float                    # distance [m]
    inc_deg: float                # inclination [deg]
    sigma_D_m: float              # distance uncertainty [m]
    sigma_inc_deg: float          # inclination uncertainty [deg]


def load_sparc_csv(
    path: str,
    name: Optional[str] = None,
) -> List[Galaxy]:
    """
    Minimal CSV reader for a single galaxy or multiple galaxies.
    Expects columns (example): r_kpc, v_kms, e_v_kms, vdisk_ms, vbulge_ms, vgas_ms, D_Mpc, inc_deg, eD_Mpc, einc_deg, name

    NOTE: This is a minimal parser; depending on your SPARC dump you may need to
    adapt column names.
    """
    import pandas as pd  # optional dependency

    df = pd.read_csv(path)
    galaxies: List[Galaxy] = []
    if name is not None:
        df = df[df["name"] == name]

    for gname, sub in df.groupby("name"):
        r_m = (sub["r_kpc"].to_numpy(float)) * 3.085677581e19
        v_obs_kms = sub["v_kms"].to_numpy(float)
        v_err_kms = sub["e_v_kms"].to_numpy(float)
        v_disk_ms = sub["vdisk_ms"].to_numpy(float)
        v_bulge_ms = sub["vbulge_ms"].to_numpy(float)
        v_gas_ms = sub["vgas_ms"].to_numpy(float)
        D_m = float(sub["D_Mpc"].iloc[0]) * 3.085677581e22
        inc_deg = float(sub["inc_deg"].iloc[0])
        sigma_D_m = float(sub["eD_Mpc"].iloc[0]) * 3.085677581e22
        sigma_inc_deg = float(sub["einc_deg"].iloc[0])

        galaxies.append(
            Galaxy(
                name=gname,
                r_m=r_m,
                v_obs_kms=v_obs_kms,
                v_err_kms=v_err_kms,
                v_disk_ms=v_disk_ms,
                v_bulge_ms=v_bulge_ms,
                v_gas_ms=v_gas_ms,
                D_m=D_m,
                inc_deg=inc_deg,
                sigma_D_m=sigma_D_m,
                sigma_inc_deg=sigma_inc_deg,
            )
        )
    return galaxies


def toy_sample(n_gal: int = 10, seed: int = 42) -> List[Galaxy]:
    """
    Create a synthetic mini-sample of galaxies suitable for quick pipeline demos.
    """
    rng = np.random.default_rng(seed)
    galaxies: List[Galaxy] = []

    for i in range(n_gal):
        name = f"Toy-{i:02d}"
        # radii in kpc→m, 20 points
        r_kpc = np.linspace(1.0, 20.0, 20)
        r_m = r_kpc * 3.085677581e19

        # toy baryons: declining disk component (m/s)
        v_disk_ms = 180_000.0 * np.exp(-r_m / (6.0e19))
        v_bulge_ms = 0.0 * v_disk_ms
        v_gas_ms = 50_000.0 * np.exp(-r_m / (8.0e19))

        # generate an "obs" speed using an a0 ~ 1.5 c H0
        H0 = 70_000.0 / 3.085677581e22
        c = 299_792_458.0
        eps_true = 1.5
        a0 = eps_true * c * H0
        v_model_ms = np.sqrt(v_disk_ms**2 + v_gas_ms**2 + a0 * r_m)
        v_obs_kms = v_model_ms / 1000.0 + rng.normal(0.0, 5.0, size=r_m.size)
        v_err_kms = 5.0 + 0 * v_obs_kms

        D_m = float(rng.normal(10.0, 1.0) * 3.085677581e22)  # 10±1 Mpc
        inc_deg = float(rng.uniform(35.0, 80.0))
        sigma_D_m = 0.2 * D_m
        sigma_inc_deg = 5.0

        galaxies.append(
            Galaxy(
                name=name,
                r_m=r_m,
                v_obs_kms=v_obs_kms,
                v_err_kms=v_err_kms,
                v_disk_ms=v_disk_ms,
                v_bulge_ms=v_bulge_ms,
                v_gas_ms=v_gas_ms,
                D_m=D_m,
                inc_deg=inc_deg,
                sigma_D_m=sigma_D_m,
                sigma_inc_deg=sigma_inc_deg,
            )
        )
    return galaxies
