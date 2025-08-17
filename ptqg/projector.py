"""
projector.py

Implements the "observable" prescription knobs for PT projection.

We provide two conceptual orders:
- VARIATION_THEN_PROJECT: perform variation in C_u, construct scalars, then Pi_PT
- PROJECT_THEN_VARIATION: project metric to real scalars first, then vary
(In this toolkit we keep both options for experimentation; the paper recommends
projecting observables at the scalar level. For C2/C3 demos we will use
VARIATION_THEN_PROJECT.)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Any, Dict

import numpy as np
from .math_core import QuU, pi_PT


class ProjectionOrder(Enum):
    VARIATION_THEN_PROJECT = auto()
    PROJECT_THEN_VARIATION = auto()


@dataclass
class ProjectorSettings:
    """
    Settings container controlling the projection order.
    """
    order: ProjectionOrder = ProjectionOrder.VARIATION_THEN_PROJECT


_DEFAULT_SETTINGS = ProjectorSettings()


def set_projector_settings(settings: ProjectorSettings) -> None:
    global _DEFAULT_SETTINGS
    _DEFAULT_SETTINGS = settings


def project_scalar(x: Any) -> np.ndarray:
    """
    Project any scalar-like quantity that may live in C_u (QuU) to R.
    For arrays/dicts, apply elementwise when meaningful.
    """
    if isinstance(x, QuU):
        return pi_PT(x)
    if isinstance(x, dict):
        return {k: project_scalar(v) for k, v in x.items()}
    return np.asarray(x)
