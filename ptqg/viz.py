"""
viz.py

C9 helper: unified plotting style and a couple of quick plotting helpers.
"""

from __future__ import annotations

from typing import Optional, Sequence
import numpy as np
import matplotlib.pyplot as plt


def set_style():
    plt.rcParams.update({
        "figure.figsize": (6.0, 4.0),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 12,
    })


def shaded_curve(
    x: np.ndarray,
    y: np.ndarray,
    ylo: Optional[np.ndarray] = None,
    yhi: Optional[np.ndarray] = None,
    label: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
):
    if ax is None:
        ax = plt.gca()
    ax.plot(x, y, label=label)
    if ylo is not None and yhi is not None:
        ax.fill_between(x, ylo, yhi, alpha=0.2)
    return ax


def posterior_hist(samples: np.ndarray, bins: int = 50, label: Optional[str] = None, ax: Optional[plt.Axes] = None):
    if ax is None:
        ax = plt.gca()
    ax.hist(samples.ravel(), bins=bins, density=True, alpha=0.6, label=label)
    return ax
