"""
mcmc.py

C4: Simple wrapper for sampling epsilon (global) and per-galaxy upsilon_*'s.

- If 'emcee' is available, use it.
- Otherwise, fall back to a lightweight random-search / grid sampler returning
  approximate posteriors (sufficient for smoke tests and demo notebooks).

The log-prob combines chosen likelihood (Gaussian or Student-t) with Gaussian
priors on upsilon_* and either a flat or Gaussian prior on epsilon.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Callable
import numpy as np

from .data_sparc import Galaxy
from .likelihood import loglike_gaussian, loglike_student_t


def _log_prior(
    eps: float,
    upsilons: Dict[str, float],
    prior_eps: Callable[[float], float],
    prior_ups: Callable[[float], float],
) -> float:
    lp = float(prior_eps(eps))
    for _, u in upsilons.items():
        lp += float(prior_ups(float(u)))
    return lp


def gaussian(mean: float, sigma: float) -> Callable[[float], float]:
    inv2 = 0.5 / (sigma * sigma)
    norm = -0.5 * np.log(2.0 * np.pi * sigma * sigma)
    return lambda x: float(norm - (x - mean) ** 2 * inv2)


def uniform(a: float, b: float) -> Callable[[float], float]:
    def lp(x: float) -> float:
        return 0.0 if (a <= x <= b) else -np.inf
    return lp


def sample(
    galaxies: List[Galaxy],
    H0: float,
    like: str = "gaussian",
    prior_eps: Optional[Callable[[float], float]] = None,
    prior_ups: Optional[Callable[[float], float]] = None,
    n_walkers: int = 24,
    n_steps: int = 2000,
    seed: int = 123,
) -> Dict[str, np.ndarray]:
    """
    Sample epsilon and per-galaxy upsilon_* using emcee if available,
    else a fallback random sampler.

    Returns dict with keys: "eps_chain", "ups_chains" (per galaxy), "logprob"
    """
    if prior_eps is None:
        prior_eps = uniform(0.0, 4.0)
    if prior_ups is None:
        prior_ups = gaussian(0.5, 0.1)

    rng = np.random.default_rng(seed)

    def logprob(theta: np.ndarray) -> float:
        eps = float(theta[0])
        ups = {galaxies[i].name: float(theta[1 + i]) for i in range(len(galaxies))}
        lp = _log_prior(eps, ups, prior_eps, prior_ups)
        if not np.isfinite(lp):
            return -np.inf
        if like == "gaussian":
            ll, _ = loglike_gaussian(galaxies, eps, H0, ups)
        else:
            ll, _ = loglike_student_t(galaxies, eps, H0, ups)
        return lp + ll

    dim = 1 + len(galaxies)
    p0 = []
    for _ in range(n_walkers):
        eps0 = rng.uniform(0.5, 2.5)
        ups0 = [rng.normal(0.5, 0.1) for _ in galaxies]
        p0.append(np.array([eps0, *ups0], dtype=float))
    p0 = np.array(p0)

    try:
        import emcee  # type: ignore
        sampler = emcee.EnsembleSampler(n_walkers, dim, logprob)
        sampler.run_mcmc(p0, n_steps, progress=False)
        chain = sampler.get_chain(flat=False)
        lnp = sampler.get_log_prob(flat=False)
        return {
            "eps_chain": chain[:, :, 0],
            "ups_chains": {galaxies[i].name: chain[:, :, 1 + i] for i in range(len(galaxies))},
            "logprob": lnp,
        }
    except Exception:
        # Fallback: random-walk Metropolis (very simple)
        chain = np.zeros((n_steps, n_walkers, dim))
        logp = np.full((n_steps, n_walkers), -np.inf)
        chain[0] = p0
        for w in range(n_walkers):
            logp[0, w] = logprob(chain[0, w])

        step_scale = np.concatenate(([0.05], np.full(len(galaxies), 0.02)))
        for s in range(1, n_steps):
            for w in range(n_walkers):
                prop = chain[s - 1, w] + rng.normal(0, step_scale, size=dim)
                lp_prop = logprob(prop)
                if np.log(rng.uniform()) < (lp_prop - logp[s - 1, w]):
                    chain[s, w] = prop
                    logp[s, w] = lp_prop
                else:
                    chain[s, w] = chain[s - 1, w]
                    logp[s, w] = logp[s - 1, w]

        return {
            "eps_chain": chain[:, :, 0],
            "ups_chains": {galaxies[i].name: chain[:, :, 1 + i] for i in range(len(galaxies))},
            "logprob": logp,
        }
