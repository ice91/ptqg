#!/usr/bin/env bash
# Run a small shared-epsilon MCMC on SPARC (or toy) data and produce results.
# Outputs in results/: eps_chain.npy, per_gal_chi2.csv, aic_bic.csv, plots.
#
# Usage:
#   scripts/run_sparc_fit.sh [--csv data/sparc.csv] [--toy]
#                            [--H0 2.268e-18] [--likelihood gaussian|studentt]
#                            [--walkers 24] [--steps 2000] [--seed 123]
#                            [--prior planck|flat]
#
set -euo pipefail

CSV=""
USE_TOY="0"
H0="2.2683e-18"          # ~ 70 km/s/Mpc in s^-1
LIKE="gaussian"
WALKERS="24"
STEPS="2000"
SEED="123"
PRIOR="flat"            # flat or planck

print_help() {
  cat <<'EOF'
run_sparc_fit.sh - Run shared-epsilon MCMC on SPARC/toy sample.

Options:
  --csv PATH        Path to sparc.csv (from fetch_sparc.sh); overrides --toy
  --toy             Use synthetic toy sample (default if --csv not provided)
  --H0 VAL          Hubble rate in s^-1 (default ~70 km/s/Mpc)
  --likelihood L    gaussian | studentt  (default gaussian)
  --walkers N       MCMC walkers (default 24)
  --steps N         MCMC steps (default 2000)
  --seed N          RNG seed (default 123)
  --prior P         planck | flat
  -h, --help        Show help

Output:
  results/eps_chain.npy        (shape [steps, walkers])
  results/logprob.npy
  results/per_gal_chi2.csv
  results/aic_bic.csv
  results/eps_posterior.png
  results/example_rc_fit.png
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --csv)
      CSV="${2:?missing path}"
      shift 2
      ;;
    --toy)
      USE_TOY="1"
      shift
      ;;
    --H0)
      H0="${2:?missing value}"
      shift 2
      ;;
    --likelihood)
      LIKE="${2:?missing (gaussian|studentt)}"
      shift 2
      ;;
    --walkers)
      WALKERS="${2:?missing int}"
      shift 2
      ;;
    --steps)
      STEPS="${2:?missing int}"
      shift 2
      ;;
    --seed)
      SEED="${2:?missing int}"
      shift 2
      ;;
    --prior)
      PRIOR="${2:?missing (planck|flat)}"
      shift 2
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      print_help
      exit 1
      ;;
  esac
done

mkdir -p results

python3 - <<PYCODE
import os, sys, json, hashlib
import numpy as np
import matplotlib.pyplot as plt

from ptqg.data_sparc import load_sparc_csv, toy_sample, Galaxy
from ptqg.likelihood import loglike_gaussian, loglike_student_t
from ptqg.mcmc import sample, gaussian, uniform
from ptqg.model_compare import aic_bic

CSV = r"${CSV}"
USE_TOY = (${USE_TOY} == 1)
H0 = float("${H0}")
LIKE = "${LIKE}".lower().strip()
WALKERS = int("${WALKERS}")
STEPS = int("${STEPS}")
SEED = int("${SEED}")
PRIOR = "${PRIOR}".lower().strip()

# Load data
if CSV and os.path.exists(CSV):
    print(f"[run_sparc_fit] Loading CSV: {CSV}")
    galaxies = load_sparc_csv(CSV)
else:
    print("[run_sparc_fit] Using toy sample")
    galaxies = toy_sample(n_gal=10, seed=SEED)

G = len(galaxies)
Ntot = sum(g.r_m.size for g in galaxies)
print(f"[run_sparc_fit] #galaxies={G}, Ntot={Ntot}")

# Priors
if PRIOR == "planck":
    # eps ~ N(1.47, 0.05^2), Upsilon* ~ N(0.5, 0.1^2)
    prior_eps = gaussian(1.47, 0.05)
else:
    prior_eps = uniform(0.0, 4.0)
prior_ups = gaussian(0.5, 0.1)

# Sample
out = sample(
    galaxies=galaxies, H0=H0, like=LIKE,
    prior_eps=prior_eps, prior_ups=prior_ups,
    n_walkers=WALKERS, n_steps=STEPS, seed=SEED
)

eps_chain = out["eps_chain"]  # [steps, walkers]
logprob = out["logprob"]

np.save("results/eps_chain.npy", eps_chain)
np.save("results/logprob.npy", logprob)

# Find MAP sample
smax, wmax = np.unravel_index(np.nanargmax(logprob), logprob.shape)
eps_map = float(eps_chain[smax, wmax])

# Recover per-galaxy Upsilon* at MAP (from same chain index)
ups_map = {}
for i, gal in enumerate(galaxies):
    name = gal.name
    ups_ch = out["ups_chains"][name]
    ups_map[name] = float(ups_ch[smax, wmax])

# Compute log-likelihood and per-gal chi2_nu at MAP (Gaussian for model comparison)
logL_map, chi2nu = loglike_gaussian(galaxies, eps_map, H0, ups_map)
k = 1 + G
AIC, BIC = aic_bic(logL_map, k, Ntot)

# Save chi2_nu CSV
import csv as _csv
with open("results/per_gal_chi2.csv", "w", newline="") as f:
    w = _csv.writer(f)
    w.writerow(["name", "chi2_nu"])
    for n, val in chi2nu.items():
        w.writerow([n, f"{val:.6f}"])

# Save AIC/BIC
with open("results/aic_bic.csv", "w", newline="") as f:
    w = _csv.writer(f)
    w.writerow(["logL_max", "k", "Ntot", "AIC", "BIC"])
    w.writerow([f"{logL_map:.6f}", k, Ntot, f"{AIC:.6f}", f"{BIC:.6f}"])

print(f"[run_sparc_fit] MAP eps={eps_map:.4f}, logL={logL_map:.3f}, AIC={AIC:.2f}, BIC={BIC:.2f}")

# Plot posterior of eps
plt.figure(figsize=(6,4))
plt.hist(eps_chain.reshape(-1), bins=60, density=True, alpha=0.7)
plt.axvline(eps_map, color="k", lw=1, label=f"MAP={eps_map:.3f}")
plt.xlabel(r"$\varepsilon$")
plt.ylabel("Posterior density")
plt.legend()
plt.tight_layout()
plt.savefig("results/eps_posterior.png", dpi=150)
plt.close()

# Make an example RC fit plot for the first galaxy
from ptqg.likelihood import model_velocity_kms
gal0 = galaxies[0]
vmod0 = model_velocity_kms(gal0, eps_map, H0, ups_map[gal0.name])
import numpy as _np
r_kpc = gal0.r_m / 3.085677581e19
plt.figure(figsize=(6,4))
plt.errorbar(r_kpc, gal0.v_obs_kms, yerr=gal0.v_err_kms, fmt="o", ms=3, label="obs")
plt.plot(r_kpc, vmod0, lw=2, label="model")
plt.xlabel("r [kpc]")
plt.ylabel("v [km/s]")
plt.title(f"Example RC fit: {gal0.name}")
plt.legend()
plt.tight_layout()
plt.savefig("results/example_rc_fit.png", dpi=150)
plt.close()

# Write a small JSON summary for reproducibility
summary = dict(
    eps_map=eps_map,
    logL_map=logL_map,
    AIC=AIC, BIC=BIC,
    walkers=WALKERS, steps=STEPS, seed=SEED, like=LIKE, prior=PRIOR,
    H0=H0, Ntot=Ntot, n_gal=G,
)
with open("results/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# checksum of eps_chain for CI
md5 = hashlib.md5(np.ascontiguousarray(eps_chain).data).hexdigest()
with open("results/eps_chain.md5", "w") as f:
    f.write(md5 + "\n")
print(f"[run_sparc_fit] Wrote results/, eps_chain md5: {md5}")
PYCODE

echo "[run_sparc_fit] Done. See results/ directory."
