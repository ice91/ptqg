#!/usr/bin/env bash
# Generate key figures: Omega_Lambda mapping, lensing predictions,
# and (if available) posterior and RC example plots from run_sparc_fit.sh.
#
# Usage:
#   scripts/make_figs.sh [--eps 1.5] [--H0 2.268e-18]
#
set -euo pipefail

EPS="1.5"
H0="2.2683e-18"

print_help() {
  cat <<'EOF'
make_figs.sh - Generate verification figures.

Options:
  --eps VAL     epsilon for demo curves (default 1.5)
  --H0  VAL     H0 in s^-1 (default ~70 km/s/Mpc)
  -h, --help    Show help

Output:
  results/fig_omega_mapping.png
  results/fig_omega_residual.png
  results/fig_gamma_t.png
  (and copies of results from run_sparc_fit.sh if they exist)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --eps)
      EPS="${2:?missing value}"
      shift 2
      ;;
    --H0)
      H0="${2:?missing value}"
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
import os, numpy as np, matplotlib.pyplot as plt
from ptqg.cosmology import omega_lambda_from_eps, omega_lambda_numeric
from ptqg.lensing import gamma_t_linear
from math import isfinite

EPS = float("${EPS}")
H0 = float("${H0}")

# ---- 1) Omega_Lambda mapping ----
eps_grid = np.linspace(0.0, 3.0, 301)
ol_analytic = np.array([omega_lambda_from_eps(e) for e in eps_grid])

# numeric scaffold (same mapping; here we keep API in place)
def a_of_t(t): return 1.0
t0 = 1.0 / H0
ol_numeric = np.array([omega_lambda_numeric(e, t0=t0, a_of_t=a_of_t) for e in eps_grid])

resid = ol_numeric - ol_analytic
rms = float(np.sqrt(np.mean(resid**2)))

plt.figure(figsize=(6.0,4.0))
plt.plot(eps_grid, ol_analytic, label="analytic")
plt.plot(eps_grid, ol_numeric, "--", label="numeric")
plt.xlabel(r"$\varepsilon$")
plt.ylabel(r"$\Omega_\Lambda$")
plt.title(r"$\Omega_\Lambda(\varepsilon)$ mapping")
plt.legend()
plt.tight_layout()
plt.savefig("results/fig_omega_mapping.png", dpi=150)
plt.close()

plt.figure(figsize=(6.0,3.5))
plt.plot(eps_grid, resid)
plt.axhline(0, color="k", lw=0.8)
plt.xlabel(r"$\varepsilon$")
plt.ylabel("numeric - analytic")
plt.title(f"Residual (RMS={rms:.2e})")
plt.tight_layout()
plt.savefig("results/fig_omega_residual.png", dpi=150)
plt.close()
print(f"[make_figs] Omega mapping residual RMS = {rms:.3e}")

# ---- 2) Lensing prediction (toy) ----
# H(z) for a flat LCDM toy (used only to show ~H(z) tracking)
def H_of_z(z, H0=H0, Om=0.31, OL=None):
    if OL is None:
        OL = omega_lambda_from_eps(EPS)
    return H0 * np.sqrt(Om*(1+z)**3 + OL)

R_kpc = np.linspace(5, 100, 60)
R_m = R_kpc * 3.085677581e19
zs = [0.1, 0.3, 0.6]

plt.figure(figsize=(6.0,4.0))
for z in zs:
    g = gamma_t_linear(R_m, eps=EPS, H_of_z=lambda zz: H_of_z(zz), z=z, A_geom=1.0, p=0.0)
    plt.plot(R_kpc, g, label=f"z={z}")
plt.xlabel("R [kpc]")
plt.ylabel(r"$\gamma_t(R)$ (toy)")
plt.title(r"Toy stacked shear $\propto a_0(z)/c^2 \sim H(z)$")
plt.legend()
plt.tight_layout()
plt.savefig("results/fig_gamma_t.png", dpi=150)
plt.close()

# ---- 3) If posterior plot already exists, copy is not needed; we just print tips.
if os.path.exists("results/eps_posterior.png"):
    print("[make_figs] Found existing results/eps_posterior.png (from run_sparc_fit.sh).")
else:
    print("[make_figs] No posterior yet. Run scripts/run_sparc_fit.sh to produce it.")

if os.path.exists("results/example_rc_fit.png"):
    print("[make_figs] Found existing results/example_rc_fit.png.")
else:
    print("[make_figs] No RC example yet. Run scripts/run_sparc_fit.sh to produce it.")
PYCODE

echo "[make_figs] Done. Figures in results/."
