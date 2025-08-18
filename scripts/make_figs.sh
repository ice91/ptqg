#!/usr/bin/env bash
set -euo pipefail

MANIFEST="manifest.yaml"

print_help() {
  cat <<'EOF'
make_figs.sh - Generate figures (manifest-driven)

Options:
  --manifest PATH   Path to manifest.yaml (default: manifest.yaml)
  -h, --help        Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) MANIFEST="${2:?missing path}"; shift 2 ;;
    -h|--help)  print_help; exit 0 ;;
    *) echo "Unknown option: $1" >&2; print_help; exit 1 ;;
  esac
done

mkdir -p results

python3 - <<PYCODE
import numpy as np, matplotlib.pyplot as plt, os
from ptqg.config import load_manifest, H0_si, annotate_string
from ptqg.cosmology import omega_lambda_from_eps, omega_lambda_numeric
from ptqg.lensing import gamma_t_linear

CFG = load_manifest(r"${MANIFEST}")
H0 = H0_si(CFG)
EPS = float(CFG["cosmology"]["prior_eps"].get("mu", 1.5))  # demo 線條用
ann = annotate_string(CFG)

# ---- Omega mapping ----
eps_grid = np.linspace(0.0, 3.0, 301)
ol_analytic = np.array([omega_lambda_from_eps(e) for e in eps_grid])
def a_of_t(t): return 1.0
t0 = 1.0 / H0
ol_numeric = np.array([omega_lambda_numeric(e, t0=t0, a_of_t=a_of_t) for e in eps_grid])
resid = ol_numeric - ol_analytic
rms = float(np.sqrt(np.mean(resid**2)))

plt.figure(figsize=(6,4))
plt.plot(eps_grid, ol_analytic, label="analytic")
plt.plot(eps_grid, ol_numeric, "--", label="numeric")
plt.xlabel(r"$\varepsilon$"); plt.ylabel(r"$\Omega_\Lambda$")
plt.title(r"$\Omega_\Lambda(\varepsilon)$ mapping")
plt.legend(); plt.tight_layout()
plt.gcf().text(0.01, 0.01, ann, fontsize=7)
plt.savefig("results/fig_omega_mapping.png", dpi=150); plt.close()

plt.figure(figsize=(6,3.4))
plt.plot(eps_grid, resid); plt.axhline(0, color="k", lw=0.8)
plt.xlabel(r"$\varepsilon$"); plt.ylabel("numeric - analytic")
plt.title(f"Residual (RMS={rms:.2e})"); plt.tight_layout()
plt.gcf().text(0.01, 0.01, ann, fontsize=7)
plt.savefig("results/fig_omega_residual.png", dpi=150); plt.close()
print(f"[make_figs] Omega mapping RMS={rms:.3e}")

# ---- Lensing toy ----
R_kpc = np.linspace(5, 100, 60)
R_m = R_kpc * 3.085677581e19
Om = 0.31; OL = omega_lambda_from_eps(EPS)
def H_of_z(z): return H0 * np.sqrt(Om*(1+z)**3 + OL)
zs = [0.1, 0.3, 0.6]
plt.figure(figsize=(6,4))
for z in zs:
    g = gamma_t_linear(R_m, eps=EPS, H_of_z=H_of_z, z=z, A_geom=1.0, p=0.0)
    plt.plot(R_kpc, g, label=f"z={z}")
plt.xlabel("R [kpc]"); plt.ylabel(r"$\gamma_t(R)$ (toy)")
plt.title(r"Stacked shear $\propto a_0(z)/c^2 \sim H(z)$")
plt.legend(); plt.tight_layout(); plt.gcf().text(0.01,0.01,ann,fontsize=7)
plt.savefig("results/fig_gamma_t.png", dpi=150); plt.close()
print("[make_figs] Done.")
PYCODE

# 若先前有 posterior/RC 圖，就留著；若沒有，提示用戶先跑 run_sparc_fit
if [[ ! -f results/eps_posterior.png ]]; then
  echo "[make_figs] Tip: run scripts/run_sparc_fit.sh --manifest ${MANIFEST} to produce posterior/RC plots."
fi
