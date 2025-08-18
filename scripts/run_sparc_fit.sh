#!/usr/bin/env bash
set -euo pipefail

MANIFEST="manifest.yaml"

print_help() {
  cat <<'EOF'
run_sparc_fit.sh - Shared-epsilon MCMC (manifest-driven)

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
import os, json, hashlib
import numpy as np
import matplotlib.pyplot as plt

from ptqg.config import load_manifest, H0_si, annotate_string, dump_run_meta
from ptqg.data_sparc import load_sparc_csv, toy_sample
from ptqg.likelihood import loglike_gaussian, loglike_student_t, model_velocity_kms
from ptqg.mcmc import sample, gaussian, uniform
from ptqg.model_compare import aic_bic

CFG = load_manifest(r"${MANIFEST}")
H0 = H0_si(CFG)
like = CFG["inference"]["likelihood"].lower()
walkers = int(CFG["inference"]["mcmc"]["walkers"])
steps   = int(CFG["inference"]["mcmc"]["steps"])
seed    = int(CFG["project"]["seed"])

# ----- data -----
csv_path = CFG["data"]["csv"]
use_toy  = CFG["data"]["use_toy_if_missing"] or (not os.path.exists(csv_path))
if use_toy:
    galaxies = toy_sample(n_gal=int(CFG["data"]["toy_n_gal"]), seed=seed)
else:
    galaxies = load_sparc_csv(csv_path)
# 可選：依 sample_list 過濾
names = CFG["data"].get("sample_list") or []
if names:
    galaxies = [g for g in galaxies if g.name in set(names)]

G = len(galaxies); Ntot = sum(g.r_m.size for g in galaxies)
print(f"[run] #gal={G}, Ntot={Ntot}, like={like}, H0={H0:.3e} s^-1")

# ----- priors -----
p = CFG["cosmology"]["prior_eps"]
if p["type"].lower() == "planck":
    prior_eps = gaussian(float(p["mu"]), float(p["sigma"]))
else:
    prior_eps = uniform(float(p["low"]), float(p["high"]))
prior_ups = gaussian(0.5, 0.1)

# ----- sample -----
out = sample(
    galaxies=galaxies, H0=H0, like=like,
    prior_eps=prior_eps, prior_ups=prior_ups,
    n_walkers=walkers, n_steps=steps, seed=seed
)
eps_chain = out["eps_chain"]; logprob = out["logprob"]
np.save("results/eps_chain.npy", eps_chain)
np.save("results/logprob.npy", logprob)

smax, wmax = np.unravel_index(np.nanargmax(logprob), logprob.shape)
eps_map = float(eps_chain[smax, wmax])
ups_map = {g.name: float(out["ups_chains"][g.name][smax, wmax]) for g in galaxies}

# 比較指標（用 Gaussian logL 做 AIC/BIC）
logL_map, chi2nu = loglike_gaussian(galaxies, eps_map, H0, ups_map)
k = 1 + G
AIC, BIC = aic_bic(logL_map, k, Ntot)
print(f"[run] MAP eps={eps_map:.4f}, logL={logL_map:.2f}, AIC={AIC:.2f}, BIC={BIC:.2f}")

# 輸出表格
import csv as _csv
with open("results/per_gal_chi2.csv", "w", newline="") as f:
    w = _csv.writer(f); w.writerow(["name", "chi2_nu"])
    for n,v in chi2nu.items(): w.writerow([n, f"{v:.6f}"])
with open("results/aic_bic.csv", "w", newline="") as f:
    w = _csv.writer(f); w.writerow(["logL_max","k","Ntot","AIC","BIC"])
    w.writerow([f"{logL_map:.6f}", k, Ntot, f"{AIC:.6f}", f"{BIC:.6f}"])

# Posterior 圖（加註釋）
plt.figure(figsize=(6,4))
plt.hist(eps_chain.reshape(-1), bins=60, density=True, alpha=0.75)
plt.axvline(eps_map, color="k", lw=1, label=f"MAP={eps_map:.3f}")
plt.xlabel(r"$\varepsilon$"); plt.ylabel("Posterior density")
plt.legend(); plt.tight_layout()
ann = annotate_string(CFG)
plt.gcf().text(0.01, 0.01, ann, fontsize=7, ha="left", va="bottom")
plt.savefig("results/eps_posterior.png", dpi=150); plt.close()

# RC 一例（加註釋）
g0 = galaxies[0]
r_kpc = g0.r_m / 3.085677581e19
vmod0 = model_velocity_kms(g0, eps_map, H0, ups_map[g0.name])
plt.figure(figsize=(6,4))
plt.errorbar(r_kpc, g0.v_obs_kms, yerr=g0.v_err_kms, fmt="o", ms=3, label="obs")
plt.plot(r_kpc, vmod0, lw=2, label="model")
plt.xlabel("r [kpc]"); plt.ylabel("v [km/s]")
plt.title(f"Example RC fit: {g0.name}")
plt.legend(); plt.tight_layout()
plt.gcf().text(0.01, 0.01, ann, fontsize=7, ha="left", va="bottom")
plt.savefig("results/example_rc_fit.png", dpi=150); plt.close()

# 總結與 checksum
md5 = hashlib.md5(np.ascontiguousarray(eps_chain).data).hexdigest()
summary = dict(
    eps_map=eps_map, logL_map=logL_map, AIC=AIC, BIC=BIC,
    walkers=walkers, steps=steps, seed=seed, like=like,
    H0=H0, Ntot=Ntot, n_gal=G, eps_chain_md5=md5
)
with open("results/summary.json", "w") as f: json.dump(summary, f, indent=2)
print(f"[run] eps_chain md5: {md5}")

# 保存本次 run meta（含 git/manifest 指紋）
dump_run_meta(CFG, extra=summary, out_path="results/run_meta.json")
PYCODE

# 構建 artifacts 指紋
python3 scripts/hash_artifacts.py --manifest "${MANIFEST}" --dir results --out results/artifacts.csv

echo "[run_sparc_fit] Done. See results/."
