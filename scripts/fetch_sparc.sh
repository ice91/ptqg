#!/usr/bin/env bash
# Fetch SPARC rotation-curve data or generate a toy CSV with the expected columns.
# Usage:
#   scripts/fetch_sparc.sh [-o data_dir] [--toy]
# Default:
#   -o data/   (folder will be created)
# Behavior:
#   1) Try to download a SPARC-like CSV into ${OUT}/sparc.csv (best-effort).
#   2) If download fails or --toy is given, generate a toy CSV via ptqg.
set -euo pipefail

OUT="data"
FORCE_TOY="0"

print_help() {
  cat <<'EOF'
fetch_sparc.sh - download SPARC-like data or generate a toy CSV.

Options:
  -o, --out DIR     Output directory (default: data)
  --toy             Skip download; create synthetic CSV from ptqg.toy_sample()
  -h, --help        Show this help

Output:
  ${OUT}/sparc.csv with columns:
    r_kpc, v_kms, e_v_kms, vdisk_ms, vbulge_ms, vgas_ms,
    D_Mpc, inc_deg, eD_Mpc, einc_deg, name
EOF
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--out)
      OUT="${2:?missing dir}"
      shift 2
      ;;
    --toy)
      FORCE_TOY="1"
      shift
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

mkdir -p "${OUT}"

CSV="${OUT}/sparc.csv"

generate_toy_csv() {
  echo "[fetch_sparc] Generating toy CSV at ${CSV}"
  python3 - <<'PYCODE'
import numpy as np
import pandas as pd
from ptqg.data_sparc import toy_sample

gals = toy_sample(n_gal=15, seed=42)
rows = []
for g in gals:
    # we will flatten per-radius rows
    r_kpc = g.r_m / 3.085677581e19
    v_kms = g.v_obs_kms
    e_v_kms = g.v_err_kms
    vdisk_ms = g.v_disk_ms
    vbulge_ms = g.v_bulge_ms
    vgas_ms = g.v_gas_ms
    D_Mpc = g.D_m / 3.085677581e22
    inc_deg = g.inc_deg
    eD_Mpc = g.sigma_D_m / 3.085677581e22
    einc_deg = g.sigma_inc_deg
    for i in range(len(r_kpc)):
        rows.append(dict(
            r_kpc=r_kpc[i], v_kms=v_kms[i], e_v_kms=e_v_kms[i],
            vdisk_ms=vdisk_ms[i], vbulge_ms=vbulge_ms[i], vgas_ms=vgas_ms[i],
            D_Mpc=D_Mpc, inc_deg=inc_deg, eD_Mpc=eD_Mpc, einc_deg=einc_deg,
            name=g.name
        ))
df = pd.DataFrame(rows)
df.to_csv(r"${CSV}", index=False)
print(f"[toy] wrote {len(df)} rows to {r\"${CSV}\"}")
PYCODE
}

try_download() {
  # Try a few candidate URLs (best-effort). If all fail, return 1.
  # Note: URLs may change; we keep this as a convenience. Toy fallback remains the default path.
  CANDIDATES=(
    "https://raw.githubusercontent.com/astrocatalogs/sparc-like-demo/main/sparc_example.csv"
    "https://raw.githubusercontent.com/someuser/somewhere/master/SPARC_massmodels_demo.csv"
  )
  for url in "${CANDIDATES[@]}"; do
    echo "[fetch_sparc] Trying ${url}"
    if command -v curl >/dev/null 2>&1; then
      if curl -fsSL "${url}" -o "${CSV}.tmp"; then
        mv "${CSV}.tmp" "${CSV}"
        echo "[fetch_sparc] Downloaded to ${CSV}"
        return 0
      fi
    elif command -v wget >/dev/null 2>&1; then
      if wget -qO "${CSV}.tmp" "${url}"; then
        mv "${CSV}.tmp" "${CSV}"
        echo "[fetch_sparc] Downloaded to ${CSV}"
        return 0
      fi
    else
      echo "[fetch_sparc] Neither curl nor wget is available."
      return 1
    fi
  done
  return 1
}

if [[ "${FORCE_TOY}" == "1" ]]; then
  generate_toy_csv
  exit 0
fi

if try_download; then
  echo "[fetch_sparc] Done."
  exit 0
else
  echo "[fetch_sparc] Download failed; generating toy CSV instead."
  generate_toy_csv
  exit 0
fi
