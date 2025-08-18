#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------------------------------------
# SPARC-like -> standardized CSV (offline)
#
# Default input:  data/sparc_processed_data.txt
# Default output: data/sparc_processed.csv
#
# Standardized header: galaxy_id,r_kpc,v_kms,err_kms,mass_msun
#
# Options:
#   --in <PATH>       input file (default: data/sparc_processed_data.txt)
#   --out <PATH>      output CSV (default: data/sparc_processed.csv)
#   --force           overwrite existing output
#   --sample <N>      also write a "small" CSV with top-N galaxies by point count
#   -h|--help         show usage
# ----------------------------------------------------------------------

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/fetch_sparc.sh [--in data/sparc_processed_data.txt] [--out data/sparc_processed.csv] [--sample 10] [--force]

Description:
  Convert a local SPARC-like table to a standardized CSV with header:
    galaxy_id,r_kpc,v_kms,err_kms,mass_msun
  The parser auto-detects delimiter (comma/whitespace), header presence,
  and common column synonyms:
    galaxy_id ~ galaxy|name|id
    r_kpc     ~ r_kpc|rad[kpc]|radius_kpc|r|radius
    v_kms     ~ v_kms|vobs[km/s]|v[km/s]|v|vobs
    err_kms   ~ err_kms|errv[km/s]|sigma_v|dv|err
    mass_msun ~ mass_msun|mass[m_sun]|mstar|mass

Options:
  --in <PATH>     Input text/CSV path. Default: data/sparc_processed_data.txt
  --out <PATH>    Output CSV path. Default: data/sparc_processed.csv
  --force         Overwrite output if exists.
  --sample <N>    Also emit a small sample CSV with top-N galaxies by row count,
                  saved as: <out_basename>_small.csv
  -h, --help      Show this help.

Examples:
  bash scripts/fetch_sparc.sh
  bash scripts/fetch_sparc.sh --in data/sparc_processed_data.txt --out data/sparc_processed.csv
  bash scripts/fetch_sparc.sh --sample 10 --force
USAGE
}

IN="data/sparc_processed_data.txt"
OUT="data/sparc_processed.csv"
FORCE=0
SAMPLE_N=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --in) IN="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --force) FORCE=1; shift;;
    --sample) SAMPLE_N="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if [[ ! -f "${IN}" ]]; then
  echo "Error: input not found: ${IN}"
  exit 1
fi

mkdir -p "$(dirname "${OUT}")"

if [[ -f "${OUT}" && ${FORCE} -ne 1 ]]; then
  echo "Output exists: ${OUT}. Use --force to overwrite."
  exit 1
fi

# Python parser for robustness / easy synonym mapping
python3 - "${IN}" "${OUT}" "${SAMPLE_N:-""}" <<'PY'
import sys, csv, re, math
from collections import defaultdict, Counter
from pathlib import Path

inp = Path(sys.argv[1])
out = Path(sys.argv[2])
sample_n = sys.argv[3].strip() if len(sys.argv) > 3 else ""
sample_n = int(sample_n) if sample_n else None

# ------------------------------
# Helpers
# ------------------------------
def is_number(x: str) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False

def detect_delim_and_header(lines):
    """
    Return (delim, has_header, header_tokens_or_None, data_start_index).
    delim: ',' or 'ws' (whitespace)
    has_header: bool
    header_tokens_or_None: list[str] or None
    data_start_index: int
    """
    # take first non-empty, non-comment lines
    clean = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("#") or s.startswith("//"):
            continue
        clean.append(s)
        if len(clean) >= 20:  # enough to detect
            break
    if not clean:
        return ("ws", False, None, 0)

    # delim guess by first line
    first = clean[0]
    delim = "," if ("," in first) else "ws"

    # tokenization function
    def tok(s):
        if delim == ",":
            return [c.strip() for c in s.split(",")]
        else:
            return re.split(r"\s+", s.strip())

    # header detection: if majority of tokens in first line are non-numeric,
    # treat as header.
    t0 = tok(first)
    # require >=5 cols
    if len(t0) < 5:
        # try next lines until >=5 cols
        i = 1
        while i < len(clean) and len(tok(clean[i])) < 5:
            i += 1
        if i < len(clean):
            t0 = tok(clean[i])
            # shift clean so data starts at that line
            clean = clean[i:]
            first = clean[0]
        else:
            # give up; return default guess
            return (delim, False, None, 0)

    nonnum = sum(1 for c in t0 if not is_number(c))
    has_header = nonnum >= 2  # at least two non-numeric columns to call it header
    if has_header:
        return (delim, True, tok(first), 1)
    else:
        return (delim, False, None, 0)

# ------------------------------
# Read file & detect
# ------------------------------
raw = inp.read_text(encoding="utf-8", errors="ignore").splitlines()
delim, has_header, header_tokens, start_idx = detect_delim_and_header(raw)

def tok(s):
    if delim == ",":
        return [c.strip() for c in s.split(",")]
    else:
        return re.split(r"\s+", s.strip())

# Synonym maps (lowercased, stripped of spaces/brackets)
syn_map = {
    "galaxy_id": {"galaxy","name","id","galaxy_id","object"},
    "r_kpc": {"r_kpc","rad[kpc]","radius_kpc","r","radius","r(kpc)","rkpc"},
    "v_kms": {"v_kms","vobs[km/s]","v[km/s]","v","vobs","vel","v_km_s"},
    "err_kms": {"err_kms","errv[km/s]","sigma_v","dv","err","e_v","verr","v_err"},
    "mass_msun": {"mass_msun","mass[m_sun]","mstar","mass","m_*","mstar_msun","m"}
}

def normalize_colname(s: str) -> str:
    s = s.strip().lower()
    s = s.replace(" ", "")
    s = s.replace("_", "")
    s = s.replace("-", "")
    s = s.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
    s = s.replace("/", "")
    return s

# map header tokens to our standard names
col_idx = {}
if has_header:
    norm_tokens = [normalize_colname(t) for t in header_tokens]
    for k, aliases in syn_map.items():
        found = None
        for i, nt in enumerate(norm_tokens):
            if nt in {normalize_colname(a) for a in aliases}:
                found = i
                break
        if found is not None:
            col_idx[k] = found

# if no header, assume order: galaxy, r, v, err, mass in the first 5 cols
if not has_header:
    col_idx = {"galaxy_id":0, "r_kpc":1, "v_kms":2, "err_kms":3, "mass_msun":4}

required = ["galaxy_id","r_kpc","v_kms","err_kms","mass_msun"]
missing = [k for k in required if k not in col_idx]
if missing:
    msg = (
        "Could not locate required columns: "
        + ", ".join(missing)
        + (f"\nDetected header tokens: {header_tokens}" if header_tokens else "")
        + "\nTip: ensure your file has these columns or recognized synonyms."
    )
    raise SystemExit(msg)

rows = []
skipped = 0
for i, line in enumerate(raw[start_idx:], start=start_idx):
    s = line.strip()
    if not s or s.startswith("#") or s.startswith("//"):
        continue
    cols = tok(s)
    # guard length
    if max(col_idx.values()) >= len(cols):
        skipped += 1
        continue
    try:
        galaxy = cols[col_idx["galaxy_id"]]
        r_kpc   = float(cols[col_idx["r_kpc"]])
        v_kms   = float(cols[col_idx["v_kms"]])
        err_kms = float(cols[col_idx["err_kms"]])
        mass    = float(cols[col_idx["mass_msun"]])
        rows.append((galaxy, r_kpc, v_kms, err_kms, mass))
    except Exception:
        skipped += 1

if not rows:
    raise SystemExit("No parsable data rows. Please check the input format/columns.")

# write output CSV atomically
out_tmp = out.with_suffix(".tmp.csv")
out_tmp.parent.mkdir(parents=True, exist_ok=True)
with out_tmp.open("w", newline="", encoding="utf-8") as fo:
    w = csv.writer(fo)
    w.writerow(["galaxy_id","r_kpc","v_kms","err_kms","mass_msun"])
    for t in rows:
        w.writerow(t)
out_tmp.replace(out)

# summary
cnt = Counter(g for g, *_ in rows)
print(f"Wrote {len(rows)} rows to {out}")
if skipped:
    print(f"Skipped {skipped} noisy/invalid lines.")
print(f"Detected delimiter: {delim}; header: {has_header}")
print(f"Galaxies: {len(cnt)} (top 10 by count)")
for g, n in cnt.most_common(10):
    print(f"  {g}: {n} pts")

# optional small sample
if sample_n and sample_n > 0:
    top_gals = [g for g, _ in cnt.most_common(sample_n)]
    small = [r for r in rows if r[0] in top_gals]
    small_out = out.with_name(out.stem + "_small.csv")
    with small_out.open("w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo)
        w.writerow(["galaxy_id","r_kpc","v_kms","err_kms","mass_msun"])
        for t in small:
            w.writerow(t)
    print(f"Wrote small sample ({len(small)} rows, {len(top_gals)} galaxies) -> {small_out}")
PY

echo "OK."
