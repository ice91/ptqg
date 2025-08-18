# ptqg/config.py
from __future__ import annotations
import os, json, hashlib, subprocess, datetime as _dt
from typing import Any, Dict
try:
    import yaml
except ImportError as e:
    raise RuntimeError("pyyaml is required. Install with `pip install pyyaml`.") from e

_MPC_M = 3.085677581e22
_C = 299_792_458.0

_DEFAULTS: Dict[str, Any] = {
    "project": {"name": "PTQG", "seed": 123, "run_id": None},
    "cosmology": {
        "H0_km_s_Mpc": 70.0,
        "prior_eps": {"type": "planck", "mu": 1.47, "sigma": 0.05, "low": 0.0, "high": 4.0},
    },
    "data": {
        "csv": "data/sparc.csv",
        "use_toy_if_missing": True,
        "toy_n_gal": 10,
        "sample_list": [],
        "filters": {"min_incl_deg": 0.0, "max_dist_frac_err": 1.0},
    },
    "inference": {
        "likelihood": "gaussian",
        "sigma_sys_kms": 3.0,
        "mcmc": {"walkers": 24, "steps": 2000, "burnin": 200, "thin": 1},
    },
    "viz": {"save_dir": "results", "annotate": {"git_sha": True, "manifest_hash": True}},
    "export": {"artifacts_csv": "results/artifacts.csv"},
}

def _deepmerge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deepmerge(out[k], v)
        else:
            out[k] = v
    return out

def load_manifest(path: str = "manifest.yaml") -> Dict[str, Any]:
    cfg = dict(_DEFAULTS)
    if os.path.exists(path):
        with open(path, "r") as f:
            user = yaml.safe_load(f) or {}
        cfg = _deepmerge(cfg, user)
    # derive run_id if missing
    if not cfg["project"].get("run_id"):
        stamp = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        sha = get_git_sha(short=True) or "nogit"
        cfg["project"]["run_id"] = f"{stamp}-{sha}"
    cfg["_meta"] = {
        "manifest_path": os.path.abspath(path),
        "manifest_hash": manifest_fingerprint(path),
        "git_sha": get_git_sha(short=True),
    }
    return cfg

def H0_si(cfg: Dict[str, Any]) -> float:
    """Convert H0 from km/s/Mpc to s^-1."""
    H0_km_s_Mpc = float(cfg["cosmology"]["H0_km_s_Mpc"])
    return (H0_km_s_Mpc * 1000.0) / _MPC_M

def get_git_sha(short: bool = True) -> str | None:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short" if short else "HEAD"]).decode().strip()
        return sha
    except Exception:
        return None

def manifest_fingerprint(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:12]
    except Exception:
        return None

def annotate_string(cfg: Dict[str, Any]) -> str:
    bits = []
    if cfg["viz"]["annotate"].get("git_sha") and cfg["_meta"]["git_sha"]:
        bits.append(f"git:{cfg['_meta']['git_sha']}")
    if cfg["viz"]["annotate"].get("manifest_hash") and cfg["_meta"]["manifest_hash"]:
        bits.append(f"manifest:{cfg['_meta']['manifest_hash']}")
    bits.append(cfg["project"]["run_id"])
    return " | ".join(bits)

def dump_run_meta(cfg: Dict[str, Any], extra: Dict[str, Any] | None = None, out_path: str = "results/run_meta.json") -> None:
    meta = {
        "project": cfg.get("project", {}),
        "cosmology": cfg.get("cosmology", {}),
        "data": {k: cfg["data"][k] for k in ["csv", "use_toy_if_missing", "toy_n_gal", "sample_list"] if k in cfg["data"]},
        "inference": {"likelihood": cfg["inference"]["likelihood"], "mcmc": cfg["inference"]["mcmc"]},
        "_meta": cfg.get("_meta", {}),
    }
    if extra:
        meta.update(extra)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)
