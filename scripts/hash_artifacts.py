#!/usr/bin/env python3
import os, argparse, hashlib, csv, time, json
from ptqg.config import load_manifest

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.yaml")
    ap.add_argument("--dir", default="results")
    ap.add_argument("--out", default="results/artifacts.csv")
    args = ap.parse_args()

    cfg = load_manifest(args.manifest)
    rows = []
    for root, _, files in os.walk(args.dir):
        for fn in files:
            p = os.path.join(root, fn)
            try:
                st = os.stat(p)
                rows.append({
                    "path": p,
                    "bytes": st.st_size,
                    "mtime": int(st.st_mtime),
                    "sha256": sha256(p),
                    "run_id": cfg["project"]["run_id"],
                    "git_sha": cfg["_meta"]["git_sha"],
                    "manifest_hash": cfg["_meta"]["manifest_hash"],
                })
            except Exception:
                pass

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
            ["path","bytes","mtime","sha256","run_id","git_sha","manifest_hash"])
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"[hash_artifacts] wrote {args.out} with {len(rows)} rows.")

if __name__ == "__main__":
    main()
