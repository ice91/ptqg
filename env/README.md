# env/ — Reproducible Environment for PTQG

This folder contains reproducible environment definitions for the **PT-Projected Quaternion Gravity (PTQG)** codebase.

We provide both **Conda** and **pip/venv** options:

---

## Option A — Conda / Mamba (Recommended)

```bash
# Using mamba (faster) or conda
mamba env create -f env/environment.yml
# or: conda env create -f env/environment.yml

conda activate ptqg

# (Optional) install pre-commit hooks for consistent style
pre-commit install
