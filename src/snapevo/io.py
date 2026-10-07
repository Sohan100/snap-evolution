"""Where files live and how they are named.

Everything a run produces goes under ``runs/<model tag>/t<t>/``. The results are tracked in git, the
ref/ caches and run logs are not:

    cells.json                       trained depth ladder (train.py)
    snap_b<budget>_D<D>.json         compiled circuit (snap.py)
    snap_b<budget>_D<D>.ckpt.pkl     checkpoint while compiling, deleted at the end
    real_b<budget>_D<D>.json         realized circuit, every rotation priced, edges (realize.py, edges.py)
    eval_b<budget>_D<D>.json         MPS infidelity at N = 100 and T counts (evaluate.py)
    xcheck_b<budget>_D<D>.json       small-ring cross-checks (crosscheck.py)
    baseline_o<order>_r<steps>_e<eps>.json   Trotter + KN baseline points (baseline.py, evaluate.py)
    baseline_scan.json               the screened product formulas and their two fronts (baseline.py)
    ref/ref_N<N>_s<seed>_k<k>_*.npz  cached MPS of e^{-iHt} on each input (mps.py)

and, shared by all models and times,

    kn_cache.json                    every KN synthesis so far, by angle and budget (synthesis.py)
    summary.csv, figures/            all evaluated points and the sweep plots (snapevo summary, plot)

Set ``SNAPEVO_RUNS`` to put runs somewhere else (for example off OneDrive). Files are written to a
per-process temporary name and renamed, so parallel jobs never see half a file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

__all__ = ["repo_root", "third_party", "run_dir", "write_json", "read_json", "budget_tag"]


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "pyproject.toml").is_file():
            return p
    raise FileNotFoundError("snapevo is not inside its repository checkout")


def third_party(name: str) -> Path:
    """Path of a vendored input under third_party/."""
    return repo_root() / "third_party" / name


def run_dir(model_tag: str, t: float, create: bool = True) -> Path:
    base = Path(os.environ.get("SNAPEVO_RUNS", repo_root() / "runs"))
    d = base / model_tag / f"t{t:g}"
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def budget_tag(budget: float) -> str:
    return f"b{budget:.1e}"


def _default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def write_json(path: Path, payload, indent: int | None = 1) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")    # one per process, then an atomic rename
    with tmp.open("w") as fh:
        json.dump(payload, fh, indent=indent, default=_default)
    os.replace(tmp, path)
    return path


def read_json(path: Path):
    with Path(path).open() as fh:
        return json.load(fh)
