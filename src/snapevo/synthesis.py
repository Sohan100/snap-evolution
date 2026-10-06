"""Kliuchnikov non-mixed (fallback) synthesis: the calibrated cost law and exact circuits.

Isaac's package (third_party/kliuchnikov_nonmixed_2026-10-05, arXiv:2203.10064,
non-mixed fallback protocol) is used verbatim. For an angle theta and a diamond
budget eps it returns a two-branch protocol:

* success, probability p: a diagonal unitary U0 = Rz(theta_s) up to phase, with
  D(U0, Rz(theta)) <= 0.9 eps;
* fallback, probability 1 - p: U1 = B F, with D(U1, Rz(theta)) <= 0.1 eps / (1 - p);

and expected T = T(V) + (1 - p) T(B). The success branch is an exact z rotation,
so "plugging in the Clifford+T circuit" for the coherent error is replacing
theta by theta_s. The fallback branch is heralded and enters the error ledger
(docs/error_metric.md).

Calls take a few seconds, so results are cached in runs/kn_cache.json.
"""

from __future__ import annotations

import cmath
import json
import math
import os
import sys
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

from snapevo import io

__all__ = ["KN_DIR", "kn_law", "kn_cost", "kn_eps", "synthesize", "unitary_diamond"]

KN_DIR = "kliuchnikov_nonmixed_2026-10-05"
_lock = threading.Lock()


@lru_cache(maxsize=1)
def kn_law() -> tuple[float, float]:
    """(a, b) of the calibrated mean cost E[T] = a log2(1/eps) + b (diamond eps).

    From the package's own calibration (40 random angles per eps, q = 0.99),
    1.096 log2(1/eps) + 4.44. The paper's fit is 1.03 log2(1/eps) + 5.75.
    """
    cal = io.read_json(io.third_party(KN_DIR) / "kliuchnikov_calibration.json")
    a, b = cal["fit"]
    return float(a), float(b)


def kn_cost(eps: float) -> float:
    """Calibrated mean expected T at diamond error eps."""
    a, b = kn_law()
    return a * math.log2(1.0 / eps) + b


def kn_eps(T: float) -> float:
    """Diamond error the calibrated law buys for an expected cost of T."""
    a, b = kn_law()
    return 2.0 ** (-(T - b) / a)


def unitary_diamond(U: np.ndarray, V: np.ndarray) -> float:
    """Diamond distance between two single-qubit unitary channels, in [0, 2].

    With eigenphases alpha, beta of U^dag V, it is 2 |sin((alpha - beta)/2)| when
    the arc is below pi, which is always the case here.
    """
    ev = np.linalg.eigvals(U.conj().T @ V)
    w = abs(cmath.phase(ev[1] / ev[0]))
    return 2.0 * math.sin(min(w, math.pi) / 2.0)


def _module():
    d = str(io.third_party(KN_DIR))
    if d not in sys.path:
        sys.path.insert(0, d)
    import synthesize as kn                      # the vendored entry point
    return kn


def _cache_path() -> Path:
    base = Path(os.environ.get("SNAPEVO_RUNS", io.repo_root() / "runs"))
    return base / "kn_cache.json"


def _key(theta: float, eps: float) -> str:
    return f"{math.remainder(theta, 2 * math.pi):.15f}|{eps:.6e}"


def synthesize(theta: float, eps: float, use_cache: bool = True) -> dict:
    """KN non-mixed synthesis of Rz(theta) at diamond budget eps.

    Returns a dict with expected_t, t_v, t_b, p, theta_s (the success-branch
    angle, on theta's branch), d0 = D(U0, Rz(theta)), d1 = D(U1, Rz(theta)) and
    d_branch = D(U1, U0), the error of the fallback relative to the success branch.
    """
    key = _key(theta, eps)
    path = _cache_path()
    if use_cache:
        with _lock:
            cache = io.read_json(path) if path.exists() else {}
        if key in cache:
            return _unwrap(cache[key], theta)
    r = _module().synthesize(math.remainder(theta, 2 * math.pi), eps)
    U0 = np.array([[complex(*z) for z in row] for row in r["unitaries"]["U0"]])
    theta_s = cmath.phase(U0[1, 1]) - cmath.phase(U0[0, 0])
    if r["unitaries"]["U1"] is not None:
        U1 = np.array([[complex(*z) for z in row] for row in r["unitaries"]["U1"]])
        d_branch = unitary_diamond(U0, U1)
    else:
        d_branch = 0.0
    out = {"theta": math.remainder(theta, 2 * math.pi), "eps": eps,
           "expected_t": r["expected_t"], "t_v": r["t_v"], "t_b": r["t_b"], "p": r["p"],
           "theta_s": theta_s, "d0": r["diamond_success"], "d1": r["diamond_fallback"],
           "d_branch": d_branch, "bound": r["diamond_bound"]}
    if use_cache:
        with _lock:
            cache = io.read_json(path) if path.exists() else {}
            cache[key] = out
            io.write_json(path, cache, indent=None)
    return _unwrap(out, theta)


def _unwrap(rec: dict, theta: float) -> dict:
    """Put theta_s on the same 2 pi branch as the caller's theta."""
    rec = dict(rec)
    rec["theta_s"] = theta + math.remainder(rec["theta_s"] - theta, 2 * math.pi)
    return rec


if __name__ == "__main__":                       # quick manual check
    print(json.dumps(synthesize(0.7, 1e-4, use_cache=False), indent=1))
