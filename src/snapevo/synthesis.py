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
import socket
import sys
import threading
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np

from snapevo import io

__all__ = ["KN_DIR", "kn_law", "kn_cost", "kn_eps", "synthesize", "branch_unitaries", "unitary_diamond"]

KN_DIR = "kliuchnikov_nonmixed_2026-10-05"
_lock = threading.RLock()                       # synthesize() holds it around _cache_lock


@contextmanager
def _cache_lock(path: Path):
    """Hold the cache across threads and processes (flock on a side file).

    Many jobs share one cache on a cluster. Where the file system has no flock
    the lock is skipped. Writes are still atomic, so the worst case is a lost
    entry that gets synthesized again.
    """
    with _lock:
        try:
            import fcntl
            fh = open(path.with_name(path.name + ".lock"), "a")
        except (ImportError, OSError):
            yield
            return
        try:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX)
            except OSError:
                pass
            yield
        finally:
            fh.close()


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


def _fast_t_count(U) -> int:
    """T count of a DOmegaUnitary by pygridsynth's own reduction, without building gate objects.

    Same steps as pygridsynth.decompose_domega_unitary: a T for every H T^m
    step with m odd, and one more if det has odd omega power. On 2,392 search
    candidates it matched tcount_vetted.synth_1q_exact every time, at 1/15 of
    the cost (that function builds an mpmath matrix per gate and verifies
    the circuit twice).
    """
    from pygridsynth.synthesis_of_cliffordT import BIT_COUNT, BIT_SHIFT
    U = U.reduce_denomexp()
    t = 0
    while U.k > 0:
        rz, rw = U.z.residue, U.w.residue
        rsq = (U.z.u * U.z.conj.u).residue
        m = BIT_SHIFT[rw] - BIT_SHIFT[rz]
        if m < 0:
            m += 4
        if rsq == 0b0000:
            U = U.mul_by_H_and_T_power_from_left(0).renew_denomexp(U.k - 1)
        elif rsq == 0b1010:
            U = U.mul_by_H_and_T_power_from_left(-m).renew_denomexp(U.k - 1)
            t += m & 1
        elif rsq == 0b0001:
            if BIT_COUNT[rz] == BIT_COUNT[rw]:
                U = U.mul_by_H_and_T_power_from_left(-m).renew_denomexp(U.k - 1)
            else:
                U = U.mul_by_H_and_T_power_from_left(-m)
            t += m & 1
        else:
            raise ValueError("residue pattern outside pygridsynth's reduction")
    return t + (U.n & 1)


class _LazyExact(dict):
    """synth_1q_exact's result with the fast T count, and the vetted synthesis on first other use.

    Isaac's projective_search reads only "t_count" of each candidate. The
    winner's word is read by fallback_protocol, which runs the full vetted
    synthesis and checks it against the fast count.
    """

    def __init__(self, U, full_fn):
        super().__init__(t_count=_fast_t_count(U))
        self._U, self._full_fn = U, full_fn

    def _full(self) -> None:
        if not dict.__contains__(self, "word"):
            full = self._full_fn(self._U)
            if full["t_count"] != dict.__getitem__(self, "t_count"):
                raise AssertionError("fast T count disagrees with tcount_vetted.synth_1q_exact")
            self.update(full)

    def __getitem__(self, key):
        if key != "t_count":
            self._full()
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        if key != "t_count":
            self._full()
        return dict.get(self, key, default)


def _patch_search() -> None:
    """Let Isaac's KN search price its candidates with _fast_t_count (results unchanged)."""
    import tcount_vetted as tv                   # vendored next to synthesize.py
    if getattr(tv.synth_1q_exact, "_snapevo_lazy", False):
        return
    full_fn = tv.synth_1q_exact

    def lazy(U):
        return _LazyExact(U, full_fn)

    lazy._snapevo_lazy = True
    tv.synth_1q_exact = lazy


def _cache_path() -> Path:
    base = Path(os.environ.get("SNAPEVO_RUNS", io.repo_root() / "runs"))
    return base / "kn_cache.json"


# The cache as this process knows it: kn_cache.json (re-read only when its size or mtime changes) plus the journals
# (kn_cache.d/<host>_<pid>.jsonl, one per process, append-only, read from where this process left off). Reading
# and rewriting the whole file under one exclusive lock on every call (22 MB by 25 Oct 2026) queued dozens of jobs
# behind each other for up to an hour; now a hit costs a dict lookup and a miss appends one line. merge_journals()
# folds the journals into kn_cache.json when nothing else is running.
_mem: dict = {"path": None, "stamp": None, "cache": {}, "journal": {}}


def _journal_dir(path: Path) -> Path:
    return path.with_name(path.stem + ".d")


def _known(path: Path, key: str, fresh: bool) -> dict | None:
    """The record for key from memory; with ``fresh`` (a miss) first reload whatever changed on disk."""
    if _mem["path"] != path:
        _mem.update(path=path, stamp=None, cache={}, journal={})
    if key in _mem["cache"] and not fresh:
        return _mem["cache"][key]
    try:
        st = path.stat()
        stamp = (st.st_size, st.st_mtime_ns)
    except FileNotFoundError:
        stamp = None
    if stamp is not None and stamp != _mem["stamp"]:
        with _cache_lock(path):
            _mem["cache"].update(io.read_json(path))
        _mem["stamp"] = stamp
    jd = _journal_dir(path)
    if jd.is_dir():
        for f in sorted(jd.glob("*.jsonl")):
            pos = _mem["journal"].get(f.name, 0)
            with open(f, "rb") as fh:
                fh.seek(pos)
                chunk = fh.read()
            end = chunk.rfind(b"\n") + 1                     # complete lines only (a writer may be mid-line)
            for line in chunk[:end].splitlines():
                if line.strip():
                    e = json.loads(line)
                    _mem["cache"][e["key"]] = e["rec"]
            _mem["journal"][f.name] = pos + end
    return _mem["cache"].get(key)


def _record(path: Path, key: str, rec: dict) -> None:
    """Keep a new synthesis: into kn_cache.json if there is none yet, else onto this process's journal."""
    _mem["cache"][key] = rec
    if not path.exists():
        with _cache_lock(path):
            cache = io.read_json(path) if path.exists() else {}
            cache[key] = rec
            io.write_json(path, cache, indent=None)
        return
    jd = _journal_dir(path)
    jd.mkdir(exist_ok=True)
    with open(jd / f"{socket.gethostname()}_{os.getpid()}.jsonl", "a") as fh:
        fh.write(json.dumps({"key": key, "rec": rec}) + "\n")


def merge_journals(path: Path | None = None) -> int:
    """Fold kn_cache.d/*.jsonl into kn_cache.json and remove them; returns the number of new entries. Run it only
    when no process is synthesizing (one still appending to a journal would lose what it writes after the merge)."""
    path = path or _cache_path()
    jd = _journal_dir(path)
    files = sorted(jd.glob("*.jsonl")) if jd.is_dir() else []
    with _cache_lock(path):
        cache = io.read_json(path) if path.exists() else {}
        n0 = len(cache)
        for f in files:
            for line in f.read_text().splitlines():
                if line.strip():
                    e = json.loads(line)
                    cache[e["key"]] = e["rec"]
        io.write_json(path, cache, indent=None)
    for f in files:
        f.unlink()
    return len(cache) - n0


def _key(theta: float, eps: float) -> str:
    return f"{math.remainder(theta, 2 * math.pi):.15f}|{eps:.6e}"


def synthesize(theta: float, eps: float, use_cache: bool = True, unitaries: bool = False) -> dict:
    """KN non-mixed synthesis of Rz(theta) at diamond budget eps.

    Returns a dict with expected_t, t_v, t_b, p, theta_s (the success-branch
    angle, on theta's branch), d0 = D(U0, Rz(theta)), d1 = D(U1, Rz(theta)) and
    d_branch = D(U1, U0), the error of the fallback relative to the success branch.
    Entries synthesized since 7 Oct 2026 also keep the branch unitaries U0 and U1
    (U1 is None when the protocol never falls back), as [re, im] pairs. With
    ``unitaries`` an older cache entry without them is synthesized again, which
    gives the same numbers, and replaced.
    """
    key = _key(theta, eps)
    path = _cache_path()
    if use_cache:
        with _lock:
            rec = _known(path, key, fresh=False)
            if rec is None or (unitaries and "U0" not in rec):
                rec = _known(path, key, fresh=True)
        if rec is not None and (not unitaries or "U0" in rec):
            return _unwrap(rec, theta)
    # Isaac's synthesize() runs best_protocol and then searches again to rebuild the gate
    # words, which we do not use. Calling best_protocol directly gives the same protocol and
    # branch unitaries in about half the time, and _patch_search cuts the search itself
    # by about 10x. Its synthesize() also failed its own check for
    # some negative angles (e.g. -0.4429453 at 1e-4), so theta goes in as theta mod 2 pi, and
    # theta_s is unwrapped below.
    _module()
    _patch_search()
    import kliuchnikov_fallback as kf            # vendored next to synthesize.py
    r = kf.best_protocol(theta % (2 * math.pi), eps, q_options=(0.99, 0.95, 0.9), split=0.1)
    U0 = r["branches"][0][1]
    U1 = r["branches"][1][1] if len(r["branches"]) > 1 else None
    theta_s = cmath.phase(U0[1, 1]) - cmath.phase(U0[0, 0])
    d_branch = unitary_diamond(U0, U1) if U1 is not None else 0.0
    mat = lambda M: None if M is None else [[[float(z.real), float(z.imag)] for z in row] for row in M]
    out = {"theta": math.remainder(theta, 2 * math.pi), "eps": eps,
           "expected_t": float(r["expected_t"]), "t_v": int(r["t_v"]), "t_b": int(r["t_b"]), "p": float(r["p"]),
           "theta_s": theta_s, "d0": float(r["d0"]), "d1": float(r["d1"]),
           "d_branch": d_branch, "bound": float(r["bound"]), "U0": mat(U0), "U1": mat(U1)}
    if use_cache:
        with _lock:
            _record(path, key, out)
    return _unwrap(out, theta)


def branch_unitaries(rec: dict) -> tuple[np.ndarray, np.ndarray | None]:
    """U0 and U1 of a synthesis record as arrays (synthesize with unitaries=True first)."""
    mat = lambda M: None if M is None else np.array([[complex(*z) for z in row] for row in M])   # noqa: E731
    return mat(rec["U0"]), mat(rec["U1"])


def _unwrap(rec: dict, theta: float) -> dict:
    """Put theta_s on the same 2 pi branch as the caller's theta."""
    rec = dict(rec)
    rec["theta_s"] = theta + math.remainder(rec["theta_s"] - theta, 2 * math.pi)
    return rec


if __name__ == "__main__":                       # quick manual check
    print(json.dumps(synthesize(0.7, 1e-4, use_cache=False), indent=1))
