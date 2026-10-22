"""Translation-invariant evaluation of every realized point: the ring per-site infidelity C(N), exact.

    python cluster/ti_ring_eval.py list    [--times 1,2,4,6,8,10] [--methods snap,chain,baseline]
    python cluster/ti_ring_eval.py targets [--times ..] [--sizes 12,14,16] [--threads 4]
    python cluster/ti_ring_eval.py run     [--times ..] [--methods ..] [--sizes ..] [--workers 4] [--only REGEX]
    python cluster/ti_ring_eval.py point   NAME [--sizes ..]
    python cluster/ti_ring_eval.py table   [--times ..] [--tol 0.05] [--json FILE]
    python cluster/ti_ring_eval.py imps    [--times 1,2,4] [--Lu 2,4,8] [--samples 8] [--cutoff 1e-14] [--workers 4]
    python cluster/ti_ring_eval.py imps-table
    python cluster/ti_ring_eval.py refcheck [--times 1,2,4] [--steps 0.2,0.1] [--max-c 1e-6]

Models: XXZ chain, Delta = 0.25, J = 1, spin units, at h_x = 1.6 and h_x = 0 (snapevo's model keys A and B).

Why. The paper reports c on a 100-site open chain (MPS, 8 random product inputs, the circuit's edge gates at the
ends). The translation-invariant convention (Gibbs and Cincio, arXiv:2506.02105) reports the per-site
infidelity in the thermodynamic limit. Our circuits are translation-invariant brickworks, so on a ring of N
sites the bulk gate also acts on the wrap bond and there is no boundary at all; C(N) on the ring converges to the
thermodynamic-limit value as N grows. This script measures C(N) at N = 12, 14, 16 for every point and sets it
against the open-chain value at N = 100.

Points (``--methods``; a point's name is method:field:t:stem):
    snap      the sweep's realized SNAP points, runs/<tag>/t<t>/real_<stem>.json (c at N = 100 from eval_<stem>.json)
    chain     the open-chain arm, runs/<tag>/t<t>/chain/newton/real_<stem>.json (and its eval_<stem>.json)
    baseline  the Trotter + KN points, runs/<tag>/t<t>/baseline_o*_r*_e*.json (c at N = 100 in the same file)
Circuits are rebuilt with snapevo.opnorm.snap_circuit and baseline_circuit (the realized success-branch angles,
no synthesis rerun). On the ring the bulk cell gate acts on the wrap bond (N-1, 0), first qubit N-1, as
Layered.apply(bc="ring") does; the fitted edge gates of the open chain play no part.

Measured. C(N) = 1 - mean_k |<V phi_k|U phi_k>|^{2/N}, V = e^{-iHt} exact (snapevo.opnorm.Propagator, Chebyshev
to 1e-15, checked against expm_multiply), on 64 random product inputs per N drawn by
circuits.product_vectors(N, 64, 5151 + N) (a seed used by no training or evaluation set). Every input's
fidelity F_k is stored, so errors paired on the same inputs can be formed across points. The exact outputs
depend only on (field, t, N) and are computed once and cached as .npy under <out>/targets/. The open-chain value
c(N = 100) (eval "c_site", the success branch, as C) and its sampling error (from the eval's 8 per-input F) are
copied into each point's file.

Output: <out>/ring/<tag>/t<t>/<method>_<stem>.json, written atomically; a point recomputes only the sizes it
lacks, so runs resume. <out> is --out, else $TI_OUT, else $SCRATCH/snapevo-scratch/ti. Nothing is written under
runs/.

run: the exact targets first (one process, --threads Python threads in the Chebyshev products), then --workers
single-threaded worker processes over the points in order of t (all of t <= 6 before t = 8 and 10). Each worker
holds the targets of one (field, t) (at N = 16, 64 inputs are 64 MB) and two 64 MB work arrays, about 0.3 GB.

table: per (field, t, method) the ratio c(N = 100, open)/C_ring(16) - 1 (median and range), the convergence
C(16)/C(14) - 1 and C(14)/C(12) - 1, and every point beyond --tol (5%). The inputs differ between sizes and
between the ring and the open chain, so each ratio carries sampling noise; the table gives its standard error
from the per-input values (``se``), and z = ratio/se for the outliers.

imps: the infinite chain (snapevo.imps). Representative points, per field and t, the sweep's SNAP point with
c(N = 100) nearest 1e-6 and the baseline point nearest 1e-5; for each, c_inf = 1 - |lambda0|^{2/L_u} from the
mixed transfer matrix of the circuit's and the reference's iMPS (iTEBD; reference mps.reference_circuit at step
--dt 0.1), on --samples random unit-cell product inputs of L_u = 2, 4, 8 sites (product_vectors(L_u, n,
6161 + L_u)); the first --n-check inputs again at --check-cutoff; and C(16) of the same unit cells placed
periodically on the ring (exact), a check paired input by input. Reference iMPS are cached in <out>/imps_ref/;
results in <out>/imps/<tag>_t<t>_Lu<L>_c<cutoff>.json; imps-table sets them against the ring table and c(100).

periodic: whether a periodic unit-cell input stands in for random product inputs. On the 16-site ring (exact),
for the imps representatives at every t: C(16) of --samples random 16-site inputs (product_vectors(16, n, 7171))
against C(16) of each input's first L_u sites repeated around the ring, L_u = 2, 4, 8, paired input by input.
On the ring an L_u-periodic input is the iMPS c_inf of that unit cell up to finite-size effects (the paired
check of imps), so the ratio is the bias of the unit-cell ensemble. <out>/periodic/<tag>_t<t>.json.

refcheck: for the points with C(16) <= --max-c, C(16) against mps.reference_circuit at the steps of the
100-site runs (0.2 in the sweep, 0.1 in the chain arm) instead of exact e^{-iHt}, on the same inputs: the shift
the reference's own error makes in c(N = 100) (cmd_refcheck). <out>/refcheck/refcheck.json.

Speed (login22, 22 Oct 2026, one thread, load about 100): a brickwork sublayer on 64 vectors costs 0.11 s at
N = 16 (eight 4x4 matmuls over the vector plus a one-qubit roll for the odd sublayers; opnorm.apply_ops takes
0.30 s for an odd one, whose wrap gate it applies with a strided loop).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # before numpy loads
    os.environ.setdefault(_v, "1")

import argparse                                                             # noqa: E402
import json                                                                 # noqa: E402
import math                                                                 # noqa: E402
import re                                                                   # noqa: E402
import socket                                                               # noqa: E402
import sys                                                                  # noqa: E402
import time                                                                 # noqa: E402
from pathlib import Path                                                    # noqa: E402

import numpy as np                                                          # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RUNS = Path(os.environ.get("SNAPEVO_RUNS", REPO / "runs"))
SCRATCH = Path(os.environ.get("SCRATCH", "/pscratch/sd/s/sohan100")) / "snapevo-scratch"
FIELDS = {"1.6": "A", "0": "B"}               # snapevo's model keys: A is h_x = 1.6, B is h_x = 0
TAGS = {"1.6": "xxz_d0.25_hx1.6", "0": "xxz_d0.25_hx0"}
TIMES = (1, 2, 4, 6, 8, 10)
METHODS = ("snap", "chain", "baseline")
SIZES = (12, 14, 16)
N_INPUTS = 64
SEED0 = 5151                                   # inputs at N: product_vectors(N, 64, 5151 + N)
VERSION = 1
# seconds of one sublayer on 64 vectors, one thread (login22, 22 Oct 2026); N <= 10 from the 4x per 2 sites law
S_LAYER = {8: 1.0e-4, 10: 4e-4, 12: 1.6e-3, 14: 2.4e-2, 16: 0.11, 18: 0.45, 20: 1.8}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _read(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"{type(o).__name__} is not JSON serializable")


def _save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, default=_default, indent=1))
    os.replace(tmp, path)


def _under_runs(path: Path) -> bool:
    path, runs = Path(path).resolve(), RUNS.resolve()
    return path == runs or runs in path.parents


def out_root(arg: str | None = None) -> Path:
    root = Path(arg or os.environ.get("TI_OUT") or SCRATCH / "ti").resolve()
    if _under_runs(root):
        raise SystemExit(f"{root} is under {RUNS}; this script never writes there")
    return root


def _model(field: str):
    from snapevo.hamiltonian import get_model
    return get_model(FIELDS[field])


def _tdir(field: str, t) -> Path:
    return RUNS / TAGS[field] / f"t{float(t):g}"


# --------------------------------------------------------------------------- the points

def _open100(ev: dict | None) -> dict | None:
    """The N = 100 open-chain numbers of an eval record (or a baseline record, which holds its own)."""
    if not ev or "c_site" not in ev:
        return None
    m = ev.get("mps") or {}
    N = int(m.get("N", ev.get("N", 100)))
    out = {"c_site": ev["c_site"], "infid": ev.get("infid"), "infid_sampled": ev.get("infid_sampled"),
           "infid_sampled_kn": ev.get("infid_sampled_kn"), "infid_total": ev.get("infid_total"), "N": N,
           "states": m.get("states"), "dt": m.get("dt"), "chi_peak": m.get("chi_peak"),
           "cutoff": m.get("cutoff"), "trunc_flag": m.get("trunc_flag"), "C_ring10": ev.get("C_ring"),
           "C_train_real": ev.get("C_train_real")}
    if out["infid_sampled"] is not None:
        out["c_sampled"] = -math.expm1(math.log1p(-out["infid_sampled"]) / N)
    rows = m.get("rows") or []
    F = np.array([r["F"] for r in rows if "F" in r], dtype=float)
    if F.size > 1:
        ck = -np.expm1(np.log(F) / N)
        out["c_se"] = float(ck.std(ddof=1) / math.sqrt(F.size))
        out["c_rows"] = ck.tolist()
    return out


def _snap_like(method: str, field: str, t: int) -> list[dict]:
    d = _tdir(field, t) / ("chain/newton" if method == "chain" else "")
    out = []
    for p in sorted(d.glob("real_b*_D*.json")):
        stem = p.stem[len("real_"):]
        rec = _read(p)
        if not rec or "angles_real" not in rec:
            continue
        ev_path = d / f"eval_{stem}.json"
        out.append({"method": method, "field": field, "t": t, "stem": stem, "D": int(rec["D"]),
                    "file": str(p), "eval_file": str(ev_path) if ev_path.exists() else None,
                    "budget": rec.get("budget"), "cell": rec.get("cell"),
                    "edges_used_open": bool((rec.get("edges") or {}).get("use"))})
    return out


def _baselines(field: str, t: int) -> list[dict]:
    from snapevo.baseline import trotter_circuit
    d = _tdir(field, t)
    out = []
    for p in sorted(d.glob("baseline_o*_r*_e*.json")):
        rec = _read(p)
        if not rec or "synth" not in rec:
            continue
        D = trotter_circuit(_model(field), float(t), int(rec["order"]), int(rec["steps"])).depth
        out.append({"method": "baseline", "field": field, "t": t, "stem": p.stem[len("baseline_"):], "D": D,
                    "file": str(p), "eval_file": str(p) if "c_site" in rec else None, "order": int(rec["order"]),
                    "steps": int(rec["steps"]), "eps": float(rec["eps"]), "edges_used_open": False})
    return out


def points(times=TIMES, methods=METHODS) -> list[dict]:
    out = []
    for t in times:
        for field in FIELDS:
            for m in methods:
                if m in ("snap", "chain"):
                    out += _snap_like(m, field, t)
                elif m == "baseline":
                    out += _baselines(field, t)
                else:
                    raise SystemExit(f"unknown method {m!r}; choose from {METHODS}")
    return out


def pname(p: dict) -> str:
    return f"{p['method']}:{p['field']}:{p['t']:g}:{p['stem']}"


def result_path(root: Path, p: dict) -> Path:
    return Path(root) / "ring" / TAGS[p["field"]] / f"t{p['t']:g}" / f"{p['method']}_{p['stem']}.json"


def find_point(name: str) -> dict:
    method, field, t, stem = name.split(":", 3)
    for p in points((int(float(t)),), (method,)):
        if p["field"] == field and p["stem"] == stem:
            return p
    raise SystemExit(f"no point {name!r}")


def circuit(p: dict, full: bool = False):
    """The realized Layered circuit of a point (its bulk; the ring has no edges), or with ``full`` the
    opnorm.Circuit with the open chain's edge gates as the 100-site evaluation ran them."""
    from snapevo import opnorm as on
    rec = _read(Path(p["file"]))
    c = on.baseline_circuit(rec, _model(p["field"])) if p["method"] == "baseline" else on.snap_circuit(rec)
    return c if full else c.layered


# --------------------------------------------------------------------------- the ring

def apply_ring(layered, X: np.ndarray, N: int) -> np.ndarray:
    """A Layered circuit on the N-site ring applied to the columns of X (2^N, n), qubit 0 most significant.

    The same unitary as Layered.apply(bc="ring") and opnorm.apply_ops(layered_ops(.., "ring")): the cell
    gate of each sublayer on every bond of its parity, a gate on (i, i+1) with its first qubit on i, the wrap
    bond (N-1, 0) with its first qubit on N-1. The qubits are kept cyclically rotated: position p holds qubit
    (p + r) mod N, so every sublayer acts on positions (0, 1), (2, 3), ...; a sublayer whose parity differs
    from r's moves the leading qubit to the back first (r -> r + 1), and the end rotates back to r = 0.
    """
    if N % 2:
        raise ValueError("the brickwork needs an even number of sites")
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[0] != 2 ** N:
        raise ValueError(f"X must be (2^{N}, n), not {X.shape}")
    n = X.shape[1]
    psi = np.array(X, dtype=complex, order="C", copy=True)
    buf = np.empty_like(psi)
    r = 0
    for parity, angles in layered.layers:
        U = np.ascontiguousarray(layered.cell.gate(angles), dtype=complex)
        if (r - parity) % 2:
            np.copyto(buf.reshape(2 ** (N - 1), 2, n), psi.reshape(2, 2 ** (N - 1), n).transpose(1, 0, 2))
            psi, buf = buf, psi
            r = (r + 1) % N
        for i in range(0, N, 2):
            np.matmul(U, psi.reshape(2 ** i, 4, -1), out=buf.reshape(2 ** i, 4, -1))
            psi, buf = buf, psi
    k = (N - r) % N
    if k:
        np.copyto(buf.reshape(2 ** (N - k), 2 ** k, n), psi.reshape(2 ** k, 2 ** (N - k), n).transpose(1, 0, 2))
        psi = buf
    return psi


def input_block(N: int, seed: int, k0: int, k1: int, vecs=None) -> np.ndarray:
    """Columns k0..k1-1 of the product inputs product_vectors(N, n, seed) as dense vectors (2^N, k1 - k0)."""
    from snapevo.circuits import product_vectors
    if vecs is None:
        vecs = product_vectors(N, k1, seed)
    X = np.empty((2 ** N, k1 - k0), dtype=complex)
    for j, sites in enumerate(vecs[k0:k1]):
        psi = np.array([1.0 + 0j])
        for v in sites:
            psi = np.kron(psi, v)
        X[:, j] = psi
    return X


def target_path(root: Path, field: str, t, N: int, seed: int, n: int, bc: str = "ring") -> Path:
    tail = "" if bc == "ring" else f"_{bc}"
    return Path(root) / "targets" / f"{TAGS[field]}_t{float(t):g}_N{N}_s{seed}_n{n}{tail}.npy"


def make_targets(root: Path, field: str, t, N: int, n: int = N_INPUTS, threads: int = 4, check: bool = True,
                 log=_log, bc: str = "ring") -> Path:
    """e^{-iHt} (ring, or open chain) applied to the inputs at N, cached as a (2^N, n) complex .npy (.json beside)."""
    from snapevo.circuits import product_vectors
    from snapevo.opnorm import Propagator
    seed = SEED0 + N
    path = target_path(root, field, t, N, seed, n, bc)
    if path.exists() and path.with_suffix(".json").exists():
        return path
    t0 = time.time()
    prop = Propagator(_model(field), N, bc, float(t), threads=threads)
    vecs = product_vectors(N, n, seed)
    Y = np.empty((2 ** N, n), dtype=complex)
    blk = 16 if N >= 16 else n
    for k0 in range(0, n, blk):
        k1 = min(n, k0 + blk)
        Y[:, k0:k1] = prop.apply(input_block(N, seed, k0, k1, vecs), -1.0)
    sec = time.time() - t0
    err = prop.check() if check else None
    norms = np.linalg.norm(Y, axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npy")
    np.save(tmp, Y)
    os.replace(tmp, path)
    _save_json(path.with_suffix(".json"), {
        "field": field, "model": _model(field).to_dict(), "t": float(t), "N": N, "bc": bc, "seed": seed, "n": n,
        "inputs": f"circuits.product_vectors({N}, {n}, {seed})", "cheb_terms": prop.K, "bounds": list(prop.bounds),
        "tol": prop.tol, "check_vs_expm_multiply": err, "max_norm_dev": float(np.abs(norms - 1).max()),
        "seconds": sec, "check_seconds": time.time() - t0 - sec, "threads": threads, "host": socket.gethostname(),
        "when": time.strftime("%Y-%m-%d %H:%M:%S")})
    log(f"targets {TAGS[field]} t = {float(t):g} N = {N} {bc}: K {prop.K}, {sec:.1f} s, check {err if err is None else f'{err:.1e}'}")
    return path


_TARGETS: dict = {}


def _targets(root: Path, field: str, t, N: int, n: int, bc: str = "ring") -> np.ndarray:
    key = (field, float(t), N, n, bc)
    if key not in _TARGETS:
        for k in [k for k in _TARGETS if (k[0], k[1]) != (field, float(t))]:
            del _TARGETS[k]                                       # hold one (field, t) at a time
        p = target_path(root, field, t, N, SEED0 + N, n, bc)
        if not p.exists():
            make_targets(root, field, t, N, n, threads=1, bc=bc)
        _TARGETS[key] = np.load(p, mmap_mode=None)
    return _TARGETS[key]


def open_fidelities(c, root: Path, field: str, t, N: int, n: int = N_INPUTS) -> np.ndarray:
    """F_k on the N-site open chain for the same inputs as the ring, the circuit with its edge gates as the
    100-site evaluation ran it (opnorm.layered_ops(bc="open")), against the open-chain e^{-iHt}."""
    from snapevo.circuits import product_vectors
    from snapevo.opnorm import apply_ops
    seed = SEED0 + N
    vecs = product_vectors(N, n, seed)
    T = _targets(root, field, t, N, n, "open")
    ops = c.ops(N, "open")
    blk = 32 if N >= 16 else n
    F = np.empty(n)
    for k0 in range(0, n, blk):
        k1 = min(n, k0 + blk)
        Y = apply_ops(ops, input_block(N, seed, k0, k1, vecs), N)
        F[k0:k1] = np.abs(np.einsum("ij,ij->j", np.conj(T[:, k0:k1]), Y)) ** 2
    return F


def ring_fidelities(layered, root: Path, field: str, t, N: int, n: int = N_INPUTS) -> np.ndarray:
    """F_k = |<V phi_k|U phi_k>|^2 on the ring for the n inputs at N."""
    seed = SEED0 + N
    from snapevo.circuits import product_vectors
    vecs = product_vectors(N, n, seed)
    T = _targets(root, field, t, N, n)
    blk = 32 if N >= 16 else n
    F = np.empty(n)
    for k0 in range(0, n, blk):
        k1 = min(n, k0 + blk)
        Y = apply_ring(layered, input_block(N, seed, k0, k1, vecs), N)
        F[k0:k1] = np.abs(np.einsum("ij,ij->j", np.conj(T[:, k0:k1]), Y)) ** 2
    return F


def per_site(F: np.ndarray, N: int) -> np.ndarray:
    """1 - F^{1/N}, without cancellation."""
    return -np.expm1(np.log(np.asarray(F, dtype=float)) / N)


def run_point(p: dict, root: Path, sizes=SIZES, n: int = N_INPUTS, log=_log, open_sizes=()) -> dict:
    """Measure (or complete) one point's file; returns its record. ``open_sizes``: also C on the open chain
    at those sizes, on the same inputs as the ring (``open``)."""
    path = result_path(root, p)
    rec = _read(path)
    if not rec or rec.get("version") != VERSION:
        ev = _read(Path(p["eval_file"])) if p.get("eval_file") else None
        rec = {"version": VERSION, "name": pname(p), "point": p, "open100": _open100(ev), "ring": {}}
    need = [N for N in sizes if str(N) not in rec["ring"]]
    need_open = [N for N in open_sizes if str(N) not in rec.get("open", {})]
    if need_open:
        full = circuit(p, full=True)
        rec.setdefault("open", {})
        for N in need_open:
            t0 = time.time()
            F = open_fidelities(full, root, p["field"], p["t"], N, n)
            ck = per_site(F, N)
            rec["open"][str(N)] = {"N": N, "seed": SEED0 + N, "n": n, "C": float(ck.mean()),
                                   "C_se": float(ck.std(ddof=1) / math.sqrt(n)), "F": F.tolist(),
                                   "edges": full.edges is not None, "seconds": time.time() - t0}
        if not need:
            rec.update({"host": socket.gethostname(), "when": time.strftime("%Y-%m-%d %H:%M:%S")})
            _save_json(path, rec)
            r, o = rec["ring"].get("16"), rec["open"].get("16")
            log(f"{pname(p)} open: " + ", ".join(f"C_open({N}) {rec['open'][str(N)]['C']:.3e}" for N in need_open)
                + (f", C_open(16)/C_ring(16) - 1 {o['C'] / r['C'] - 1:+.2%}" if r and o else "")
                + f", {sum(rec['open'][str(N)]['seconds'] for N in need_open):.1f} s")
    if not need:
        return rec
    lay = circuit(p)
    for N in need:
        t0 = time.time()
        F = ring_fidelities(lay, root, p["field"], p["t"], N, n)
        ck = per_site(F, N)
        rec["ring"][str(N)] = {"N": N, "seed": SEED0 + N, "n": n, "C": float(ck.mean()),
                               "C_se": float(ck.std(ddof=1) / math.sqrt(n)), "F": F.tolist(),
                               "mean_infid": float(1 - F.mean()), "seconds": time.time() - t0}
    rec.update({"host": socket.gethostname(), "when": time.strftime("%Y-%m-%d %H:%M:%S"),
                "threads": int(os.environ.get("OMP_NUM_THREADS", "1"))})
    _save_json(path, rec)
    o = rec["open100"] or {}
    C = {N: rec["ring"][str(N)]["C"] for N in sizes if str(N) in rec["ring"]}
    ratio = (o.get("c_site") / C[16] - 1) if o.get("c_site") and C.get(16) else None
    log(f"{pname(p)} D {p['D']}: " + ", ".join(f"C({N}) {c:.3e}" for N, c in C.items())
        + (f", c100 {o['c_site']:.3e}" if o.get("c_site") else ", no c100")
        + (f", c100/C16 - 1 {ratio:+.2%}" if ratio is not None else "")
        + f", {sum(rec['ring'][str(N)]['seconds'] for N in need):.1f} s")
    return rec


def _worker(args):
    p, root, sizes, n = args[:4]
    open_sizes = args[4] if len(args) > 4 else ()
    try:
        rec = run_point(p, Path(root), sizes, n, open_sizes=open_sizes)
        return pname(p), True, sum(rec["ring"][str(N)]["seconds"] for N in sizes if str(N) in rec["ring"])
    except Exception as e:                                       # noqa: BLE001 (logged, counted)
        import traceback
        traceback.print_exc()
        return pname(p), False, repr(e)


def estimate(p: dict, sizes=SIZES) -> float:
    return sum(p["D"] * S_LAYER.get(N, 0.11 * 4 ** ((N - 16) / 2)) * N_INPUTS / 64 for N in sizes)


# --------------------------------------------------------------------------- commands

def _sel(args) -> list[dict]:
    ps = points(tuple(int(v) for v in args.times.split(",")), tuple(args.methods.split(",")))
    if getattr(args, "only", None):
        rx = re.compile(args.only)
        ps = [p for p in ps if rx.search(pname(p))]
    return ps


def _sizes(args) -> tuple[int, ...]:
    return tuple(int(v) for v in args.sizes.split(","))


def _done(root: Path, p: dict, sizes, open_sizes=()) -> bool:
    rec = _read(result_path(root, p))
    return (bool(rec) and rec.get("version") == VERSION and all(str(N) in rec["ring"] for N in sizes)
            and all(str(N) in rec.get("open", {}) for N in open_sizes))


def cmd_list(args) -> int:
    root = out_root(args.out)
    sizes = _sizes(args)
    ps = _sel(args)
    by: dict = {}
    for p in ps:
        by.setdefault((p["t"], p["field"], p["method"]), []).append(p)
    tot = 0.0
    print(f"{len(ps)} points, results in {root / 'ring'}")
    for (t, f, m), sel in sorted(by.items()):
        est = sum(estimate(p, sizes) for p in sel)
        tot += est
        done = sum(_done(root, p, sizes) for p in sel)
        print(f"  t = {t:2d}  h_x = {f:3s}  {m:8s}: {len(sel):3d} points ({done} done), D up to "
              f"{max(p['D'] for p in sel):3d}, {sum(p['eval_file'] is not None for p in sel)} with c(100), "
              f"estimate {est / 60:.1f} min on one thread")
    print(f"estimate: {tot / 3600:.2f} h on one thread (login-node speeds), plus the targets")
    return 0


def cmd_targets(args) -> int:
    root = out_root(args.out)
    for t in (int(v) for v in args.times.split(",")):
        for field in FIELDS:
            for N in _sizes(args):
                make_targets(root, field, t, N, N_INPUTS, threads=args.threads, check=not args.no_check)
    return 0


def cmd_point(args) -> int:
    root = out_root(args.out)
    rec = run_point(find_point(args.name), root, _sizes(args))
    print(json.dumps({k: v for k, v in rec.items() if k != "ring"}, default=_default)[:2000])
    return 0


def cmd_run(args) -> int:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp
    root = out_root(args.out)
    sizes = _sizes(args)
    open_sizes = tuple(int(v) for v in args.open_sizes.split(",")) if args.open_sizes else ()
    ps = [p for p in _sel(args) if not _done(root, p, sizes, open_sizes)]
    order = {m: i for i, m in enumerate(METHODS)}
    ps.sort(key=lambda p: (p["t"], p["field"], order[p["method"]], -p["D"]))
    _log(f"ti ring run on {socket.gethostname()}: {len(ps)} points to do at N = {sizes}, {args.workers} workers, "
         f"results in {root / 'ring'}; estimate {sum(estimate(p, sizes) for p in ps) / 3600 / args.workers:.2f} h wall")
    for t in sorted({p["t"] for p in ps}):
        for field in FIELDS:
            if any(p["t"] == t and p["field"] == field for p in ps):
                for N in sizes:
                    make_targets(root, field, t, N, N_INPUTS, threads=args.threads, check=not args.no_check)
                for N in open_sizes:
                    make_targets(root, field, t, N, N_INPUTS, threads=args.threads, check=not args.no_check,
                                 bc="open")
    _log("targets ready")
    failed = []
    t0 = time.time()
    done_s = 0.0
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as ex:
        futs = [ex.submit(_worker, (p, str(root), sizes, N_INPUTS, open_sizes)) for p in ps]
        for i, f in enumerate(as_completed(futs)):
            name, ok, info = f.result()
            if not ok:
                failed.append(name)
                _log(f"FAILED {name}: {info}")
            else:
                done_s += info
            if (i + 1) % 25 == 0:
                _log(f"progress: {i + 1} of {len(ps)} points, {time.time() - t0:.0f} s wall, "
                     f"{done_s:.0f} s of point time")
    _log(f"ti ring run done: {len(ps) - len(failed)} of {len(ps)} points" + (f"; failed: {failed}" if failed else ""))
    return 1 if failed else 0


# --------------------------------------------------------------------------- the table

def collect(root: Path, times=TIMES, methods=METHODS) -> list[dict]:
    out = []
    for p in sorted((Path(root) / "ring").glob("*/t*/*.json")):
        rec = _read(p)
        if rec and rec.get("version") == VERSION and rec["point"]["t"] in times and rec["point"]["method"] in methods:
            out.append(rec)
    return out


def _ratio(a: float, sa: float, b: float, sb: float) -> tuple[float, float]:
    """a/b - 1 and its standard error from independent relative errors."""
    r = a / b
    return r - 1.0, r * math.sqrt((sa / a) ** 2 + (sb / b) ** 2)


def rows(recs: list[dict]) -> list[dict]:
    out = []
    for rec in recs:
        p, R, o = rec["point"], rec["ring"], rec["open100"] or {}
        row = {"name": rec["name"], "method": p["method"], "field": p["field"], "t": p["t"], "stem": p["stem"],
               "D": p["D"], "c100": o.get("c_site"), "c100_se": o.get("c_se"), "C": {}, "C_se": {}}
        for N, r in R.items():
            row["C"][int(N)], row["C_se"][int(N)] = r["C"], r["C_se"]
        C, S = row["C"], row["C_se"]
        if row["c100"] and 16 in C:
            row["r100"], row["r100_se"] = _ratio(row["c100"], row["c100_se"] or 0.0, C[16], S[16])
        if 16 in C and 14 in C:
            row["r16"], row["r16_se"] = _ratio(C[16], S[16], C[14], S[14])
        if 14 in C and 12 in C:
            row["r14"], row["r14_se"] = _ratio(C[14], S[14], C[12], S[12])
        op = (rec.get("open") or {}).get("16")
        if op and "16" in R:
            cr, co = per_site(np.array(R["16"]["F"]), 16), per_site(np.array(op["F"]), 16)
            n = cr.size
            row["Co16"] = float(co.mean())
            row["rb16"] = float(co.mean() / cr.mean() - 1)                        # paired: same inputs
            row["rb16_se"] = float((co - cr).std(ddof=1) / math.sqrt(n) / cr.mean())
            row["b_sites"] = 16 * row["rb16"]                                      # boundary excess, in sites of c
            if row["c100"]:
                pred = float(cr.mean()) * (1 + row["b_sites"] / 100)               # c(100) = c_ring + b c / 100
                row["r100p"], row["r100p_se"] = _ratio(row["c100"], row["c100_se"] or 0.0, pred, S[16])
        out.append(row)
    return out


def _stat(v: list[float]) -> str:
    if not v:
        return "-"
    return f"{np.median(v):+.1%} ({min(v):+.1%} to {max(v):+.1%})"


def cmd_table(args) -> int:
    root = out_root(args.out)
    times = tuple(int(v) for v in args.times.split(","))
    R = rows(collect(root, times, tuple(args.methods.split(","))))
    if not R:
        raise SystemExit(f"no results under {root / 'ring'}")
    tol = args.tol
    print(f"# Ring per-site infidelity C(N) (exact, 64 random product inputs per N, seed 5151 + N) against "
          f"c(N = 100, open chain, MPS, 8 inputs)\n")
    print(f"{len(R)} points, results in {root / 'ring'}. Ratios as median (range); se = median standard error "
          f"of the ratio from the per-input values (sampling noise; the inputs differ between sizes).\n")
    print("| h_x | t | method | n | c100/C(16) - 1 | se | C(16)/C(14) - 1 | se | C(14)/C(12) - 1 | se | "
          f"> {tol:.0%} |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    outl = []
    groups: dict = {}
    for r in R:
        groups.setdefault((r["field"], r["t"], r["method"]), []).append(r)
    order = {m: i for i, m in enumerate(METHODS)}
    for (f, t, m), sel in sorted(groups.items(), key=lambda kv: (-float(kv[0][0]), kv[0][1], order[kv[0][2]])):
        cols = []
        nbad = 0
        for key in ("r100", "r16", "r14"):
            v = [r[key] for r in sel if key in r]
            se = [r[key + "_se"] for r in sel if key in r]
            cols += [_stat(v), f"{np.median(se):.1%}" if se else "-"]
        for r in sel:
            bad = [k for k in ("r100", "r16", "r14") if k in r and abs(r[k]) > tol]
            if bad:
                nbad += 1
                outl.append((r, bad))
        print(f"| {f} | {t} | {m} | {len(sel)} | " + " | ".join(cols) + f" | {nbad} |")
    print(f"\nOutliers (|ratio| > {tol:.0%}); z = ratio / se:\n")
    print("| point | D | c100 | C(12) | C(14) | C(16) | c100/C16 - 1 (z) | C16/C14 - 1 (z) | C14/C12 - 1 (z) |")
    print("|---|---|---|---|---|---|---|---|---|")

    def cell(r, k):
        if k not in r:
            return "-"
        z = r[k] / r[k + "_se"] if r[k + "_se"] else math.inf
        return f"{r[k]:+.1%} ({z:+.1f})"
    for r, bad in sorted(outl, key=lambda x: (-float(x[0]["field"]), x[0]["t"], x[0]["method"], x[0]["stem"])):
        C = r["C"]
        print(f"| {r['name']} | {r['D']} | {r['c100']:.2e} | " if r["c100"] else f"| {r['name']} | {r['D']} | - | ",
              end="")
        print(" | ".join(f"{C[N]:.2e}" if N in C else "-" for N in (12, 14, 16)) + " | "
              + " | ".join(cell(r, k) for k in ("r100", "r16", "r14")) + " |")
    if any("rb16" in r for r in R):
        print("\nThe open chain at N = 16 on the same 64 inputs as the ring (paired, the circuit's edge gates as at "
              "N = 100): C_open(16)/C_ring(16) - 1 = b/16 with b the boundary excess in sites of c; the open chain "
              "at N = 100 then predicts c100 = C_ring(16) (1 + b/100).\n")
        print("| h_x | t | method | n | C_open(16)/C_ring(16) - 1 | se (paired) | b (sites) | predicted c100/C16 - 1 "
              "| measured c100/C16 - 1 | c100/predicted - 1 | se |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for (f, t, m), sel in sorted(groups.items(), key=lambda kv: (-float(kv[0][0]), kv[0][1], order[kv[0][2]])):
            sel = [r for r in sel if "rb16" in r]
            if not sel:
                continue
            rb = [r["rb16"] for r in sel]
            b = [r["b_sites"] for r in sel]
            print(f"| {f} | {t} | {m} | {len(sel)} | {_stat(rb)} | {np.median([r['rb16_se'] for r in sel]):.2%} | "
                  f"{np.median(b):+.2f} ({min(b):+.2f} to {max(b):+.2f}) | {_stat([x / 100 for x in b])} | "
                  f"{_stat([r['r100'] for r in sel if 'r100' in r])} | {_stat([r['r100p'] for r in sel if 'r100p' in r])} | "
                  f"{np.median([r['r100p_se'] for r in sel if 'r100p' in r]):.1%} |")
    if args.json:
        _save_json(Path(args.json), {"rows": R, "tol": tol})
        print(f"\nwrote {args.json}")
    return 0


# --------------------------------------------------------------------------- the infinite chain (snapevo.imps)

IMPS_SEED0 = 6161                         # unit-cell inputs: product_vectors(L_u, n, 6161 + L_u)
IMPS_TARGETS = (("snap", 1e-6), ("baseline", 1e-5))   # representatives: c(N = 100) nearest these
IMPS_RING_N = 16                          # the paired ring check: the same unit cells placed periodically


def imps_points(times=(1, 2, 4), targets=IMPS_TARGETS) -> list[dict]:
    """Per field and t, the sweep's SNAP point with c(N = 100) nearest 1e-6 and the baseline point nearest 1e-5."""
    out = []
    for t in times:
        for field in FIELDS:
            for method, target in targets:
                best = None
                for p in points((t,), (method,)):
                    if p["field"] != field or not p.get("eval_file"):
                        continue
                    o = _open100(_read(Path(p["eval_file"])))
                    if o and o["c_site"] > 0:
                        d = abs(math.log(o["c_site"] / target))
                        if best is None or d < best[0]:
                            best = (d, p)
                if best:
                    out.append(best[1])
    return out


def imps_path(root: Path, field: str, t, L: int, cutoff: float) -> Path:
    return Path(root) / "imps" / f"{TAGS[field]}_t{float(t):g}_Lu{L}_c{cutoff:.0e}.json"


def _imps_task(args) -> dict:
    """Every representative point of one (field, t) on the n unit-cell inputs of L sites, at one cutoff."""
    field, t, L, n, pts, root, dt, cutoff, chi_max, n_check, check_cutoff = args
    from snapevo import imps as im
    from snapevo.opnorm import Propagator
    root = Path(root)
    path = imps_path(root, field, t, L, cutoff)
    old = _read(path) or {}
    model = _model(field)
    seed = IMPS_SEED0 + L
    vecs = im.unit_cell_inputs(L, n, seed)
    rec = {"version": VERSION, "field": field, "t": float(t), "L": L, "seed": seed, "n": n, "dt": dt,
           "cutoff": cutoff, "chi_max": chi_max, "inputs": f"circuits.product_vectors({L}, {n}, {seed})",
           "refs": [], "points": dict(old.get("points", {}))}
    t0 = time.time()
    refs = []
    for k, v in enumerate(vecs):
        cache = root / "imps_ref" / f"{TAGS[field]}_t{float(t):g}_Lu{L}_s{seed}_k{k}_dt{dt:g}_c{cutoff:.0e}_x{chi_max}.npz"
        fresh = not cache.exists()
        ref, meta = im.reference_state(model, float(t), v, dt, chi_max, cutoff, cache)
        refs.append(ref)
        rec["refs"].append({"k": k, "chi_peak": ref.chi_peak, "chi_end": ref.chi, "discarded": ref.discarded,
                            "seconds": meta.get("seconds"), "sublayers": meta.get("sublayers"), "cached": not fresh})
    # the paired ring: the same unit cells placed periodically on 16 sites, exact e^{-iHt}
    N = IMPS_RING_N
    ring_ok = N % L == 0
    if ring_ok:
        X = np.stack([im.ring_state(v, N) for v in vecs], axis=1)
        V = Propagator(model, N, "ring", float(t), threads=1).apply(X, -1.0)
    for p in pts:
        name = pname(p)
        if name in rec["points"] and len(rec["points"][name]["rows"]) == n:
            continue
        lay = circuit(p)
        rows = []
        Cring = per_site(np.abs(np.einsum("ij,ij->j", V.conj(), apply_ring(lay, X, N))) ** 2, N) if ring_ok else None
        for k, v in enumerate(vecs):
            r = im.c_inf(lay, model, float(t), v, dt, chi_max, cutoff, ref=refs[k])
            row = {"k": k, "c_inf": r["c_inf"], "residual": r["residual"], "gap": r["gap"], "dim": r["dim"],
                   "method": r["method"], "chi_u": r["chi_u"], "chi_v": r["chi_v"], "discarded_u": r["discarded_u"],
                   "discarded_v": r["discarded_v"], "seconds_circuit": r["seconds_circuit"],
                   "seconds_fid": r["seconds"], "lambda_aa": abs(r["lambda_aa"]), "lambda_bb": abs(r["lambda_bb"])}
            if ring_ok:
                row[f"C_ring{N}"] = float(Cring[k])
            if k < n_check and check_cutoff:
                vc, _ = im.reference_state(model, float(t), v, dt, chi_max, check_cutoff, root / "imps_ref" /
                                           f"{TAGS[field]}_t{float(t):g}_Lu{L}_s{seed}_k{k}_dt{dt:g}_c{check_cutoff:.0e}_x{chi_max}.npz")
                rc = im.c_inf(lay, model, float(t), v, dt, chi_max, check_cutoff, ref=vc)
                row["check"] = {"cutoff": check_cutoff, "c_inf": rc["c_inf"], "rel_change": rc["c_inf"] / r["c_inf"] - 1,
                                "chi_u": rc["chi_u"], "chi_v": rc["chi_v"]}
            rows.append(row)
        c = np.array([r["c_inf"] for r in rows])
        rec["points"][name] = {"point": p, "rows": rows, "c_inf": float(c.mean()),
                               "c_inf_se": float(c.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
                               f"C_ring{N}": float(np.mean(Cring)) if ring_ok else None}
        rec["seconds"] = time.time() - t0
        _save_json(path, rec)
        q = rec["points"][name]
        _log(f"imps {name} L_u = {L}: c_inf {q['c_inf']:.3e} +- {q['c_inf_se']:.1e}"
             + (f", ring({N}, same cells) {q[f'C_ring{N}']:.3e} ({q['c_inf'] / q[f'C_ring{N}'] - 1:+.2e})" if ring_ok else "")
             + f", chi {max(r['chi_u'] for r in rows)}/{max(r['chi_v'] for r in rows)}, "
               f"{sum(r['seconds_circuit'] + r['seconds_fid'] for r in rows):.1f} s")
    rec["seconds"] = time.time() - t0
    _save_json(path, rec)
    return {"field": field, "t": t, "L": L, "seconds": rec["seconds"],
            "ref_seconds": sum((r["seconds"] or 0) for r in rec["refs"] if not r["cached"])}


def cmd_imps(args) -> int:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp
    root = out_root(args.out)
    times = tuple(int(v) for v in args.times.split(","))
    pts = imps_points(times)
    if args.only:
        rx = re.compile(args.only)
        pts = [p for p in pts if rx.search(pname(p))]
    Ls = tuple(int(v) for v in args.Lu.split(","))
    _log(f"imps on {socket.gethostname()}: {len(pts)} points ({', '.join(pname(p) for p in pts)}), L_u = {Ls}, "
         f"{args.samples} inputs each, dt {args.dt}, cutoff {args.cutoff:.0e}, chi_max {args.chi_max}, "
         f"{args.workers} workers")
    tasks = []
    for t in times:
        for field in FIELDS:
            sel = [p for p in pts if p["t"] == t and p["field"] == field]
            for L in Ls:
                if sel:
                    tasks.append((field, t, L, args.samples, sel, str(root), args.dt, args.cutoff, args.chi_max,
                                  args.n_check, args.check_cutoff))
    tasks.sort(key=lambda a: (a[1], -a[2]))
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(_imps_task, a): a for a in tasks}
        for f in as_completed(futs):
            a = futs[f]
            try:
                r = f.result()
                _log(f"imps task h_x = {a[0]} t = {a[1]} L_u = {a[2]} done: {r['seconds']:.0f} s "
                     f"(references {r['ref_seconds']:.0f} s)")
            except Exception as e:                                # noqa: BLE001
                _log(f"FAILED imps task h_x = {a[0]} t = {a[1]} L_u = {a[2]}: {e!r}")
    return 0


def cmd_imps_table(args) -> int:
    root = out_root(args.out)
    recs = [r for r in (_read(p) for p in sorted((root / "imps").glob("*.json"))) if r]
    if not recs:
        raise SystemExit(f"no results under {root / 'imps'}")
    ring = {r["name"]: r for r in collect(root)}
    by: dict = {}
    for rec in recs:
        for name, q in rec["points"].items():
            by.setdefault((name, rec["cutoff"]), {})[rec["L"]] = (rec, q)
    print("# Infinite chain (iTEBD, mixed transfer matrix) against the ring and the 100-site open chain\n")
    print("c_inf per L_u: mean over the inputs (standard error). ring16 same cells: C(16) of the same unit cells "
          "placed periodically (exact), as c_inf/C - 1 over all inputs. C16 random: the 64 random inputs of the "
          "ring table. c100: the open chain, 8 inputs.\n")
    print("| point | D | cutoff | c_inf L_u = 2 | c_inf L_u = 4 | c_inf L_u = 8 | c_inf all | c_inf/ring16 same cells - 1 "
          "| C16 random | c_inf/C16 - 1 | c100 | c100/c_inf - 1 | cutoff check | chi (circuit/ref) | s per input |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (name, cut), byL in sorted(by.items(), key=lambda kv: (kv[0][0].split(":")[2], kv[0][0])):
        cells, allc, pair, checks, chis, secs = [], [], [], [], [], []
        D = None
        for L in (2, 4, 8):
            if L not in byL:
                cells.append("-")
                continue
            rec, q = byL[L]
            D = q["point"]["D"]
            cells.append(f"{q['c_inf']:.3e} ({q['c_inf_se'] / q['c_inf']:.0%})")
            for r in q["rows"]:
                allc.append(r["c_inf"])
                if "C_ring16" in r:
                    pair.append(r["c_inf"] / r["C_ring16"] - 1)
                if "check" in r:
                    checks.append(abs(r["check"]["rel_change"]))
                chis.append((r["chi_u"], r["chi_v"]))
                secs.append(r["seconds_circuit"] + r["seconds_fid"])
        ca = float(np.mean(allc))
        rg = ring.get(name)
        C16 = rg["ring"]["16"]["C"] if rg and "16" in rg["ring"] else None
        o = (rg or {}).get("open100") or {}
        c100 = o.get("c_site")
        pr = f"{np.median(pair):+.1e} ({min(pair):+.1e} to {max(pair):+.1e})" if pair else "-"
        print(f"| {name} | {D} | {cut:.0e} | " + " | ".join(cells) + f" | {ca:.3e} | {pr} | "
              + (f"{C16:.3e} | {ca / C16 - 1:+.1%}" if C16 else "- | -") + " | "
              + (f"{c100:.3e} | {c100 / ca - 1:+.1%}" if c100 else "- | -") + " | "
              + (f"{max(checks):.1e}" if checks else "-") + f" | {max(c[0] for c in chis)}/{max(c[1] for c in chis)} | "
              f"{np.median(secs):.2f} |")
    print("\nReferences (iTEBD of mps.reference_circuit), per (h_x, t, L_u):\n")
    print("| h_x | t | L_u | cutoff | inputs | chi peak | sublayers | seconds per input (median, max) | discarded (max) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for rec in sorted(recs, key=lambda r: (-float(r["field"]), r["t"], r["L"], r["cutoff"])):
        s = [r["seconds"] for r in rec["refs"] if r["seconds"] is not None]
        print(f"| {rec['field']} | {rec['t']:g} | {rec['L']} | {rec['cutoff']:.0e} | {len(rec['refs'])} | "
              f"{max(r['chi_peak'] for r in rec['refs'])} | {rec['refs'][0]['sublayers']} | "
              f"{np.median(s):.2f}, {max(s):.2f} | {max(r['discarded'] for r in rec['refs']):.1e} |")
    return 0


# --------------------------------------------------------------------------- the reference step of the N = 100 runs

def cmd_refcheck(args) -> int:
    """C_ring(16) of the tightest points against mps.reference_circuit at the 100-site runs' steps, not exact V.

    The 100-site evaluations compare with the order-4 product formula at step 0.2 (the sweep) or 0.1 (the
    chain arm), not with e^{-iHt}. The reference's own per-site error c_ref is small, but it enters the measured
    fidelity at the amplitude level: c_meas = c + c_ref - 2 rho sqrt(c c_ref), |rho| <= 1. On the ring the same
    inputs can be compared against both, which measures that shift for each point directly.
    """
    from snapevo.mps import reference_circuit
    root = out_root(args.out)
    N = 16
    n = N_INPUTS
    seed = SEED0 + N
    out_dir = root / "refcheck"
    times = tuple(int(v) for v in args.times.split(","))
    steps = tuple(float(v) for v in args.steps.split(","))
    rows = []
    for t in times:
        for field in FIELDS:
            sel = []
            for rec in collect(root, (t,), tuple(args.methods.split(","))):
                p = rec["point"]
                c16 = rec["ring"].get("16", {}).get("C")
                if p["field"] == field and c16 is not None and c16 <= args.max_c:
                    sel.append(rec)
            if not sel:
                continue
            V = _targets(root, field, t, N, n)
            from snapevo.circuits import product_vectors
            vecs = product_vectors(N, n, seed)
            R = {}
            for dt in steps:
                t0 = time.time()
                circ = reference_circuit(_model(field), float(t), dt)
                Y = np.concatenate([apply_ring(circ, input_block(N, seed, k0, min(n, k0 + 32), vecs), N)
                                    for k0 in range(0, n, 32)], axis=1)
                cref = per_site(np.abs(np.einsum("ij,ij->j", V.conj(), Y)) ** 2, N)
                R[dt] = Y
                _log(f"refcheck h_x = {field} t = {t}: reference step {dt}: c_ref {cref.mean():.2e} "
                     f"({circ.depth} sublayers, {time.time() - t0:.0f} s)")
                rows.append({"field": field, "t": t, "kind": "reference", "dt": dt, "c_ref": float(cref.mean())})
            for rec in sel:
                p = rec["point"]
                lay = circuit(p)
                Y = np.concatenate([apply_ring(lay, input_block(N, seed, k0, min(n, k0 + 32), vecs), N)
                                    for k0 in range(0, n, 32)], axis=1)
                row = {"field": field, "t": t, "kind": "point", "name": rec["name"], "D": p["D"],
                       "C_exact": float(per_site(np.abs(np.einsum("ij,ij->j", V.conj(), Y)) ** 2, N).mean()),
                       "c100": (rec["open100"] or {}).get("c_site"), "dt100": (rec["open100"] or {}).get("dt")}
                for dt, W in R.items():
                    row[f"C_ref{dt:g}"] = float(per_site(np.abs(np.einsum("ij,ij->j", W.conj(), Y)) ** 2, N).mean())
                rows.append(row)
                _log(f"refcheck {rec['name']}: C(16) exact {row['C_exact']:.3e}, "
                     + ", ".join(f"vs step {dt:g} {row[f'C_ref{dt:g}'] / row['C_exact'] - 1:+.2%}" for dt in R)
                     + (f"; c100 (step {row['dt100']}) / C16 - 1 {row['c100'] / row['C_exact'] - 1:+.2%}"
                        if row["c100"] else ""))
    _save_json(out_dir / "refcheck.json", {"N": N, "inputs": f"product_vectors({N}, {n}, {seed})", "rows": rows,
                                           "when": time.strftime("%Y-%m-%d %H:%M:%S")})
    return 0


# --------------------------------------------------------------------------- periodic inputs on the ring

PERIODIC_SEED = 7171                      # 16-site inputs product_vectors(16, n, 7171); L_u cells are their first sites


def _periodic_task(args) -> dict:
    """C(16) on the ring for n random 16-site inputs and for their first L_u sites repeated, every L_u, paired."""
    field, t, pts, root, n, Ls, blk = args
    from snapevo.circuits import product_vectors
    from snapevo.opnorm import Propagator
    N = 16
    root = Path(root)
    path = root / "periodic" / f"{TAGS[field]}_t{float(t):g}.json"
    t0 = time.time()
    vecs = product_vectors(N, n, PERIODIC_SEED)
    prop = Propagator(_model(field), N, "ring", float(t), threads=1)
    lays = {pname(p): circuit(p) for p in pts}
    c = {name: {L: np.empty(n) for L in Ls} for name in lays}
    sec = {"targets": 0.0, "circuits": 0.0}
    for k0 in range(0, n, blk):
        k1 = min(n, k0 + blk)
        for L in Ls:
            per = np.array([[v[i % L] for i in range(N)] for v in vecs[k0:k1]])
            X = input_block(N, 0, 0, k1 - k0, per)
            a = time.time()
            V = prop.apply(X, -1.0)
            sec["targets"] += time.time() - a
            a = time.time()
            for name, lay in lays.items():
                Y = apply_ring(lay, X, N)
                c[name][L][k0:k1] = per_site(np.abs(np.einsum("ij,ij->j", V.conj(), Y)) ** 2, N)
            sec["circuits"] += time.time() - a
    out = {"version": VERSION, "field": field, "t": float(t), "N": N, "n": n, "seed": PERIODIC_SEED, "Ls": list(Ls),
           "inputs": f"product_vectors({N}, {n}, {PERIODIC_SEED}); L_u cells: sites 0..L_u-1 of each, repeated",
           "points": {}, "seconds": time.time() - t0, "split": sec}
    for name in lays:
        base = c[name][N]
        row = {"D": next(p["D"] for p in pts if pname(p) == name), "c": {str(L): c[name][L].tolist() for L in Ls}, "mean": {}, "bias": {}}
        for L in Ls:
            row["mean"][str(L)] = float(c[name][L].mean())
            d = c[name][L] - base
            row["bias"][str(L)] = {"ratio": float(c[name][L].mean() / base.mean() - 1),
                                   "se": float(d.std(ddof=1) / math.sqrt(n) / base.mean())}
        out["points"][name] = row
    _save_json(path, out)
    return {"field": field, "t": t, "seconds": out["seconds"]}


def cmd_periodic(args) -> int:
    """Is a periodic unit-cell input an unbiased stand-in for random product inputs? On the 16-site ring (exact),
    C of n random inputs against C of the same inputs' first L_u sites repeated, L_u = 2, 4, 8 (paired)."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp
    root = out_root(args.out)
    times = tuple(int(v) for v in args.times.split(","))
    pts = imps_points(times)
    Ls = tuple(int(v) for v in args.Lu.split(",")) + (16,)
    tasks = []
    for t in times:
        for field in FIELDS:
            sel = [p for p in pts if p["t"] == t and p["field"] == field]
            if sel and not (root / "periodic" / f"{TAGS[field]}_t{float(t):g}.json").exists():
                tasks.append((field, t, sel, str(root), args.samples, Ls, 32))
    tasks.sort(key=lambda a: a[1])
    _log(f"periodic on {socket.gethostname()}: {len(tasks)} (field, t) tasks, {args.samples} inputs, L_u = {Ls}, "
         f"{args.workers} workers")
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(_periodic_task, a): a for a in tasks}
        for f in as_completed(futs):
            a = futs[f]
            try:
                r = f.result()
                _log(f"periodic h_x = {a[0]} t = {a[1]} done: {r['seconds']:.0f} s")
            except Exception as e:                                # noqa: BLE001
                _log(f"FAILED periodic h_x = {a[0]} t = {a[1]}: {e!r}")
    recs = [r for r in (_read(q) for q in sorted((root / "periodic").glob("*.json"))) if r]
    print("\n| point | D | C(16) random | " + " | ".join(f"L_u = {L}: C/C_random - 1 (se)" for L in Ls if L != 16)
          + " |")
    print("|---|---|---|" + "---|" * (len(Ls) - 1))
    for rec in sorted(recs, key=lambda r: (r["t"], -float(r["field"]))):
        for name, row in rec["points"].items():
            print(f"| {name} | {row['D']} | {row['mean']['16']:.3e} | "
                  + " | ".join(f"{row['bias'][str(L)]['ratio']:+.1%} ({row['bias'][str(L)]['se']:.1%})"
                               for L in Ls if L != 16) + " |")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ti_ring_eval.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, sizes=True):
        p.add_argument("--out", default=None, help="results root (default $TI_OUT, else $SCRATCH/snapevo-scratch/ti)")
        p.add_argument("--times", default=",".join(map(str, TIMES)))
        p.add_argument("--methods", default=",".join(METHODS))
        if sizes:
            p.add_argument("--sizes", default=",".join(map(str, SIZES)))

    p = sub.add_parser("list")
    common(p)
    p.add_argument("--only", default=None)
    p = sub.add_parser("targets")
    common(p)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--no-check", action="store_true", dest="no_check")
    p = sub.add_parser("run")
    common(p)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--open-sizes", default="", dest="open_sizes",
                   help="also the open chain at these sizes, same inputs (e.g. 16)")
    p.add_argument("--threads", type=int, default=4, help="Python threads for the targets")
    p.add_argument("--only", default=None, help="regex on point names")
    p.add_argument("--no-check", action="store_true", dest="no_check")
    p = sub.add_parser("point")
    common(p)
    p.add_argument("name")
    p = sub.add_parser("table")
    common(p, sizes=False)
    p.add_argument("--tol", type=float, default=0.05)
    p.add_argument("--json", default=None)
    p = sub.add_parser("imps")
    p.add_argument("--out", default=None)
    p.add_argument("--times", default="1,2,4")
    p.add_argument("--Lu", default="2,4,8")
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--cutoff", type=float, default=1e-14)
    p.add_argument("--chi-max", type=int, default=2048, dest="chi_max")
    p.add_argument("--check-cutoff", type=float, default=1e-12, dest="check_cutoff",
                   help="second cutoff for the first --n-check inputs of each L_u (0: none)")
    p.add_argument("--n-check", type=int, default=2, dest="n_check")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--only", default=None, help="regex on point names")
    p = sub.add_parser("imps-table")
    p.add_argument("--out", default=None)
    p = sub.add_parser("periodic")
    p.add_argument("--out", default=None)
    p.add_argument("--times", default="1,2,4,6,8,10")
    p.add_argument("--Lu", default="2,4,8")
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p = sub.add_parser("refcheck")
    p.add_argument("--out", default=None)
    p.add_argument("--times", default="1,2,4")
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--steps", default="0.2,0.1")
    p.add_argument("--max-c", type=float, default=1e-6, dest="max_c")
    args = ap.parse_args(argv)
    return {"list": cmd_list, "targets": cmd_targets, "run": cmd_run, "point": cmd_point, "table": cmd_table,
            "imps": cmd_imps, "imps-table": cmd_imps_table, "refcheck": cmd_refcheck,
            "periodic": cmd_periodic}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
