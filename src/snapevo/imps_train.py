"""Train the translation-invariant brickwork on the infinite chain (2-site unit cell) by trust-region Newton.

Models: XXZ chain, Delta = 0.25, J = 1, spin units, at h_x = 1.6 (cell xxz+x) and h_x = 0 (cell xxz+stag).

The method of Gibbs and Cincio (arXiv:2506.02105, Secs. II-III; L. Cincio, 23 Oct 2026: "We did the training on
infinite systems. By maximizing the overlap per lattice site computed from mixed transfer matrix"; "You should be
fine with 2-site"), with our optimizer:

Inputs. Random iMPS of bond dimension 2 on the 2-site cell (imps.random_imps: random isometries from the SVD of a
complex Gaussian matrix), N_TRAIN = 4 for training and N_TEST = 100 for testing, as Gibbs and Cincio. Seeds
(TRAIN_SEED, k) and (TEST_SEED, k), disjoint. Each input's target e^{-iHt}|phi_k> is imps.reference_state
(order-4 Suzuki, step DT, iTEBD at CUTOFF and CHI_MAX), cached on disk.

Cost. For input k, lambda_k is the leading eigenvalue of the mixed transfer matrix of one 2-site cell of
<psi_k| V(theta) |phi_k>, the overlap per cell; |lambda_k|^{2/2} is the squared fidelity per site, and

    c(theta) = 1 - mean_k |lambda_k|          (per site, the convention of the rest of the pipeline)
    C3       = 1 - mean_k |lambda_k|^2        (Gibbs and Cincio's Eq. 12, per 2-site cell; about 2c)

Both states are kept right-canonical by iTEBD, so |lambda| needs no normalization beyond the truncation defect
(~ the cutoff).

Gradient (exact up to the truncation). For layer l of the circuit, |phi~_l> = U_{l-1} ... U_0 |phi> and
|psi~_l> = U_{l+1}^dag ... U_{D-1}^dag |psi> (the environment iMPS of Gibbs and Cincio, Eq. psi_tilde; both by
iTEBD at the training truncation), and lambda is the leading eigenvalue of imps.GateTransfer(phi~_l, psi~_l, U_l):
the gate layer U_l sits inside the transfer matrix, exactly. Its derivative comes from the left and right fixed
points, d lambda = l^dag (dE) r / (l^dag r) = sum(dU * G) / (l^dag r) with G = GateTransfer.env(l, r), and dU is
exact (gate_derivatives: every angle enters as a rotation, d rot(theta)/d theta = rot(theta + pi)/2). Gibbs and
Cincio difference lambda instead. The backward environments are stored, the forward one is updated layer by layer.
The value is lambda at the top layer.

Hessian. Central differences of the exact gradient, step FD_STEP = 1e-4, symmetrized (our paper's Eq. hess_fd);
its 2n gradients (n angles) times N_TRAIN inputs run as separate tasks on a process pool.

Optimizer. Trust-region Newton only (snapevo.newton_fast.newton_reuse_minimize: the full Hessian reused across
steps and refreshed when the model stops predicting, the stopping rule of chain_snap.newton_minimize with tol =
TOL_REL x c, no step cap, a checkpoint per run); the Hessian is a must (Sohan, 22 Oct 2026). optimizer="lbfgs" is
a comparison arm only (Sohan, 24 Oct 2026): newton_fast.lbfgs_hessian_minimize steps by L-BFGS-B and keeps the
Hessian for the start, the convergence test (the Newton runs' test) and the end, into imps_train*_lbfgs/.

Ladder. As the protocol: depths ascending; starts "trotter" (the order-1 Trotter brickwork of that depth), "warm"
(the previous depth padded with identity sublayers) and "random" (trotter + 0.1 rad noise seeded by D); the best
of the three is kept; stop once the floor falls by less than 1.5x over 8 sublayers or reaches TARGET = 3.3e-7, a
third of the smallest budget, 1e-6 (Sohan, 24 Oct 2026; it was 1e-10 for a 1e-9 budget, which no compression paper
approaches: Gibbs and Cincio's deepest unitary is C3 ~ 3e-7 per cell). Output
runs/<tag>/t<T>/imps_train/ladder.json in cells.json's layout (model, cell, t, N = "inf", train_set, val_set,
results with D, params, C_train, C_val), C_val being the cost on the test inputs; checkpoints in imps_train/ckpt/.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import math
import multiprocessing as mp
import os
import threading
import time
from pathlib import Path

import numpy as np

from snapevo import imps as im
from snapevo.cells import Cell, cell_for, get_cell
from snapevo.circuits import Layered
from snapevo.hamiltonian import XXZ

__all__ = ["N_TRAIN", "N_TEST", "TRAIN_SEED", "TEST_SEED", "CHI_IN", "DT", "CUTOFF", "CHI_MAX", "TOL_REL",
           "FD_STEP", "gate_derivatives", "make_input", "input_eval", "Problem", "train_imps_ladder"]

N_TRAIN, N_TEST = 4, 100
TRAIN_SEED, TEST_SEED = 7101, 7202
CHI_IN = 2
DT, CUTOFF, CHI_MAX = 0.1, 1e-14, 2048
TOL_REL = 1e-6
FD_STEP = 1e-4
STALL = (1.5, 8)
TARGET = 3.3e-7
STARTS = ("trotter", "warm", "random")
CACHE = Path(os.environ.get("SCRATCH", "/pscratch/sd/s/sohan100")) / "snapevo-scratch" / "imps_train" / "refs"


# --------------------------------------------------------------------------- pieces

def gate_derivatives(cell: Cell, angles) -> list[np.ndarray]:
    """The exact derivatives dU/d theta_k of the cell gate, one 4x4 per angle."""
    a = np.asarray(angles, dtype=float)
    out = []
    n_shift = 5 if cell.name == "xxz+x" else 3
    for k in range(n_shift):                    # each of these angles enters one rotation exactly once
        s = a.copy()
        s[k] += math.pi
        out.append(0.5 * cell.gate(s))
    if cell.name == "xxz+stag":                 # d enters Rz(d) x Rz(-d)
        from snapevo.cells import _I2, _Z, _rot
        d = float(a[3])
        dg = 0.5 * np.kron(_rot(d + math.pi, _Z, _I2), _rot(-d, _Z, _I2)) \
            - 0.5 * np.kron(_rot(d, _Z, _I2), _rot(-d + math.pi, _Z, _I2))
        out.append(cell.two_qubit_part(a) @ dg)
    return out


def make_input(kind: str, k: int, chi_max: int = CHI_MAX, cutoff: float = CUTOFF) -> im.IMPS:
    seed = TRAIN_SEED if kind == "train" else TEST_SEED
    return im.random_imps(2, CHI_IN, seed=(seed, k), chi_max=chi_max, cutoff=cutoff)


def ref_path(model: XXZ, t: float, kind: str, k: int, dt: float = DT, cutoff: float = CUTOFF,
             chi_max: int = CHI_MAX, root: Path = CACHE) -> Path:
    seed = TRAIN_SEED if kind == "train" else TEST_SEED
    return Path(root) / f"{model.tag}_t{float(t):g}_rimps{CHI_IN}_s{seed}_k{k}_dt{dt:g}_c{cutoff:.0e}_x{chi_max}.npz"


def _fresh(psi: im.IMPS, chi_max: int, cutoff: float) -> im.IMPS:
    out = psi.copy()
    out.chi_max, out.cutoff = int(chi_max), float(cutoff)
    out.discarded, out.truncations, out.updates = 0.0, 0, 0
    out.chi_peak = max(out.chi)
    return out


def input_eval(cell: Cell, layers, phi: im.IMPS, psi: im.IMPS, chi_max: int = CHI_MAX, cutoff: float = CUTOFF,
               grad: bool = True, warm: dict | None = None, key=None, tol: float = 1e-14) -> dict:
    """lambda of <psi| V |phi> per 2-site cell for the circuit ``layers`` [(parity, angles)], and d|lambda|/d angles.

    Returns lambda (top layer), |lambda|, the gradient (D, n_angles) of |lambda| if ``grad``, chi peaks, the
    discarded weights and the largest eigen-residual. ``warm`` (a dict) keeps the eigenvectors per (key, layer)
    to start ARPACK from the last ones.
    """
    D = len(layers)
    gates = [cell.gate(a) for _, a in layers]
    out = {"chi_phi": 0, "chi_psi": psi.chi_peak, "discarded_phi": 0.0, "discarded_psi": 0.0, "residual": 0.0}
    back = [None] * D
    if grad:
        cur = _fresh(psi, chi_max, cutoff)
        back[D - 1] = cur.copy()
        for l in range(D - 1, 0, -1):
            cur.apply_layer(gates[l].conj().T, layers[l][0])
            back[l - 1] = cur.copy()
        out["discarded_psi"] = cur.discarded
        out["chi_psi"] = max(out["chi_psi"], cur.chi_peak)
    fwd = _fresh(phi, chi_max, cutoff)
    g = np.zeros((D, cell.n_angles)) if grad else None
    lam = None
    for l in range(D):
        if grad or l == D - 1:
            T = im.GateTransfer(fwd, back[l] if grad else psi, gates[l], layers[l][0])
            w = (warm or {}).get((key, l))
            e = T.eig(v0r=None if w is None else w[0], v0l=None if w is None else w[1], tol=tol)
            if warm is not None:
                warm[(key, l)] = (e["r"], e["l"])
            out["residual"] = max(out["residual"], e["residual"])
            lam = e["lambda"]
            if grad:
                G = T.env(e["l"], e["r"])
                for k, dU in enumerate(gate_derivatives(cell, layers[l][1])):
                    dlam = complex(np.sum(dU.reshape(2, 2, 2, 2) * G)) / e["lr"]
                    g[l, k] = (np.conj(lam) * dlam).real / abs(lam)
                back[l] = None
        if l < D - 1:
            fwd.apply_layer(gates[l], layers[l][0])
    out.update(chi_phi=fwd.chi_peak, discarded_phi=fwd.discarded, **{"lambda": lam, "abs": abs(lam)})
    out["grad"] = g
    return out


# --------------------------------------------------------------------------- the problem, on a process pool

_W: dict = {}                                    # per worker process: the problem's inputs, references, warm starts


def _worker_init(spec: dict) -> None:
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    _W.clear()
    _W.update(spec=spec, model=XXZ(**spec["model"]), cell=get_cell(spec["cell"]), inputs={}, refs={}, warm={})


def _get(kind: str, k: int):
    s = _W["spec"]
    if (kind, k) not in _W["inputs"]:
        phi = make_input(kind, k, s["chi_max"], s["cutoff"])
        path = ref_path(_W["model"], s["t"], kind, k, s["dt"], s["ref_cutoff"], s["ref_chi_max"], Path(s["cache"]))
        psi, _ = im.reference_state(_W["model"], s["t"], phi, s["dt"], s["ref_chi_max"], s["ref_cutoff"], path)
        _W["inputs"][(kind, k)], _W["refs"][(kind, k)] = phi, psi
    return _W["inputs"][(kind, k)], _W["refs"][(kind, k)]


def _ref_task(args):
    kind, k = args
    t0 = time.time()
    _phi, psi = _get(kind, k)
    return {"kind": kind, "k": k, "chi": psi.chi_peak, "discarded": psi.discarded, "seconds": time.time() - t0}


def _eval_task(args):
    kind, k, x, grad = args
    s = _W["spec"]
    phi, psi = _get(kind, k)
    cell = _W["cell"]
    lay = Layered.brickwork(cell, x).layers
    t0 = time.time()
    r = input_eval(cell, lay, phi, psi, s["chi_max"], s["cutoff"], grad=grad, warm=_W["warm"],
                   key=(kind, k, len(lay), grad))
    r["seconds"] = time.time() - t0
    r["grad"] = None if r["grad"] is None else r["grad"].reshape(-1)
    r["lambda"] = complex(r["lambda"])
    return r


class Problem:
    """The training problem of one (model, t): cost, exact gradient and difference Hessian over the pool."""

    def __init__(self, model: XXZ, t: float, workers: int = 4, n_train: int = N_TRAIN, n_test: int = N_TEST,
                 chi_max: int = CHI_MAX, cutoff: float = CUTOFF, dt: float = DT, cache: Path = CACHE, log=print):
        self.model, self.t, self.cell = model, float(t), cell_for(model)
        self.n_train, self.n_test = int(n_train), int(n_test)
        self.spec = {"model": model.to_dict(), "cell": self.cell.name, "t": self.t, "chi_max": int(chi_max),
                     "cutoff": float(cutoff), "dt": float(dt), "ref_chi_max": int(chi_max), "ref_cutoff": float(cutoff),
                     "cache": str(cache)}
        self.log = log
        self.pool = cf.ProcessPoolExecutor(int(workers), mp_context=mp.get_context("spawn"),
                                           initializer=_worker_init, initargs=(self.spec,))
        self.stats = {"evals": 0, "grads": 0, "hessians": 0, "seconds_grad": 0.0, "seconds_hess": 0.0}
        self._lock = threading.Lock()

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)

    def build_references(self, test: bool = True) -> list[dict]:
        """Every reference e^{-iHt}|phi_k> on the pool (cached on disk), the longest first."""
        jobs = [("train", k) for k in range(self.n_train)] + ([("test", k) for k in range(self.n_test)] if test else [])
        out = list(self.pool.map(_ref_task, jobs))
        return out

    def _run(self, tasks):
        return list(self.pool.map(_eval_task, tasks))

    def value(self, x, kind: str = "train") -> dict:
        n = self.n_train if kind == "train" else self.n_test
        R = self._run([(kind, k, np.asarray(x, dtype=float), False) for k in range(n)])
        a = np.array([r["abs"] for r in R])
        return {"c": float(1 - a.mean()), "C3": float(1 - (a ** 2).mean()), "abs": a.tolist(),
                "chi_phi": max(r["chi_phi"] for r in R), "chi_psi": max(r["chi_psi"] for r in R),
                "discarded": max(r["discarded_phi"] for r in R), "residual": max(r["residual"] for r in R)}

    def cost_grad(self, x) -> tuple[float, np.ndarray]:
        t0 = time.time()
        R = self._run([("train", k, np.asarray(x, dtype=float), True) for k in range(self.n_train)])
        c = 1.0 - float(np.mean([r["abs"] for r in R]))
        g = -np.mean([r["grad"] for r in R], axis=0)
        with self._lock:
            self.stats["grads"] += 1
            self.stats["seconds_grad"] += time.time() - t0
        return c, g

    def hessian(self, x, rows=None) -> np.ndarray:
        """Central differences of the exact gradient, 2n points x n_train inputs on the pool, symmetrized."""
        t0 = time.time()
        x = np.asarray(x, dtype=float)
        n = x.size
        tasks = []
        for j in range(n):
            for sgn in (1.0, -1.0):
                xj = x.copy()
                xj[j] += sgn * FD_STEP
                tasks += [("train", k, xj, True) for k in range(self.n_train)]
        R = self._run(tasks)
        G = np.array([r["grad"] for r in R]).reshape(n, 2, self.n_train, n).mean(axis=2)   # (j, sign, n)
        H = -(G[:, 0] - G[:, 1]) / (2 * FD_STEP)
        with self._lock:
            self.stats["hessians"] += 1
            self.stats["seconds_hess"] += time.time() - t0
        return 0.5 * (H + H.T)


# --------------------------------------------------------------------------- the ladder

def _starts(cell: Cell, model: XXZ, t: float, D: int, prev) -> dict:
    na = cell.n_angles
    z0 = {"trotter": cell.trotter_angles(model, t, D)}
    if prev is not None and int(prev["D"]) < D:
        p = np.asarray(prev["params"], dtype=float).reshape(-1, na)
        z0["warm"] = np.vstack([p, np.zeros((D - p.shape[0], na))]).reshape(-1)
    z0["random"] = z0["trotter"] + 0.1 * np.random.default_rng(D).normal(size=na * D)
    return z0


def train_imps_ladder(model: XXZ, t: float, depths, runs: Path = Path("runs"), workers: int = 4,
                      n_train: int = N_TRAIN, n_test: int = N_TEST, stall=STALL, target=TARGET,
                      deadline: float | None = None, parallel_starts: bool = True, out_sub: str | None = None,
                      optimizer: str = "newton", only_start: str | None = None, hold_lock: bool = True,
                      log=print) -> Path:
    """Train every depth (ascending) on the infinite chain; resumes ladder.json and the checkpoints.

    ``out_sub`` defaults to imps_train for the default 4 training states and imps_train_n<N> otherwise, with
    _lbfgs appended for ``optimizer`` "lbfgs" (the comparison arm; "newton" is the protocol).

    Split runs (cluster/imps_split.py, one node per start): ``only_start`` trains just that start of the first
    unfinished depth to its stop, under its own lock (lock_<start>), and returns without recording the depth; a
    normal run then records it, reusing every finished start checkpoint as is (any run does: a checkpoint whose
    run stopped with "converged", "noise floor" or "stalled" from the same start point is not optimized again).
    ``hold_lock`` False skips the ladder lock, for runs under a driver that holds it."""
    from snapevo.chain_snap import DeadlineReached
    from snapevo.newton_fast import lbfgs_hessian_minimize, newton_reuse_minimize
    from snapevo.train import _stalled
    from snapevo.train_newton import _lock, _write

    if optimizer not in ("newton", "lbfgs"):
        raise ValueError(f"optimizer must be newton or lbfgs, not {optimizer!r}")
    cell = cell_for(model)
    na = cell.n_angles
    if out_sub is None:
        out_sub = ("imps_train" if n_train == N_TRAIN else f"imps_train_n{n_train}") \
            + ("_lbfgs" if optimizer == "lbfgs" else "")
    out_dir = Path(runs) / model.tag / f"t{float(t):g}" / out_sub
    ck_dir = out_dir / "ckpt"
    ck_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "ladder.json"
    header = {"model": model.to_dict(), "cell": cell.name, "t": float(t), "N": "inf", "unit_cell": 2,
              "train_set": [TRAIN_SEED, n_train], "val_set": [TEST_SEED, n_test],
              "inputs": f"imps.random_imps(2, {CHI_IN}, seed=(seed, k))", "dt": DT, "cutoff": CUTOFF,
              "chi_max": CHI_MAX, "cost": "1 - mean_k |lambda0_k|^{2/2} per site; C3 = 1 - mean |lambda0|^2 per cell",
              "optimizer": ("trust-region Newton (newton_fast.newton_reuse_minimize), difference Hessian of the exact "
                            "gradient" if optimizer == "newton" else
                            "L-BFGS-B steps (newton_fast.lbfgs_hessian_minimize, memory 10), difference Hessian of the "
                            "exact gradient at the start, for the Newton convergence test, and at the end; stalled once "
                            "1000 steps lower c by less than 1e-5 c"),
              "tol_rel": TOL_REL, "fd_step": FD_STEP, "starts": list(STARTS), "target": target,
              "stall": None if stall is None else list(stall)}
    if only_start is not None and only_start not in STARTS:
        raise ValueError(f"only_start must be one of {STARTS}, not {only_start!r}")
    lock = _lock(out_dir) if hold_lock and only_start is None else None
    start_lock = _named_lock(out_dir, f"lock_{only_start}") if only_start is not None else None
    prob = Problem(model, t, workers=workers, n_train=n_train, n_test=n_test, log=log)
    try:
        t0 = time.time()
        refs = prob.build_references(test=only_start is None)
        log(f"references: {len(refs)} built or loaded in {time.time() - t0:.0f}s, chi up to "
            f"{max(r['chi'] for r in refs)}")
        results = []
        if out.exists():
            doc = json.loads(out.read_text())
            for k in ("model", "cell", "t", "train_set", "val_set"):
                if doc.get(k) != header[k]:
                    raise ValueError(f"{out} has {k} = {doc.get(k)}, not {header[k]}")
            results = doc["results"]
        prev = None
        for D in sorted(int(d) for d in depths):
            done = [r for r in results if int(r["D"]) == D]
            if done:
                prev = done[0]
                if target is not None and prev["C_train"] <= target:
                    log(f"D={D}: floor {prev['C_train']:.2e} at or below the target (done before)")
                    break
                if stall is not None and _stalled(results, D, *stall):
                    log(f"D={D}: the ladder stopped here before (stall rule)")
                    break
                continue
            if deadline is not None and time.time() > deadline:
                log(f"deadline: stop before D = {D}")
                break
            lower = [r for r in results if int(r["D"]) < D]
            if prev is None and lower:
                prev = max(lower, key=lambda r: r["D"])
            z0 = _starts(cell, model, t, D, prev)
            t_depth = time.time()

            def run(name):
                say = (lambda m, nm=name: log(f"D={D} {nm}: {m}"))
                ck = ck_dir / f"D{D}_{name}.json"
                done_run = _finished(ck, z0[name])
                if done_run is not None:
                    say(f"reused the finished checkpoint ({done_run[1].get('stop')}, c = {done_run[1]['c_after']:.6e})")
                    return name, done_run[0], done_run[1]
                minimize = newton_reuse_minimize if optimizer == "newton" else lbfgs_hessian_minimize
                z, info = minimize(z0[name], prob.cost_grad, prob.hessian, lambda c: TOL_REL * c,
                                   log=say, checkpoint=str(ck),
                                   deadline=deadline, label=f"{optimizer} {name}")
                return name, np.asarray(z, dtype=float), info
            if only_start is not None:
                if only_start not in z0:
                    log(f"D={D}: no {only_start} start at this depth")
                    break
                try:
                    _nm, _z, info = run(only_start)
                except DeadlineReached:
                    log(f"D={D} {only_start}: deadline reached; checkpoint kept in {ck_dir}")
                    break
                log(f"D={D} {only_start}: finished, c = {info['c_after']:.6e} ({info.get('stop')}); the depth is "
                    f"recorded by the next full run")
                break
            try:
                if parallel_starts:
                    with cf.ThreadPoolExecutor(len(z0)) as ex:
                        res = list(ex.map(run, list(z0)))
                else:
                    res = [run(nm) for nm in z0]
            except DeadlineReached:
                log(f"D={D}: deadline reached during training; checkpoints kept in {ck_dir}")
                break
            name, z, info = min(res, key=lambda r: r[2]["c_after"])
            test = prob.value(z, "test") if n_test else None
            train = prob.value(z, "train")
            rec = {"D": D, "params": z.reshape(D, na).tolist(), "C_train": float(info["c_after"]),
                   "C_val": None if test is None else test["c"], "C3_train": train["C3"],
                   "C3_val": None if test is None else test["C3"], "start": name, "optimizer": optimizer,
                   "chi_phi": train["chi_phi"], "chi_psi": train["chi_psi"], "discarded": train["discarded"],
                   "residual": train["residual"], "seconds": time.time() - t_depth,
                   "runs": {nm: {"c": float(i["c_after"]), "c_start": float(i["c_before"]), "steps": int(i["steps"]),
                                 "hessians": int(i.get("hessians", 0)), "gradients": int(i.get("gradients", 0)),
                                 "stop": i.get("stop"), "seconds": float(i.get("seconds", 0.0)),
                                 "n_neg_after": i.get("n_neg_after"), "eig_min_after": i.get("eig_min_after")}
                            for nm, _z, i in res}}
            ctest = "-" if test is None else f"{test['c']:.3e}"
            log(f"D={D} best={name} C_train={rec['C_train']:.3e} C_test={ctest}"
                f" | " + ", ".join(f"{nm} {i['c_after']:.3e} ({i['steps']} it, {i.get('stop')})" for nm, _z, i in res)
                + f" | {rec['seconds']:.0f}s")
            results = [r for r in results if int(r["D"]) != D] + [rec]
            results.sort(key=lambda r: r["D"])
            _write(out, {**header, "results": results})
            prev = rec
            if target is not None and rec["C_train"] <= target:
                log(f"D={D}: floor at or below the target {target:.0e}, stopping the ladder")
                break
            if stall is not None and _stalled(results, D, *stall):
                log(f"D={D}: floor fell by less than {stall[0]:g}x over {stall[1]} sublayers, stopping the ladder")
                break
        return out
    finally:
        prob.close()
        for lk in (lock, start_lock):
            if lk is not None:
                lk.close()


# --------------------------------------------------------------------------- split runs

def _named_lock(out_dir: Path, name: str):
    import fcntl
    out_dir.mkdir(parents=True, exist_ok=True)
    fh = open(out_dir / name, "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        raise RuntimeError(f"another run holds {out_dir / name}") from None
    return fh


def _finished(ck: Path, z_in) -> tuple[np.ndarray, dict] | None:
    """The result of a finished run from its checkpoint (same start point, a final stop), else None."""
    if not ck.exists():
        return None
    try:
        doc = json.loads(ck.read_text())
    except (OSError, ValueError):
        return None
    info = doc.get("info", {})
    if info.get("stop") not in ("converged", "noise floor", "stalled") or "c_after" not in info:
        return None
    if np.asarray(doc["z_in"]).shape != np.asarray(z_in).shape or not np.array_equal(np.asarray(doc["z_in"]), z_in):
        return None
    return np.asarray(doc["z"], dtype=float), info


def ladder_status(model: XXZ, t: float, depths, runs: Path = Path("runs"), n_train: int = N_TRAIN,
                  optimizer: str = "newton", stall=STALL, target=TARGET, out_sub: str | None = None) -> dict:
    """Where a ladder stands: {"finished": bool, "why", "D": next depth, "starts": the starts that depth has}."""
    if out_sub is None:
        out_sub = ("imps_train" if n_train == N_TRAIN else f"imps_train_n{n_train}") \
            + ("_lbfgs" if optimizer == "lbfgs" else "")
    from snapevo.train import _stalled
    out = Path(runs) / model.tag / f"t{float(t):g}" / out_sub / "ladder.json"
    results = json.loads(out.read_text())["results"] if out.exists() else []
    have = False
    for D in sorted(int(d) for d in depths):
        done = [r for r in results if int(r["D"]) == D]
        if done:
            have = True
            if target is not None and done[0]["C_train"] <= target:
                return {"finished": True, "why": f"target at D={D}", "D": None, "starts": []}
            if stall is not None and _stalled(results, D, *stall):
                return {"finished": True, "why": f"stall at D={D}", "D": None, "starts": []}
            continue
        starts = [s for s in STARTS if s != "warm" or have]
        return {"finished": False, "why": None, "D": D, "starts": starts}
    return {"finished": True, "why": "depth grid exhausted", "D": None, "starts": []}
