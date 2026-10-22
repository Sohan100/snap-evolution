"""Chain workers beyond one node's processes: GPU workers, and a farm of workers on many nodes.

The open-chain cost's work is per input (chain_snap's worker functions: _cw_run for the cost and its gradient,
_cw_env_* for the curve environments), and a Newton step needs 2 len(z) gradients at once (chain_snap.fd_hessian_rows).
ChainPool runs that on G input groups x K replicas of processes on one node. Here the same worker functions run on

    GPU workers   processes bound to one GPU each, in which chain_snap's MPS kernels (_run_input, _env_states,
                  _sandwich, apply_layered, _snapshot) are swapped for mps_gpu's (CuPy) before any work;
    remote workers processes on other nodes (started by srun: ``python -m snapevo.chain_farm worker``) that
                  connect to the master over TCP (multiprocessing.connection, an authentication key) and run
                  whatever the master sends, CPU or GPU.

A Farm is a fixed set of such workers for a whole job. FarmPool leases G x K of them (blocking until they are
free), initializes each with its group's inputs and targets (chain_snap._cw_init, the targets moved to the
GPU on GPU workers), and is a ChainPool from then on (submit, rows, gather, close: close returns the workers to
the farm). Unlike ChainPool, a task that names no replica waits in its group's queue until one of the group's
workers is idle, so fast and slow workers (GPU and CPU, short and long gradients) share the work by speed; a
task whose worker is lost (its connection drops) is run again on another worker of its group, and lost
workers are no longer leased. ``use_farm(farm, ...)`` sets chain_snap.POOL_FACTORY, so every ChainProblem of the process (the
ladder's, precompute's, chain_point's compiles) leases its pool from the farm; nothing else changes, and the
CPU path is untouched when no farm is set.

Remote protocol (one connection per worker process): the master sends (task id, function, args) tuples
(functions by reference, pickled), the worker runs them in order and answers (task id, ok, result or the
exception); None ends the worker. The address file holds the master's host, port and a random key (hex),
written atomically; workers wait for it, so they can start before or after the master.
"""

from __future__ import annotations

import concurrent.futures as cf
import itertools
import json
import multiprocessing as mp
import os
import socket
import threading
import time
from multiprocessing.connection import Client, Listener
from pathlib import Path

import numpy as np

__all__ = ["Farm", "LocalFarm", "SocketFarm", "FarmPool", "use_farm", "worker_main", "gpu_setup"]


# ============================================================================ worker side

def gpu_setup(device: int | None) -> dict:
    """Bind this process to a GPU and route chain_snap's MPS kernels to mps_gpu (idempotent). Returns the device."""
    import cupy as cp

    import snapevo.chain_snap as cs
    import snapevo.mps_gpu as mg
    if device is not None:
        cp.cuda.Device(int(device)).use()
    cs._run_input = mg.run_input_gpu
    cs._env_states = mg.env_states_gpu
    cs._sandwich = mg.sandwich_gpu
    cs.apply_layered = mg.apply_layered_gpu
    cs._snapshot = mg.snapshot_gpu
    os.environ["SNAPEVO_GPU_WORKER"] = "1"
    return {"pid": os.getpid(), "host": socket.gethostname(), "device": int(cp.cuda.Device().id)}


def _farm_init(cell_name: str, inputs, targets, chi_max: int, cutoff: float, target_paths=None) -> dict:
    """chain_snap._cw_init for one pool lease; targets from ``target_paths`` (npz files) when given."""
    import snapevo.chain_snap as cs
    if target_paths is not None:
        from snapevo.mps import MPS
        targets = [MPS.load(p)[0] for p in target_paths]
    cs._cw_init(cell_name, inputs, targets, chi_max, cutoff)
    if os.environ.get("SNAPEVO_GPU_WORKER") == "1":
        from snapevo.mps_gpu import to_gpu
        cs._CW["targets"] = [to_gpu(t) for t in cs._CW["targets"]]
    return {"pid": os.getpid(), "host": socket.gethostname()}


def _farm_release() -> None:
    """Drop the lease's state (inputs, targets, environments) and the GPU's cached blocks."""
    import snapevo.chain_snap as cs
    cs._CW.clear()
    if os.environ.get("SNAPEVO_GPU_WORKER") == "1":
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()


def _local_worker_init(device: int | None, threads: int) -> None:
    import signal
    signal.signal(signal.SIGTERM, signal.SIG_IGN)               # the parent checkpoints on SIGTERM and needs us
    if device is not None:
        gpu_setup(device)


def _hello() -> dict:
    return {"pid": os.getpid(), "host": socket.gethostname(), "gpu": os.environ.get("SNAPEVO_GPU_WORKER") == "1"}


def _read_address(path: Path, wait: float) -> tuple:
    t0 = time.time()
    while True:
        try:
            doc = json.loads(Path(path).read_text())
            return (doc["host"], int(doc["port"])), bytes.fromhex(doc["key"])
        except (OSError, ValueError, KeyError):
            if time.time() - t0 > wait:
                raise TimeoutError(f"no farm address in {path} after {wait:.0f}s") from None
            time.sleep(2)


def worker_main(address_file: str, gpu: str | None = None, threads: int = 1, wait: float = 3600.0) -> int:
    """A remote worker: connect to the master, run its tasks in order until it says stop (or disconnects).

    gpu: None (CPU), "local" (the GPU SLURM_LOCALID mod the visible count), or a device number.
    """
    import signal
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = str(int(threads))
    signal.signal(signal.SIGTERM, signal.SIG_IGN)               # the master stops us; SLURM kills at the end
    if gpu is not None:
        import cupy as cp
        n = cp.cuda.runtime.getDeviceCount()
        dev = (int(os.environ.get("SLURM_LOCALID", "0")) % n) if gpu == "local" else int(gpu)
        gpu_setup(dev)
    address, key = _read_address(Path(address_file), wait)
    conn = None
    for _ in range(60):
        try:
            conn = Client(address, authkey=key)
            break
        except (ConnectionRefusedError, OSError):
            time.sleep(2)
    if conn is None:
        raise ConnectionError(f"cannot reach the farm master at {address}")
    conn.send(("hello", _hello()))
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            return 0
        if msg is None:
            conn.close()
            return 0
        tid, fn, args = msg
        try:
            res = (tid, True, fn(*args))
        except BaseException as exc:                              # noqa: BLE001 (sent to the master)
            res = (tid, False, exc)
        try:
            conn.send(res)
        except Exception as exc:                                  # noqa: BLE001 (an unpicklable result)
            conn.send((tid, False, RuntimeError(f"cannot send the result: {exc!r}")))


# ============================================================================ master side

class RemoteExecutor:
    """One remote worker process seen as a single-process executor (submit -> Future, in order)."""

    def __init__(self, conn, hello: dict):
        self.conn, self.hello = conn, hello
        self._send = threading.Lock()
        self._pending: dict[int, cf.Future] = {}
        self._ids = itertools.count()
        self._dead: BaseException | None = None
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self) -> None:
        while True:
            try:
                tid, ok, payload = self.conn.recv()
            except (EOFError, OSError) as exc:
                self._dead = ConnectionError(f"farm worker {self.hello} disconnected: {exc!r}")
                for f in list(self._pending.values()):
                    if not f.done():
                        f.set_exception(self._dead)
                self._pending.clear()
                return
            f = self._pending.pop(tid, None)
            if f is not None:
                (f.set_result if ok else f.set_exception)(payload)

    def submit(self, fn, *args) -> cf.Future:
        f: cf.Future = cf.Future()
        if self._dead is not None:
            f.set_exception(self._dead)
            return f
        tid = next(self._ids)
        self._pending[tid] = f
        try:
            with self._send:
                self.conn.send((tid, fn, args))
        except Exception as exc:                                  # noqa: BLE001
            self._pending.pop(tid, None)
            f.set_exception(exc)
        return f

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        try:
            with self._send:
                self.conn.send(None)
            self.conn.close()
        except OSError:
            pass


class Farm:
    """A fixed set of executors (each one worker process) leased to pools; size, gpu count, lease/release."""

    def __init__(self, executors: list, gpu_flags: list[bool]):
        self.executors = list(executors)
        self.gpu = list(gpu_flags)
        self._free = list(range(len(self.executors)))
        self._cv = threading.Condition()

    @property
    def size(self) -> int:
        return len(self.executors)

    def lease(self, n: int, deadline: float | None = None) -> list[int]:
        """n free workers (blocking until they are; GPU workers spread over devices round-robin)."""
        if n > self.size:
            raise ValueError(f"a pool of {n} workers on a farm of {self.size}")
        with self._cv:
            while True:
                self._free = [i for i in self._free if getattr(self.executors[i], "_dead", None) is None]
                alive = sum(getattr(e, "_dead", None) is None for e in self.executors)
                if n > alive:
                    raise RuntimeError(f"a pool of {n} workers, but only {alive} of the farm's are alive")
                if len(self._free) >= n:
                    break
                left = None if deadline is None else deadline - time.time()
                if left is not None and left <= 0:
                    raise TimeoutError("no free farm workers before the deadline")
                self._cv.wait(timeout=60 if left is None else min(60, left))
            got, self._free = self._free[:n], self._free[n:]
            return got

    def release(self, ids) -> None:
        with self._cv:
            self._free.extend(i for i in ids if i not in self._free)
            self._free.sort()
            self._cv.notify_all()

    def close(self) -> None:
        for ex in self.executors:
            try:
                ex.shutdown(wait=False, cancel_futures=True)
            except Exception:                                     # noqa: BLE001
                pass
        self.executors = []


class LocalFarm(Farm):
    """Worker processes on this node: ``per_device`` per GPU in ``devices`` (GPU farm), or ``cpus`` CPU workers."""

    def __init__(self, devices=(), per_device: int = 1, cpus: int = 0, threads: int = 1):
        from snapevo.mps_train import _threads_env
        ctx = mp.get_context("spawn")
        slots = [d for _ in range(int(per_device)) for d in devices] + [None] * int(cpus)
        with _threads_env(threads):
            exs = [cf.ProcessPoolExecutor(1, mp_context=ctx, initializer=_local_worker_init, initargs=(d, threads))
                   for d in slots]
            for f in [e.submit(_hello) for e in exs]:
                f.result()
        super().__init__(exs, [d is not None for d in slots])
        self.devices = list(slots)


class SocketFarm(Farm):
    """Workers on other nodes: listen, publish the address in ``address_file``, accept ``expected`` workers.

    Waits up to ``wait`` seconds for all of them; proceeds with those that came if at least ``minimum``.
    """

    def __init__(self, address_file, expected: int, minimum: int | None = None, wait: float = 1800.0,
                 port: int = 0, log=print):
        key = os.urandom(32)
        self.listener = Listener(("0.0.0.0", int(port)), authkey=key, backlog=4096)
        host = socket.gethostname()
        addr = {"host": host, "port": self.listener.address[1], "key": key.hex(), "pid": os.getpid()}
        p = Path(address_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(addr))
        os.chmod(tmp, 0o600)
        tmp.replace(p)
        conns, hellos = [], []
        stop = threading.Event()

        def accept():
            while len(conns) < expected and not stop.is_set():
                try:
                    c = self.listener.accept()
                except OSError:
                    return
                kind, hello = c.recv()
                conns.append(c)
                hellos.append(hello)

        th = threading.Thread(target=accept, daemon=True)
        th.start()
        t0 = time.time()
        while len(conns) < expected and time.time() - t0 < wait:
            time.sleep(1)
        stop.set()
        need = expected if minimum is None else int(minimum)
        if len(conns) < need:
            raise TimeoutError(f"{len(conns)} of {expected} farm workers connected in {wait:.0f}s")
        log(f"farm: {len(conns)} workers on {len({h['host'] for h in hellos})} nodes "
            f"({sum(h['gpu'] for h in hellos)} on GPUs), {time.time() - t0:.0f}s")
        exs = [RemoteExecutor(c, h) for c, h in zip(conns, hellos)]
        super().__init__(exs, [h["gpu"] for h in hellos])
        self.hellos = hellos

    def close(self) -> None:
        super().close()
        try:
            self.listener.close()
        except OSError:
            pass


# ============================================================================ the pool

class FarmPool:
    """chain_snap.ChainPool on leased farm workers (same attributes and methods; close() returns them)."""

    def __init__(self, farm: Farm, cell, inputs, targets, chi_max: int, cutoff: float, groups: int | None = None,
                 replicas: int = 1, target_paths=None, deadline: float | None = None):
        from snapevo.chain_snap import ChainPool
        inputs = np.asarray(inputs, dtype=complex)
        n = len(targets)
        G = max(1, min(int(groups or n), n))
        K = max(1, int(replicas))
        self.n, self.G, self.K = n, G, K
        self.groups = [list(range(g, n, G)) for g in range(G)]
        self.chi_max, self.cutoff, self.threads, self.local = int(chi_max), float(cutoff), 1, False
        self._lock = threading.Lock()
        self._load = [[0] * K for _ in range(G)]
        self.farm = farm
        self._ids = farm.lease(G * K, deadline)
        try:
            self.workers = [[farm.executors[self._ids[g * K + r]] for r in range(K)] for g in range(G)]
            futs = []
            for g, grp in enumerate(self.groups):
                tp = None if target_paths is None else [str(target_paths[k]) for k in grp]
                tg = None if tp is not None else [targets[k] for k in grp]
                for w in self.workers[g]:
                    futs.append(w.submit(_farm_init, cell.name, inputs[grp], tg, self.chi_max, self.cutoff, tp))
            for f in futs:
                f.result()
        except BaseException:
            farm.release(self._ids)
            raise
        # ChainPool's methods work on these attributes as they are
        self.rows = ChainPool.rows.__get__(self)
        self.gather = ChainPool.gather.__get__(self)
        self._queue: list[list] = [[] for _ in range(G)]           # (fn, args, future) waiting per group
        self._busy = [[0] * K for _ in range(G)]
        self._gone = [[False] * K for _ in range(G)]              # workers lost (their tasks go to the others)
        flags = {id(e): f for e, f in zip(farm.executors, farm.gpu)}
        # free workers are taken GPU first: a GPU worker finishes a task 8-30x sooner than a CPU core
        self._order = [sorted(range(K), key=lambda r, g=g: not flags.get(id(self.workers[g][r]), False))
                       for g in range(G)]

    @property
    def size(self) -> int:
        return self.G * self.K

    def submit(self, g: int, fn, *args, replica: int | None = None) -> cf.Future:
        """A task for group g: on ``replica`` if named (environments), else on the first of its workers to be free.

        Unlike ChainPool, unnamed tasks wait here and go to a worker only when it has nothing to do, so fast
        (GPU) and slow (CPU) workers, and short and long tasks, share a Hessian's gradients by speed.
        """
        if replica is not None:
            return self.workers[g][int(replica)].submit(fn, *args)
        out: cf.Future = cf.Future()
        with self._lock:
            self._queue[g].append((fn, args, out))
        self._dispatch(g)
        return out

    def _dispatch(self, g: int) -> None:
        while True:
            with self._lock:
                if not self._queue[g]:
                    return
                if all(self._gone[g]):
                    lost = self._queue[g]
                    self._queue[g] = []
                    for _fn, _a, out in lost:
                        out.set_exception(ConnectionError(f"every worker of input group {g} is lost"))
                    return
                free = [r for r in self._order[g] if self._busy[g][r] == 0 and not self._gone[g][r]]
                if not free:
                    return
                r = free[0]
                fn, args, out = self._queue[g].pop(0)
                self._busy[g][r] += 1
            try:
                inner = self.workers[g][r].submit(fn, *args)
            except BaseException as exc:                          # noqa: BLE001
                with self._lock:
                    self._busy[g][r] -= 1
                out.set_exception(exc)
                continue

            def done(f, g=g, r=r, out=out, fn=fn, args=args):
                exc = f.exception()
                with self._lock:
                    self._busy[g][r] -= 1
                    if isinstance(exc, ConnectionError):          # the worker died: run the task elsewhere
                        self._gone[g][r] = True
                        self._queue[g].insert(0, (fn, args, out))
                        exc = None
                        lost = True
                    else:
                        lost = False
                if not lost:
                    if exc is not None:
                        out.set_exception(exc)
                    else:
                        out.set_result(f.result())
                self._dispatch(g)
            inner.add_done_callback(done)

    def close(self) -> None:
        if not getattr(self, "_ids", None):
            return
        futs = [w.submit(_farm_release) for row in self.workers for w in row]
        for f in futs:
            try:
                f.result()
            except Exception:                                     # noqa: BLE001 (a dead worker stays leased)
                pass
        self.farm.release(self._ids)
        self._ids = []
        self.workers = []


def use_farm(farm: Farm, replicas: int | None = None, groups: int | None = None, target_paths_of=None,
             deadline: float | None = None):
    """Make every ChainProblem of this process lease its pool from ``farm`` (chain_snap.POOL_FACTORY).

    replicas: K per pool (default: all the farm's workers over the G groups, i.e. one pool at a time); groups:
    G (default the caller's, one input per group). target_paths_of(targets) -> paths or None lets workers load
    targets from disk instead of receiving them.
    """
    import snapevo.chain_snap as cs

    def factory(cell, inputs, targets, chi_max, cutoff, groups=None, replicas=1, threads=1, local=False):
        G = int(groups_ or groups or len(targets))
        K = int(replicas_) if replicas_ else max(1, farm.size // max(1, G))
        tp = target_paths_of(targets) if target_paths_of is not None else None
        return FarmPool(farm, cell, inputs, targets, chi_max, cutoff, groups=G, replicas=K, target_paths=tp,
                        deadline=deadline)

    groups_, replicas_ = groups, replicas
    cs.POOL_FACTORY = factory
    return factory


if __name__ == "__main__":                                         # pragma: no cover (srun entry point)
    import argparse
    ap = argparse.ArgumentParser(description="a chain_farm worker")
    ap.add_argument("role", choices=("worker",))
    ap.add_argument("--address-file", required=True, dest="address_file")
    ap.add_argument("--gpu", default=None, help="'local' (by SLURM_LOCALID) or a device number; omit for CPU")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--wait", type=float, default=3600.0)
    a = ap.parse_args()
    raise SystemExit(worker_main(a.address_file, a.gpu, a.threads, a.wait))
