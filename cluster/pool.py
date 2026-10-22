"""Run the job lists as one shared work pool, on whole nodes and on login nodes.

    python cluster/pool.py work [--cores N] [--mem-gb M] [--max-task-hours H]
    python cluster/pool.py status [-v]
    python cluster/pool.py retry ID ...

Perlmutter's preempt, debug, debug_preempt and interactive QOS hand out whole
nodes (128 cores, 512 GB), so a job array of single-process tasks would hold one
node per task. Here a worker runs on each node (pool.slurm) or login node and
pulls lines from cluster/jobs/*.txt, as many at once as its cores allow, each
pinned to its own cores. The job lists and the commands are the ones
make_jobs.py writes, unchanged.

A line waits for the lines it depends on, at the level of one model and time:
a SNAP point for its training ladder and references, a baseline evaluation for
its scan and references, and a fallback draw for the point and the baseline
evaluation of its budget. A line is done when its command exits 0. Every
command skips work whose result exists and resumes compiles from their
checkpoints, so a line cut off by a time limit or a preemption runs again from
where it stopped. A line that fails twice waits for `retry`.

A worker starts a line only if its remaining wall time covers the line's
minimum (`need` below), so the 30 min debug jobs take only short lines. A line
cut off by a deadline gets a larger minimum next time (minwall/). Writing a
huge minimum there holds a line back from every worker except one started with
`--force <list>`, which is how a changed command reaches only new workers.

State lives in cluster/logs/pool/ (ignored by git): run/ (claims with a
heartbeat), done/, fail/, tasks/ (one log per line), attempts.jsonl, and stop/
(touch stop/<host or job id> or stop/all to stop workers; their lines resume).

Workers never submit jobs. Each node job is submitted by hand, sized to the
lines that are ready (`status`), with pool.slurm on preempt (48 h, requeued
when preempted), debug or debug_preempt (30 min), or under salloc on the
interactive QOS (4 h).
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import math
import os
import re
import signal
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
JOBS = HERE / "jobs"
STATE = HERE / "logs" / "pool"
LISTS = ("train", "reference", "scan", "points", "baselines", "samples")
STALE = 300            # seconds without a heartbeat before a claim is taken over
MAX_FAILS = 2
HOST = socket.gethostname()
JOB = os.environ.get("SLURM_JOB_ID", "")
WHO = f"{HOST}:{JOB or 'login'}:{os.getpid()}"
SETUP = f"cd {REPO} && source .venv/bin/activate && export SNAPEVO_RUNS={REPO}/runs"


@dataclass
class Task:
    list: str
    line: int
    cmd: str
    model: str = ""
    t: float = 0.0
    budget: str = ""
    deps: list = field(default_factory=list)

    @property
    def id(self) -> str:
        return f"{self.list}-{self.line}"


def _opt(cmd: str, name: str) -> str:
    m = re.search(rf"--{name} (\S+)", cmd)
    return m.group(1) if m else ""


def load_tasks() -> list[Task]:
    tasks = []
    for name in LISTS:
        path = JOBS / f"{name}.txt"
        if not path.exists():
            continue
        for i, cmd in enumerate(path.read_text().splitlines(), 1):
            if cmd.strip():
                tasks.append(Task(name, i, cmd, _opt(cmd, "model"), float(_opt(cmd, "t") or 0),
                                  _opt(cmd, "budget") or _opt(cmd, "targets")))
    key = {}
    for k in tasks:
        key.setdefault((k.list, k.model, k.t, k.budget if k.list in ("points", "baselines") else ""), []).append(k.id)
    for k in tasks:
        mt = (k.model, k.t, "")
        refs = key.get(("reference", *mt), [])
        if k.list == "points":
            k.deps = key.get(("train", *mt), []) + refs
        elif k.list == "baselines":
            k.deps = key.get(("scan", *mt), []) + refs
        elif k.list == "samples":
            k.deps = key.get(("points", k.model, k.t, k.budget), []) + key.get(("baselines", k.model, k.t, k.budget), [])
    return tasks


def need(k: Task) -> tuple[int, float, float]:
    """Cores, memory (GB) and the minimum wall time (s) worth starting the line with."""
    t = k.t
    if "--chi-max 2048" in k.cmd and t >= 8:   # chi up to 2048: about 8x the time, 4x the memory of 1024
        if k.list == "reference":              # not checkpointed, so the minimum covers the whole run
            return 16, 48, (8 if t > 8 else 3) * 3600
        return (4, 48, 4 * 3600) if k.list == "points" else (8, 48, 4 * 3600)
    if k.list == "train":
        return 2, 2, 600 if t <= 2 else 3600
    if k.list == "reference":                  # an MPS at chi 1024 holds about 2 GB
        if t <= 4:
            return 2, 2, 900
        if t <= 6:
            return 4, 4, 1800
        return (8, 8, 7200) if t <= 8 else (16, 16, 7200)
    if k.list == "scan":
        return 2, 4, 900 if t <= 2 else 3600
    if k.list == "points":
        if t <= 2:
            return 2, 4, 1200
        return (2, 4, 1500) if t <= 4 else ((4, 8, 7200) if t <= 6 else (4, 12, 7200))
    # baselines and samples: MPS of whole circuits
    if t <= 4:
        return 2, 4, 1200
    return (4, 8, 1500) if t <= 6 else (8, 16, 7200)


def priority(k: Task) -> tuple:
    """Training first (the deepest ladders are the long pole), then references, scans, points, baselines, draws."""
    b = float(k.budget) if k.budget else 1.0
    return (LISTS.index(k.list), -k.t, b, k.line)


# --------------------------------------------------------------------------- state

def _p(kind: str, tid: str) -> Path:
    return STATE / kind / tid


def _ensure_dirs() -> None:
    for d in ("run", "done", "fail", "tasks", "stop", "minwall"):
        (STATE / d).mkdir(parents=True, exist_ok=True)


@contextlib.contextmanager
def _locked():
    with open(STATE / "queue.lock", "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _fails(tid: str) -> int:
    f = _p("fail", tid)
    return len(f.read_text().splitlines()) if f.exists() else 0


def _running(tid: str, now: float) -> bool:
    f = _p("run", tid)
    try:
        return now - f.stat().st_mtime < STALE
    except FileNotFoundError:
        return False


def _minwall(k: Task, force: tuple = ()) -> float:
    """The line's minimum wall time, raised by earlier deadline cuts unless its list is forced."""
    f = _p("minwall", k.id)
    base = need(k)[2]
    return max(base, float(f.read_text())) if f.exists() and k.list not in force else base


def classify(tasks: list[Task]) -> dict[str, str]:
    """done, running, failed, ready or blocked, per line."""
    now = time.time()
    st = {}
    for k in tasks:
        if _p("done", k.id).exists():
            st[k.id] = "done"
        elif _running(k.id, now):
            st[k.id] = "running"
        elif _fails(k.id) >= MAX_FAILS:
            st[k.id] = "failed"
        else:
            st[k.id] = None
    for k in tasks:
        if st[k.id] is None:
            st[k.id] = "ready" if all(st.get(d) == "done" for d in k.deps) else "blocked"
    return st


def _attempt(rec: dict) -> None:
    with open(STATE / "attempts.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")


# --------------------------------------------------------------------------- worker

def _deadline() -> float:
    """End of this SLURM job minus a margin, or infinity outside SLURM."""
    if not JOB:
        return math.inf
    end = os.environ.get("SLURM_JOB_END_TIME")
    if end and end.isdigit():
        return float(end) - 150
    try:
        out = subprocess.run(["squeue", "-h", "-j", JOB, "-o", "%e"], capture_output=True, text=True,
                             timeout=60).stdout.strip()
        return time.mktime(time.strptime(out, "%Y-%m-%dT%H:%M:%S")) - 150
    except Exception:
        return time.time() + 1800 - 150


class Worker:
    def __init__(self, cores: int, mem_gb: float, pin: bool, max_task_hours: float | None, idle_exit: float,
                 force: tuple = ()):
        self.cores, self.mem, self.pin, self.force = cores, mem_gb, pin, force
        self.max_task = max_task_hours * 3600 if max_task_hours else math.inf
        self.idle_exit = idle_exit
        self.deadline = _deadline()
        self.free = list(range(cores))             # physical core ids
        self.running = {}                          # tid -> dict(proc, cores, mem, start, log)
        self.stopping = False
        self.reason = ""
        for s in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
            signal.signal(s, self._on_signal)

    def _on_signal(self, signum, _frame):
        self.stopping = True
        self.reason = f"signal {signal.Signals(signum).name}"

    def log(self, msg: str) -> None:
        print(time.strftime("%m-%d %H:%M:%S"), WHO, msg, flush=True)

    def _cpus(self, cores: list[int]) -> str:
        n = os.cpu_count() or 2 * self.cores
        sib = n // 2 if n >= 2 * self.cores else 0     # hyperthread siblings on Perlmutter: c and c + 128
        ids = cores + ([c + sib for c in cores] if sib else [])
        return ",".join(map(str, sorted(ids)))

    def _launch(self, k: Task) -> None:
        cores, mem, _ = need(k)
        mine = self.free[:cores]
        del self.free[:cores]
        th = str(cores)
        env = {**os.environ, "OMP_NUM_THREADS": th, "OPENBLAS_NUM_THREADS": th, "MKL_NUM_THREADS": th,
               "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
               "SNAPEVO_RUNS": os.environ.get("SNAPEVO_RUNS", str(REPO / "runs")), "PYTHONUNBUFFERED": "1"}
        cmd = k.cmd
        if self.pin:
            cmd = f"taskset -c {self._cpus(mine)} {cmd}"
        logf = open(_p("tasks", k.id + ".log"), "a")
        logf.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {WHO} cores {len(mine)}: {k.cmd}\n")
        logf.flush()
        proc = subprocess.Popen(cmd, shell=True, cwd=REPO, env=env, stdout=logf, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
        _p("run", k.id).write_text(json.dumps({"who": WHO, "pid": proc.pid, "start": time.time(), "cmd": k.cmd}))
        self.running[k.id] = {"proc": proc, "cores": mine, "mem": mem, "start": time.time(), "log": logf, "task": k}
        self.log(f"start {k.id} on {len(mine)} cores: {k.cmd}")

    def _finish(self, tid: str, rc: int | None, interrupted: str = "") -> None:
        r = self.running.pop(tid)
        self.free = sorted(self.free + r["cores"])
        el = time.time() - r["start"]
        r["log"].write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] exit {rc} after {el:.0f} s"
                       f"{' (' + interrupted + ')' if interrupted else ''}\n")
        r["log"].close()
        rec = {"id": tid, "who": WHO, "start": r["start"], "elapsed": el, "rc": rc, "interrupted": interrupted}
        with _locked():
            if interrupted:
                if interrupted == "deadline":
                    _p("minwall", tid).write_text(f"{min(4 * 3600, 2 * el + need(r['task'])[2]):.0f}")
            elif rc == 0:
                _p("done", tid).write_text(json.dumps(rec))
            else:
                with open(_p("fail", tid), "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
            _p("run", tid).unlink(missing_ok=True)
        _attempt(rec)
        self.log(f"{'interrupted' if interrupted else ('done' if rc == 0 else 'FAILED')} {tid} "
                 f"rc {rc} after {el / 60:.1f} min")

    def _kill_all(self, why: str) -> None:
        for r in self.running.values():
            with contextlib.suppress(ProcessLookupError):
                os.killpg(r["proc"].pid, signal.SIGTERM)
        t0 = time.time()
        while time.time() - t0 < 20 and any(r["proc"].poll() is None for r in self.running.values()):
            time.sleep(0.5)
        for tid in list(self.running):
            p = self.running[tid]["proc"]
            if p.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            self._finish(tid, p.returncode, why)

    def _stop_requested(self) -> bool:
        d = STATE / "stop"
        return any((d / n).exists() for n in ("all", HOST, JOB) if n)

    def _fill(self, tasks: list[Task]) -> bool:
        """Start every ready line that fits, best first. False when no line is left to run anywhere."""
        st = classify(tasks)
        if not any(v in ("ready", "blocked", "running") for v in st.values()):
            return False
        left = self.deadline - time.time()
        mem_used = sum(r["mem"] for r in self.running.values())
        for k in sorted((k for k in tasks if st[k.id] == "ready"), key=priority):
            if len(self.free) < 2 or self.stopping:
                break
            cores, mem, _ = need(k)
            mw = _minwall(k, self.force)
            if cores <= len(self.free) and mem_used + mem <= self.mem and mw <= left and mw <= self.max_task:
                self._launch(k)
                mem_used += mem
        return True

    def run(self) -> None:
        _ensure_dirs()
        tasks = load_tasks()
        self.log(f"worker up: {self.cores} cores, {self.mem:.0f} GB, "
                 f"deadline {'none' if self.deadline == math.inf else time.ctime(self.deadline)}, {len(tasks)} lines")
        idle_since = time.time()
        while True:
            for tid, r in list(self.running.items()):
                rc = r["proc"].poll()
                if rc is not None:
                    self._finish(tid, rc, "signal" if rc in (-15, 143) else "")
            if self.stopping or self._stop_requested() or time.time() >= self.deadline:
                why = self.reason or ("deadline" if time.time() >= self.deadline else "stop file")
                self.log(f"stopping ({why}), {len(self.running)} lines running")
                self._kill_all("deadline" if why == "deadline" else "stopped")
                return
            for r in self.running.values():
                with contextlib.suppress(FileNotFoundError):
                    os.utime(_p("run", r["task"].id))
            if len(self.free) >= 2 and not self.stopping:
                tasks = load_tasks()                  # the lists may change while the sweep runs
                with _locked():
                    left_anywhere = self._fill(tasks)
                if not left_anywhere and not self.running:
                    self.log("every line is done or failed, exiting")
                    return
            if self.running:
                idle_since = time.time()
            elif time.time() - idle_since > self.idle_exit:
                self.log(f"nothing to run for {self.idle_exit / 60:.0f} min, exiting")
                return
            time.sleep(20)


# --------------------------------------------------------------------------- status

def status(verbose: bool) -> dict:
    _ensure_dirs()
    tasks = load_tasks()
    st = classify(tasks)
    counts = {}
    for k in tasks:
        counts.setdefault(k.list, {}).setdefault(st[k.id], 0)
        counts[k.list][st[k.id]] += 1
    order = ("done", "running", "ready", "blocked", "failed")
    print(f"{'list':10s} " + " ".join(f"{s:>8s}" for s in order))
    for name in LISTS:
        c = counts.get(name, {})
        print(f"{name:10s} " + " ".join(f"{c.get(s, 0):8d}" for s in order))
    now = time.time()
    if verbose:
        for k in tasks:
            if st[k.id] == "running":
                info = json.loads(_p("run", k.id).read_text())
                print(f"  running {k.id:14s} {(now - info['start']) / 3600:6.1f} h  {info['who']}  {k.cmd}")
        for k in tasks:
            if st[k.id] == "failed" or _fails(k.id):
                last = json.loads(_p("fail", k.id).read_text().splitlines()[-1])
                print(f"  failures {k.id:14s} {_fails(k.id)}x, last rc {last['rc']}  {k.cmd}")
    return {"tasks": tasks, "state": st}


# --------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("work")
    w.add_argument("--cores", type=int, default=None, help="physical cores to use (default: all on a node)")
    w.add_argument("--mem-gb", type=float, default=None, dest="mem_gb")
    w.add_argument("--no-pin", action="store_true", dest="no_pin", help="do not pin lines to cores (login nodes)")
    w.add_argument("--max-task-hours", type=float, default=None, dest="max_task_hours")
    w.add_argument("--idle-exit", type=float, default=600, dest="idle_exit", help="seconds idle before exiting")
    w.add_argument("--force", default="", help="comma-separated lists whose raised minimums this worker ignores")
    s = sub.add_parser("status")
    s.add_argument("-v", action="store_true")
    r = sub.add_parser("retry")
    r.add_argument("ids", nargs="+")
    a = ap.parse_args()
    if a.cmd == "work":
        n = os.cpu_count() or 2
        cores = a.cores or (n // 2 if JOB else max(1, n // 8))
        mem = a.mem_gb or (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30 * 0.9)
        Worker(cores, mem, not a.no_pin and bool(JOB), a.max_task_hours, a.idle_exit,
               tuple(v for v in a.force.split(",") if v)).run()
    elif a.cmd == "status":
        status(a.v)
    elif a.cmd == "retry":
        _ensure_dirs()
        for tid in a.ids:
            _p("fail", tid).unlink(missing_ok=True)
            _p("minwall", tid).unlink(missing_ok=True)
            print(f"cleared {tid}")


if __name__ == "__main__":
    main()
