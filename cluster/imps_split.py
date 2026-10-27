"""Run one infinite-chain ladder with its starts split over nodes, one node per start, inside a Slurm allocation.

    python cluster/imps_split.py FIELD T [--n-train N] [--workers 250] [--deadline EPOCH]
                                 [--depths 2,3,...] [--runs DIR] [--logs DIR] [--hosts h1,h2,h3] [--interactive]
                                 [--optimizer newton|lbfgs] [--ssh]

FIELD is 1.6 or 0 (XXZ chain, Delta = 0.25, J = 1, spin units, h_x = FIELD); T is the evolution time; depths are
cluster/make_jobs.py's DEPTHS[T]. The driver holds the ladder lock for its whole run. On start it claims the ladder
(cluster/logs/imps_train/hx<F>_t<T>_n<N>.split, which cluster/imps_interactive.sh skips) and stops that ladder's run
in the interactive allocation (job imps-int), if one is running, so the lock frees. Then, depth by depth
(snapevo.imps_train.ladder_status), one srun step per start of the next depth (cluster/imps_train.py --only-start S
--no-lock), each on its own node, and when all have stopped one step on the first node that records the depth (a
normal run over the depths up to it, which reuses the finished start checkpoints and adds the test cost).

--hosts limits the nodes (default: all of the allocation); with fewer nodes than starts, starts share a node and
split its workers. --ssh runs the steps on login nodes (--hosts, required) by ssh instead of srun, each start under a
login node's per-user limits (32 cores); --optimizer is the ladder's training optimizer (claim, .done and log files
then carry an _lbfgs suffix). --interactive is for cluster/imps_interactive.sh, which runs the driver inside its own allocation:
it neither claims the ladder nor stops an interactive run, and it hands the ladder over (stops its steps, which
checkpoint, and exits 4) as soon as a split job on another QOS claims it.

Exit 0 once the ladder is finished (target or stall rule; touches the .done file), 3 at the deadline (the batch
script requeues the job), 1 after repeated failed steps.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "cluster"))
LOGS = REPO / "cluster" / "logs" / "imps_train"


def nodes() -> list[str]:
    nl = os.environ.get("SLURM_JOB_NODELIST")
    if not nl:
        raise SystemExit("run inside a Slurm allocation")
    return subprocess.run(["scontrol", "show", "hostnames", nl], capture_output=True, text=True,
                          check=True).stdout.split()


def stop_interactive(field: str, t: str, n: int, log) -> None:
    """Stop this ladder's run in the imps-int allocation, if any (its checkpoints are kept)."""
    q = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-n", "imps-int", "-o", "%N"],
                       capture_output=True, text=True).stdout.split()
    if not q:
        return
    hosts = subprocess.run(["scontrol", "show", "hostnames", ",".join(q)], capture_output=True, text=True).stdout.split()
    pat = f"python [c]luster/imps_train.py {field} {t} --n-train {n} --workers"
    for h in hosts:
        r = subprocess.run(["ssh", "-n", "-q", "-o", "BatchMode=yes", "-o", "LogLevel=ERROR", h,
                            f"for p in $(pgrep -u $USER -f '{pat}'); do kill -TERM $(pgrep -P $p) $p; "
                            f"echo stopped $p; done"], capture_output=True, text=True, timeout=60)
        if r.stdout.strip():
            log(f"stopped the interactive run on {h}: {r.stdout.strip()}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("field")
    ap.add_argument("t")
    ap.add_argument("--n-train", type=int, default=4, dest="n_train")
    ap.add_argument("--workers", type=int, default=250)
    ap.add_argument("--deadline", type=float, default=None)
    ap.add_argument("--depths", default=None, help="default: cluster/make_jobs.py's DEPTHS[T]")
    ap.add_argument("--runs", default=str(REPO / "runs"))
    ap.add_argument("--logs", default=str(LOGS))
    ap.add_argument("--hosts", default=None)
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--optimizer", default="newton", choices=["newton", "lbfgs"])
    ap.add_argument("--ssh", action="store_true")
    a = ap.parse_args()
    if a.ssh and not a.hosts:
        raise SystemExit("--ssh needs --hosts")

    from imps_train import depth_grid
    from snapevo.hamiltonian import XXZ
    from snapevo.imps_train import N_TRAIN, ladder_status

    t = int(a.t) if float(a.t).is_integer() else float(a.t)
    model = XXZ(0.25, float(a.field))
    depths = [int(d) for d in a.depths.split(",")] if a.depths else depth_grid(t)
    runs = Path(a.runs)
    sub = ("imps_train" if a.n_train == N_TRAIN else f"imps_train_n{a.n_train}") \
        + ("_lbfgs" if a.optimizer == "lbfgs" else "")
    out_dir = runs / model.tag / f"t{float(t):g}" / sub
    Path(a.logs).mkdir(parents=True, exist_ok=True)
    stem = Path(a.logs) / f"hx{a.field}_t{a.t}_n{a.n_train}{'_lbfgs' if a.optimizer == 'lbfgs' else ''}"
    ladder_log = open(f"{stem}.log", "a")

    def log(s):
        print(f"{time.strftime('%m-%d %H:%M:%S')} [{model.tag} t={t:g} split] {s}", file=ladder_log, flush=True)

    hosts = a.hosts.split(",") if a.hosts else nodes()
    log(f"split driver: job {os.environ.get('SLURM_JOB_ID')}, nodes {hosts}, workers {a.workers} per node, "
        f"deadline {a.deadline}{', interactive' if a.interactive else ''}")
    if not a.interactive:
        Path(f"{stem}.split").write_text(f"{os.environ.get('SLURM_JOB_ID')}\n")
        stop_interactive(a.field, a.t, a.n_train, log)
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = open(out_dir / "lock", "w")
    t_wait = time.time()
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except OSError:
            if time.time() - t_wait > 1800:
                log("another run still holds the ladder lock after 30 minutes; giving up")
                return 1
            time.sleep(10)
    log(f"holding the ladder lock (waited {time.time() - t_wait:.0f}s)")

    py = [sys.executable, str(REPO / "cluster" / "imps_train.py"), a.field, a.t, "--n-train", str(a.n_train),
          "--no-lock", "--runs", str(runs), "--optimizer", a.optimizer] + ([] if a.deadline is None else ["--deadline", str(a.deadline)])

    def step(host, extra, workers=a.workers):
        if a.ssh:
            env = "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
            cmd = ["ssh", "-n", "-q", "-o", "BatchMode=yes", "-o", "LogLevel=ERROR", host,
                   f"cd {REPO} && {env} exec " + " ".join(py + ["--workers", str(workers)] + extra)]
            return subprocess.Popen(cmd, cwd=REPO, stdout=ladder_log, stderr=subprocess.STDOUT)
        cmd = ["srun", "--nodes=1", "--ntasks=1", "--cpus-per-task=256", "--cpu-bind=none", "--overlap", "-w",
               host] + py + ["--workers", str(workers)] + extra
        return subprocess.Popen(cmd, cwd=REPO, stdout=ladder_log, stderr=subprocess.STDOUT)

    claim = Path(f"{stem}.split")

    def wait_all(procs):
        """Exit codes of the steps; None if another job claimed the ladder meanwhile (steps stopped)."""
        while any(p.poll() is None for p in procs):
            if a.interactive and claim.exists():
                for p in procs:
                    if p.poll() is None:
                        p.terminate()
                for p in procs:
                    p.wait()
                return None
            time.sleep(5)
        return [p.returncode for p in procs]

    fails = 0
    while True:
        if a.interactive and claim.exists():
            log(f"claimed by split job {claim.read_text().strip()}; handing the ladder over")
            return 4
        if a.deadline is not None and time.time() > a.deadline - 60:
            log("deadline: stopping before the next step")
            return 3
        st = ladder_status(model, t, depths, runs=runs, n_train=a.n_train, optimizer=a.optimizer)
        if st["finished"]:
            Path(f"{stem}.done").touch()
            log(f"ladder finished ({st['why']})")
            return 0
        D, starts = st["D"], st["starts"]
        place = [hosts[i % len(hosts)] for i in range(len(starts))]   # starts sharing a node split its workers
        log(f"D={D}: starts {starts} on {place}")
        grid = ["--depths", ",".join(str(d) for d in depths)]
        procs = [step(h, grid + ["--only-start", s], max(1, a.workers // place.count(h)))
                 for s, h in zip(starts, place)]
        rcs = wait_all(procs)
        if rcs is None:
            log(f"D={D}: claimed by split job {claim.read_text().strip()}; starts stopped, handing over")
            return 4
        if 3 in rcs or (a.deadline is not None and time.time() > a.deadline):
            log(f"D={D}: deadline during the starts; checkpoints kept")
            return 3
        if any(rc != 0 for rc in rcs):
            fails += 1
            log(f"D={D}: start steps exited {rcs} (failure {fails})")
            if fails >= 3:
                return 1
            continue
        rcs = wait_all([step(hosts[0], ["--depths", ",".join(str(d) for d in depths if d <= D)])])
        if rcs is None:
            log(f"D={D}: claimed by split job {claim.read_text().strip()}; handing over")
            return 4
        rc = rcs[0]
        if rc == 3:
            return 3
        if rc != 0:
            fails += 1
            log(f"D={D}: the recording step exited {rc} (failure {fails})")
            if fails >= 3:
                return 1
            continue
        fails = 0


if __name__ == "__main__":
    sys.exit(main())
