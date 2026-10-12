# Running the t sweep on a cluster

The sweep is models A and B at $t = 1, 2, 4, 6, 8, 10$ (spin units), SNAP at 15 training budgets in half decades from $10^{-1}$ to $10^{-8}$, and a product-formula baseline evaluated at the point nearest each budget. Every point is evaluated by MPS on a 100-site open chain with 8 random product inputs (`docs/error_metric.md`). The grid lives in `make_jobs.py`.

## Set up once

```
git clone <repo> snap-evolution && cd snap-evolution
python -m venv .venv && source .venv/bin/activate
pip install -e .[test]
pytest                                  # about 2 min
cp -r runs /scratch/$USER/snapevo-runs  # optional: keep the runs on scratch
export SNAPEVO_RUNS=/scratch/$USER/snapevo-runs
```

The repo tracks the run results under `runs/`, so a clone already holds every ladder, compile and evaluation done so far. Training, compiles, cached references and baseline scans pick up where they stopped. Without `SNAPEVO_RUNS` the runs go straight into the clone's `runs/`. All jobs share one KN cache, `runs/kn_cache.json`. They lock it with `flock` on a side file, and every JSON and MPS file is written to a per-process temporary name and then renamed, so parallel jobs cannot corrupt it. On a file system without `flock` the lock is skipped, and the worst case is a lost cache entry that gets synthesized again.

Edit the site lines of `array.slurm` (modules, virtual environment, partition, account) before the first submission.

## Submit

```
python cluster/make_jobs.py             # writes cluster/jobs/*.txt
cluster/submit.sh
```

`submit.sh` sends six arrays. Training (12 tasks), references (66 tasks) and baseline scans (12 tasks) start at once. The 180 SNAP points wait for training and references, and each compiles its budget at up to two depths (`--margins 3,30` in `make_jobs.py`), the 180 baseline evaluations wait for the scans and references, and the 180 fallback draws (`snapevo sample`, `fallbacks.py`), one per model, time and budget, wait for the points and the baseline evaluations. Two budgets can pick the same baseline point, and the job that gets there second skips it (a lock file per point). Each draw is one more circuit MPS, so the draws cost about four evaluations per baseline point at $t \le 4$ (4 draws per input) and two beyond (2 draws). A point whose ladder does not reach far enough below its budget is skipped in seconds, and so is a point that already has its evaluation (the $t = 1$ and 2 points in the repo). Each model and time gets one extra reference, input 0 at a 100 times smaller cutoff, which every evaluation uses to check its truncation. Logs go to `cluster/logs/`. A task that hits its time limit can be resubmitted as is. Compiles resume from their checkpoint, and references and scans skip what is cached.

On a workstation, `cluster/run_local.sh cluster/jobs/points.txt` runs a list in order.

## On NERSC Perlmutter

`array.slurm` carries the Perlmutter site lines (account `m4980`, `-C cpu`, the shared QOS, the module and the virtual environment), and `submit.sh` asks for at most 48 h, the cap of the shared and regular QOS. We ran the sweep on the preempt, debug, debug_preempt and interactive QOS instead. These hand out whole nodes (128 cores, 512 GB), so an array of single-process tasks would hold one node per task. The sweep runs as one work pool instead (`pool.py`):

```
cd cluster
sbatch -q preempt --requeue -t 48:00:00 pool.slurm    # one node, requeued when preempted
sbatch -q debug -t 00:30:00 pool.slurm                # one node, short lines only
salloc -A m4980 -C cpu -q interactive -N 1 -t 04:00:00 srun -n 1 -c 256 --cpu-bind=none bash pool.slurm
python pool.py status -v                              # done, running, ready, blocked, failed per list
```

Each node job runs one worker. It pulls lines from the same six job lists, runs as many at once as its cores allow, and pins each to its own cores (2 cores for training, compiles and short MPS runs, up to 16 for the references at $t = 10$). A line waits only for what it needs at its own model and time. A point waits for its ladder and references, a baseline evaluation for its scan and references, and a draw for the point and baseline evaluation of its budget. So the points at $t = 4$ start while the ladder at $t = 10$ still trains. A worker starts a line only if its wall time covers the line's minimum, so the 30 min debug nodes take only short lines, and it stops its lines 150 s before its wall time. They resume in the next worker, since every command skips finished work and compiles resume from their checkpoints. A line that fails twice waits for `pool.py retry`. Workers never submit jobs. Every node job is submitted by hand, sized to the lines that are ready. The pool keeps its state in `cluster/logs/pool/`, and `cluster/logs/submitted.txt` lists the node jobs.

## What to expect

Measured on the cloud box (one core per job, often shared), $N = 100$, cutoff $10^{-12}$. Entries in italics are estimates.

| $t$ | reference bond dimension | one reference input | training ladder | one compile |
|---|---|---|---|---|
| 1 | 9 | 1 s | 6 min | 1 min |
| 2 | 16 | 1 to 2 s | *under 1 h* | 4 to 14 min |
| 4 | 43 | 5 to 10 s | 55 min | 14 min at $10^{-4}$ |
| 6 | 110 | 45 to 65 s | *a few hours* | *hours* |
| 8 | 275 | 6 min | *many hours* | *up to a day* |
| 10 | 448 at cutoff $10^{-10}$ | 28 min at $10^{-10}$, *1 to 2 h at $10^{-12}$* | *a day or more* | *days* |

The compile times at large $t$ come from the Sep 2026 runs (16 h for a 61-sublayer cell). Edge rotations add a few minutes per point at the tight budgets. Their KN syntheses at $\epsilon = 0.1\sqrt{B}$ take a second or two each with the faster candidate count in `synthesis.py` (Isaac's own search took up to 10 min for angles near multiples of $\pi/4$, which now get set to those multiples instead). The reference cost grows as the bond dimension cubed, and 2 to 4 threads per task help from $t = 8$ on (`submit.sh` asks for 4). At $t = 10$ each cached reference input takes a few hundred MB on disk, so the 8 inputs of one model need a few GB.

## After the runs

```
python -m snapevo report --model B --t 10
```

prints every SNAP and baseline point of one model and time, `python -m snapevo summary` writes all of them to `$SNAPEVO_RUNS/summary.csv`, and `python -m snapevo plot` draws T per qubit and T per rotation against the per-site infidelity for every model and time (`pip install -e .[plot]` first). Copy the results back into the repo's `runs/` and commit them. The `ref/` folders are caches, ignored by git, and can be left behind once all points are evaluated.
