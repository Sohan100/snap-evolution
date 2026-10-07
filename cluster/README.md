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

`submit.sh` sends five arrays. Training (12 tasks), references (66 tasks) and baseline scans (12 tasks) start at once. The 180 SNAP points wait for training and references, and the 180 baseline evaluations wait for the scans and references. A point whose ladder does not reach far enough below its budget is skipped in seconds, and so is a point that already has its evaluation (the $t = 1$ and 2 points in the repo). Each model and time gets one extra reference, input 0 at a 100 times smaller cutoff, which every evaluation uses to check its truncation. Logs go to `cluster/logs/`. A task that hits its time limit can be resubmitted as is. Compiles resume from their checkpoint, and references and scans skip what is cached.

On a workstation, `cluster/run_local.sh cluster/jobs/points.txt` runs a list in order.

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

The compile times at large $t$ come from the Sep 2026 runs (16 h for a 61-sublayer cell). At the tight budgets the edge rotations add to that. Each of the 24 to 36 edge angles is synthesized with KN at $\epsilon = 0.1\sqrt{B}$, which takes about half a minute at $5 \times 10^{-5}$ and more below, so 15 to 60 min per point for $B \le 3 \times 10^{-7}$. The reference cost grows as the bond dimension cubed, and 2 to 4 threads per task help from $t = 8$ on (`submit.sh` asks for 4). At $t = 10$ each cached reference input takes a few hundred MB on disk, so the 8 inputs of one model need a few GB.

## After the runs

```
python -m snapevo report --model B --t 10
```

prints every SNAP and baseline point of one model and time, `python -m snapevo summary` writes all of them to `$SNAPEVO_RUNS/summary.csv`, and `python -m snapevo plot` draws T per qubit and T per rotation against the per-site infidelity for every model and time (`pip install -e .[plot]` first). Copy the results back into the repo's `runs/` and commit them. The `ref/` folders are caches, ignored by git, and can be left behind once all points are evaluated.
