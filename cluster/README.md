# Running the t sweep on a cluster

The sweep is models A and B at $t = 1, 2, 4, 6, 8, 10$ (spin units), SNAP at training budgets $10^{-4}$, $10^{-5}$ and $10^{-6}$, and a product-formula baseline at each time. Every point is evaluated by MPS on a 100-site open chain with 8 random product inputs (`docs/error_metric.md`). The grid lives in `make_jobs.py`.

## Set up once

```
git clone <repo> snap-evolution && cd snap-evolution
python -m venv .venv && source .venv/bin/activate
pip install -e .[test]
pytest                                  # about 2 min
export SNAPEVO_RUNS=/scratch/$USER/snapevo-runs
```

To reuse what is already done, copy the `runs/` folder from the PC to `$SNAPEVO_RUNS`. Training, compiles, cached references and baseline scans pick up where they stopped.

Edit the site lines of `array.slurm` (modules, virtual environment, partition, account) before the first submission.

## Submit

```
python cluster/make_jobs.py             # writes cluster/jobs/*.txt
cluster/submit.sh
```

`submit.sh` sends four arrays. Training (12 tasks) and references (66 tasks) start at once. The 36 SNAP points wait for both, and the 12 baseline scans wait for the references. Each model and time gets one extra reference, input 0 at a 100 times smaller cutoff, which every evaluation uses to check its truncation. Logs go to `cluster/logs/`. A task that hits its time limit can be resubmitted as is. Compiles resume from their checkpoint, and references and scans skip what is cached.

On a workstation, `cluster/run_local.sh cluster/jobs/points.txt` runs a list in order.

## What to expect

Measured on one core of the cloud box (2026-10-06), $N = 100$, cutoff $10^{-12}$:

| $t$ | reference bond dimension | one reference | training ladder | one compile |
|---|---|---|---|---|
| 1 | 9 | 1 s | 6 min | 1 min |
| 2 | 16 | 1 s | 15 min | 5 to 30 min |
| 4 | 43 | 10 s | about 1 h | up to a few hours |
| 6 | 110 | 45 s | hours | hours |
| 8 | 275 | 6 min | many hours | up to a day |
| 10 | see `docs/validation.md` | about 1 h | a day or more | days |

The compile times at large $t$ come from the Sep 2026 runs (16 h for a 61-sublayer cell). The reference cost grows as the bond dimension cubed, and 2 to 4 threads per task help from $t = 8$ on (`submit.sh` asks for 4). Each reference of an 8-input set at $t = 10$ takes a few hundred MB on disk.

## After the runs

```
python -m snapevo report --model B --t 10
```

prints every SNAP and baseline point of one model and time. Copy `$SNAPEVO_RUNS` back to the PC's `runs/` to keep everything in one place. The `ref/` folders can be left behind once all points are evaluated.
