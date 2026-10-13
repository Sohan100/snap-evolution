#!/bin/bash
#SBATCH -A m4980
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --job-name=snapevo-pool
#SBATCH --output=logs/%x_%j.out
#
# One pool worker on a whole node (pool.py). Submit it by hand, from cluster/:
#   sbatch -q preempt --requeue -t 48:00:00 pool.slurm
#   sbatch -q debug -t 00:30:00 pool.slurm
# or run it under salloc for the interactive QOS (cluster/README.md). The worker
# pulls lines from cluster/jobs/*.txt, runs as many at once as its 128 cores
# allow, and stops 150 s before the wall time. Its lines resume in the next
# worker. POOL_ARGS passes options to the worker, for example
#   sbatch -q preempt --requeue -t 48:00:00 --export=ALL,POOL_ARGS=--force=train pool.slurm
set -euo pipefail
cd "$SLURM_SUBMIT_DIR/.."

# Site setup, NERSC Perlmutter.
module load python
source .venv/bin/activate
export SNAPEVO_RUNS=${SNAPEVO_RUNS:-$PWD/runs}

exec python cluster/pool.py work ${POOL_ARGS:-}
