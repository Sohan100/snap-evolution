#!/bin/bash
#SBATCH -A m4980
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --job-name=snapevo-size
#SBATCH --output=logs/%x_%j.out
#SBATCH --open-mode=append
#
# The ring-size experiment (size_transfer.py) on one whole node. Submit from cluster/:
#   sbatch -q preempt --requeue -t 12:00:00 size_transfer.slurm
# A requeued run skips every task whose result exists.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR/.."

# Site setup, NERSC Perlmutter.
module load python
source .venv/bin/activate
export SNAPEVO_RUNS=${SNAPEVO_RUNS:-$PWD/runs}

exec python cluster/size_transfer.py run ${WORKERS:-60}
