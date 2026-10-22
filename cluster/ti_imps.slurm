#!/bin/bash
#SBATCH -A m4980
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --job-name=snapevo-ti
#SBATCH --output=logs/%x_%j.out
#SBATCH --open-mode=append
#
# Infinite-chain (TI) evaluation, ti_imps_run.py, on one whole node: the times in $TIMES, this job's share
# $SHARD (i/n) of the (field, t, input) units, $WORKERS single-threaded workers. From cluster/:
#   TIMES=8 SHARD=0/1 WORKERS=128 sbatch -q preempt --requeue -t 24:00:00 ti_imps.slurm
#   TIMES=10 WORKERS=64 sbatch -q preempt --requeue -t 24:00:00 --array=0-3 ti_imps.slurm
# (as an array, task i of n takes shard i/n unless SHARD is set.)
# (sbatch exports the submitting environment.) Every (point, input) is its own result file, so a preempted and
# requeued job resumes where it stopped; a unit's reference is cached once done. At t = 10 the iMPS reach bond
# dimension ~2000 (about 1 GB a state with the 8-site cell), so t = 10 runs fewer workers than cores.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR/.."

# Site setup, NERSC Perlmutter.
module load python
source .venv/bin/activate
export SNAPEVO_RUNS=${SNAPEVO_RUNS:-$PWD/runs}

if [ -z "${SHARD:-}" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then SHARD=$SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT; fi
echo "ti_imps.slurm on $(hostname): TIMES=${TIMES:-8,10} SHARD=${SHARD:-0/1} WORKERS=${WORKERS:-96}"
exec python cluster/ti_imps_run.py run --times "${TIMES:-8,10}" --shard "${SHARD:-0/1}" --workers "${WORKERS:-96}"
