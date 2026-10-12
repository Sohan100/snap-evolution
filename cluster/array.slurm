#!/bin/bash
#SBATCH --job-name=snapevo
#SBATCH -A m4980
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%A_%a.out
#
# Runs line $SLURM_ARRAY_TASK_ID of the job list $JOBS (one shell command per line).
# submit.sh sets JOBS and the array range. By hand, from cluster/:
#   sbatch --array=1-$(wc -l < jobs/points.txt) --export=ALL,JOBS=$PWD/jobs/points.txt array.slurm
set -euo pipefail
: "${JOBS:?set JOBS to a job list}"
cd "$SLURM_SUBMIT_DIR/.."

# Site setup, NERSC Perlmutter (account, constraint and QOS go in the #SBATCH lines above).
module load python
source .venv/bin/activate
export SNAPEVO_RUNS=${SNAPEVO_RUNS:-$PWD/runs}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$OMP_NUM_THREADS
cmd=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOBS")
echo "[$(date)] $cmd"
eval "$cmd"
