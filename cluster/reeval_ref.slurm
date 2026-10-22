#!/bin/bash
#SBATCH -A m4980
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --job-name=snapevo-reeval
#SBATCH --output=logs/%x_%j.out
#SBATCH --open-mode=append
#
# Re-evaluate the tight points against step-0.1 references (cluster/reeval_ref.py). Submit by hand from cluster/:
#   sbatch -q preempt --requeue -t 48:00:00 --export=ALL,TIMES="6 8 10" reeval_ref.slurm
# Phase 1 builds the missing references, one process per reference with THREADS threads (more at larger t).
# Phase 2 re-evaluates the points (and redraws their fallbacks), 8 threads each. Both skip finished work, so a
# requeued or repeated job continues; rerun after the pool's last t >= 6 points finish to catch them too.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR/.."
module load python
source .venv/bin/activate
mkdir -p cluster/logs/reeval
T="${TIMES:-6 8 10}"
for t in $T; do
  case $t in 6) th=4 ;; 8) th=8 ;; *) th=16 ;; esac
  python cluster/reeval_ref.py refs --times $t | sed "s#^python#env OMP_NUM_THREADS=$th OPENBLAS_NUM_THREADS=$th MKL_NUM_THREADS=$th python#"
done > cluster/logs/reeval/refs_$SLURM_JOB_ID.txt
# 8 references at once (refs sorted by t): 32 threads at t = 6 up to 128 at t = 10; chi 2048 needs ~40 GB each
xargs -P ${NREF:-8} -I{} sh -c '{}' < cluster/logs/reeval/refs_$SLURM_JOB_ID.txt >> cluster/logs/reeval/refs.log 2>&1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
python cluster/reeval_ref.py files --times $T | xargs -P ${NPT:-10} -I{} python cluster/reeval_ref.py point {} \
  >> cluster/logs/reeval/points.log 2>&1
echo "$(date '+%m-%d %H:%M:%S') reeval done: $(python cluster/reeval_ref.py files --times $T | wc -l) points left"
