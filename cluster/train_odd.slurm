#!/bin/bash
#SBATCH -A m4980
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --job-name=snapevo-train-odd
#SBATCH --output=logs/%x_%j.out
#SBATCH --open-mode=append
#
# Ring ladders for the times the sweep never ran, t = 3, 5, 7, 9, both fields, on one node: eight ladders at once,
# 16 cores each. The settings are the sweep's for every t (make_jobs.py: maxiter 30000, stall 1.5,8, target 1e-10),
# with no inline polish; the Newton polish is a separate pass over every depth at every t (polish_ladder.py), so
# all times get one recipe. From cluster/:
#   sbatch -q preempt --requeue -t 48:00:00 train_odd.slurm
# A requeued job resumes each ladder from its cells.json (train_ladder keeps finished depths).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR/.."

# Site setup, NERSC Perlmutter.
module load python
source .venv/bin/activate
export SNAPEVO_RUNS=${SNAPEVO_RUNS:-$PWD/runs}

declare -A DEPTHS=(
  [3]="2,3,5,7,9,11,13,15,17,19,21,25,29,33,37,41,45,49,53,57,61,65,73,81"
  [5]="3,5,7,9,11,13,15,17,21,25,29,33,37,41,45,49,53,57,61,65,73,81,89,97,105,113"
  [7]="5,7,9,13,17,21,25,29,33,37,41,45,49,53,57,61,65,73,81,89,97,105,113,121,129,137"
  [9]="5,7,9,13,17,21,25,29,33,37,41,45,49,53,57,61,65,73,81,89,97,105,113,121,129,137,145,153"
)
mkdir -p cluster/logs/train_odd
slot=0
for t in ${TIMES:-3 5 7 9}; do
  for m in A B; do                                   # A is h_x = 1.6, B is h_x = 0
    lo=$((slot * 16)); hi=$((lo + 15))
    OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 \
    XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=16" \
      taskset -c $lo-$hi python -m snapevo train --model $m --t $t --depths "${DEPTHS[$t]}" \
        --maxiter 30000 --stall 1.5,8 --target 1e-10 \
        >> cluster/logs/train_odd/train_${m}_t${t}.log 2>&1 &
    slot=$((slot + 1))
  done
done
wait
echo "train_odd.slurm: all ladders done"
