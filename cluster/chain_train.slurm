#!/bin/bash
#SBATCH -A m4980
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --job-name=snapevo-chain
#SBATCH --output=logs/%x_%j.out
#SBATCH --open-mode=append
#
# Open-chain arm: several chain_train.py ladders at once on one node, each start on its own workers.
# Submit by hand from cluster/, e.g.
#   sbatch -q preempt --requeue -t 48:00:00 --export=ALL,LADDERS="0:1:24 1.6:1:24 0:2:36 1.6:2:36" chain_train.slurm
# LADDERS lists FIELD:T[:WORKERS], WORKERS the ladder's worker pool (a multiple of the 12 training inputs;
# default $WORKERS or 24), shared by a depth's 6 runs (3 starts x Newton and L-BFGS, no iteration caps) and
# their Hessians. Every ladder stops 5 min before the wall time and resumes from its checkpoints in the next
# job (e.g. --dependency=afterany:ID). OPTIMIZER=newton runs the Newton trainer alone.
# PRECOMPUTE=K runs, instead of the ladders, K precompute runs at once per LADDERS entry, each on WORKERS
# workers (chain_train.py --precompute): the depth-independent starts, ahead of the ladders, on extra nodes.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR/.."
module load python
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
END=${SLURM_JOB_END_TIME:-$(date -d "$(squeue -h -j $SLURM_JOB_ID -o %e 2>/dev/null)" +%s 2>/dev/null)}
DL=()
if [ -n "$END" ] && [ "$END" -gt $(( $(date +%s) + 900 )) ]; then
  DL=(--deadline $((END - 300)))
else
  echo "$(date '+%m-%d %H:%M:%S') could not read the job's end time; running without a deadline"
fi
mkdir -p cluster/logs/chain
for lt in ${LADDERS:?set LADDERS, e.g. "0:1:6 1.6:1:6"}; do
  IFS=: read f t p <<< "$lt"
  if [ -n "${PRECOMPUTE:-}" ]; then
    python cluster/chain_train.py $f $t --precompute --tasks $PRECOMPUTE --workers ${p:-${WORKERS:-24}} "${DL[@]}" \
      >> cluster/logs/chain/precompute_${f}_t${t}.log 2>&1 &
  else
    python cluster/chain_train.py $f $t --workers ${p:-${WORKERS:-24}} --optimizer ${OPTIMIZER:-best} "${DL[@]}" \
      >> cluster/logs/chain/ladder_${f}_t${t}.log 2>&1 &
  fi
done
wait
echo "$(date '+%m-%d %H:%M:%S') all ladders stopped"
