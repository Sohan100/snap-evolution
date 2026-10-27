#!/bin/bash
#SBATCH -A m4980
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --job-name=snapevo-titrain
#SBATCH --output=logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --signal=B:TERM@1800
#
# TI training with the protocol's optimizer: ring ladders (10-site ring, the production training set) where every
# start (Trotter, warm, Trotter + 0.1 rad noise) at every depth is minimized by trust-region Newton on the Hessian
# (exact autodiff up to 300 angles, central differences of the exact gradient above, spread over HESSIAN_WORKERS
# helpers), with no iteration cap, until the predicted Newton decrease and the last step's are below 1e-6 c
# (cluster/ring_train.py --optimizers newton, snapevo.train_ring_best). Same settings at every t. Output:
# runs/<tag>/t<T>/ring_newton/ladder.json (cells.json's format) and ring_newton/ckpt/; logs in
# cluster/logs/ti_train/. LADDERS picks the ladders (FIELD:T), e.g. from cluster/:
#   LADDERS="1.6:10 0:10 1.6:1 0:1" sbatch -q preempt --requeue -t 48:00:00 ti_train.slurm
# Stopping and requeue as cluster/ring_train.slurm: deadline 30 minutes before the job's end, SIGTERM forwarded,
# the job requeues itself if a ladder stopped unfinished, and a rerun continues every run from its checkpoint.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR/.."
module load python
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export JAX_ENABLE_X64=True
stamp() { date '+%m-%d %H:%M:%S'; }
END=${SLURM_JOB_END_TIME:-$(date -d "$(squeue -h -j "$SLURM_JOB_ID" -o %e 2>/dev/null)" +%s 2>/dev/null)}
DL=()
if [ -n "$END" ] && [ "$END" -gt $(( $(date +%s) + 3600 )) ]; then
  DL=(--deadline $((END - 1800)))
else
  echo "$(stamp) could not read the job's end time; running without a deadline (SIGTERM still checkpoints)"
fi
HESSIAN_WORKERS=${HESSIAN_WORKERS:-8}
LADDERS=${LADDERS:?set LADDERS, e.g. \"1.6:10 0:10\"}
declare -A ODD=(                                     # the times the sweep never ran; the others use its grid
  [3]="2,3,5,7,9,11,13,15,17,19,21,25,29,33,37,41,45,49,53,57,61,65,73,81"
  [5]="3,5,7,9,11,13,15,17,21,25,29,33,37,41,45,49,53,57,61,65,73,81,89,97,105,113"
  [7]="5,7,9,13,17,21,25,29,33,37,41,45,49,53,57,61,65,73,81,89,97,105,113,121,129,137"
  [9]="5,7,9,13,17,21,25,29,33,37,41,45,49,53,57,61,65,73,81,89,97,105,113,121,129,137,145,153"
)
mkdir -p cluster/logs/ti_train
pids=()
names=()
trap 'echo "$(stamp) SIGTERM: telling the ladders to checkpoint"; kill -TERM "${pids[@]}" 2>/dev/null' TERM
for L in $LADDERS; do
  IFS=: read -r f t <<< "$L"
  name="ladder_${f}_t${t}"
  DEP=()
  if [ -n "${ODD[$t]:-}" ]; then DEP=(--depths "${ODD[$t]}"); fi
  python cluster/ring_train.py "$f" "$t" --optimizers newton --out-sub ring_newton --workers 3 \
    --hessian-workers "$HESSIAN_WORKERS" "${DEP[@]}" "${DL[@]}" >> "cluster/logs/ti_train/${name}.log" 2>&1 &
  pids+=($!)
  names+=("$name")
done
requeue=0
failed=0
for k in "${!pids[@]}"; do
  pid=${pids[$k]}
  wait "$pid"
  rc=$?
  while [ $rc -gt 128 ] && kill -0 "$pid" 2>/dev/null; do   # wait was interrupted by the trapped signal
    wait "$pid"
    rc=$?
  done
  echo "$(stamp) ${names[$k]}: exit $rc"
  if [ $rc -eq 3 ] || [ $rc -eq 143 ]; then
    requeue=1
  elif [ $rc -ne 0 ]; then
    failed=1
  fi
done
echo "$(stamp) all ladders stopped (unfinished: $requeue, failed: $failed)"
if [ $requeue -eq 1 ]; then
  scontrol requeue "$SLURM_JOB_ID" || echo "$(stamp) requeue failed; resubmit (the ladders resume)"
fi
