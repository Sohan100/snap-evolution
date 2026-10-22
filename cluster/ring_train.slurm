#!/bin/bash
#SBATCH -A m4980
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --job-name=snapevo-rtrain
#SBATCH --output=logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --signal=B:TERM@1800
#
# Ring arm, best-of ladders (snapevo.train_ring_best): all 12 ladders (h_x = 1.6 and 0; t = 1, 2, 4, 6, 8, 10) at
# once on one node, each with WORKERS spawned processes (default 6: every start x optimizer of a depth at once,
# 72 single-threaded JAX processes on 128 cores; above 300 angles each Newton run adds HESSIAN_WORKERS (default 4)
# helpers for its difference Hessian, using the cores the short ladders free). Submit by hand from cluster/, e.g.
#   sbatch -q preempt --requeue -t 48:00:00 ring_train.slurm
# LADDERS (default all 12) restricts the set, e.g. --export=ALL,LADDERS="1.6:10 0:10". Output:
# runs/<tag>/t<T>/ring2/ladder.json (cells.json's format) and ring2/ckpt/; logs in cluster/logs/ring_train/.
# Stopping: every ladder gets --deadline = the job's end minus 30 minutes (each run saves its checkpoint at its
# next cost evaluation; the margin covers a deep Hessian or an XLA compile in progress); SIGTERM (sent 30
# minutes before the end by --signal, and by Slurm at preemption) is forwarded to every ladder, which forwards it
# to its runs. If any ladder stopped unfinished (exit 3 or 143) the job requeues itself (needs --requeue);
# otherwise resubmit. A rerun keeps every finished depth and continues every run from its checkpoint.
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
WORKERS=${WORKERS:-6}
HESSIAN_WORKERS=${HESSIAN_WORKERS:-4}
LADDERS=${LADDERS:-"1.6:1 1.6:2 1.6:4 1.6:6 1.6:8 1.6:10 0:1 0:2 0:4 0:6 0:8 0:10"}
mkdir -p cluster/logs/ring_train
pids=()
names=()
trap 'echo "$(stamp) SIGTERM: telling the ladders to checkpoint"; kill -TERM "${pids[@]}" 2>/dev/null' TERM
for L in $LADDERS; do
  IFS=: read -r f t <<< "$L"
  name="ladder_${f}_t${t}"
  python cluster/ring_train.py "$f" "$t" --workers "$WORKERS" --hessian-workers "$HESSIAN_WORKERS" "${DL[@]}" \
    >> "cluster/logs/ring_train/${name}.log" 2>&1 &
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
