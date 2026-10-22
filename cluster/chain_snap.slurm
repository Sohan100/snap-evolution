#!/bin/bash
#SBATCH -A m4980
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --job-name=snapevo-csnap
#SBATCH --output=logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --signal=B:TERM@600
#
# Open-chain arm: several chain_snap.py points at once on one node, each with its own pool of 12 workers (one
# per training input) plus its parent process. Submit by hand from cluster/, e.g.
#   sbatch -q preempt --requeue -t 48:00:00 --export=ALL,REOPT=newton,POINTS="0:2:1e-4 1.6:2:1e-4:7" chain_snap.slurm
# REOPT (required, no default) is the re-optimizer: newton, lbfgs-converged or coordinate.
# POINTS lists FIELD:T:BUDGET[:D]; without D the floor margins (3,30) pick the depths, one compile after the
# other within that point. About 10 points fill a 128-core node (re-optimizations keep only 12 cores per point
# busy); memory per point is about 3 GB at t = 4, D = 17, and at t = 6 pass --env-layers 4 (all 25 sublayers of
# a D = 25 cell would need about 3 GB per worker, 40 GB per point). CSNAP_ARGS passes
# options to every point, e.g. --export=ALL,REOPT=...,POINTS=...,CSNAP_ARGS="--no-polish --max-round-trials 48".
# Every point stops at the deadline (the job's end minus 10 min) or when SIGTERM arrives (sent 10 min before
# the end by --signal and forwarded here), writes its checkpoint and exits 3. If any point exited 3 (or was
# killed by SIGTERM in its short realize or evaluate stage) the job requeues itself, which needs --requeue at
# submission; otherwise resubmit, or chain jobs with --dependency=afterany:ID. Points with an evaluation are
# skipped, a compile resumes exactly from its checkpoint, and a finished polish is reused.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR/.."
module load python
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
stamp() { date '+%m-%d %H:%M:%S'; }
END=${SLURM_JOB_END_TIME:-$(date -d "$(squeue -h -j "$SLURM_JOB_ID" -o %e 2>/dev/null)" +%s 2>/dev/null)}
DL=()
if [ -n "$END" ] && [ "$END" -gt $(( $(date +%s) + 1800 )) ]; then
  DL=(--deadline $((END - 600)))
else
  echo "$(stamp) could not read the job's end time; running without a deadline (SIGTERM still checkpoints)"
fi
: "${REOPT:?set REOPT to newton, lbfgs-converged or coordinate (there is no default)}"
mkdir -p cluster/logs/chain_snap
pids=()
names=()
trap 'echo "$(stamp) SIGTERM: telling the points to checkpoint"; kill -TERM "${pids[@]}" 2>/dev/null' TERM
for p in ${POINTS:?set POINTS, e.g. "0:2:1e-4 1.6:2:1e-4"}; do
  IFS=: read -r f t b d <<< "$p"
  name="point_${f}_t${t}_b${b}${d:+_D$d}_${REOPT}"
  # shellcheck disable=SC2086
  python cluster/chain_snap.py "$f" "$t" "$b" --reopt "$REOPT" ${d:+--D "$d"} "${DL[@]}" ${CSNAP_ARGS:-} \
    >> "cluster/logs/chain_snap/${name}.log" 2>&1 &
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
echo "$(stamp) all points stopped (unfinished: $requeue, failed: $failed)"
if [ $requeue -eq 1 ]; then
  scontrol requeue "$SLURM_JOB_ID" || echo "$(stamp) requeue failed; resubmit the same POINTS (they resume)"
fi
exit $failed
