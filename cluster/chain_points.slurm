#!/bin/bash
#SBATCH -A m4980
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --job-name=snapevo-cpts
#SBATCH --output=logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --signal=B:TERM@600
#
# Open-chain arm on one compute node: compiles, N = 100 evaluations and fallback draws of every point the finished
# depths allow, for TIMES (comma-separated, default 1,2,4). SLOTS loops (default 10; a compile keeps 12 cores busy)
# take FIELD:T:BUDGET:D points from cluster/ring_grid_tasks.py chain-points, longest first (the login-node runners,
# cluster/logs/chain/login_compile.sh, take the cheapest first), under the same per-point flocks in
# cluster/logs/chain/point_locks, so no point is compiled twice. A loop with no point left draws the KN fallbacks
# (cluster/sample_chain.py) of an evaluation that lacks them. Points come back as ladders finish depths; after
# IDLE_MAX seconds (default 3600) with nothing to do, the job ends. Submit from cluster/, e.g.
#   sbatch -q preempt --requeue -t 48:00:00 --time-min 4:00:00 --export=ALL,TIMES=1,2,4 chain_points.slurm
# SIGTERM (preemption, or 10 minutes before the end): every compile checkpoints and exits 3, and the job requeues
# itself (needs --requeue); a compile resumes from its checkpoint.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR/.."
module load python
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
TIMES=${TIMES:-1,2,4} SLOTS=${SLOTS:-10} IDLE_MAX=${IDLE_MAX:-3600}
LOCKS=cluster/logs/chain/point_locks
LOGS=cluster/logs/chain_points
mkdir -p $LOCKS $LOGS
STOP=$LOGS/stop_${SLURM_JOB_ID}
rm -f $STOP
stamp() { date '+%m-%d %H:%M:%S'; }
END=${SLURM_JOB_END_TIME:-$(date -d "$(squeue -h -j "$SLURM_JOB_ID" -o %e 2>/dev/null)" +%s 2>/dev/null)}
DL=()
[ -n "$END" ] && [ "$END" -gt $(( $(date +%s) + 1800 )) ] && DL=(--deadline $((END - 600)))

slot() {                       # one loop: a compile if any point is free, else a draw, else wait
  local s=$1 L=$LOGS/slot_${SLURM_JOB_ID}_$1.log idle=0 p f t b d ev did rc e tt
  while [ ! -e $STOP ]; do
    did=0
    for p in $(python cluster/ring_grid_tasks.py chain-points --times $TIMES --chain-out newton --per-node 1 \
               2>/dev/null | grep -v '^#'); do
      [ -e $STOP ] && break
      IFS=: read f t b d <<< "$p"
      ev=$(printf "runs/xxz_d0.25_hx%s/t%s/chain/newton/eval_b%.1e_D%s.json" $f $t $b $d)
      [ -e "$ev" ] && continue
      [ -e $LOCKS/$p.failed_$SLURM_JOB_ID ] && continue          # failed once in this job: leave it to the next
      exec 9> $LOCKS/$p.lock
      flock -n 9 || { exec 9>&-; continue; }
      if [ ! -e "$ev" ]; then
        echo "$(stamp) start $p" >> $L
        python cluster/chain_snap.py $f $t $b --D $d --reopt newton --chain-out newton --edge-zero-t "${DL[@]}" \
          >> $L 2>&1 &
        wait $!; rc=$?
        echo "$(stamp) done $p rc $rc" >> $L
        [ $rc != 0 ] && [ ! -e $STOP ] && touch $LOCKS/$p.failed_$SLURM_JOB_ID
        did=1
      fi
      exec 9>&-
      break                    # re-read the list: other slots and the login runners take points too
    done
    if [ $did = 0 ] && [ ! -e $STOP ]; then
      for e in $(for tt in ${TIMES//,/ }; do ls runs/xxz_d0.25_hx*/t$tt/chain/newton/eval_*.json 2>/dev/null; done); do
        python -c "import json,sys; sys.exit(0 if 'sampled' in json.load(open('$e')) else 1)" && continue
        exec 8> $LOCKS/sample_$(echo $e | tr / _).lock
        flock -n 8 || { exec 8>&-; continue; }
        echo "$(stamp) draw $e" >> $L
        OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 python cluster/sample_chain.py $e >> $L 2>&1
        exec 8>&-
        did=1
        break
      done
    fi
    if [ $did = 0 ]; then
      idle=$((idle + 300)); [ $idle -ge $IDLE_MAX ] && { echo "$(stamp) slot $s idle ${IDLE_MAX}s: done" >> $L; return; }
      sleep 300
    else
      idle=0
    fi
  done
}

pids=()
# slurm's --signal reaches only this script: stop the loops and pass TERM to what each slot is running
trap 'echo "$(stamp) SIGTERM: compiles checkpoint"; touch $STOP; for q in "${pids[@]}"; do pkill -TERM -P $q 2>/dev/null; done' TERM
echo "$(stamp) chain points t = $TIMES: $SLOTS slots on $(hostname)"
for s in $(seq 1 $SLOTS); do
  slot $s &
  pids+=($!)
  sleep 5                      # stagger, so the slots do not all lock the same first point
done
for q in "${pids[@]}"; do
  wait $q; while [ $? -gt 128 ] && kill -0 $q 2>/dev/null; do wait $q; done
done
echo "$(stamp) all slots finished"
if [ -e $STOP ]; then
  scontrol requeue "$SLURM_JOB_ID" || echo "$(stamp) requeue failed; resubmit (compiles resume from checkpoints)"
fi
