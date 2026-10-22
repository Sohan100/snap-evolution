#!/bin/bash
#SBATCH -A m4980
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --job-name=snapevo-rcmp
#SBATCH --output=logs/%x_%j.out
#SBATCH --open-mode=append
#
# Ring re-optimizer comparison: WORKERS single-core workers on one node pulling tasks from TASKS, a file of lines
#   VARIANT FIELD T BUDGET D
# VARIANT is newton, lbfgsc (lbfgs-converged) or coord (coordinate), optionally with a search-breadth suffix
# +top12, +trials96, +gentle or +alldepths (D ignored); each task runs cluster/chain_snap.py --arm ring with that
# re-optimizer, production's realize (edges.fit_edges) and evaluation, records in runs/<tag>/t<T>/ring2/<VARIANT>/.
# VARIANT prod reruns the production compile (snapevo.cli compile) for its time only, into $PRODTIME (a scratch
# runs directory holding copies of the cells.json files), never into runs/. Submit by hand from cluster/, e.g.
#   sbatch -q regular -t 24:00:00 --export=ALL,TASKS=$PWD/jobs/ring_compare.txt,WORKERS=64 ring_compare.slurm
# Several jobs may share one TASKS file (claims are locked); a task whose eval_ record exists is skipped quickly.
# Each task stops 15 minutes before the job's end (chain_snap.py --deadline: checkpoint, exit 3; the production
# compile by timeout, resuming from its own periodic checkpoint) and its line is put back at the end of TASKS,
# so resubmitting the same file continues every unfinished task where it stopped. Memory: a ring
# compile with its realize and evaluation stays under 2 GB, except the evaluation of t >= 6 tight points (chi
# 1024-2048, up to ~8 GB): keep WORKERS <= 32 for files with those.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR/.."
module load python
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
: "${TASKS:?set TASKS to a task file (VARIANT FIELD T BUDGET D per line)}"
WORKERS=${WORKERS:-64}
PRODTIME=${PRODTIME:-$SCRATCH/snapevo-scratch/chain_snap_dev/prodtime}
LOGS=cluster/logs/ring_compare
mkdir -p "$LOGS"
stamp() { date '+%m-%d %H:%M:%S'; }
END=${SLURM_JOB_END_TIME:-$(date -d "$(squeue -h -j "$SLURM_JOB_ID" -o %e 2>/dev/null)" +%s 2>/dev/null)}
STOP=$(( ${END:-$(( $(date +%s) + 86400 ))} - 900 ))
requeue_task() { flock "$TASKS.lock" bash -c 'echo "$2" >> "$1"' _ "$TASKS" "$1"; }
claim() {
  flock "$TASKS.lock" bash -c '
    n=$(cat "$1.next" 2>/dev/null || echo 0)
    l=$(sed -n "$((n + 1))p" "$1")
    if [ -n "$l" ]; then echo $((n + 1)) > "$1.next"; fi
    echo "$l"' _ "$TASKS"
}
worker() {
  local core=$1 line v f t b D mode extra reopt log rc s0
  local -a args depth
  while [ "$(date +%s)" -lt "$STOP" ]; do
    line=$(claim)
    [ -z "$line" ] && return 0
    read -r v f t b D <<< "$line"
    mode=${v%%+*}
    extra=${v#"$mode"}
    log="$LOGS/${v}_${f}_t${t}_b${b}_D${D}.log"
    case $mode in
      newton) reopt=newton ;; lbfgsc) reopt=lbfgs-converged ;; coord) reopt=coordinate ;; prod) reopt= ;;
      *) echo "$(stamp) unknown variant $v"; continue ;;
    esac
    args=()
    depth=(--D "$D")
    case $extra in
      "") ;; +top12) args=(--top-k 12) ;; +trials96) args=(--max-round-trials 96) ;;
      +gentle) args=(--relax-ladder gentle) ;; +alldepths) args=(--all-depths); depth=() ;;
      *) echo "$(stamp) unknown variant $v"; continue ;;
    esac
    s0=$(date +%s)
    if [ "$mode" = prod ]; then
      m=$([ "$f" = 1.6 ] && echo A || echo B)
      SNAPEVO_RUNS=$PRODTIME timeout $(( STOP - $(date +%s) )) taskset -c "$core" python -m snapevo.cli compile \
        --model "$m" --t "$t" --budget "$b" --D "$D" >> "$log" 2>&1
    else
      taskset -c "$core" python cluster/chain_snap.py "$f" "$t" "$b" --arm ring "${depth[@]}" --reopt "$reopt" \
        --ring-out "ring2/$v" --edge-fit lbfgs --deadline "$STOP" "${args[@]}" >> "$log" 2>&1
    fi
    rc=$?
    if [ $rc -eq 3 ] || [ $rc -eq 124 ] || [ $rc -eq 143 ]; then
      requeue_task "$line"
    fi
    echo "$(stamp) core $core: $line exit $rc after $(( $(date +%s) - s0 ))s"
  done
}
for ((k = 0; k < WORKERS; k++)); do
  worker $((k * 128 / WORKERS)) &
done
wait
echo "$(stamp) all workers stopped (no task left, or the job's end is near: resubmit the same TASKS)"
