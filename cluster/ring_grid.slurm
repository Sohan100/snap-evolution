#!/bin/bash
#SBATCH -A m4980
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --job-name=snapevo-rgrid
#SBATCH --output=logs/%x_%j.out
#SBATCH --open-mode=append
#
# Production ring grid (Sohan, 18 Oct): WORKERS single-core workers pulling lines from TASKS
# (cluster/ring_grid_tasks.py writes them):
#   ref FIELD T DT CUTOFF CHI K   build one N = 100 reference into runs/<tag>/t<T>/ref (snapevo.cli reference)
#   point FIELD T BUDGET D        compile depth D of runs/<tag>/t<T>/ring2/ladder.json at BUDGET with
#                                 compile_chain's protocol and the newton re-optimizer, realize with the edge fit
#                                 EDGE_FIT (default lbfgs-uncapped: production's L-BFGS-B fit without its cap; it
#                                 leaves most edge angles at multiples of pi/4, the Newton fit none, which cost
#                                 2 T/q more at t = 1 for the same c, and a Newton fit at N_fit = 18 takes tens of
#                                 hours per point; EDGE_FIT=newton or auto selects it), evaluate at N = 100
#                                 (make_jobs' cutoff/chi rules, reference step 0.1 at h_x = 1.6), records in
#                                 runs/<tag>/t<T>/ring2/protocol/
# Submit by hand from cluster/, e.g.
#   mkdir -p jobs_ring2 && python ring_grid_tasks.py refs --out jobs_ring2/refs.txt
#   sbatch -q preempt --requeue -t 24:00:00 --export=ALL,TASKS=$PWD/jobs_ring2/refs.txt,WORKERS=54 ring_grid.slurm
#   (once ladders are trained; --times restricts to the finished ones:)
#   python ring_grid_tasks.py points --out jobs_ring2/points.txt
#   sbatch -q preempt --requeue -t 48:00:00 --export=ALL,TASKS=$PWD/jobs_ring2/points.txt ring_grid.slurm
#   (a second job with the same TASKS uses a second node)
# The job requeues itself (needs --requeue) while TASKS has unclaimed lines.
# Several jobs (or nodes) may share one TASKS file: claims are locked. Each point stops 15 minutes before the
# job's end (chain_snap.py --deadline: checkpoint, exit 3) and a reference is cut by timeout; either way the
# line goes back to the end of TASKS, so resubmitting the same file continues (a compile resumes exactly from its
# checkpoint; a point whose eval_ record exists is skipped at once). A point cut inside its realize or
# evaluation by the job's end may not get its line back: before resubmitting, regenerate the list
# (ring_grid_tasks.py points skips every point with an eval_ record). Memory: up to ~8 GB for the evaluations and
# references at chi 2048 (t >= 6, budgets 3e-9 and 1e-9, REEVAL points); WORKERS <= 64 keeps that within a node.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR/.."
module load python
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
: "${TASKS:?set TASKS to a task file from cluster/ring_grid_tasks.py}"
WORKERS=${WORKERS:-64}
RING_OUT=${RING_OUT:-ring2/protocol}
EDGE_FIT=${EDGE_FIT:-lbfgs-uncapped}
LOGS=cluster/logs/ring_grid
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
  local core=$1 line kind rc s0 log
  while [ "$(date +%s)" -lt "$STOP" ]; do
    line=$(claim)
    [ -z "$line" ] && return 0
    read -r kind a1 a2 a3 a4 a5 a6 <<< "$line"
    s0=$(date +%s)
    case $kind in
      ref)
        m=$([ "$a1" = 1.6 ] && echo A || echo B)
        log="$LOGS/ref_${a1}_t${a2}_dt${a3}_c${a4}_x${a5}_k${a6}.log"
        timeout $(( STOP - $(date +%s) )) taskset -c "$core" python -m snapevo.cli reference --model "$m" --t "$a2" \
          --dt "$a3" --cutoff "$a4" --chi-max "$a5" --only "$a6" >> "$log" 2>&1
        ;;
      point)
        log="$LOGS/point_${a1}_t${a2}_b${a3}_D${a4}.log"
        taskset -c "$core" python cluster/chain_snap.py "$a1" "$a2" "$a3" --arm ring --D "$a4" --reopt newton \
          --ring-cells ring2/ladder.json --ring-out "$RING_OUT" --edge-fit "$EDGE_FIT" --deadline "$STOP" >> "$log" 2>&1
        ;;
      *)
        echo "$(stamp) unknown task: $line"; continue ;;
    esac
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
left=$(( $(wc -l < "$TASKS") - $(cat "$TASKS.next" 2>/dev/null || echo 0) ))
echo "$(stamp) all workers stopped, $left task(s) not yet claimed"
if [ "$left" -gt 0 ]; then
  scontrol requeue "$SLURM_JOB_ID" || echo "$(stamp) requeue failed; resubmit the same TASKS"
fi
