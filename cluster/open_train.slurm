#!/bin/bash
#SBATCH -A m4980
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --job-name=snapevo-open
#SBATCH --output=logs/%x_%j.out
#SBATCH --open-mode=append
#
# Fine-tune the ladder cells on the 100-site open chain (open_train.py) on one whole node, for the times in
# $TIMES (default: every time in the frozen list). The two halves are separate jobs and may run at once; each
# run holds a lock per time, so a second job on the same times fails at once. From cluster/, once
# pool.py status shows the points at those times done:
#   python open_train.py freeze --times 1,2,4,6       # the task list, with each depth's settings
#   python open_train.py list --times 1,2,4,6         # the tasks, their estimates and the simulated wall time
#   TIMES=1,2,4,6 sbatch -q preempt --requeue -t 48:00:00 --job-name=snapevo-open-lo open_train.slurm
#   python open_train.py freeze --times 8,10
#   TIMES=8,10 sbatch -q preempt --requeue -t 48:00:00 --job-name=snapevo-open-hi open_train.slurm
# (sbatch exports the submitting environment; with --export, quote the commas: --export=ALL,TIMES='8,10',
# or write the times as 8:10.)
# Tasks run pinned to their own cores, longest estimated remaining time first, within the node's cores and memory.
# Training stops in time to evaluate before the job ends and saves a provisional result; a requeued or
# resubmitted job resumes every task from its stage files in $SCRATCH/snapevo-scratch/open_train/.
# Exit 0: all done; 1: a task failed (see the summary at the end of the log); 2: some unfinished.
# OT_ARGS passes more options to the runner, for example --export=ALL,TIMES=8:10,OT_ARGS="--mem-gb 400".
set -euo pipefail
cd "$SLURM_SUBMIT_DIR/.."

# Site setup, NERSC Perlmutter.
module load python
source .venv/bin/activate
export SNAPEVO_RUNS=${SNAPEVO_RUNS:-$PWD/runs}

echo "open_train.slurm: TIMES=${TIMES:-all} OT_ARGS=${OT_ARGS:-}"
exec python cluster/open_train.py run ${TIMES:+--times "$TIMES"} ${OT_ARGS:-}
