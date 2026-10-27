#!/bin/bash
#SBATCH -A m4980
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --output=logs/%x_%j.out
#SBATCH --open-mode=append
#
# One more SNAP compile worker for a ladder's budget points, on a whole compute node with WORKERS (default 250):
#   FIELD=1.6 T=4 NTRAIN=4 OPT=newton sbatch -q regular -t 12:00:00 --job-name=snapevo-pts-hx1.6-t4-n4 imps_points.slurm
# It runs cluster/imps_snap.py snap over the ladder's budgets: imps_snap.ti_point locks each (budget, depth) point, so
# any number of these jobs and the ladder's own chain (cluster/imps_after_train.sh) share the points, each taking the
# next one nobody holds. 10 minutes before the time limit it stops (SIGTERM: the point checkpoints) and, if points are
# left, submits a fresh copy of itself (the regular QOS refuses requeues). When no point is left it exits.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR/.."
module load python
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
lad="hx${FIELD}_t${T}_n${NTRAIN}_${OPT}"
log="/pscratch/sd/s/sohan100/snapevo-scratch/imps_snap/logs/${lad}.log"
END=${SLURM_JOB_END_TIME:-$(date -d "$(squeue -h -j "$SLURM_JOB_ID" -o %e)" +%s)}
left=$(( END - $(date +%s) - 600 ))
echo "$(date '+%m-%d %H:%M:%S') compile worker $SLURM_JOB_ID on $(hostname) (${WORKERS:-250} workers)" >> "$log"
timeout "$left" python cluster/imps_snap.py snap "$FIELD" "$T" --n-train "$NTRAIN" --optimizer "$OPT" \
  --workers "${WORKERS:-250}" >> "$log" 2>&1
rc=$?
echo "$(date '+%m-%d %H:%M:%S') compile worker $SLURM_JOB_ID exit $rc" >> "$log"
if [ $rc -eq 124 ]; then
  (cd "$SLURM_SUBMIT_DIR" && FIELD=$FIELD T=$T NTRAIN=$NTRAIN OPT=$OPT WORKERS=${WORKERS:-250} \
    sbatch -q regular -t 12:00:00 --job-name="$SLURM_JOB_NAME" --export=ALL imps_points.slurm)
fi
