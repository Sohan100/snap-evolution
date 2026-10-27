#!/bin/bash
#SBATCH -A m4980
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --job-name=snapevo-imps
#SBATCH --output=logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --signal=B:TERM@1800
#
# One infinite-chain Newton ladder (cluster/imps_train.py) on a whole node: FIELD, T, NTRAIN (4 or 8), WORKERS.
#   FIELD=1.6 T=4 NTRAIN=8 sbatch -q preempt --requeue -t 48:00:00 imps_train.slurm
# Deadline 30 minutes before the job's end (checkpoints are saved every step); exit 3 requeues the job.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR/.."
module load python
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
END=${SLURM_JOB_END_TIME:-$(date -d "$(squeue -h -j "$SLURM_JOB_ID" -o %e 2>/dev/null)" +%s 2>/dev/null)}
DL=()
if [ -n "$END" ]; then DL=(--deadline $((END - 1800))); fi
mkdir -p cluster/logs/imps_train
python cluster/imps_train.py "$FIELD" "$T" --n-train "${NTRAIN:-4}" --workers "${WORKERS:-120}" "${DL[@]}" \
  >> "cluster/logs/imps_train/hx${FIELD}_t${T}_n${NTRAIN:-4}.log" 2>&1
rc=$?
echo "$(date '+%m-%d %H:%M:%S') ladder exit $rc"
if [ $rc -eq 3 ] || [ $rc -eq 143 ]; then scontrol requeue "$SLURM_JOB_ID"; fi
