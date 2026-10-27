#!/bin/bash
#SBATCH -A m4980
#SBATCH -C cpu
#SBATCH -N 3
#SBATCH --exclusive
#SBATCH --job-name=snapevo-isplit
#SBATCH --output=logs/%x_%j.out
#SBATCH --open-mode=append
#
# One infinite-chain Newton ladder with its three starts on three nodes (cluster/imps_split.py): FIELD, T, NTRAIN,
# WORKERS (per node, default 250 of the node's 256 hardware threads; each worker holds ~0.1-0.2 GB at t = 4).
#   FIELD=1.6 T=4 NTRAIN=8 sbatch -q preempt --requeue -t 10:00:00 --time-min 03:00:00 imps_split.slurm
# Deadline 20 minutes before the job's end (checkpoints are saved every step); exit 3 requeues the job.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR/.."
module load python
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
END=${SLURM_JOB_END_TIME:-$(date -d "$(squeue -h -j "$SLURM_JOB_ID" -o %e 2>/dev/null)" +%s 2>/dev/null)}
DL=()
if [ -n "$END" ]; then DL=(--deadline $((END - 1200))); fi
mkdir -p cluster/logs/imps_train
python cluster/imps_split.py "$FIELD" "$T" --n-train "${NTRAIN:-4}" --workers "${WORKERS:-250}" "${DL[@]}"
rc=$?
echo "$(date '+%m-%d %H:%M:%S') split ladder hx${FIELD} t${T} n${NTRAIN:-4} exit $rc"
if [ $rc -eq 3 ] || [ $rc -eq 143 ]; then scontrol requeue "$SLURM_JOB_ID"; fi
