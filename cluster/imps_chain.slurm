#!/bin/bash
#SBATCH -A m4980
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --output=logs/%x_%j.out
#SBATCH --open-mode=append
#
# One ladder's post-training chain (cluster/imps_after_train.sh: SNAP, Gibbs-Cincio uniform, reeval) on a whole compute
# node with WORKERS (default 250) workers: FIELD, T, NTRAIN, OPT (newton or lbfgs).
#   FIELD=0 T=3 NTRAIN=4 OPT=newton sbatch -q regular -t 12:00:00 --job-name=snapevo-ti-hx0-t3-n4-newton imps_chain.slurm
# A login node may run the same chain; cluster/imps_chain_coordinator.sh stops it when this job starts and leaves
# cluster/logs/imps_train/handover_<ladder>.go, which this job waits for (two compiles never write one ladder). Every
# step resumes from its checkpoints, so 10 minutes before the time limit the chain is stopped and a fresh copy of the
# job submitted (the regular QOS refuses scontrol requeue); cluster/imps_chain_coordinator.sh runs the ladder on its
# login node while that copy waits in the queue.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR/.."
module load python
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
lad="hx${FIELD}_t${T}_n${NTRAIN}_${OPT}"
go="cluster/logs/imps_train/handover_${lad}.go"
for i in $(seq 1 90); do [ -e "$go" ] && break; sleep 10; done
[ -e "$go" ] || { echo "$(date '+%m-%d %H:%M:%S') no handover for $lad after 15 min; stopping"; exit 1; }
END=${SLURM_JOB_END_TIME:-$(date -d "$(squeue -h -j "$SLURM_JOB_ID" -o %e)" +%s)}
left=$(( END - $(date +%s) - 600 ))
timeout "$left" bash cluster/imps_after_train.sh "$FIELD" "$T" "$NTRAIN" "$OPT" "${WORKERS:-250}" \
  >> "/pscratch/sd/s/sohan100/snapevo-scratch/imps_snap/logs/${lad}.log" 2>&1
rc=$?
echo "$(date '+%m-%d %H:%M:%S') chain $lad exit $rc"
if [ $rc -eq 124 ]; then              # the regular QOS refuses requeues: submit a fresh copy of this job
  (cd "$SLURM_SUBMIT_DIR" && FIELD=$FIELD T=$T NTRAIN=$NTRAIN OPT=$OPT WORKERS=${WORKERS:-250} \
    sbatch -q regular -t 12:00:00 --job-name="$SLURM_JOB_NAME" --export=ALL imps_chain.slurm)
fi
