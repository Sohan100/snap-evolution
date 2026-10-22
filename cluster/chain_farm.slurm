#!/bin/bash
#SBATCH -A m4980
#SBATCH -C cpu
#SBATCH --exclusive
#SBATCH --job-name=snapevo-cfarm
#SBATCH --output=logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --signal=B:TERM@600
#
# Open-chain arm on a farm of workers over all of the job's nodes (cluster/chain_farm.py, snapevo.chain_farm):
# srun starts WPN worker processes per node (THREADS BLAS threads each; on GPU nodes GPN per GPU, --gpu local,
# plus CPN CPU workers on the host cores), and one master process on the first node runs MODE (ladder |
# precompute | compile | bench) for FIELD and T, leasing its pools from them, so a Newton step's 2 len(z)
# gradients run over every node at once. Submit by hand from cluster/ with the settings in the environment
# (ARGS may hold commas, so pass everything with --export=ALL), e.g.
#   MODE=ladder FIELD=1.6 T=8 sbatch --export=ALL -N 20 -q regular -t 24:00:00 --time-min 4:00:00 --requeue \
#     chain_farm.slurm
#   MODE=precompute FIELD=1.6 T=8 ARGS="--tasks 2" sbatch --export=ALL -N 10 ... chain_farm.slurm
#   GPU=1 MODE=ladder FIELD=1.6 T=10 sbatch --export=ALL -N 20 -C gpu -A m4980_g --gpus-per-node 4 ... chain_farm.slurm
#   MODE=compile FIELD=1.6 T=8 ARGS="1e-3,1e-4 --all-depths --edge-zero-t --chain-out newton" sbatch ... (budgets
#     first)
# Defaults: CPU nodes WPN=128 single-threaded workers (80 at t = 10: a worker needs ~4.6 GB at D = 37 and more
# with D, so lower WPN beyond D ~ 60); GPU nodes GPN=1 worker per GPU (one process saturates an A100 at t >= 8:
# 2, 4 and 8 processes per GPU gave the same throughput) plus CPN=48 CPU workers (32 at t = 10). A task goes to
# whichever worker of its input group is free, so GPU and CPU workers share each Hessian by speed, and the
# tasks of a lost worker run on the others. The master stops 10 minutes before the end (and on SIGTERM:
# --signal), every run checkpoints, the workers are told to stop, and the job requeues itself when the master
# exits 3 (needs --requeue); a rerun continues where it stopped.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR/.."
module load python
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
: "${MODE:?set MODE to ladder, precompute or compile}" "${FIELD:?set FIELD (1.6 or 0)}" "${T:?set T}"
stamp() { date '+%m-%d %H:%M:%S'; }
END=${SLURM_JOB_END_TIME:-$(date -d "$(squeue -h -j "$SLURM_JOB_ID" -o %e 2>/dev/null)" +%s 2>/dev/null)}
DL=()
if [ -n "$END" ] && [ "$END" -gt $(( $(date +%s) + 1800 )) ]; then
  DL=(--deadline $((END - 600)))
fi
LOGS=cluster/logs/chain_farm
mkdir -p "$LOGS"
ADDR="$LOGS/farm_${SLURM_JOB_ID}.json"
rm -f "$ADDR"
NN=${SLURM_JOB_NUM_NODES:-1}
if [ "${GPU:-0}" = 1 ]; then
  GPN=${GPN:-1}
  NG=4
  WPN=$((GPN * NG))
  srun --exact --nodes="$NN" --ntasks-per-node="$WPN" --cpus-per-task=2 --gpus-per-node="$NG" --gpu-bind=none \
    python -m snapevo.chain_farm worker --address-file "$ADDR" --gpu local >> "$LOGS/workers_${SLURM_JOB_ID}.log" 2>&1 &
  CPN=${CPN:-$([ "${T%.*}" -ge 10 ] && echo 32 || echo 48)}
  if [ "$CPN" -gt 0 ]; then
    srun --overlap --nodes="$NN" --ntasks-per-node="$CPN" --cpus-per-task=1 \
      python -m snapevo.chain_farm worker --address-file "$ADDR" >> "$LOGS/workers_${SLURM_JOB_ID}.log" 2>&1 &
  fi
  EXPECTED=$(( NN * (WPN + CPN) ))
else
  THREADS=${THREADS:-1}
  WPN=${WPN:-$([ "${T%.*}" -ge 10 ] && echo 80 || echo 128)}
  srun --nodes="$NN" --ntasks-per-node="$WPN" --cpus-per-task="$THREADS" \
    python -m snapevo.chain_farm worker --address-file "$ADDR" --threads "$THREADS" \
    >> "$LOGS/workers_${SLURM_JOB_ID}.log" 2>&1 &
  EXPECTED=$(( NN * WPN ))
fi
WORKERS_PID=$!
echo "$(stamp) $MODE h_x = $FIELD, t = $T: $EXPECTED workers on $NN node(s)"
# shellcheck disable=SC2086
python cluster/chain_farm.py "$MODE" "$FIELD" "$T" ${ARGS:-} --farm-file "$ADDR" --farm-expected "$EXPECTED" \
  --farm-minimum $(( EXPECTED * 9 / 10 )) "${DL[@]}" >> "$LOGS/${MODE}_${FIELD}_t${T}.log" 2>&1 &
MASTER=$!
trap 'echo "$(stamp) SIGTERM: telling the master to checkpoint"; kill -TERM "$MASTER" 2>/dev/null' TERM
wait "$MASTER"
rc=$?
while [ $rc -gt 128 ] && kill -0 "$MASTER" 2>/dev/null; do
  wait "$MASTER"
  rc=$?
done
echo "$(stamp) master exit $rc"
sleep 20
kill "$WORKERS_PID" 2>/dev/null
wait 2>/dev/null
if [ $rc -eq 3 ] || [ $rc -eq 143 ]; then
  scontrol requeue "$SLURM_JOB_ID" || echo "$(stamp) requeue failed; resubmit (the runs resume)"
fi
