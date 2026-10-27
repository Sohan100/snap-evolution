#!/bin/bash
# One ladder's post-training chain (cluster/imps_chain.slurm) through back-to-back 1-node, 4-hour interactive-QOS
# allocations, until the chain logs that it is done. Run from the repo root on a login node, under nohup:
#   nohup bash cluster/imps_chain_interactive.sh FIELD T NTRAIN OPT >> cluster/logs/imps_train/chain_int.log 2>&1 &
# The job is named like the regular compile job (snapevo-ti-hx<F>-t<T>-n<N>-<OPT>), so cluster/imps_chain_coordinator.sh
# hands the ladder over from its login node when it runs (the ladder needs a line in chain_map.txt). The interactive QOS
# allows 4 nodes per user: this takes one beside cluster/imps_interactive.sh's training allocation.
f=$1; t=$2; n=$3; opt=$4
cd "$(dirname "$0")"
lad="hx${f}_t${t}_n${n}_${opt}"
log="/pscratch/sd/s/sohan100/snapevo-scratch/imps_snap/logs/${lad}.log"
while true; do
  since=$(wc -l < "$log")
  echo "$(date '+%m-%d %H:%M:%S') $lad: requesting an interactive node"
  FIELD=$f T=$t NTRAIN=$n OPT=$opt WORKERS=250 salloc -A m4980 -C cpu -q interactive -N 1 -t 04:00:00 \
    --job-name="snapevo-ti-hx${f}-t${t}-n${n}-${opt}" srun --cpu-bind=none bash imps_chain.slurm
  echo "$(date '+%m-%d %H:%M:%S') $lad: allocation ended, exit $?"
  tail -n +"$((since + 1))" "$log" | grep -q "snap, uniform and reeval done" && break
  sleep 30
done
echo "$(date '+%m-%d %H:%M:%S') $lad: chain done"
