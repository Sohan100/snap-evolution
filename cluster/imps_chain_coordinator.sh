#!/bin/bash
# Hand ladders' post-training chains from login nodes to compute jobs (cluster/imps_chain.slurm) and back. Run on a
# login node:
#   nohup bash cluster/imps_chain_coordinator.sh MAPFILE >> cluster/logs/imps_train/coordinator.log 2>&1 &
# MAPFILE lines: FIELD T NTRAIN OPT LOGIN_NODE. Every minute, for each ladder:
#   - its job snapevo-ti-hx<F>-t<T>-n<N>-<OPT> is running: stop that ladder's chain on its login node
#     (imps_after_train.sh and its imps_snap.py) and touch cluster/logs/imps_train/handover_<ladder>.go, which the job
#     waits for; while the job runs, keep stopping any new login instance of it;
#   - handed over (.go) but no job running (requeued at its time limit, or ended) and the chain not done: remove the
#     .go and start the chain again on its login node, so the ladder keeps compiling while the job waits in the queue
#     (every step resumes from its checkpoints; the next start of the job takes it back the same way).
# Done means the chain's last "finished training; compiling" line in its log is followed by "snap, uniform and reeval
# done".
map=$1
cd "$(dirname "$0")/.."
LOGS=/pscratch/sd/s/sohan100/snapevo-scratch/imps_snap/logs
chain_done() {
  awk '/finished training; compiling/ {d = 0} /snap, uniform and reeval done/ {d = 1} END {exit !d}' "$1" 2>/dev/null
}
while true; do
  while read -r f t n opt host; do
    [ -z "$f" ] && continue
    lad="hx${f}_t${t}_n${n}_${opt}"
    go="cluster/logs/imps_train/handover_${lad}.go"
    running=0
    squeue -u "$USER" -h -t R -n "snapevo-ti-hx${f}-t${t}-n${n}-${opt}" | grep -q . && running=1
    if [ $running -eq 0 ]; then
      if [ -e "$go" ] && ! chain_done "$LOGS/${lad}.log"; then
        rm -f "$go"
        ssh -n -f -q -o BatchMode=yes -o LogLevel=ERROR "$host" "cd $PWD && setsid nohup bash cluster/imps_after_train.sh $f $t $n $opt 30 >> $LOGS/${lad}.log 2>&1 < /dev/null &" \
          && echo "$(date '+%m-%d %H:%M:%S') $lad: no job running and not done; back on $host"
      fi
      continue
    fi
    out=$(timeout 30 ssh -n -q -o BatchMode=yes -o LogLevel=ERROR "$host" "
      for p in \$(pgrep -u \$USER -f 'imps_after_train.sh $f $t $n [${opt:0:1}]${opt:1}') \$(pgrep -u \$USER -f 'cluster/imps_snap.py [a-z]* $f $t --n-train $n --optimizer [${opt:0:1}]${opt:1}'); do
        kill -TERM \$(pgrep -P \$p) \$p 2>/dev/null && echo stopped \$p; done")
    [ -n "$out" ] && echo "$(date '+%m-%d %H:%M:%S') $lad: on $host $out"
    if [ ! -e "$go" ]; then
      sleep 5
      touch "$go"
      echo "$(date '+%m-%d %H:%M:%S') $lad handed to its compute job"
    fi
  done < "$map"
  sleep 60
done
