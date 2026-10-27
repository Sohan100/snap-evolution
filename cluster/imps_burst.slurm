#!/bin/bash
#SBATCH -A m4980
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH --exclusive
#SBATCH --output=logs/%x_%j.out
#
# Debug-QOS bursts for compile chains that live on login nodes (cluster/imps_after_train.sh). One chain per node:
#   CHAINS="FIELD:T:NTRAIN:OPT:LOGIN_HOST:LOGIN_WORKERS ..." sbatch -N <number of chains> imps_burst.slurm
# For each chain: stop it on its login node (SIGTERM: the compile checkpoints), run it here with 250 workers until
# 4 minutes before the time limit, stop it (SIGTERM to its process group: checkpoint), and start it again on its login
# node unless it finished. With RESUBMIT=1 the job submits itself again while any chain is unfinished. Never two
# compiles of one ladder at once: the login chain is gone before this one starts, and back only after this one exits.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR/.."
module load python
source .venv/bin/activate
LOGS=/pscratch/sd/s/sohan100/snapevo-scratch/imps_snap/logs
END=${SLURM_JOB_END_TIME:-$(date -d "$(squeue -h -j "$SLURM_JOB_ID" -o %e)" +%s)}
nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
say() { echo "$(date '+%m-%d %H:%M:%S') [burst $SLURM_JOB_ID] $*"; }

stop_login() {   # FIELD T NTRAIN OPT HOST
  timeout 330 ssh -n -q -o BatchMode=yes -o LogLevel=ERROR "$5" "
    ps=\$(pgrep -u \$USER -f 'imps_after_train.sh $1 $2 $3 [${4:0:1}]${4:1}')
    py=\$(pgrep -u \$USER -f 'cluster/imps_snap.py [a-z]* $1 $2 --n-train $3 --optimizer [${4:0:1}]${4:1}')
    kill -TERM \$ps \$py 2>/dev/null
    for i in \$(seq 300); do [ -z \"\$py\" ] && break; kill -0 \$py 2>/dev/null || break; sleep 1; done
    for p in \$py; do kill -0 \$p 2>/dev/null && kill -KILL \$(pgrep -P \$p) \$p; done
    sleep 2; o=\$(ps -u \$USER -o pid=,ppid=,args= | awk '\$2==1 && /snap-evolution\/.venv\/bin\/python -c from multiprocessing.spawn/ {print \$1}'); [ -n \"\$o\" ] && kill -KILL \$o
    echo \$ps \$py"
}

run_chain() {    # index FIELD:T:NTRAIN:OPT:HOST:LW
  IFS=: read -r f t n opt host lw <<< "$2"
  lad="hx${f}_t${t}_n${n}_${opt}"
  log="$LOGS/${lad}.log"
  since=$(wc -l < "$log")
  say "$lad: stopping it on $host (stopped: $(stop_login "$f" "$t" "$n" "$opt" "$host"))"
  echo "$(date '+%m-%d %H:%M:%S') debug burst $SLURM_JOB_ID on ${nodes[$1]} (250 workers)" >> "$log"
  left=$(( END - $(date +%s) - 240 ))
  srun --nodes=1 --ntasks=1 --cpus-per-task=256 --cpu-bind=none -w "${nodes[$1]}" bash -c "
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    setsid bash cluster/imps_after_train.sh $f $t $n $opt 250 >> $log 2>&1 &
    pid=\$!
    ( sleep $left; kill -TERM -- -\$pid 2>/dev/null ) & w=\$!
    wait \$pid; kill \$w 2>/dev/null
    for i in \$(seq 150); do pgrep -g \$pid > /dev/null || break; sleep 1; done
    pkill -KILL -g \$pid 2>/dev/null; true"
  if tail -n +"$((since + 1))" "$log" | grep -q "snap, uniform and reeval done"; then
    say "$lad: chain done"
    return 0
  fi
  ssh -n -f -q -o BatchMode=yes "$host" "cd $PWD && setsid nohup bash cluster/imps_after_train.sh $f $t $n $opt $lw >> $log 2>&1 < /dev/null &"
  say "$lad: back on $host"
  return 1
}

i=0; pids=()
for c in $CHAINS; do
  [ $i -ge ${#nodes[@]} ] && break
  run_chain $i "$c" & pids+=($!)
  i=$((i + 1))
done
left_chains=()
i=0
for c in $CHAINS; do
  if [ $i -lt ${#pids[@]} ]; then wait "${pids[$i]}" || left_chains+=("$c"); else left_chains+=("$c"); fi
  i=$((i + 1))
done
if [ "${RESUBMIT:-0}" = 1 ] && [ ${#left_chains[@]} -gt 0 ]; then
  n=${#left_chains[@]}; [ $n -gt 8 ] && n=8
  cd cluster && CHAINS="${left_chains[*]}" RESUBMIT=1 sbatch -N "$n" --job-name="$SLURM_JOB_NAME" --export=ALL imps_burst.slurm
fi
