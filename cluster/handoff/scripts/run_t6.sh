#!/bin/bash
# Run the t = 6 open-chain arm on this workstation: training, SNAP compiles, N = 100 evaluations, fallback draws.
#
#   bash scripts/run_t6.sh start      start everything in the background (safe to rerun: it resumes)
#   bash scripts/run_t6.sh stop       stop cleanly; every piece resumes from its checkpoint on the next start
#   bash scripts/status.sh            what is done and what is running
#
# Cores (48) and memory (256 GB), defaults:
#   2 ladders (h_x = 0 and h_x = 1.6) x LADDER_WORKERS = 12 workers    24 cores, up to ~2.6 GB per worker
#                                                                      (shallow depths; under 1 GB at D >= 33)
#   COMPILE_SLOTS = 1 compile at a time, 12 workers                     12 cores, up to ~40 GB
#   SAMPLE_SLOTS = 2 fallback draws at a time, 4 threads each           8 cores, up to ~40 GB each
# Override with environment variables, e.g. COMPILE_SLOTS=2 bash scripts/run_t6.sh start once a ladder has
# finished (its log says "stopping the ladder").
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SELF=$ROOT/scripts/run_t6.sh
L=$ROOT/logs
mkdir -p $L/locks
cd $ROOT/snap-evolution
source .venv/bin/activate
export SCRATCH=$ROOT/data                # training references: $SCRATCH/snapevo-scratch/open_train/refs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
T=6
stamp() { date '+%m-%d %H:%M:%S'; }
# Pool workers ignore SIGTERM (their parent checkpoints first) and outlive a parent that was killed (OOM killer,
# kill -9). Kill such orphans, workers of this package's python whose parent is no longer python; nothing else.
orphans() {
  local pid ppid args n=0
  while read -r pid ppid args; do
    case "$args" in "$ROOT/snap-evolution/.venv/bin/python"*spawn_main*) ;; *) continue ;; esac
    case "$(ps -o args= -p $ppid 2>/dev/null)" in *python*) continue ;; esac
    kill -KILL $pid 2>/dev/null && n=$((n + 1))
  done < <(ps -u $USER -o pid=,ppid=,args=)
  [ $n -gt 0 ] && echo "$(stamp) killed $n orphaned workers of an earlier run" | tee -a $L/run.log
  return 0
}

case "${1:-}" in
start)
  rm -f $L/STOP
  orphans
  for f in 0 1.6; do
    if pgrep -u $USER -f "chain_train.py $f $T " > /dev/null; then echo "ladder h_x = $f already running"; continue; fi
    setsid nohup python cluster/chain_train.py $f $T --workers ${LADDER_WORKERS:-12} >> $L/ladder_${f}_t$T.log 2>&1 < /dev/null &
    echo "$(stamp) started ladder h_x = $f (pid $!)" | tee -a $L/run.log
  done
  for s in $(seq 1 ${COMPILE_SLOTS:-1}); do
    pgrep -u $USER -f "run_t6.sh compile-loop $s\$" > /dev/null && continue
    setsid nohup bash $SELF compile-loop $s >> $L/run.log 2>&1 < /dev/null &
  done
  for s in $(seq 1 ${SAMPLE_SLOTS:-2}); do
    pgrep -u $USER -f "run_t6.sh sample-loop $s\$" > /dev/null && continue
    setsid nohup bash $SELF sample-loop $s >> $L/run.log 2>&1 < /dev/null &
  done
  echo "$(stamp) started; logs in $L, progress: bash scripts/status.sh"
  ;;
stop)
  touch $L/STOP
  pkill -TERM -u $USER -f "chain_train.py (0|1.6) $T( |$)" || true
  echo "$(stamp) stop requested: ladders checkpoint and exit; compile and draw loops exit after their current task" \
    | tee -a $L/run.log
  ;;
compile-loop)        # SNAP compile + realize + N = 100 evaluation of every point the finished depths allow
  s=$2
  echo "$(stamp) compile loop $s up"
  while [ ! -e $L/STOP ]; do
    points=$(python cluster/ring_grid_tasks.py chain-points --times $T --chain-out newton --per-node 1 2>/dev/null \
             | grep -v '^#' | tac)
    did=0
    for p in $points; do
      [ -e $L/STOP ] && break
      IFS=: read f t b d <<< "$p"
      exec 9> $L/locks/$p.lock
      flock -n 9 || { exec 9>&-; continue; }
      ev=$(printf "runs/xxz_d0.25_hx%s/t%s/chain/newton/eval_b%.1e_D%s.json" $f $t $b $d)
      if [ ! -e "$ev" ]; then
        echo "$(stamp) start $p" >> $L/compile_$s.log
        python cluster/chain_snap.py $f $t $b --D $d --reopt newton --chain-out newton --edge-zero-t --env-layers 4 \
          >> $L/compile_$s.log 2>&1
        echo "$(stamp) done $p rc $?" >> $L/compile_$s.log
        did=1
      fi
      exec 9>&-
    done
    [ $did = 0 ] && sleep 900
  done
  echo "$(stamp) compile loop $s stopped"
  ;;
sample-loop)         # fallback draws of every evaluated point (the main metric with the KN fallbacks drawn)
  s=$2
  echo "$(stamp) draw loop $s up"
  while [ ! -e $L/STOP ]; do
    did=0
    for e in runs/xxz_d0.25_hx*/t$T/chain/newton/eval_*.json; do
      [ -e "$e" ] || continue
      [ -e $L/STOP ] && break
      python -c "import json,sys; sys.exit(0 if 'sampled' in json.load(open('$e')) else 1)" && continue
      OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 python cluster/sample_chain.py $e >> $L/sample_$s.log 2>&1
      did=1
    done
    [ $did = 0 ] && sleep 900
  done
  echo "$(stamp) draw loop $s stopped"
  ;;
*) echo "usage: bash scripts/run_t6.sh start|stop"; exit 2 ;;
esac
