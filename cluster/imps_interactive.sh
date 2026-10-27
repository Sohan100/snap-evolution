#!/bin/bash
# Run infinite-chain ladders through back-to-back 4-hour interactive-QOS allocations (at most 4 nodes per job and two
# jobs per user), resuming from the checkpoints each time, until every ladder has finished. Run from the repo root,
# under nohup:
#   nohup bash cluster/imps_interactive.sh "1.6:4:8 0:4:8 1.6:4:4" >> cluster/logs/imps_train/interactive.log 2>&1 &
# Each entry is FIELD:T:NTRAIN, in priority order. In each allocation the first unfinished ladder runs split over
# SPLIT_NODES nodes (cluster/imps_split.py, one start per node) and the next ones get one node each
# (cluster/imps_train.py), all with WORKERS workers per node; the job asks for at most 4 nodes
# (cluster/imps_interactive_alloc.sh does the work inside it). A ladder stops 15 minutes before the allocation ends
# (exit 3). A finished ladder (.done) is skipped, and so is one a split job on another QOS has claimed (.split).
# A new allocation is requested only once no imps-int job is left.
LADDERS=$1; SPLIT_NODES=${SPLIT_NODES:-3}; export WORKERS=${WORKERS:-250}
cd "$(dirname "$0")/.."
mkdir -p cluster/logs/imps_train
stem() { echo "cluster/logs/imps_train/hx${1}_t${2}_n${3}"; }
while true; do
  todo=()
  for L in $LADDERS; do
    IFS=: read -r f t n <<< "$L"
    s=$(stem $f $t $n)
    [ -e "$s.done" ] || [ -e "$s.split" ] || todo+=("$L")
  done
  [ ${#todo[@]} -eq 0 ] && { echo "$(date '+%m-%d %H:%M:%S') all ladders finished or claimed by split jobs"; break; }
  if squeue -u "$USER" -h -n imps-int | grep -q .; then sleep 60; continue; fi
  N=$(( SPLIT_NODES + ${#todo[@]} - 1 )); [ $N -gt 4 ] && N=4
  echo "$(date '+%m-%d %H:%M:%S') requesting $N interactive nodes: ${todo[0]} split over $SPLIT_NODES, then ${todo[*]:1}"
  salloc -A m4980 -C cpu -q interactive -N "$N" -t 04:00:00 --job-name=imps-int \
    bash cluster/imps_interactive_alloc.sh "$SPLIT_NODES" "${todo[@]}"
  echo "$(date '+%m-%d %H:%M:%S') allocation ended, exit $?"
  sleep 30
done
