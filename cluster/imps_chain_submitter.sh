#!/bin/bash
# Submit a compute-node compile job (cluster/imps_chain.slurm) for each ladder as soon as it finishes training. Run on a
# login node:  nohup bash cluster/imps_chain_submitter.sh LISTFILE >> cluster/logs/imps_train/submitter.log 2>&1 &
# LISTFILE lines: FIELD T NTRAIN OPT LOGIN_NODE (the login node whose watcher would otherwise compile it; the line is
# also appended to chain_map.txt so cluster/imps_chain_coordinator.sh hands that work over when the job starts).
list=$1
cd "$(dirname "$0")/.."
source .venv/bin/activate
finished() {
  python -c "
import sys; sys.path.insert(0, 'cluster')
from imps_train import depth_grid
from snapevo.hamiltonian import XXZ
from snapevo.imps_train import ladder_status
sys.exit(0 if ladder_status(XXZ(0.25, $1), $2, depth_grid($2), n_train=$3, optimizer='$4')['finished'] else 1)"
}
while true; do
  left=0
  while read -r f t n opt host; do
    [ -z "$f" ] && continue
    name="snapevo-ti-hx${f}-t${t}-n${n}-${opt}"
    done_file="cluster/logs/imps_train/submitted_hx${f}_t${t}_n${n}_${opt}"
    [ -e "$done_file" ] && continue
    left=$((left + 1))
    finished "$f" "$t" "$n" "$opt" || continue
    grep -q "^$f $t $n $opt " cluster/logs/imps_train/chain_map.txt || echo "$f $t $n $opt $host" >> cluster/logs/imps_train/chain_map.txt
    (cd cluster && FIELD=$f T=$t NTRAIN=$n OPT=$opt WORKERS=250 sbatch -q regular -t 12:00:00 --job-name="$name" \
      --export=ALL imps_chain.slurm) && touch "$done_file"
    echo "$(date '+%m-%d %H:%M:%S') hx$f t$t n$n $opt finished training: submitted $name"
  done < "$list"
  [ $left -eq 0 ] && { echo "$(date '+%m-%d %H:%M:%S') every listed ladder submitted"; break; }
  sleep 600
done
