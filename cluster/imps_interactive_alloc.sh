#!/bin/bash
# The work inside one imps-int allocation (cluster/imps_interactive.sh): SPLIT_NODES then FIELD:T:NTRAIN entries.
# The first ladder runs split over the first SPLIT_NODES nodes (cluster/imps_split.py --interactive, driven from
# here); each further ladder gets one of the remaining nodes (cluster/imps_train.py). Ladders without a node wait for
# the next allocation.
SPLIT_NODES=$1; shift
cd "$(dirname "$0")/.."
module load python >/dev/null 2>&1
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mapfile -t hosts < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
DL=$(( $(date +%s) + 4*3600 - 900 ))
stem() { echo "cluster/logs/imps_train/hx${1}_t${2}_n${3}"; }
k=$SPLIT_NODES; [ $k -gt ${#hosts[@]} ] && k=${#hosts[@]}
IFS=: read -r f t n <<< "$1"
split_hosts=$(IFS=,; echo "${hosts[*]:0:$k}")
( python cluster/imps_split.py "$f" "$t" --n-train "$n" --workers "$WORKERS" --hosts "$split_hosts" \
    --deadline "$DL" --interactive; echo "$(date '+%m-%d %H:%M:%S') split ladder $1 on $split_hosts exit $?" ) &
shift
i=$k
for L in "$@"; do
  [ $i -ge ${#hosts[@]} ] && break
  IFS=: read -r f t n <<< "$L"
  s=$(stem $f $t $n)
  h=${hosts[$i]}; i=$((i + 1))
  ( srun -N 1 -n 1 -c 256 --cpu-bind=none --overlap -w "$h" python cluster/imps_train.py "$f" "$t" --n-train "$n" \
      --workers "$WORKERS" --deadline "$DL" >> "$s.log" 2>&1
    rc=$?; [ $rc -eq 0 ] && touch "$s.done"; echo "$(date '+%m-%d %H:%M:%S') ladder $L on $h exit $rc" ) &
done
wait
