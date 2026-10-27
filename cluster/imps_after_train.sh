#!/bin/bash
# Wait for an infinite-chain ladder to finish training (snapevo.imps_train.ladder_status: target or stall rule), then
# run SNAP, the uniform arm and the re-evaluation on it (cluster/imps_snap.py). Run from the repo root, under nohup:
#   nohup bash cluster/imps_after_train.sh FIELD T NTRAIN OPTIMIZER [WORKERS] >> LOG 2>&1 &
# FIELD 1.6 or 0, OPTIMIZER newton or lbfgs (the ladder's training optimizer; the compile re-optimizes by Newton).
f=$1; t=$2; n=$3; opt=$4; W=${5:-30}
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
done_training() {
  .venv/bin/python -c "
import sys; sys.path.insert(0, 'cluster')
from imps_train import depth_grid
from snapevo.hamiltonian import XXZ
from snapevo.imps_train import ladder_status
st = ladder_status(XXZ(0.25, $f), $t, depth_grid($t), n_train=$n, optimizer='$opt')
sys.exit(0 if st['finished'] else 1)"
}
until done_training; do sleep 600; done
echo "$(date '+%m-%d %H:%M:%S') ladder hx$f t$t n$n $opt finished training; compiling"
rc=0
for cmd in snap uniform reeval; do
  .venv/bin/python cluster/imps_snap.py $cmd $f $t --n-train $n --optimizer $opt --workers $W || { rc=$?; echo "$(date '+%m-%d %H:%M:%S') $cmd failed (exit $rc)"; }
done
[ $rc -eq 0 ] && echo "$(date '+%m-%d %H:%M:%S') ladder hx$f t$t n$n $opt: snap, uniform and reeval done"
exit $rc
