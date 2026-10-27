#!/bin/bash
# SNAP compile workers (cluster/imps_points.slurm's work) in back-to-back 4-hour interactive-QOS allocations, one worker
# per node, until an allocation finds nothing left to do. Run from the repo root on a login node, under nohup:
#   nohup bash cluster/imps_points_interactive.sh "1.6:4:4 1.6:4:4 0:4:4 0:4:4" >> cluster/logs/imps_train/points_int.log 2>&1 &
# Each entry FIELD:T:NTRAIN (Newton ladders) gets one node; imps_snap.ti_point's per-point locks share the points with
# the ladder's other workers. Stops when an allocation lasts under 15 minutes (every worker found no point left).
ENTRIES=$1
cd "$(dirname "$0")/.."
N=$(wc -w <<< "$ENTRIES"); [ "$N" -gt 4 ] && N=4
while true; do
  t0=$(date +%s)
  echo "$(date '+%m-%d %H:%M:%S') requesting $N interactive nodes for $ENTRIES"
  salloc -A m4980 -C cpu -q interactive -N "$N" -t 04:00:00 --job-name=snapevo-pts-int bash -c '
    nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
    end=$(( $(date +%s) + 4 * 3600 - 600 )); i=0
    for e in '"$ENTRIES"'; do
      [ $i -ge ${#nodes[@]} ] && break
      IFS=: read -r f t n <<< "$e"
      log=/pscratch/sd/s/sohan100/snapevo-scratch/imps_snap/logs/hx${f}_t${t}_n${n}_newton.log
      srun --nodes=1 --ntasks=1 --cpus-per-task=256 --cpu-bind=none -w "${nodes[$i]}" bash -c "
        source .venv/bin/activate; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
        echo \"\$(date +%m-%d\ %H:%M:%S) interactive compile worker on \$(hostname)\" >> $log
        timeout \$(( $end - \$(date +%s) )) python cluster/imps_snap.py snap $f $t --n-train $n --workers 250 >> $log 2>&1" &
      i=$((i + 1))
    done
    wait'
  echo "$(date '+%m-%d %H:%M:%S') allocation ended"
  [ $(( $(date +%s) - t0 )) -lt 900 ] && { echo "$(date '+%m-%d %H:%M:%S') nothing left to compile"; break; }
  sleep 30
done
