#!/bin/bash
#SBATCH -A m4980
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --output=logs/%x_%j.out
#SBATCH --open-mode=append
#
# cluster/ti_trotter.py FIELD 4 on a whole compute node (250 workers). LOGIN_HOST runs the same scan with fewer workers:
# it is stopped first (the scan resumes from its .partial.json), so one scan writes the file at a time.
#   FIELD=0 LOGIN_HOST=login03 sbatch -q regular -t 08:00:00 --job-name=snapevo-ti-trotter-hx0-t4 ti_trotter.slurm
set -uo pipefail
cd "$SLURM_SUBMIT_DIR/.."
module load python
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
log="/pscratch/sd/s/sohan100/snapevo-scratch/imps_snap/logs/trotter_hx${FIELD}_t4.log"
timeout 30 ssh -n -q -o BatchMode=yes "$LOGIN_HOST" "pkill -u \$USER -f 'cluster/ti_trotter.py $FIELD 4 '" && sleep 10
echo "$(date '+%m-%d %H:%M:%S') compute job $SLURM_JOB_ID takes over from $LOGIN_HOST" >> "$log"
python cluster/ti_trotter.py "$FIELD" 4 --workers 250 >> "$log" 2>&1
