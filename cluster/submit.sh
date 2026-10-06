#!/bin/bash
# Submit the whole sweep as four SLURM arrays. Training and references start at
# once; the SNAP points wait for both, the baselines wait for the references.
# afterany, not afterok, so one failed task does not block the rest.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
sub() {
    local list=$1; shift
    sbatch --parsable --array=1-"$(wc -l < "$list")" --export=ALL,JOBS="$PWD/$list" "$@" array.slurm
}
train=$(sub jobs/train.txt --job-name=train --time=48:00:00)
ref=$(sub jobs/reference.txt --job-name=reference --time=24:00:00 --cpus-per-task=4 --mem=16G)
points=$(sub jobs/points.txt --job-name=point --time=72:00:00 --dependency=afterany:"$train":"$ref")
base=$(sub jobs/baselines.txt --job-name=baseline --time=72:00:00 --cpus-per-task=4 --mem=16G \
    --dependency=afterany:"$ref")
echo "train $train, reference $ref, points $points, baselines $base"
