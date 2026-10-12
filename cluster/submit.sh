#!/bin/bash
# Submit the whole sweep as six SLURM arrays. Training, references and the
# baseline scans start at once. The SNAP points wait for training and references,
# the baseline evaluations wait for the scans and references, and the fallback
# draws wait for both the points and the baseline evaluations. afterany, not
# afterok, so one failed task does not block the rest.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
sub() {
    local list=$1; shift
    sbatch --parsable --array=1-"$(wc -l < "$list")" --export=ALL,JOBS="$PWD/$list" "$@" array.slurm
}
train=$(sub jobs/train.txt --job-name=train --time=48:00:00)
ref=$(sub jobs/reference.txt --job-name=reference --time=24:00:00 --cpus-per-task=4 --mem=16G)
scan=$(sub jobs/scan.txt --job-name=scan --time=24:00:00)
points=$(sub jobs/points.txt --job-name=point --time=48:00:00 --dependency=afterany:"$train":"$ref")
base=$(sub jobs/baselines.txt --job-name=baseline --time=48:00:00 --cpus-per-task=4 --mem=16G \
    --dependency=afterany:"$scan":"$ref")
samp=$(sub jobs/samples.txt --job-name=sample --time=48:00:00 --cpus-per-task=4 --mem=16G \
    --dependency=afterany:"$points":"$base")
echo "train $train, reference $ref, scan $scan, points $points, baselines $base, samples $samp"
