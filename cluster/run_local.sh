#!/bin/bash
# Run a job list in order on this machine:  cluster/run_local.sh cluster/jobs/points.txt
# A failed line is reported and skipped. Training, compiles, references and baseline scans
# resume or skip what is already done, so the list can be rerun after an interruption.
set -uo pipefail
cd "$(dirname "$0")/.."
while IFS= read -r cmd; do
    [ -z "$cmd" ] && continue
    echo "[$(date +%T)] $cmd"
    eval "$cmd" < /dev/null || echo "[$(date +%T)] FAILED: $cmd"
done < "$1"
