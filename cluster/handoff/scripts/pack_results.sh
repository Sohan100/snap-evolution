#!/bin/bash
# Pack the results to send back to NERSC (Sohan merges them with cluster/handoff/merge_back.py).
# Safe to run while things are running; send the newest archive (each one is complete on its own).
ROOT=$(cd "$(dirname "$0")/.." && pwd)
out=$ROOT/results_t6_$(date +%Y%m%d_%H%M).tar.gz
tar -C $ROOT -czf $out --exclude '*.lock' --exclude 'lock' --exclude '*.tmp' \
  snap-evolution/runs/xxz_d0.25_hx0/t6/chain snap-evolution/runs/xxz_d0.25_hx1.6/t6/chain \
  snap-evolution/runs/kn_cache.json logs
du -sh $out
echo "send this file to Sohan: $out"
