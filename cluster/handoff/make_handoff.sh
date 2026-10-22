#!/bin/bash
# Build the t = 6 open-chain handoff package for Isaac's workstation (48 cores, 256 GB, 1 TB SSD).
#
#   cluster/handoff/make_handoff.sh            build or refresh the package (NERSC keeps training t = 6)
#   cluster/handoff/make_handoff.sh --final    stop the t = 6 work on the NERSC login nodes first, so the
#                                              package carries the latest t = 6 state and nothing runs twice
#
# Output: $SCRATCH/snapevo-scratch/handoff_isaac_t6/ and handoff_isaac_t6.zip next to it (the file to send).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
SCR=${SCRATCH:-/pscratch/sd/s/sohan100}/snapevo-scratch
PKG=$SCR/handoff_isaac_t6
LOGINS="02 03 04 06 07 08"          # where the t = 6 ladders and precompute run on NERSC (cluster/logs/submitted.txt)

if [ "${1:-}" = "--final" ]; then
  echo "stopping t = 6 open-chain work on login nodes $LOGINS"
  for n in $LOGINS; do
    timeout 20 ssh -n -o BatchMode=yes -o LogLevel=ERROR login$n \
      'pkill -TERM -u $USER -f "chain_train.py (0|1.6) 6( |$)"; sleep 5; pkill -KILL -u $USER -f "chain_train.py (0|1.6) 6( |$)"' \
      >/dev/null 2>&1 || true
    # the pool workers ignore SIGTERM (their parent checkpoints first), so a killed master leaves them behind
    sleep 2
    timeout 20 ssh -n -o BatchMode=yes -o LogLevel=ERROR login$n "bash $REPO/cluster/logs/chain/orphans.sh kill" || true
  done
  sleep 10
  for f in 0 1.6; do
    lk=$REPO/runs/xxz_d0.25_hx$f/t6/chain/lock
    [ -e "$lk" ] && ! flock -n "$lk" true && { echo "t = 6 h_x = $f is still locked; stop it by hand"; exit 1; }
  done
  echo "$(date '+%Y-%m-%d %H:%M')  t = 6 open-chain work stopped on NERSC; handed to Isaac's workstation" \
    >> $REPO/cluster/logs/submitted.txt
fi

mkdir -p $PKG/data/snapevo-scratch/open_train/refs $PKG/scripts
echo "code -> $PKG/snap-evolution"
rsync -a --delete --exclude .venv --exclude __pycache__ --exclude '*.pyc' --exclude .git --exclude runs \
  --exclude 'cluster/logs' --exclude 'cluster/jobs_chain' --exclude '.pytest_cache' \
  $REPO/ $PKG/snap-evolution/
echo "runs (t = 6 only) and the KN cache"
mkdir -p $PKG/snap-evolution/runs
cp $REPO/runs/kn_cache.json $PKG/snap-evolution/runs/
for f in 0 1.6; do
  d=runs/xxz_d0.25_hx$f/t6
  mkdir -p $PKG/snap-evolution/$d
  rsync -a $REPO/$d/ref $PKG/snap-evolution/$d/
  [ -d $REPO/$d/chain ] && rsync -a --exclude lock --exclude '*.lock' $REPO/$d/chain $PKG/snap-evolution/$d/
done
echo "training references (t = 6)"
for f in 0 1.6; do
  rsync -a $SCR/open_train/refs/xxz_d0.25_hx$f/t6 $PKG/data/snapevo-scratch/open_train/refs/xxz_d0.25_hx$f/
done
echo "python requirements"
$REPO/.venv/bin/pip freeze 2>/dev/null | grep -v -i -E '^(-e|snapevo|cupy|nvidia|fastrlock)' > $PKG/requirements.txt
cp $REPO/cluster/handoff/HANDOFF.md $PKG/HANDOFF.md
cp $REPO/cluster/handoff/scripts/*.sh $PKG/scripts/
chmod +x $PKG/scripts/*.sh
{ echo "built $(date '+%Y-%m-%d %H:%M %Z') on $(hostname) from $REPO"
  echo "git HEAD $(git -C $REPO rev-parse HEAD) on $(git -C $REPO rev-parse --abbrev-ref HEAD), with uncommitted changes:"
  git -C $REPO status --short -- src cluster tests | head -100
  for f in 0 1.6; do
    l=$REPO/runs/xxz_d0.25_hx$f/t6/chain/ladder.json
    [ -f "$l" ] && python3 -c "import json;d=json.load(open('$l'));print('t = 6, h_x = $f: depths done', [r['D'] for r in d['results']])"
  done
  [ "${1:-}" = "--final" ] && echo "FINAL: NERSC stopped its t = 6 work; Isaac's workstation owns t = 6 from here." \
    || echo "NOT FINAL: NERSC is still training t = 6; rebuild with --final before sending."
} > $PKG/PROVENANCE.txt
echo "checksums"
(cd $PKG && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256)
echo "zip"
rm -f $SCR/handoff_isaac_t6.zip $SCR/handoff_isaac_t6.tar
(cd $SCR && zip -r -q -n .npz:.npy handoff_isaac_t6.zip handoff_isaac_t6)
du -sh $PKG $SCR/handoff_isaac_t6.zip
cat $PKG/PROVENANCE.txt | tail -4
