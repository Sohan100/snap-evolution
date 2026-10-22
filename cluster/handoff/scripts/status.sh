#!/bin/bash
# Progress of the t = 6 open-chain run: ladder depths and floors, compiles, evaluations, draws, what is running.
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd $ROOT/snap-evolution
source .venv/bin/activate 2>/dev/null
date
python - <<'EOF'
import glob, json
for f in ("0", "1.6"):
    p = f"runs/xxz_d0.25_hx{f}/t6/chain/ladder.json"
    try:
        rs = json.load(open(p))["results"]
    except (OSError, ValueError):
        print(f"h_x = {f}: no finished depth yet"); continue
    print(f"h_x = {f}: depths " + ", ".join(f"D={r['D']} c={r['c_train']:.2e}" for r in rs))
    ev = glob.glob(f"runs/xxz_d0.25_hx{f}/t6/chain/newton/eval_*.json")
    smp = sum(1 for e in ev if "sampled" in json.load(open(e)))
    best = min((json.load(open(e))["c_site"] for e in ev), default=None)
    print(f"   {len(ev)} points evaluated at N = 100, {smp} with fallbacks drawn"
          + (f", tightest c = {best:.2e}" if best else ""))
EOF
echo "running:"
pgrep -u $USER -fa "chain_train.py|chain_snap.py|sample_chain.py|run_t6.sh (compile|sample)-loop" | cut -c1-140
[ -e $ROOT/logs/STOP ] && echo "(STOP file present: stopping)"
for f in $ROOT/logs/ladder_*_t6.log; do echo "--- $(basename $f)"; tail -n 2 $f | cut -c1-160; done
