"""Merge results returned from Isaac's workstation (scripts/pack_results.sh) into this checkout.

    python cluster/handoff/merge_back.py results_t6_YYYYMMDD_HHMM.tar.gz [--dry-run]

The workstation owns t = 6 of the open-chain arm after a --final handoff, so its runs/<tag>/t6/chain/ trees
(ladder.json, ckpt/, newton/ snap_, real_ and eval_ records) replace ours file by file; a file is only taken when
it is new here or the returned copy is newer. Its KN syntheses are added to runs/kn_cache.json (union; entries
already here are kept), under the cache's own lock. Its logs go to cluster/logs/handoff_t6/.
"""

import argparse
import fcntl
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archive")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory(dir=ROOT / "cluster" / "logs") as tmp:
        with tarfile.open(a.archive) as tf:
            tf.extractall(tmp, filter="data")
        src = Path(tmp) / "snap-evolution" / "runs"
        n_new = n_upd = 0
        for tag in ("xxz_d0.25_hx0", "xxz_d0.25_hx1.6"):
            base = src / tag / "t6" / "chain"
            if not base.exists():
                continue
            for f in base.rglob("*"):
                if not f.is_file():
                    continue
                dst = ROOT / "runs" / tag / "t6" / "chain" / f.relative_to(base)
                if dst.exists() and dst.stat().st_mtime >= f.stat().st_mtime:
                    continue
                n_new += not dst.exists()
                n_upd += dst.exists()
                if not a.dry_run:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dst)
        print(f"t = 6 chain files: {n_new} new, {n_upd} updated" + (" (dry run)" if a.dry_run else ""))
        theirs = src / "kn_cache.json"
        if theirs.exists():
            ours = ROOT / "runs" / "kn_cache.json"
            with open(str(ours) + ".lock", "w") as lk:
                fcntl.flock(lk, fcntl.LOCK_EX)
                mine = json.loads(ours.read_text())
                add = {k: v for k, v in json.loads(theirs.read_text()).items() if k not in mine}
                print(f"KN cache: {len(add)} new syntheses" + (" (dry run)" if a.dry_run else ""))
                if add and not a.dry_run:
                    mine.update(add)
                    tmpf = ours.with_name(f"kn_cache.json.{os.getpid()}.tmp")
                    tmpf.write_text(json.dumps(mine))
                    tmpf.replace(ours)
        logs = Path(tmp) / "logs"
        if logs.exists() and not a.dry_run:
            shutil.copytree(logs, ROOT / "cluster" / "logs" / "handoff_t6", dirs_exist_ok=True)


if __name__ == "__main__":
    main()
