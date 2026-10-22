#!/bin/bash
# One-time setup on the workstation: a virtual environment, the package, and checks. Run from anywhere:
#   bash scripts/setup.sh
# Needs Python 3.10 or newer (3.13 was used on NERSC) and about 10 GB of free disk for the package itself.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd $ROOT
sha256sum -c --quiet MANIFEST.sha256 && echo "checksums OK"
PY=${PYTHON:-python3}
$PY -c 'import sys; assert sys.version_info >= (3, 10), sys.version'
cd snap-evolution
[ -d .venv ] || $PY -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel > /dev/null
pip install -r $ROOT/requirements.txt
pip install -e '.[test]'
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m snapevo verify-library                       # independent check of Isaac's library file (~30 s)
python -m pytest -q -p no:cacheprovider tests/test_train_chain.py tests/test_train_chain_best.py tests/test_newton_fast.py
echo "setup done; next: bash scripts/run_t6.sh start"
