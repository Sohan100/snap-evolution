# Source this before running any command by hand (bash scripts/run_t6.sh already does the same):
#   source scripts/env.sh
# It activates the package's virtual environment, points the training references inside the package, and keeps
# every numerical library single-threaded (the code parallelizes over processes).
_HROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$_HROOT/snap-evolution/.venv/bin/activate"
export SCRATCH="$_HROOT/data"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
cd "$_HROOT/snap-evolution"
