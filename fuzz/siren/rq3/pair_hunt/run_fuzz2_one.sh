#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/tiffanyb/Fun/robot/spark"
BASE="$ROOT/fuzz/siren/rq3/pair_hunt"
PICKERS="random,random_seeded,cem,cem_seeded"
BUDGET=125
BATCH=25

case "${1:-}" in
  pssa)
    TARGET="G1MobileBase_D2_WG_SO_v0_pssa_s17_lam1.0_base_INSERTION1.json"
    ;;
  sss)
    TARGET="G1FixedBase_D1_AG_SO_v0_sss_s1_lam10.0_arm_INSERTION90.json"
    ;;
  *)
    echo "usage: $0 {pssa|sss}" >&2
    exit 2
    ;;
esac

mkdir -p "$BASE/fuzz2_results" "$BASE/fuzz2_logs"

LOG="$BASE/fuzz2_logs/${TARGET%.json}.log"
OUT="$BASE/fuzz2_results/$TARGET"
exec >"$LOG" 2>&1

cd "$ROOT"
echo "target=$TARGET"
echo "pickers=$PICKERS budget=$BUDGET batch=$BATCH"
echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=.

exec /Users/tiffanyb/Tools/miniconda3/envs/spark/bin/python \
  -m fuzz.siren.pipeline.stage3_fuzz \
  --target "$BASE/targets/$TARGET" \
  --pickers "$PICKERS" \
  --budget "$BUDGET" \
  --batch "$BATCH" \
  --out "$OUT"
