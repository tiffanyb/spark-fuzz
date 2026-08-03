#!/bin/bash
# End-to-end smoke test of the whole pipeline: one scenario family, one filter,
# one target, one fuzz run, one analysis. Every artifact lands in pipeline/example.
#
# The configuration is chosen to be fast AND known-productive:
#   G1FixedBase_D2_AG_SO_v0  seed 1  rcbf  lambda=0.3
#     - fixed base, so run_from_state is exact (56/56 in the fidelity test)
#     - this exact (scene, filter, seed, gain) previously yielded 17 controls
#   deliberately small: grid 4, budget 24, 2 search seeds
#
# Usage:  bash fuzz/siren/pipeline/run_example.sh
set -u

ROOT=/Users/tiffanyb/Fun/robot/spark
OUT=$ROOT/fuzz/siren/pipeline/example
PY=/Users/tiffanyb/Tools/miniconda3/envs/spark/bin/python
export PYTHONPATH=$ROOT KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$ROOT" || exit 1

CASE=G1FixedBase_D2_AG_SO_v0
ALGO=rcbf
SEEDS=1
LAM=0.3

rm -rf "$OUT"; mkdir -p "$OUT"/{stage1,stage2,stage3,stage4}

step () { echo; echo "=================== $* ==================="; }

step "STAGE 1  find attacks"
$PY -m fuzz.siren.pipeline.stage1_search \
    --algo $ALGO --scenes $CASE --seeds $SEEDS --n-worlds 1 \
    --grid 4 --lam $LAM --out-dir "$OUT/stage1" \
    2>&1 | grep -vE "^(Frame |Initializing|BaseSafe|\*\*|This program| Ipopt|         For)" \
    | tee "$OUT/stage1/stage1.log" | tail -20
ls "$OUT"/stage1/control_*.json >/dev/null 2>&1 || { echo "STAGE 1 FOUND NOTHING"; exit 1; }

step "STAGE 2  verify, trace, render"
$PY -m fuzz.siren.pipeline.stage2_verify \
    --src "$OUT/stage1/control_*.json" --out-dir "$OUT/stage2" \
    --repeats 3 --trace full --render \
    2>&1 | grep -vE "^(Frame |Initializing|BaseSafe|\*\*|This program| Ipopt|         For)" \
    | tee "$OUT/stage2/stage2.log" | tail -25

step "STAGE 3a  build targets"
$PY -m fuzz.siren.pipeline.stage3_targets \
    --verified "$OUT/stage2/*.json" --out-dir "$OUT/stage3" \
    2>&1 | grep -vE "^(Frame |Initializing|BaseSafe|\*\*|This program| Ipopt|         For)" \
    | tee "$OUT/stage3/stage3a.log" | tail -15
TARGET=$(ls "$OUT"/stage3/*.json 2>/dev/null | head -1)
[ -n "$TARGET" ] || { echo "NO TARGET BUILT"; exit 1; }
echo "using target: $TARGET"

step "STAGE 3b  fuzz the target"
$PY -m fuzz.siren.pipeline.stage3_fuzz \
    --target "$TARGET" --pickers random,cem,bo \
    --budget 24 --batch 6 --search-seeds 0,1 \
    --out "$OUT/stage3/fuzz_result.json" \
    2>&1 | grep -vE "^(Frame |Initializing|BaseSafe|\*\*|This program| Ipopt|         For)" \
    | tee "$OUT/stage3/stage3b.log" | tail -20

step "STAGE 4a  discovery curves"
$PY -m fuzz.siren.pipeline.stage4_curves \
    --src "$OUT/stage3/fuzz_result.json" --out-dir "$OUT/stage4" \
    2>&1 | tee "$OUT/stage4/stage4a.log" | tail -15

step "STAGE 4b  render searched locations"
$PY -m fuzz.siren.pipeline.stage4_render \
    --fuzz "$OUT/stage3/fuzz_result.json" --targets-dir "$OUT/stage3" \
    --out-dir "$OUT/stage4" \
    2>&1 | grep -vE "^(Frame |Initializing|BaseSafe|\*\*|This program| Ipopt|         For)" \
    | tee "$OUT/stage4/stage4b.log" | tail -8

step "ARTIFACTS"
find "$OUT" -type f | sed "s|$OUT/||" | sort
