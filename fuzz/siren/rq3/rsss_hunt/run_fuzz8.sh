#!/bin/bash
# Fuzz the eight randomly sampled rsss_hunt targets.
#
#   ./run_fuzz8.sh <worker-id>
set -u
WID="${1:-0}"
ROOT=/Users/tiffanyb/Fun/robot/spark
PY=/Users/tiffanyb/Tools/miniconda3/envs/spark/bin/python
BASE=$ROOT/fuzz/siren/rq3/rsss_hunt
TGT=$BASE/targets
SELECTED=$BASE/selected_fuzz8.txt
OUT=$BASE/fuzz8_results
LOG=$BASE/fuzz8_logs
CLAIM=$BASE/fuzz8_claims
FLOOR_GB=22
STALE_MIN=180
BUDGET=125
PICKERS="random,random_seeded,cem,cem_seeded"

mkdir -p "$OUT" "$LOG" "$CLAIM"
cd "$ROOT" || exit 1
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       OPENBLAS_NUM_THREADS=1 PYTHONPATH=.

freegb () {
  vm_stat | awk -v pg=16384 '/Pages free/{f=$3}/Pages inactive/{i=$3}/Pages speculative/{s=$3}
    END{gsub(/\./,"",f);gsub(/\./,"",i);gsub(/\./,"",s);printf "%d",(f+i+s)*pg/1073741824}'
}

while IFS= read -r fname; do
  case "$fname" in
    ""|\#*) continue ;;
  esac
  t="$TGT/$fname"
  name="${fname%.json}"
  [ -f "$OUT/${name}.json" ] && continue
  lock="$CLAIM/$name"
  if [ -d "$lock" ] && [ -n "$(find "$lock" -maxdepth 0 -mmin +$STALE_MIN 2>/dev/null)" ]; then
    unlink "$lock/owner" 2>/dev/null
    rmdir "$lock" 2>/dev/null
  fi
  mkdir "$lock" 2>/dev/null || continue
  echo "$WID $$" > "$lock/owner"
  free=$(freegb)
  if [ "$free" -lt "$FLOOR_GB" ]; then
    echo "[gate] w$WID free ${free}GB < ${FLOOR_GB}GB -- stopping"
    unlink "$lock/owner" 2>/dev/null
    rmdir "$lock" 2>/dev/null
    exit 0
  fi
  echo "[run ] w$WID $name (free ${free}GB)"
  $PY -m fuzz.siren.pipeline.stage3_fuzz \
      --target "$t" --pickers "$PICKERS" --budget "$BUDGET" --batch 25 \
      --out "$OUT/${name}.json" > "$LOG/${name}.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[FAIL] w$WID $name rc=$rc"
  else
    echo "[done] w$WID $name"
  fi
  unlink "$lock/owner" 2>/dev/null
  rmdir "$lock" 2>/dev/null
done < "$SELECTED"
echo "[end ] w$WID no targets left to claim"
