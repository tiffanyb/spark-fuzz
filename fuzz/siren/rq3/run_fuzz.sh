#!/bin/bash
# RQ3 stage-3 fuzzing over the re-validated INSERTION targets.
#
# Workers CLAIM targets atomically rather than taking a fixed shard: `mkdir` is
# atomic on POSIX, so exactly one worker wins a given target and any number of
# workers can be added or restarted without duplicating work. A fixed
# shard/of split cannot do that -- a target in progress has no result file yet,
# so a second wave would redo it.
#
# Resumable: a finished target is skipped by its result file, and a claim whose
# worker died is reclaimed after STALE_MIN.
#
# The memory gate is not decoration -- an earlier unbounded parallel MuJoCo run
# on this machine caused a kernel panic (WindowServer watchdog, 122 s).
#
#   ./run_fuzz.sh <worker-id>
set -u
WID="${1:-0}"
ROOT=/Users/tiffanyb/Fun/robot/spark
PY=/Users/tiffanyb/Tools/miniconda3/envs/spark/bin/python
TGT=$ROOT/fuzz/siren/rq3/targets
OUT=$ROOT/fuzz/siren/rq3/results
LOG=$ROOT/fuzz/siren/rq3/logs/fuzz
CLAIM=$ROOT/fuzz/siren/rq3/logs/claims
FLOOR_GB=22
STALE_MIN=180
BUDGET=125
PICKERS="${PICKERS:-random,random_seeded,cem,cem_seeded}"

mkdir -p "$OUT" "$LOG" "$CLAIM"
cd "$ROOT" || exit 1
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       OPENBLAS_NUM_THREADS=1 PYTHONPATH=.

freegb () {
  vm_stat | awk -v pg=16384 '/Pages free/{f=$3}/Pages inactive/{i=$3}/Pages speculative/{s=$3}
    END{gsub(/\./,"",f);gsub(/\./,"",i);gsub(/\./,"",s);printf "%d",(f+i+s)*pg/1073741824}'
}

for t in "$TGT"/INSERTION_*.json; do
  name=$(basename "$t" .json)
  [ -f "$OUT/${name}.json" ] && continue
  lock="$CLAIM/$name"
  # reclaim a lock whose worker died
  if [ -d "$lock" ] && [ -n "$(find "$lock" -maxdepth 0 -mmin +$STALE_MIN 2>/dev/null)" ]; then
    rmdir "$lock" 2>/dev/null
  fi
  mkdir "$lock" 2>/dev/null || continue      # someone else owns it
  echo "$WID $$" > "$lock/owner"
  free=$(freegb)
  if [ "$free" -lt "$FLOOR_GB" ]; then
    echo "[gate] w$WID free ${free}GB < ${FLOOR_GB}GB -- stopping"
    rmdir "$lock" 2>/dev/null; rm -rf "$lock" 2>/dev/null
    exit 0
  fi
  echo "[run ] w$WID $name (free ${free}GB)"
  $PY -m fuzz.siren.pipeline.stage3_fuzz \
      --target "$t" --pickers "$PICKERS" --budget "$BUDGET" --batch 25 \
      --out "$OUT/${name}.json" > "$LOG/${name}.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[FAIL] w$WID $name rc=$rc"
    rm -f "$OUT/${name}.json"
  else
    echo "[done] w$WID $name"
  fi
  rm -rf "$lock"
done
echo "[end ] w$WID no targets left to claim"
