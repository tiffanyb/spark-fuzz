#!/bin/bash
# Two trials of target generation, run INTERLEAVED:
#   config1    SPARK's shipped parameters, verbatim
#   config2            the values this project has been using
#
# 7 filters x 16 usable scenario families x 2 trials = 224 jobs.
#
# Both trials share ONE work queue rather than running one after the other. The
# trials only mean something compared against each other, and running them in
# sequence leaves you with a complete picture of one configuration and nothing
# at all about the other for most of the wall-clock. Interleaved, the comparison
# becomes readable as soon as matching (filter, family) pairs land.
#
# Jobs already finished are skipped, so this is resumable: kill it, rerun it,
# and it picks up where it stopped.
#
# Usage:  bash fuzz/siren/pipeline/run_trials.sh [PARALLEL]
set -u

ROOT=/Users/tiffanyb/Fun/robot/spark
PY=/Users/tiffanyb/Tools/miniconda3/envs/spark/bin/python
PAR=${1:-10}
export PYTHONPATH=$ROOT KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$ROOT" || exit 1

FILTERS="ssa rssa pssa cbf rcbf sss rsss"
# ALL FOUR G1FixedBase_*_AG_DO_* families are DROPPED. A fixed base cannot
# evade a 0.400 m/s obstacle, whatever the index order:
#   D1  filter is blind to obstacle motion (Cartesian_Lf = 0) AND has no
#       authority (C = 0.0268, ratio 14.9) -> 0 of 47 G0 pass C1
#   D2  index SEES the motion, but torso_link_1/2/3 are guarded while the
#       pelvis/waist volumes are in env_collision_vol_ignore -- the torso is
#       driven only by three waist joints and cannot get out of the way
#       -> 0 of 55 G0 pass C1, every failure a COLLISION
# Original D1 note follows.
# G1FixedBase_D1_AG_DO_v0 / _v1 are DROPPED: structurally unwinnable, not
# merely hard. The obstacle closes at 0.400 m/s against a first-order arm
# whose control authority is C = 0.0268 -- a ratio of 14.9 -- and
# collision_safety_index_1 hardcodes Cartesian_Lf = 0, so the filter cannot
# anticipate the motion either. Measured: 0 of 47 G0 candidates pass C1,
# every failure at the SAME clearance as the gate, i.e. G0 has no influence
# at all. The mobile-base D1+DO families are KEPT: C = 1.098 vs 0.400 means
# reacting late is still fast enough, and the filter measurably acts there
# (engaged 18-27% of steps, max deviation 0.57-0.80).
SCENES="G1FixedBase_D1_AG_SO_v0 G1FixedBase_D1_AG_SO_v1 \
G1FixedBase_D2_AG_SO_v0 G1FixedBase_D2_AG_SO_v1 \
G1MobileBase_D1_WG_SO_v0 G1MobileBase_D1_WG_SO_v1 \
G1MobileBase_D1_WG_DO_v0 G1MobileBase_D1_WG_DO_v1 \
G1MobileBase_D2_WG_SO_v0 G1MobileBase_D2_WG_SO_v1 \
G1MobileBase_D2_WG_DO_v0 G1MobileBase_D2_WG_DO_v1"

QUEUE=$ROOT/fuzz/siren/pipeline/trials_0803_1/queue.txt
mkdir -p "$ROOT/fuzz/siren/pipeline/trials_0803_1"
: > "$QUEUE"
NSKIP=0
# interleave: the two trials for a given (filter, scene) sit next to each other,
# so a partial run still yields matched pairs to compare
for f in $FILTERS; do
  for s in $SCENES; do
    # TRIALS: only config2 for now. The SPARK-default trial is parked --
# it opened many more gates (103/112 vs 77) but produced almost no usable
# targets, because d_min = 0.1 puts the robot inside the keep-out shell
# before it moves, so the plain task collides on its own and C1 then
# rejects nearly every candidate. Its config and logs are kept; re-add it
# here to resume.
for T in config2; do
      mkdir -p "$ROOT/fuzz/siren/pipeline/trials_0803_1/$T/logs"
      log="$ROOT/fuzz/siren/pipeline/trials_0803_1/$T/logs/${f}_${s}.log"
      # Resume on TARGETS, not on "the job ran". A job that completed without
      # finding anything is exactly the job that must be retried now that the
      # stop condition is controls-found and the seed range is wider.
      # files are control_<scene>_<filter>_s<seed>_lam<lam>.json

      # arm channel on every family; base channel only where SPARK actually
      # commands a base goal (the 8 G1MobileBase families) -- on FixedBase there
      # is no second channel to attack
      echo "$T $f $s arm" >> "$QUEUE"
      case "$s" in
        G1MobileBase_*) echo "$T $f $s base" >> "$QUEUE" ;;
      esac
    done
  done
done
echo "queued $(wc -l < "$QUEUE") jobs, skipped $NSKIP already-finished, parallel $PAR"

xargs -P "$PAR" -L 1 bash -c '
  T=$0; f=$1; s=$2; ch=$3
  ROOT='"$ROOT"'; PY='"$PY"'
  OUT=$ROOT/fuzz/siren/pipeline/trials_0803_1/$T
  CONF=$ROOT/fuzz/siren/pipeline/configs/$T.yaml
  log="$OUT/logs/${f}_${s}_${ch}.log"
  $PY -m fuzz.siren.pipeline.stage1_search \
      --config "$CONF" --algo "$f" --scenes "$s" --goal-channel "$ch" \
      --out-dir "$OUT" > "$log" 2>&1
  n=$(grep -c "\*\*\* CONTROL" "$log" 2>/dev/null || echo 0)
  g=$(grep -c "gate PASSED" "$log" 2>/dev/null || echo 0)
  printf "%-22s %-5s %-26s %-4s %s controls, %s gates\n" "$T" "$f" "$s" "$ch" "$n" "$g"
' < "$QUEUE"

echo
echo "############ SUMMARY ############"
for T in config2; do
  d=$ROOT/fuzz/siren/pipeline/trials_0803_1/$T
  echo "$T:"
  echo "   jobs run          $(ls $d/logs/*.log 2>/dev/null | wc -l)/$(wc -l < "$QUEUE")"
  echo "   gates passed      $(grep -l 'gate PASSED' $d/logs/*.log 2>/dev/null | wc -l)"
  echo "   control files     $(ls $d/control_*.json 2>/dev/null | wc -l)"
  echo "   controls found    $(grep -h '\*\*\* CONTROL' $d/logs/*.log 2>/dev/null | wc -l)"
done
