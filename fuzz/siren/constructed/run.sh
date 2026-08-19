#!/usr/bin/env bash
#
# End-to-end runner for the goal-insertion experiments.
#
#   ./run.sh                 run every stage, in order
#   ./run.sh arma analyze    run only the named stages
#   ./run.sh --list          show stages, runtimes and outputs
#   FORCE=1 ./run.sh spots   re-run a stage whose output already exists
#
# Stages are STRICTLY SEQUENTIAL and each builds MuJoCo worlds, which are memory
# hungry; nothing here is parallelised on purpose, and the script refuses to
# start if another run is already going.
#
# Total runtime for a full run is roughly 9 hours, dominated by `deployment`.
# Every stage is skipped if its output exists, so an interrupted run can simply
# be restarted; `deployment` additionally resumes mid-stage from its own records.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY=/Users/tiffanyb/Tools/miniconda3/envs/spark/bin/python
CDIR="fuzz/siren/constructed"
EDIR="$CDIR/experiment"
LOGS="$CDIR/logs"
SEEDS=${SEEDS:-1,2,3,4,5,6}
FORCE=${FORCE:-0}

# The system python3 is x86_64 and its numpy will not import on this arm64
# machine, so the conda interpreter is not optional.
[ -x "$PY" ] || { echo "FATAL: interpreter not found: $PY" >&2; exit 1; }
cd "$REPO" || exit 1
[ -d "$CDIR" ] || { echo "FATAL: run from inside the spark repo" >&2; exit 1; }

# MuJoCo and OSQP each load an OpenMP runtime; without the first line the
# process aborts on the duplicate. The thread pins keep runs comparable and stop
# one job saturating the machine.
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=.

mkdir -p "$LOGS"

if pgrep -f "envs/spark/bin/python -m fuzz.siren" >/dev/null 2>&1; then
    echo "FATAL: a fuzz.siren job is already running. These stages must not run"
    echo "       concurrently (memory). Wait for it, or kill it first:"
    pgrep -fl "envs/spark/bin/python -m fuzz.siren" | sed 's/^/       /'
    exit 1
fi

step() {   # step <name> <output> <~mins> <deps-csv> <cmd...>
    #
    # Skips only if the output exists AND nothing it depends on is newer. Keying
    # on existence alone is what lets a corrected search sit beside a stale
    # artifact -- this project has been bitten by exactly that twice (a corpus
    # regenerated after the attacks that referenced it, and a corpus built with
    # a since-fixed ordering). A stale output is re-run, not skipped.
    local name="$1" out="$2" mins="$3" deps="$4"; shift 4
    if [ "$FORCE" != "1" ] && [ -e "$out" ]; then
        local stale=""
        local IFS=','
        for d in $deps; do
            [ -n "$d" ] && [ -e "$d" ] && [ "$d" -nt "$out" ] && stale="$stale $d"
        done
        unset IFS
        if [ -z "$stale" ]; then
            echo "== $name: SKIP (up to date: $out)"
            return 0
        fi
        echo "== $name: STALE, inputs newer than output:$stale"
    fi
    echo "== $name: running (~${mins} min) -> $LOGS/$name.log"
    local t0=$SECONDS
    if ! "$@" > "$LOGS/$name.log" 2>&1; then
        echo "== $name: FAILED after $(( (SECONDS-t0)/60 )) min. Last lines:" >&2
        tail -20 "$LOGS/$name.log" >&2
        return 1
    fi
    echo "== $name: done in $(( (SECONDS-t0)/60 )) min"
    grep -E "wrote|summary ->|placements at|STATIC INSERTION|scenes informative|usable in" \
         "$LOGS/$name.log" 2>/dev/null | tail -4 | sed 's/^/     /'
}

# Source files whose change should invalidate downstream outputs.
SRC_GEOM="$CDIR/stock_radius.py,$CDIR/swept.py,$CDIR/separated_search.py"
SRC_DEPLOY="$EDIR/deployment.py,$EDIR/__init__.py,fuzz/siren/pipeline/configs/trial1_spark_default.yaml"

# ---- attack construction (Arm B: existence, constructed obstacles) -------- #

do_spots() {
    # ONE corpus wide enough for every hunt pass. The original runs rebuilt the
    # corpus between passes with a narrower band, which silently orphaned six of
    # the seven attacks found earlier -- their placements were no longer members.
    # Band is metric v2 (true surface separation): the historical v1 bands
    # 0.016-0.040 and 0.002-0.022 are 0.05 lower here, so -0.050..-0.005 covers
    # both, and the passes differ only by --gap-target.
    step spots "$CDIR/stock_spots.json" 20 "$SRC_GEOM" \
        $PY -m fuzz.siren.constructed.stock_radius --phase spots \
            --grid 5 --gap-lo -0.050 --gap-hi -0.005 --per-goal 8
}

do_hunt() {
    step hunt_main "$CDIR/stock_attacks/rssa_0.json" 40 "$CDIR/stock_spots.json,$SRC_GEOM" \
        $PY -m fuzz.siren.constructed.stock_radius --phase hunt --want 1 \
            --steps 900 --gap-target -0.023 --ladder 1,0.5,0.2,0.05 \
            --dmins 0.020,0.015 --max-spots 14
    # ssa, cbf and sss also need the velocity term phi_k lowered
    step hunt_k "$CDIR/stock_attacks/ssa_0.json" 60 "$CDIR/stock_spots.json,$SRC_GEOM" \
        $PY -m fuzz.siren.constructed.stock_radius --phase hunt --want 1 \
            --steps 900 --ladder 1,0.5,0.2,0.05 --dmins 0.020,0.015 \
            --ks 1.0,0.3,0.1 --max-spots 20 --algos ssa,cbf,sss
    # cbf only falls in a much tighter band than the rest
    step hunt_cbf "$CDIR/stock_attacks/cbf_0.json" 40 "$CDIR/stock_spots.json,$SRC_GEOM" \
        $PY -m fuzz.siren.constructed.stock_radius --phase hunt --want 1 \
            --steps 900 --gap-target -0.046 --ladder 0.02,0.05,0.2,1 \
            --dmins 0.015,0.018,0.020 --ks 0.1,0.3,1.0 --max-spots 30 --algos cbf
}

do_render() {
    step render "$CDIR/stock_visualizations/ssa_0_attack.mp4" 10 "$CDIR/stock_attacks,$CDIR/render_stock.py" \
        $PY -m fuzz.siren.constructed.render_stock
}

# ---- deployment condition: unmodified scenes, only the goal varies ----------------------- #

do_preflight() {
    # Gate: if SPARK's generated scenes never engage the filter, Arm A cannot
    # distinguish a filter from no filter and the budget belongs elsewhere.
    step preflight "$EDIR/preflight.json" 10 "$EDIR/preflight.py" \
        $PY -m fuzz.siren.constructed.experiment.preflight --seeds "$SEEDS"
}

do_shipped() {
    # Is SPARK's shipped configuration viable here, or degenerate? Decides
    # whether results are about SPARK as distributed or about our tuning.
    step shipped "$EDIR/shipped_check.json" 15 "$EDIR/shipped_check.py" \
        $PY -m fuzz.siren.constructed.experiment.shipped_check --seeds "$SEEDS"
}

do_reach() {
    # Kinematic reachability screen. Filter-independent (obstacles parked), so
    # one screen serves every filter and configuration.
    step reach "$EDIR/reachable.json" 90 "$EDIR/reachable.py" \
        $PY -m fuzz.siren.constructed.experiment.reachable --seeds "$SEEDS" --grid 5
}

do_deployment() {
    # The main experiment. Resumes from its own records.jsonl, so re-running
    # after an interruption picks up where it stopped.
    local resume=""
    [ -e "$EDIR/deployment_out/records.jsonl" ] && resume="--resume"
    step deployment "$EDIR/deployment_out/summary.json" 360 "$EDIR/reachable.json,$SRC_ARMA" \
        $PY -m fuzz.siren.constructed.experiment.deployment --seeds "$SEEDS" \
            --base-config trial1_spark_default $resume
}

do_repeat() {
    # Effects are 18-360 um; this measures the label-flip rate across fresh
    # processes, i.e. the tolerance band every binary label assumes.
    step repeat "$EDIR/deployment_out/repeatability.json" 30 "$EDIR/deployment_out/records.jsonl,$EDIR/repeat.py" \
        $PY -m fuzz.siren.constructed.experiment.repeat --run "$EDIR/deployment_out"
}

do_analyze() {
    step analyze "$EDIR/deployment_out/RESULTS.md" 2 "$EDIR/deployment_out/records.jsonl,$EDIR/analyze.py" \
        $PY -m fuzz.siren.constructed.experiment.analyze --run "$EDIR/deployment_out"
}

ALL="spots hunt render preflight shipped reach deployment repeat analyze"

if [ "${1:-}" = "--list" ]; then
    cat <<'EOT'
stage      ~min  output                                       what it does
---------  ----  -------------------------------------------  ------------------------------
spots        20  constructed/stock_spots.json                  placement corpus (surface metric)
hunt        140  constructed/stock_attacks/*.json              per-filter attack search
render       10  constructed/stock_visualizations/*.mp4         attack + legitimate-task videos
preflight    10  experiment/preflight.json                     engagement gate
shipped      15  experiment/shipped_check.json                 shipped-config viability
reach        90  experiment/reachable.json                     kinematic reachability screen
deployment  360  experiment/deployment_out/summary.json                  main experiment (resumable)
repeat       30  experiment/deployment_out/repeatability.json            label-flip + determinism
analyze       2  experiment/deployment_out/RESULTS.md                    tables and figures
EOT
    exit 0
fi

STAGES="${*:-$ALL}"
echo "repo    : $REPO"
echo "python  : $PY"
echo "seeds   : $SEEDS"
echo "stages  : $STAGES"
echo "logs    : $LOGS"
echo "started : $(date '+%F %T')"
echo

T0=$SECONDS
for s in $STAGES; do
    case "$s" in
        spots)     do_spots ;;
        hunt)      do_hunt ;;
        render)    do_render ;;
        preflight) do_preflight ;;
        shipped)   do_shipped ;;
        reach)     do_reach ;;
        deployment)      do_deployment ;;
        repeat)    do_repeat ;;
        analyze)   do_analyze ;;
        *) echo "unknown stage: $s (try --list)" >&2; exit 1 ;;
    esac || { echo; echo "ABORTED at stage '$s' after $(( (SECONDS-T0)/60 )) min"; exit 1; }
done

echo
echo "all stages finished in $(( (SECONDS-T0)/60 )) min at $(date '+%F %T')"
echo "results  : $EDIR/deployment_out/RESULTS.md, $EDIR/FINDINGS.md, $CDIR/STOCK_RESULT.md"
echo "metadata : meta.json in each output directory (timestamps, argv, config text, file hashes)"
