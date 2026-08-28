#!/usr/bin/env bash
#
# End-to-end runner for the rq1 constructed goal-insertion attacks.
#
#   ./run.sh                 build the attacks: spots -> hunt -> render
#   ./run.sh <stage> ...     run named stages (see --list for the full set)
#   ./run.sh --list          show stages, runtimes and outputs
#   FORCE=1 ./run.sh spots   re-run a stage whose output already exists
#
# Stages are STRICTLY SEQUENTIAL and each builds MuJoCo worlds, which are memory
# hungry; nothing here is parallelised on purpose, and the script refuses to
# start if another run is already going.
#
# The default run is roughly 3 hours, dominated by `hunt`. A stage is skipped
# when its output exists AND nothing it depends on is newer, so an interrupted
# run can simply be restarted.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
# Absolute path to this script. $0 is relative and the script cd's to $REPO
# below, so --sweep could not re-invoke itself through $0.
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
PY=/Users/tiffanyb/Tools/miniconda3/envs/spark/bin/python
CDIR="fuzz/siren/experiment/rq1"
LOGS="$CDIR/logs"
SEED=${SEED:-1}                  # attack-construction stages (one scene)
RQ1="$CDIR/rq1_results"
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

# Anchored at the start of the command line so this matches only a real
# interpreter process. Unanchored, `pgrep -f` also matches any SHELL whose
# command line happens to contain the pattern -- including the very shell
# running this check, which made the guard fire against itself.
if [ "${ALLOW_PARALLEL:-0}" != "1" ] \
   && pgrep -f "^$PY -m fuzz.siren" >/dev/null 2>&1; then
    echo "FATAL: a fuzz.siren job is already running. These stages are memory"
    echo "       hungry, so one at a time by default. Wait, kill it, or set"
    echo "       ALLOW_PARALLEL=1 if you are fanning out deliberately:"
    pgrep -fl "^$PY -m fuzz.siren" | sed 's/^/       /'
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
    local log="$LOGS/${name}_s${SEED}.log"
    echo "== $name: running (~${mins} min) -> $log"
    local t0=$SECONDS
    if ! "$@" > "$log" 2>&1; then
        echo "== $name: FAILED after $(( (SECONDS-t0)/60 )) min. Last lines:" >&2
        tail -20 "$log" >&2
        return 1
    fi
    echo "== $name: done in $(( (SECONDS-t0)/60 )) min"
    # `|| true`: this is a cosmetic summary. With `set -o pipefail` a grep that
    # matches nothing makes the whole pipeline non-zero, which becomes `step`'s
    # return status and aborts the stage loop -- that is exactly what happened
    # when the log path went per-seed and this line still pointed at the old one.
    grep -E "wrote|summary ->|placements at|STATIC INSERTION|scenes informative|usable in" \
         "$log" 2>/dev/null | tail -4 | sed 's/^/     /' || true
    return 0
}

# Source files whose change should invalidate downstream outputs.
SRC_GEOM="$CDIR/run_rq1.py,$CDIR/swept.py,$CDIR/separated_search.py"

# ---- attack construction: existence, constructed obstacles ---------------- #

do_spots() {
    # ONE corpus wide enough for every hunt pass. The original runs rebuilt the
    # corpus between passes with a narrower band, which silently orphaned six of
    # the seven attacks found earlier -- their placements were no longer members.
    # Band is metric v2 (true surface separation): the historical v1 bands
    # 0.016-0.040 and 0.002-0.022 are 0.05 lower here, so -0.050..-0.005 covers
    # both, and the passes differ only by --gap-target.
    step spots "$RQ1/stock_spots_${SEED}.json" 20 "$SRC_GEOM" \
        $PY -m fuzz.siren.experiment.rq1.run_rq1 --phase spots \
            --seed "$SEED" --grid 5 --gap-lo -0.050 --gap-hi -0.005 --per-goal 8
}

do_hunt() {
    step hunt_main "$RQ1/stock_attacks/rssa_0_${SEED}.json" 40 "$RQ1/stock_spots_${SEED}.json,$SRC_GEOM" \
        $PY -m fuzz.siren.experiment.rq1.run_rq1 --phase hunt --want 1 \
            --seed "$SEED" --steps 900 --gap-target -0.023 --ladder 1,0.5,0.2,0.05 \
            --dmins 0.020,0.015 --max-spots 14
    step hunt_k "$RQ1/stock_attacks/ssa_0_${SEED}.json" 60 "$RQ1/stock_spots_${SEED}.json,$SRC_GEOM" \
        $PY -m fuzz.siren.experiment.rq1.run_rq1 --phase hunt --want 1 \
            --seed "$SEED" --steps 900 --ladder 1,0.5,0.2,0.05 --dmins 0.020,0.015 \
            --ks 1.0,0.3,0.1 --max-spots 20 --algos ssa,cbf,sss
    step hunt_cbf "$RQ1/stock_attacks/cbf_0_${SEED}.json" 40 "$RQ1/stock_spots_${SEED}.json,$SRC_GEOM" \
        $PY -m fuzz.siren.experiment.rq1.run_rq1 --phase hunt --want 1 \
            --seed "$SEED" --steps 900 --gap-target -0.046 --ladder 0.02,0.05,0.2,1 \
            --dmins 0.015,0.018,0.020 --ks 0.1,0.3,1.0 --max-spots 30 --algos cbf
}

do_render() {
    step render "$CDIR/stock_visualizations/ssa_0_${SEED}_attack.mp4" 10 "$RQ1/stock_attacks,$CDIR/render_stock.py" \
        $PY -m fuzz.siren.experiment.rq1.render_stock
}

# Default set: attack construction only.
ALL="spots hunt render"

if [ "${1:-}" = "--list" ]; then
    cat <<'EOT'
DEFAULT (./run.sh runs these, in this order) -- constructed attacks
stage      ~min  output                                  what it does
---------  ----  --------------------------------------  ------------------------------
spots        20  rq1_results/stock_spots_<seed>.json     placement corpus (surface metric)
hunt        140  rq1_results/stock_attacks/*.json        per-filter attack search
render       10  stock_visualizations/*.mp4              attack + legitimate-task videos
EOT
    exit 0
fi

if [ "${1:-}" = "--sweep" ]; then
    # Fan out over seeds. Each worker is a full single-seed run of this script,
    # so the per-stage skip/staleness logic still applies per seed. JOBS bounds
    # concurrency: these are MuJoCo processes and running one per seed at once
    # is how a previous sweep exhausted memory.
    range="${2:-1-10}"; lo="${range%%-*}"; hi="${range##*-}"
    jobs_max="${JOBS:-6}"
    shift 2 || shift 1
    sweep_stages="${*:-spots hunt}"
    echo "sweep   : seeds $lo..$hi, $jobs_max at a time, stages: $sweep_stages"
    mkdir -p "$LOGS/sweep"
    pids=""
    for sd in $(seq "$lo" "$hi"); do
        while [ "$(jobs -rp | wc -l)" -ge "$jobs_max" ]; do sleep 5; done
        echo "  -> seed $sd started"
        ALLOW_PARALLEL=1 SEED="$sd" bash "$SELF" $sweep_stages \
            > "$LOGS/sweep/seed${sd}.log" 2>&1 &
    done
    wait
    echo
    echo "sweep finished; per-seed logs in $LOGS/sweep/"
    exit 0
fi

STAGES="${*:-$ALL}"
echo "repo    : $REPO"
echo "python  : $PY"
echo "seed    : $SEED (attack construction)"
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
        *) echo "unknown stage: $s (try --list)" >&2; exit 1 ;;
    esac || { echo; echo "ABORTED at stage '$s' after $(( (SECONDS-T0)/60 )) min"; exit 1; }
done

echo
echo "all stages finished in $(( (SECONDS-T0)/60 )) min at $(date '+%F %T')"
echo "results  : $RQ1/stock_attacks/ (attacks), $CDIR/stock_visualizations/ (videos)"
echo "metadata : meta.json in each output directory (timestamps, argv, config text, file hashes)"
