#!/usr/bin/env bash
# Drive the goal: >=1 attack for every (seed, filter) over seeds 1-10.
#
# Two phases, each bounded-parallel:
#   1. build a placement corpus for any seed that lacks one
#   2. hunt, per seed, ONLY the filters that seed is still missing
#
# Phase 2 is driven off coverage.py rather than run.sh's stage skipping, because
# a stage is "done" when its one check-file exists, which cannot express "all
# seven filters covered".
set -uo pipefail
cd /Users/tiffanyb/Fun/robot/spark || exit 1
PY=/Users/tiffanyb/Tools/miniconda3/envs/spark/bin/python
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       OPENBLAS_NUM_THREADS=1 PYTHONPATH=.
C=fuzz/siren/experiment/rq1
RQ1=$C/rq1_results
LOGS=$C/logs/goal
# Sized from measurement, not guesswork. One hunt worker holds ~4.7 GB steady
# after the World-release fix (it was 15 GB and climbing before). Four workers
# is ~19 GB steady, ~32 GB worst case, on a 128 GB machine -- deliberately
# conservative, because the failure mode here is not a slow job but a kernel
# panic: six unfixed workers starved WindowServer for 122 s and the userspace
# watchdog took the machine down.
# Raised 4 -> 8 only after the per-worker footprint was MEASURED at ~4.7-5.4 GB
# steady (it was 15 GB and climbing before the World-release fix). 8 workers is
# ~40 GB of 128, leaving ~88 GB; each is one thread at ~79% CPU, so 8 of 16
# cores. The MIN_FREE_GB gate below is the backstop, since the failure mode is
# a kernel panic rather than a slow job.
JOBS=${JOBS:-8}
# Refuse to add a worker unless this much memory is free (GB).
MIN_FREE_GB=${MIN_FREE_GB:-24}
# Seed 2 is excluded by construction, not by giving up on it: its legitimate
# task ends in one step (|G1-G0| = 0.032 < reach_eps 0.05), so there is
# essentially no return leg for an inserted goal to act on. Its corpus came back
# with 0 placements even at the widest band tried (-0.070 to +0.005).
EXCLUDE=${EXCLUDE:-2}
SEEDS=${SEEDS:-"1 3 4 5 6 7 8 9 10"}
mkdir -p "$LOGS" "$RQ1"

free_gb() {
    # free + inactive + speculative pages; macOS reclaims inactive under demand
    vm_stat | awk -v pg=16384 '
        /Pages free/        {f=$3}
        /Pages inactive/    {i=$3}
        /Pages speculative/ {s=$3}
        END {gsub(/\./,"",f); gsub(/\./,"",i); gsub(/\./,"",s);
             printf "%d", (f+i+s)*pg/1073741824}'
}

throttle() {
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done
    # A second gate on actual free memory, in case a worker drifts above its
    # measured steady state. Without this, the job count alone is a proxy that
    # can be wrong.
    while [ "$(free_gb)" -lt "$MIN_FREE_GB" ]; do
        echo "    (waiting: only $(free_gb) GB free, need $MIN_FREE_GB)"
        sleep 20
    done
}

echo "=== phase 1: corpora ==="
for s in $SEEDS; do
    if [ -s "$RQ1/stock_spots_$s.json" ]; then
        n=$($PY -c "import json;print(len(json.load(open('$RQ1/stock_spots_$s.json'))['spots']))" 2>/dev/null || echo 0)
        [ "$n" -gt 0 ] && { echo "  seed $s: $n placements, keeping"; continue; }
        echo "  seed $s: corpus empty, regenerating wider"
    fi
    throttle
    echo "  seed $s: building"
    $PY -m fuzz.siren.experiment.rq1.run_rq1 --phase spots --seed "$s" \
        --grid 5 --gap-lo -0.070 --gap-hi 0.005 --per-goal 10 \
        > "$LOGS/spots_s$s.log" 2>&1 &
done
wait
echo "  corpora: $(ls $RQ1/stock_spots_*.json 2>/dev/null | wc -l | tr -d ' ')"

echo "=== phase 2: hunt the gaps ==="
round=0
while [ "$round" -lt "${ROUNDS:-5}" ]; do
    round=$((round+1))
    mapfile -t gaps < <($PY -m fuzz.siren.experiment.rq1.coverage --seeds 1-10 --exclude "$EXCLUDE" --gaps-only 2>/dev/null \
                        | sed -n 's/^   seed \([0-9]*\): \(.*\)$/\1 \2/p')
    [ "${#gaps[@]}" -eq 0 ] && { echo "  no gaps left"; break; }
    echo "  round $round: ${#gaps[@]} seeds with gaps"
    for g in "${gaps[@]}"; do
        s="${g%% *}"; algos="${g#* }"
        algos="${algos%% \[*}"                     # drop a [no corpus] marker
        algos="$(echo "$algos" | tr ' ' ',')"
        [ -s "$RQ1/stock_spots_$s.json" ] || { echo "  seed $s: no corpus, skip"; continue; }
        # widen the search each round: more spots, then more settings
        # Each round widens the search AND moves the placement band.
        #
        # GT (--gap-target) is the free-room value placements are tried nearest
        # to first. It matters as much as the ladder: every attack confirmed so
        # far sits at gap -0.020..-0.033, but cbf historically fell in a much
        # tighter band (-0.045 in this metric). With max-spots capped, ordering
        # by distance from a single GT means cbf's band is simply never reached,
        # no matter how wide the eta/lambda ladder gets. Sweeping GT across
        # rounds is what gives the proportional filters a chance.
        case $round in
          1) MS=12; GT=-0.023; LAD="1,0.5,0.2,0.05";       DM="0.020,0.018,0.015"; KS="1.0,0.3,0.1" ;;
          2) MS=30; GT=-0.035; LAD="1,0.5,0.2,0.05,0.02";  DM="0.020,0.018,0.015"; KS="1.0,0.5,0.3,0.1" ;;
          3) MS=45; GT=-0.045; LAD="1,0.5,0.2,0.05,0.02,0.01"; DM="0.020,0.018,0.015"; KS="1.0,0.5,0.3,0.1,0.03" ;;
          *) MS=60; GT=-0.012; LAD="1,0.5,0.2,0.05,0.02,0.01"; DM="0.020,0.018,0.015"; KS="1.0,0.5,0.3,0.1,0.03" ;;
        esac
        # ONE JOB PER (seed, filter), not one per seed.
        #
        # Spotting is per seed and already cached, but hunting is per filter:
        # phase_hunt loops filters and shares nothing between them, so bundling
        # a seed's seven filters into one process buys no efficiency and costs
        # three things -- coarse load balancing, a long-lived process that
        # accumulates memory, and one filter's failure sitting in the same log
        # as six others. Splitting also keeps each process short, which matters
        # after the leak that panicked this machine.
        for one in $(echo "$algos" | tr ',' ' '); do
            throttle
            echo "  seed $s / $one (round $round)"
            # Per-filter PRIORS, taken from the settings that actually
            # produced the 12 confirmed attacks. run_rq1 iterates
            # `for d in dmins: for m in ladder: for k in ks`, so whatever is
            # first in each list is the first combination tried. The uniform
            # cross-product tried the STOCK combination first, which is the one
            # least likely to work -- round 1 spent 27 jobs to find 1 attack.
            # Putting each filter's known-good values first makes that its first
            # trial instead of its 36th. Values still cover the whole ladder, so
            # nothing is excluded; only the order changes.
            case "$one" in
              ssa)  L="0.2,0.5,1,0.05,0.02";  D="0.018,0.015,0.020"; K="0.3,0.1,1.0" ;;
              rssa) L="1,0.2,0.5,0.05,0.02";  D="0.020,0.018,0.015"; K="0.1,0.3,1.0" ;;
              pssa) L="0.2,0.05,1,0.5,0.02";  D="0.020,0.018,0.015"; K="0.1,0.3,1.0" ;;
              sss)  L="0.05,0.02,0.2,0.5,1";  D="0.015,0.018,0.020"; K="0.3,0.1,1.0" ;;
              rsss) L="0.05,0.2,0.02,1,0.5";  D="0.018,0.020,0.015"; K="0.1,0.3,1.0" ;;
              rcbf) L="0.05,0.2,0.02,1,0.5";  D="0.020,0.015,0.018"; K="0.1,0.3,1.0" ;;
              # cbf has never fallen here; lead with the proportional family's
              # working region rather than with stock lambda=10.
              cbf)  L="0.05,0.02,0.2,0.01,1"; D="0.015,0.018,0.020"; K="0.3,0.1,1.0" ;;
              *)    L="$LAD"; D="$DM"; K="$KS" ;;
            esac
            $PY -m fuzz.siren.experiment.rq1.run_rq1 --phase hunt --want 1 \
                --seed "$s" --steps 900 --algos "$one" --max-spots "$MS" \
                --ladder "$L" --dmins "$D" --ks "$K" --gap-target "$GT" \
                > "$LOGS/hunt_s${s}_${one}_r${round}.log" 2>&1 &
        done
    done
    wait
    $PY -m fuzz.siren.experiment.rq1.coverage --seeds 1-10 --exclude "$EXCLUDE" 2>&1 | grep -E "^coverage"
done
echo "=== final ==="
$PY -m fuzz.siren.experiment.rq1.coverage --seeds 1-10 --exclude "$EXCLUDE"
