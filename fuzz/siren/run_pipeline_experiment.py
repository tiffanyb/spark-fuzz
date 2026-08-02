"""
The clean experiment: one pipeline, six search strategies, every scenario+seed.

The ONLY thing that differs between arms is which picker proposes the next
candidate. The evaluation, the scoring, the classification and the bookkeeping
are the same code path for all of them (fuzz/siren/pipeline.py), so a difference
between arms is a difference between search strategies and nothing else.

    arms          random / cem / bo, each unseeded and obstacle-seeded
    scenarios     all 8 G1FixedBase benchmark cases
    seeds         whatever --seeds says

Every scene is gated before any arm runs on it, because an ungated scene poisons
the comparison silently:

    baseline reachable    the legitimate goal must be achievable unattacked
    filter not inert      a filter already giving up is not being defeated
    demand achievable     an over-demanded filter fails whenever it engages, so
                          "attack" degenerates to "get close enough to engage"

Results are written as JSON LINES, one record per (case, seed, arm), appended as
they complete -- a nine-hour sweep that dies at hour eight still leaves eight
hours of usable data. Every Evaluation is kept whole, not just the counts, so
later questions can be answered without re-running anything.

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    python -m fuzz.siren.run_pipeline_experiment --seeds 0-9 --budget 30
"""

import argparse
import json
import time
import traceback

import numpy as np

from .pipeline import run_search, summarize
from .search.pick import ARMS, make_picker
from .world.run import World
from .world.types import real_filter

CASES = [
    "G1FixedBase_D1_AG_SO_v0", "G1FixedBase_D1_AG_SO_v1",
    "G1FixedBase_D1_AG_DO_v0", "G1FixedBase_D1_AG_DO_v1",
    "G1FixedBase_D2_AG_SO_v0", "G1FixedBase_D2_AG_SO_v1",
    "G1FixedBase_D2_AG_DO_v0", "G1FixedBase_D2_AG_DO_v1",
]


def parse_seeds(spec):
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def index_for(case):
    """Dictated by the control mode, not chosen — SPARK asserts it."""
    return "velocity" if "_D2_" in case else "distance"


def max_steps_for(case, override=None):
    """D1 needs a longer horizon than D2 and it is not a free parameter.

    Measured: at 300 steps every D1 candidate ends in leg1_timeout and the
    baseline itself times out, so the whole case silently produces no data. D1 is
    velocity-controlled, so the end-effector covers ground per step at the rate
    the reference controller commands, and reaching a goal across the workspace
    simply takes more steps than the acceleration-controlled D2 arm driven by the
    same controller. Too SHORT a horizon does not bias the result, it deletes it.
    """
    if override is not None:
        return override
    # Raised 3x (was 300/500). Measured: of 129 leg-2 timeouts re-run at 2500
    # steps, 127 REACHED with a median of 555 steps -- they were slow, not
    # trapped. And 828 D1 candidates were discarded as leg-1 timeouts, never
    # reaching the inserted goal at all, which silently capped how FAR an
    # inserted goal could be placed and still count. A longer horizon widens the
    # admissible-goal set rather than changing any outcome.
    return 900 if "_D2_" in case else 1500


def gate(world, scene, spec, max_steps):
    """Is this scene worth running six arms on? Returns (ok, reason, detail)."""
    from .search.loop import probe_demand
    base = world.run([np.asarray(scene.G1)], spec, max_steps=max_steps)
    detail = {"baseline_label": base.label, "baseline_gave_up": int(base.n_gave_up),
              "baseline_steps": int(base.n_steps),
              "baseline_min_clearance": float(base.min_clearance)}
    if not base.reached:
        return False, f"baseline_unreachable({base.label})", detail
    if base.n_gave_up > 0:
        return False, f"filter_inert({base.n_gave_up} give-ups)", detail
    # A baseline that "reaches" in a couple of steps means the legitimate goal was
    # already satisfied at reset: there is no journey to interfere with, and an
    # inserted goal cannot be said to have diverted anything. Observed on
    # D1_AG_SO_v0 seed 2, where the gate passed with baseline_steps == 1 and all
    # 30 candidates then timed out.
    if base.n_steps < 5:
        return False, f"degenerate_scene(baseline reached in {base.n_steps} steps)", detail

    feas = probe_demand(world, scene, spec, max_steps=max_steps)
    detail["demand_verdict"] = feas.get("verdict")
    detail["frac_feasible_when_engaged"] = feas.get("frac_feasible_when_engaged")
    detail["C_min"] = feas.get("C_min")
    detail["C_max"] = feas.get("C_max")
    if feas.get("verdict") in ("never_feasible", "mostly_infeasible"):
        return False, f"demand_not_achievable({feas.get('verdict')})", detail
    return True, None, detail


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--cases", default=",".join(CASES))
    p.add_argument("--seeds", default="0-9")
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--eta", type=float, default=0.02)
    p.add_argument("--d-min", type=float, default=0.02)
    p.add_argument("--budget", type=int, default=30)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=None,
                   help="override; default 500 for D1 cases, 300 for D2")
    p.add_argument("--search-seed", type=int, default=0)
    p.add_argument("--search-seeds", default=None,
                   help="comma/range list; repeats every arm once per seed so "
                        "the arm effect can be separated from RNG luck")
    p.add_argument("--shard", default="0/1", help="i/n, split cases across procs")
    p.add_argument("--out", default="/tmp/pipeline_experiment.jsonl")
    a = p.parse_args(argv)

    cases = [c.strip() for c in a.cases.split(",") if c.strip()]
    _si, _sn = (int(x) for x in a.shard.split("/"))
    cases = [c for i, c in enumerate(cases) if i % _sn == _si]
    seeds = parse_seeds(a.seeds)
    search_seeds = (parse_seeds(a.search_seeds) if a.search_seeds
                    else [a.search_seed])
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    sink = open(a.out, "a", buffering=1)

    print(f"cases={len(cases)} seeds={len(seeds)} arms={len(arms)} "
          f"=> {len(cases)*len(seeds)} scenes, {len(cases)*len(seeds)*len(arms)} runs",
          flush=True)
    t_start = time.time()
    n_gated, n_run = 0, 0

    for case in cases:
        index = index_for(case)
        steps = max_steps_for(case, a.max_steps)
        for seed in seeds:
            spec = real_filter(algo="ssa", index=index, d_min=a.d_min,
                               eta=a.eta, k=0.1)
            t_scene = time.time()
            try:
                world = World.build(seed=seed, spec=spec, test_case=case,
                                    max_steps=steps)
                scene = world.scene()
                ok, reason, detail = gate(world, scene, spec, steps)
            except Exception as e:
                sink.write(json.dumps({"case": case, "seed": seed,
                                       "status": "error",
                                       "error": f"{type(e).__name__}: {e}",
                                       "traceback": traceback.format_exc()[-2000:]}) + "\n")
                print(f"{case} seed={seed}  ERROR {type(e).__name__}: {e}", flush=True)
                continue

            meta = {
                "case": case, "seed": seed, "index": index, "eta": a.eta,
                "d_min": a.d_min, "budget": a.budget, "max_steps": steps,
                "scene": {
                    "G0": [float(x) for x in scene.G0],
                    "G1": [float(x) for x in scene.G1],
                    "bounds": [[float(l), float(h)] for l, h in scene.bounds],
                    "keepout": float(scene.keepout),
                    "n_obstacles": int(len(scene.obstacles_world)),
                    "obstacles_world": [[float(v) for v in np.asarray(o)[:3, 3]]
                                        for o in scene.obstacles_world],
                },
                "gate": detail,
            }

            if not ok:
                n_gated += 1
                sink.write(json.dumps({**meta, "status": "gated_out",
                                       "reason": reason}) + "\n")
                print(f"{case} seed={seed:<3} GATED: {reason}", flush=True)
                continue

            for arm in arms:
              for ss in search_seeds:
                t0 = time.time()
                try:
                    picker = make_picker(arm, scene, seed=ss)
                    evals = run_search(world, scene, picker, budget=a.budget,
                                       batch=a.batch, max_steps=steps)
                    s = summarize(evals)
                    rec = {**meta, "arm": arm, "status": "ok",
                           "search_seed": ss,
                           "elapsed_s": round(time.time() - t0, 1),
                           "summary": s,
                           "evaluations": [e.as_row() for e in evals]}
                    n_run += 1
                    print(f"{case} seed={seed:<3} {arm:<14} ss={ss} "
                          f"coll2={s['n_collisions_leg2']}/{s['n_full']:<3} "
                          f"mod={s['n_modification_hits']:<3} "
                          f"K1={s['kind_KIND_1']} K2={s['kind_KIND_2']} "
                          f"noinf={s['kind_NO_INFEASIBILITY']} "
                          f"({rec['elapsed_s']}s)", flush=True)
                except Exception as e:
                    rec = {**meta, "arm": arm, "search_seed": ss, "status": "error",
                           "error": f"{type(e).__name__}: {e}",
                           "traceback": traceback.format_exc()[-2000:]}
                    print(f"{case} seed={seed:<3} {arm:<14} ERROR "
                          f"{type(e).__name__}: {e}", flush=True)
                sink.write(json.dumps(rec) + "\n")

            print(f"   [scene done in {time.time()-t_scene:.0f}s; "
                  f"elapsed {(time.time()-t_start)/60:.0f} min]", flush=True)

    sink.close()
    print(f"\nDONE  {n_run} arm-runs, {n_gated} scenes gated out, "
          f"{(time.time()-t_start)/60:.1f} min total -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
