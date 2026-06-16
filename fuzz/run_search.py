"""
Search for a goal-insertion attack against a FEASIBLE safety filter.

Unlike the original hunt (which ran with the benchmark's oversized keep-out shell,
d_min=0.10, leaving SSA in permanent give-up mode), this runs with a d_min small
enough that the goals sit OUTSIDE the keep-out shell -- so the SSA QP is feasible
and the filter is genuinely active. A confirmed attack here (especially one with
~0 infeasible-QP steps) is a real defeat of the working filter, not an artifact.

Two strategies:
  --strategy random    : keep sampling admissible G1' until the first confirmed
                         attack (DEADLOCK/COLLISION) or until --max-candidates.
  --strategy targeted  : Cross-Entropy-Method search that steers G1' toward goals
                         that break the filter (see GoalInsertionFuzzer.run_targeted).

Run (inside the SPARK conda env, single-thread env vars set):
    python -m fuzz.run_search --strategy random   --seed 17 --d-min 0.02 --out /tmp/r.json
    python -m fuzz.run_search --strategy targeted --seed 17 --d-min 0.02 --out /tmp/t.json
"""

import argparse
import json
import os

import numpy as np

from .config import build_single_arm_config
from .harness import SingleArmHarness
from .fuzzer import GoalInsertionFuzzer, instrument_infeasibility


def _serialize(rec):
    if rec is None:
        return None
    o = rec["outcome"]
    return {"candidate": [round(float(x), 4) for x in rec["candidate"]],
            "label": o.label, "final_dist": round(float(o.final_dist), 4),
            "worst_min_dist_env": round(float(o.worst_min_dist_env), 4),
            "n_steps": o.n_steps, "infeasible_QP": rec.get("infeasible", 0),
            "score": round(float(rec["score"]), 4)}


def main():
    ap = argparse.ArgumentParser(description="Goal-insertion attack search vs a feasible SSA")
    ap.add_argument("--strategy", choices=["random", "targeted"], default="random")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--safe-algo", default="ssa")
    ap.add_argument("--d-min", type=float, default=0.02,
                    help="safety-index keep-out distance; small => goals outside shell => SSA feasible")
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--max-candidates", type=int, default=1000,
                    help="(random) cap on candidates before giving up the until-hit search")
    ap.add_argument("--iterations", type=int, default=12, help="(targeted) CEM rounds")
    ap.add_argument("--pop", type=int, default=12, help="(targeted) candidates per round")
    ap.add_argument("--search-seed", type=int, default=0, help="RNG seed for the search itself")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = build_single_arm_config(seed=args.seed, safe_algo=args.safe_algo,
                                  max_steps=args.max_steps, d_min_env=args.d_min)
    h = SingleArmHarness(cfg)
    feas = instrument_infeasibility(h)
    sc = h.scene_info()
    print(f"[setup] seed={args.seed} algo={args.safe_algo} d_min={args.d_min} "
          f"strategy={args.strategy} infeasibility_instrumented={feas}", flush=True)

    fz = GoalInsertionFuzzer(h, sc, seed=args.search_seed)
    if args.strategy == "random":
        rep = fz.run(n_candidates=args.max_candidates, max_steps=args.max_steps,
                     stop_on_first_hit=True)
    else:
        rep = fz.run_targeted(iterations=args.iterations, pop=args.pop,
                              max_steps=args.max_steps, stop_on_first_hit=True)

    hit = rep.get("first_hit")
    print("\n==================== SEARCH RESULT ====================")
    print(f"strategy           : {args.strategy}  (d_min={args.d_min})")
    print(f"baseline           : {rep['baseline'].label} "
          f"(infeasible_QP={rep.get('baseline_infeasible')}/{rep['baseline'].n_steps})")
    print(f"candidates screened: {rep.get('n_screened')}")
    if hit:
        print(f"ATTACK FOUND       : {_serialize(hit)}")
    else:
        print(f"NO ATTACK FOUND    : searched {rep.get('n_screened')} screened candidates, "
              f"no DEADLOCK/COLLISION against the feasible filter")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        out = {"args": vars(args),
               "baseline": {"label": rep["baseline"].label,
                            "infeasible_QP": rep.get("baseline_infeasible")},
               "n_screened": rep.get("n_screened"),
               "first_hit": _serialize(hit),
               "all_results": [_serialize(r) for r in rep.get("results", [])][:50]}
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
