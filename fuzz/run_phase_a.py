"""
Phase-A entry point: single-arm adversarial goal insertion on the SPARK
benchmark (FixedBase G1, static obstacles, r-SSA reactive safe controller).

Usage (from the spark repo root, inside the SPARK conda env):

    python -m fuzz.run_phase_a --safe-algo rssa --seed 0 --candidates 50

It (1) fixes a scene, (2) confirms the legitimate goal G1 is reachable
(baseline), (3) searches for an admissible inserted goal G1' that traps the
controller, and (4) prints + saves a ranked report.

Note: requires MuJoCo and the SPARK dependencies; it is not runnable from a bare
Python install. This script does no training and is deterministic given --seed.
"""

import argparse
import json
import os

import numpy as np

from .config import build_single_arm_config
from .harness import SingleArmHarness
from .fuzzer import GoalInsertionFuzzer


def main():
    ap = argparse.ArgumentParser(description="Phase-A single-arm goal-insertion fuzzer")
    ap.add_argument("--test-case", default="G1FixedBase_D1_AG_SO_v0")
    ap.add_argument("--safe-algo", default="rssa",
                    choices=["ssa", "rssa", "pssa", "cbf", "rcbf", "sss"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--candidates", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=400,
                    help="full horizon; DEADLOCK requires a full-length attempt")
    ap.add_argument("--reach-eps", type=float, default=0.05)
    ap.add_argument("--ik-check", action="store_true",
                    help="require IK-solvable candidates (slower, stronger admissibility)")
    ap.add_argument("--viewer", action="store_true", help="show the MuJoCo viewer")
    ap.add_argument("--out", default=None, help="path to write the JSON report")
    args = ap.parse_args()

    cfg = build_single_arm_config(
        test_case=args.test_case,
        safe_algo=args.safe_algo,
        seed=args.seed,
        max_steps=args.max_steps,
        reach_eps=args.reach_eps,
        enable_viewer=args.viewer,
    )

    harness = SingleArmHarness(cfg)
    scene = harness.scene_info()
    print(f"[scene] G0={np.round(scene['G0_base'], 3)}  G1={np.round(scene['G1_base'], 3)}  "
          f"obstacles={len(scene['obstacles_world'])}  bounds={scene['bounds']}")

    fuzzer = GoalInsertionFuzzer(harness, scene, seed=args.seed, ik_check=args.ik_check)
    report = fuzzer.run(n_candidates=args.candidates, max_steps=args.max_steps)

    baseline = report["baseline"]
    results = report["results"]
    n_attack = sum(1 for r in results if r["outcome"].is_attack_success())
    print("\n==================== SUMMARY ====================")
    print(f"controller        : {args.safe_algo}")
    print(f"baseline (G0->G1) : {baseline.label} (reached={baseline.reached_final})")
    if report.get("aborted"):
        print(f"ABORTED           : {report.get('abort_reason')} "
              "(baseline could not reach G1; nothing fuzzed)")
    print(f"one-hop skipped   : {report.get('n_one_hop_skipped', 0)} "
          "(G1' not individually reachable -> trivial)")
    print(f"sequence attacks  : {len(results)} two-hop candidates evaluated")
    n_timeout = sum(1 for r in results if r["outcome"].label == "TIMEOUT")
    print(f"attacks found     : {n_attack}/{len(results)} (DEADLOCK/COLLISION; "
          f"confirmed only)")
    print(f"inconclusive      : {n_timeout} TIMEOUT (did not reach, not a confirmed trap)")
    if results:
        best = results[0]
        bo = best["outcome"]
        print(f"best attack       : G1'={np.round(best['candidate'], 3)} -> {bo.label} "
              f"(steps={bo.n_steps}, peak_slack={bo.peak_slack:.3g}, "
              f"worst_dist={bo.worst_min_dist_env:.3g})")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        serializable = {
            "args": vars(args),
            "scene": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                      for k, v in scene.items()},
            "baseline": baseline.__dict__,
            "results": [
                {"candidate": r["candidate"].tolist(),
                 "score": r["score"],
                 "outcome": {k: v for k, v in r["outcome"].__dict__.items()
                             if k != "schedule"}}
                for r in results
            ],
        }
        with open(args.out, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        print(f"\nreport written to {args.out}")


if __name__ == "__main__":
    main()
