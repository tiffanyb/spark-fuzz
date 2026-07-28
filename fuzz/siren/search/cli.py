"""
SIREN entry point.

    python -m fuzz.siren.search.cli \
        --attack        {insertion|modification} \
        --threat        {white|gray|gray-proportional|weak-black|strict-black|random} \
        --observability {full|coarse} \
        --picker        {random|cem} \
        --seed 20 --budget 120 --out results.json

The four axes are independent: any attack type x any threat tier x any
observability x any picker is a legal combination.
"""

import argparse
import json
import os

import numpy as np

from ..world.run import World
from ..world.types import real_filter
from .attacks import make_attack
from .loop import search
from .pick import make_picker
from .threat import threat_model


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return [round(float(x), 5) for x in o.reshape(-1)]
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if not np.isfinite(v) else round(v, 6)
    if isinstance(o, (np.integer, int)):
        return int(o)
    return str(o)


def report_to_dict(report, args) -> dict:
    return {
        "args": vars(args),
        "scene": {
            "seed": report.scene.seed,
            "test_case": report.scene.test_case,
            "G0": _jsonable(report.scene.G0),
            "G1": _jsonable(report.scene.G1),
            "n_obstacles": report.scene.n_obstacles,
            "keepout": _jsonable(report.scene.keepout),
        },
        "baseline": (None if report.baseline is None else {
            "label": report.baseline.label,
            "min_g": _jsonable(report.baseline.min_g),
            "n_gave_up": report.baseline.n_gave_up,
        }),
        "n_proposed": report.n_proposed,
        "n_screened": report.n_screened,
        "n_inadmissible": report.n_inadmissible,
        "n_gated_out": report.n_gated_out,
        "n_success": report.n_success,
        "elapsed_s": round(report.elapsed_s, 2),
        "aborted": report.aborted,
        "results": [
            {
                "candidate": _jsonable(r.candidate),
                "score": _jsonable(r.score),
                "success": r.success,
                "labels": r.labels,
                "member_scores": [_jsonable(s) for s in r.member_scores],
                "min_C_d": [_jsonable(rec.min_C_d) for rec in r.records],
                "min_g": [_jsonable(rec.min_g) for rec in r.records],
                "max_penetration": [_jsonable(rec.max_penetration) for rec in r.records],
                "n_gave_up": [rec.n_gave_up for rec in r.records],
            }
            for r in report.ranked()
        ],
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="SIREN — goal-attack search against safety filters")
    p.add_argument("--attack", default="insertion",
                   choices=["insertion", "modification"])
    p.add_argument("--threat", default="white",
                   choices=["white", "gray", "gray-proportional",
                            "weak-black", "strict-black", "random"])
    p.add_argument("--observability", default=None, choices=["full", "coarse"])
    p.add_argument("--picker", default="random", choices=["random", "cem"])

    p.add_argument("--seed", type=int, default=20)
    p.add_argument("--test-case", default="G1FixedBase_D1_AG_SO_v0")
    p.add_argument("--index", default="distance", choices=["distance", "velocity"])
    p.add_argument("--algo", default="ssa",
                   help="the REAL deployed filter (what white-box knows)")
    p.add_argument("--eta", type=float, default=0.5)
    p.add_argument("--lam", type=float, default=10.0)
    p.add_argument("--d-min", type=float, default=0.02)

    p.add_argument("--budget", type=int, default=100)
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--beta", type=float, default=1.0, help="penetration hedge weight")
    p.add_argument("--search-seed", type=int, default=0)
    p.add_argument("--stop-on-first", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    real = real_filter(algo=args.algo, index=args.index, d_min=args.d_min,
                       eta=args.eta, lam=args.lam, k=0.1)
    threat = threat_model(args.threat, d_min=args.d_min, index=args.index,
                          real=real, beta=args.beta,
                          observability=args.observability)

    print(f"[setup] {threat.describe()}", flush=True)
    world = World.build(seed=args.seed, spec=real, test_case=args.test_case,
                        max_steps=args.max_steps)
    scene = world.scene()
    print(f"[setup] {scene.summary()}  world_index={world.supports_index}", flush=True)

    attack = make_attack(args.attack)
    picker = make_picker(args.picker, scene, seed=args.search_seed)

    report = search(world, attack, picker, threat, budget=args.budget,
                    batch=args.batch, max_steps=args.max_steps,
                    verbose=not args.quiet, stop_on_first=args.stop_on_first)

    print("\n==================== SIREN RESULT ====================")
    print(report.summary())

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report_to_dict(report, args), f, indent=2)
        print(f"\nwrote {args.out}")
    return report


if __name__ == "__main__":
    main()
