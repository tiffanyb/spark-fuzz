"""
Find a lambda where the CBF-family filter is FEASIBLE and the task still fails.

Every catalogued collision for cbf / rcbf / sss / rsss (21 of 21) occurs with the
filter infeasible for 60-100% of the trajectory. Those four share the lambda*phi
demand shape, and at the shipped lambda = 10 the demand exceeds control authority,
so the constraint set is empty and a collision is guaranteed before any attacker
exists. Controls built there measure the gain, not the filter.

Lowering lambda restores feasibility. The open question is whether anything is
left to attack once it is restored -- on the G1 fixed-base scene the collision
vanished entirely at every feasible gain, which closed that line of enquiry.

This sweeps lambda per (case, algo, seed) and looks for a WINDOW:

    %infeasible ~ 0      the filter can satisfy its constraints
    baseline still fails COLLISION or DEADLOCK on the plain task

A triple with a non-empty window is genuine target material for that filter. A
triple with none says the filter is either over-asking (high lambda) or fully
capable (low lambda), with no regime in between -- also a real result, and the
one that would confine the attack to the eta-demand SSA family.

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 python -m fuzz.siren.scenario.lambda_window \
        --case R1LiteUpper_D2_AG_SO_v0 --algos cbf,rcbf,sss,rsss
"""

import argparse
import json

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="R1LiteUpper_D2_AG_SO_v0")
    p.add_argument("--algos", default="cbf,rcbf,sss,rsss")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--lambdas", default="10.0,3.0,1.0,0.3,0.1")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    from ..world.run import World
    from ..world.types import real_filter

    case = a.case
    index = "velocity" if "_D2_" in case else "distance"
    steps = a.max_steps or (900 if "_D2_" in case else 1500)
    algos = [x.strip() for x in a.algos.split(",") if x.strip()]
    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    lams = [float(x) for x in a.lambdas.split(",") if x.strip()]

    print(f"{case}   index={index}  max_steps={steps}\n", flush=True)
    print(f"{'algo':<6}{'sd':>3}{'lam':>7}{'outcome':>10}{'steps':>7}"
          f"{'%infeas':>9}{'min clear':>11}{'gaveup':>7}   window", flush=True)

    rows = []
    for algo in algos:
        for sd in seeds:
            for lam in lams:
                spec = real_filter(algo=algo, index=index, d_min=0.02,
                                   eta=0.02, lam=lam, k=0.1)
                try:
                    w = World.build(seed=sd, spec=spec, test_case=case,
                                    max_steps=steps)
                    sc = w.scene()
                    rec = w.run([np.asarray(sc.G1, float)], max_steps=steps,
                                exact_margin=True)
                except Exception as e:
                    print(f"{algo:<6}{sd:>3}{lam:>7.1f}   failed "
                          f"{type(e).__name__}", flush=True)
                    continue

                mus = np.array([s.mu for s in rec.steps], float)
                fin = np.isfinite(mus)
                pct = 100.0 * (mus[fin] > 0).mean() if fin.any() else np.nan
                feasible = np.isfinite(pct) and pct < 5.0
                fails = rec.label in ("COLLISION", "DEADLOCK")
                degenerate = rec.n_steps <= 1
                win = feasible and fails and not degenerate

                rows.append({"case": case, "algo": algo, "seed": sd, "lam": lam,
                             "label": rec.label, "n_steps": int(rec.n_steps),
                             "pct_infeasible": float(pct),
                             "min_clearance": float(rec.min_clearance),
                             "n_gave_up": int(rec.n_gave_up),
                             "degenerate": bool(degenerate),
                             "window": bool(win)})
                print(f"{algo:<6}{sd:>3}{lam:>7.1f}{rec.label:>10}"
                      f"{rec.n_steps:>7}{pct:>9.1f}{rec.min_clearance:>+11.6f}"
                      f"{rec.n_gave_up:>7}   {'*** WINDOW ***' if win else ''}",
                      flush=True)

    out = a.out or f"fuzz/siren/experiment/lambda_window_{case}.json"
    json.dump(rows, open(out, "w"), indent=2, default=float)

    wins = [r for r in rows if r["window"]]
    print(f"\n{'='*84}")
    print(f"{len(wins)}/{len(rows)} (algo, seed, lambda) combos are FEASIBLE "
          f"and still fail")
    for r in wins:
        print(f"   {r['algo']:<6} seed {r['seed']}  lam {r['lam']:<5} "
              f"{r['label']}  {r['pct_infeasible']:.1f}% infeasible  "
              f"clearance {r['min_clearance']:+.6f}")
    if not wins:
        print("   NONE -- for every filter tried, feasibility and failure are "
              "mutually exclusive\n   in this scene: over-asking causes the "
              "collision, and a gain that can be met\n   also completes the "
              "task.")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
