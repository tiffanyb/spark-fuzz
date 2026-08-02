"""
Build a layout where the PROPORTIONAL-demand filters fail while still feasible.

Diagnosis this is designed against:

  * cbf / rcbf / sss / rsss all use demand_shape = proportional (lambda*phi);
    ssa / rssa / pssa use constant (eta). All 21 catalogued collisions for the
    proportional four occur with the filter 60-100% INFEASIBLE at the shipped
    lambda = 10 -- artifacts of the gain, not attack surface.
  * Lowering lambda restores feasibility, but on every stock STATIC scene the
    task then simply succeeds: min clearance is identical (+0.003966) from
    lambda = 3.0 down to 0.03, i.e. the filter is never the binding constraint
    because the nominal path never comes close to an obstacle.
  * The one scene with a genuine window is dynamic, and there a G0 detour shifts
    the robot's arrival PHASE relative to the moving obstacle, so the known
    collision does not survive insertion -- 0 controls from 668 G0 candidates.

So: keep the obstacle STATIC (no phase to break) and put it where the arm is
actually moving fast, so arresting needs more distance than remains. That is the
relative-degree-2 failure the proportional demand is vulnerable to, because
lambda*phi -> 0 as phi -> 0: the demand vanishes exactly when the robot is
closest, permitting an approach it cannot then stop.

METHOD. Run the NOMINAL (essentially unfiltered) path home -> G1, find where the
end-effector is fastest, and place an obstacle offset from the path at that
point. Offset is swept: too far and the filter never engages, too near and the
robot starts in contact or the goal becomes inadmissible.

Every candidate layout is checked for legitimacy before being reported -- goal
admissible, obstacles disjoint, not in contact at reset, feasible at reset.

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 python -m fuzz.siren.scenario.design_braking_scene
"""

import argparse
import json

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="G1FixedBase_D2_AG_SO_v0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--algos", default="cbf,rcbf,sss,rsss")
    p.add_argument("--lambdas", default="10.0,3.0,1.0,0.3,0.1")
    p.add_argument("--offsets", default="0.02,0.035,0.05,0.065,0.08")
    p.add_argument("--fracs", default="0.4,0.55,0.7",
                   help="fraction along the nominal path to place the obstacle")
    p.add_argument("--max-steps", type=int, default=900)
    p.add_argument("--out", default="fuzz/siren/experiment/braking_scene.json")
    a = p.parse_args(argv)

    from ..world.run import World
    from ..world.types import real_filter
    from .scenes import Scenario, apply_scenario
    from ..search.pick import is_admissible

    algos = [x.strip() for x in a.algos.split(",") if x.strip()]
    lams = [float(x) for x in a.lambdas.split(",") if x.strip()]
    offsets = [float(x) for x in a.offsets.split(",") if x.strip()]
    fracs = [float(x) for x in a.fracs.split(",") if x.strip()]
    index = "velocity" if "_D2_" in a.case else "distance"

    # ---- 1. the path to place against -------------------------------- #
    # StepMeasurement records no end-effector position, so the actual nominal
    # trajectory is not recoverable from a run record. Use the straight line
    # home -> G1 instead. That is a real approximation, not a silent fallback:
    # the reference controller drives roughly toward the goal, so the line is a
    # good proxy for WHERE the arm goes, and the mid-path point is where a
    # trapezoidal speed profile is fastest -- which is what the placement needs.
    spec0 = real_filter(algo="ssa", index=index, d_min=0.02, eta=0.02,
                        lam=1.0, k=0.1)
    w0 = World.build(seed=a.seed, spec=spec0, test_case=a.case,
                     max_steps=a.max_steps)
    sc0 = w0.scene()
    G1 = np.asarray(sc0.G1, float)
    home = np.asarray(sc0.G0, float)
    ee = np.array([home + t * (G1 - home) for t in np.linspace(0, 1, 101)])
    print(f"{a.case} seed {a.seed}: placing against the straight line "
          f"home -> G1 ({np.linalg.norm(G1-home):.3f} m)", flush=True)
    print(f"  home {np.round(home,3)}  G1 {np.round(G1,3)}", flush=True)
    print(f"  existing obstacles: {len(sc0.obstacles_world)}", flush=True)

    # a frame around the path direction at the placement point
    def placement(frac, off):
        j = int(np.clip(frac * (len(ee) - 1), 1, len(ee) - 2))
        pt = ee[j]
        d = ee[min(j + 1, len(ee) - 1)] - ee[max(j - 1, 0)]
        n = np.linalg.norm(d)
        d = d / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
        # any unit vector perpendicular to the direction of travel
        tmp = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(tmp, d)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        perp = np.cross(d, tmp)
        perp /= np.linalg.norm(perp)
        return pt + off * perp, pt

    rows, good = [], []
    for frac in fracs:
        for off in offsets:
            centre, pt = placement(frac, off)
            scn = Scenario(name=f"brake_f{frac}_o{off}",
                           obstacles=[centre.tolist()], G1=G1.tolist(),
                           base_case=a.case,
                           note="obstacle offset from the nominal path at the "
                                "arm's fastest point")
            for algo in algos:
                for lam in lams:
                    spec = real_filter(algo=algo, index=index, d_min=0.02,
                                       eta=0.02, lam=lam, k=0.1)
                    try:
                        w = World.build(seed=a.seed, spec=spec,
                                        test_case=a.case,
                                        max_steps=a.max_steps)
                        apply_scenario(w.harness, scn)
                        w._scene = None
                        sc = w.scene()
                        if not is_admissible(G1, sc)[0]:
                            continue
                        rec = w.run([G1], max_steps=a.max_steps,
                                    exact_margin=True)
                    except Exception as e:
                        continue

                    mus = np.array([s.mu for s in rec.steps], float)
                    fin = np.isfinite(mus)
                    pct = 100.0 * (mus[fin] > 0).mean() if fin.any() else np.nan
                    cl = np.array([s.clearance for s in rec.steps], float)
                    eng = np.array([bool(s.engaged) for s in rec.steps])
                    feasible = np.isfinite(pct) and pct < 5.0
                    fails = rec.label in ("COLLISION", "DEADLOCK")
                    degenerate = rec.n_steps <= 1 or (len(cl) and cl[0] <= 0)
                    win = feasible and fails and not degenerate and eng.any()

                    r = {"frac": frac, "offset": off, "algo": algo, "lam": lam,
                         "obstacle": centre.tolist(),
                         "label": rec.label, "n_steps": int(rec.n_steps),
                         "pct_infeasible": float(pct),
                         "min_clearance": float(rec.min_clearance),
                         "clearance_at_0": float(cl[0]) if len(cl) else np.nan,
                         "engaged_frac": float(eng.mean()) if len(eng) else 0.0,
                         "degenerate": bool(degenerate), "window": bool(win)}
                    rows.append(r)
                    if win:
                        good.append(r)
                        print(f"  *** WINDOW  frac={frac} off={off} {algo} "
                              f"lam={lam}: {rec.label} {rec.n_steps} steps, "
                              f"{pct:.1f}% infeas, clearance "
                              f"{rec.min_clearance:+.6f}, engaged "
                              f"{100*eng.mean():.0f}%", flush=True)

    json.dump(rows, open(a.out, "w"), indent=2, default=float)
    print(f"\n{'='*84}")
    print(f"{len(good)}/{len(rows)} (layout, filter, lambda) combos are "
          f"FEASIBLE and still fail")
    by_algo = {}
    for r in good:
        by_algo.setdefault(r["algo"], []).append(r)
    for algo in algos:
        v = by_algo.get(algo, [])
        print(f"   {algo:<6} {len(v):>3} windows"
              + (f"   best: frac={v[0]['frac']} off={v[0]['offset']} "
                 f"lam={v[0]['lam']} {v[0]['label']}" if v else ""))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
