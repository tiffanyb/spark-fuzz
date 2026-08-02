"""
Does a hand-built scene actually admit an insertion attack?

Two stages, and the order matters:

  1. LEGITIMACY + BASELINE. The layout must be fair (obstacles disjoint, G1
     inside the workspace and outside the attacker's own keepout) and SSA must
     complete G0 -> G1 unattacked. A scene failing either is not a test of
     anything.

  2. BRUTE FORCE. Sweep a dense grid of inserted goals and run each one. This is
     not a search — no guidance, no optimiser, no budget — so what it finds is
     what EXISTS, up to grid resolution. That is the whole point: it establishes
     ground truth independently of the fuzzer being evaluated.

If brute force finds an attack, the scene becomes a positive control and SIREN's
own search can be scored against it. If brute force finds none, the scene is
simply not attackable and needs redesigning — which is also worth knowing, since
it says something about how hard these attacks are to construct at all.

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 python -m fuzz.siren.scenario.verify --scenario corridor
"""

import argparse
import json
import time

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="corridor")
    p.add_argument("--algo", default="ssa")
    p.add_argument("--eta", type=float, default=0.02)
    p.add_argument("--d-min", type=float, default=0.02)
    p.add_argument("--grid", type=int, default=7, help="points per axis")
    p.add_argument("--max-steps", type=int, default=900)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    from ..world.types import real_filter
    from ..search.pick import is_admissible
    from .scenes import SCENARIOS, build_scenario_world, check_legitimacy

    scn = SCENARIOS[a.scenario]
    index = "velocity" if "_D2_" in scn.base_case else "distance"
    spec = real_filter(algo=a.algo, index=index, d_min=a.d_min,
                       eta=a.eta, lam=10.0, k=0.1)

    w = build_scenario_world(scn, spec, max_steps=a.max_steps)
    sc = w.scene()
    print(f"scenario '{scn.name}' on {scn.base_case} with {a.algo}")
    print(f"  {scn.note}")
    print(f"  G0 {np.round(sc.G0,3)}   G1 {np.round(sc.G1,3)}   "
          f"obstacles {len(scn.obstacles)}   keepout {sc.keepout}")

    # ---------------- stage 1: is the scene fair, and does SSA work? -------- #
    faults = check_legitimacy(scn, sc.keepout, 0.05, sc.G0, sc.bounds)
    # Geometry check against the REAL robot, not just the end-effector. The
    # analytic check above compares obstacle centres to G0, which is one point;
    # the arm is 25 collision volumes and an obstacle 0.09 m from the fingertip
    # can sit inside the forearm. Three hand-built layouts passed the point
    # check and then collided at step 1 with the robot untouched by any goal.
    _af, _ti = w.harness.reset()
    start_clear = w.harness.clearance(_ti)
    if start_clear < 0.0:
        faults.append(f"robot is already in contact at reset "
                      f"(clearance {start_clear:+.4f}) — an obstacle overlaps "
                      f"the ARM, not just the end-effector")
    elif start_clear < a.d_min:
        faults.append(f"robot starts inside the keep-out shell "
                      f"(clearance {start_clear:+.4f} < d_min {a.d_min}) — the "
                      f"filter is engaged before the task begins")
    print(f"\n[1] legitimacy: {'OK' if not faults else 'FAULTS'}   "
          f"(clearance at reset {start_clear:+.4f} m)")
    for f in faults:
        print(f"      - {f}")
    if faults:
        print("  scene is not legitimate; fix the layout before drawing conclusions")
        return 1

    base = w.run([np.asarray(sc.G1)], spec, max_steps=a.max_steps)
    print(f"[1] baseline G0 -> G1: {base.label}  steps={base.n_steps}  "
          f"gave_up={base.n_gave_up}  min_clearance={base.min_clearance:+.4f}")
    if not base.reached:
        print("  the filter cannot do the ORDINARY task here — not a usable scene")
        return 1

    # ---------------- stage 2: exhaustive sweep over inserted goals --------- #
    lo = [b[0] for b in sc.bounds]
    hi = [b[1] for b in sc.bounds]
    axes = [np.linspace(lo[d], hi[d], a.grid) for d in range(3)]
    cands = [np.array([x, y, z]) for x in axes[0] for y in axes[1] for z in axes[2]]
    cands = [c for c in cands if is_admissible(c, sc)[0]]
    print(f"\n[2] brute force: {a.grid}^3 grid -> {len(cands)} admissible "
          f"inserted goals (no guidance, no budget)", flush=True)

    t0 = time.time()
    rows, hits = [], []
    for i, c in enumerate(cands):
        one = w.run([c], max_steps=a.max_steps)
        if not one.reached:
            rows.append({"cand": c.tolist(), "leg1": one.label, "leg2": None})
            continue
        full = w.run([c, np.asarray(sc.G1)], max_steps=a.max_steps)
        rows.append({"cand": c.tolist(), "leg1": "REACHED", "leg2": full.label,
                     "min_clearance": float(full.min_clearance),
                     "n_gave_up": int(full.n_gave_up)})
        if full.label in ("COLLISION", "DEADLOCK"):
            hits.append(rows[-1])
            print(f"    HIT {full.label} at {np.round(c,3)}  "
                  f"clearance {full.min_clearance:+.5f}", flush=True)
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{len(cands)} swept, {len(hits)} hits "
                  f"({time.time()-t0:.0f}s)", flush=True)

    from collections import Counter
    print(f"\n{'='*70}")
    print(f"leg-1 outcomes: {dict(Counter(r['leg1'] for r in rows))}")
    print(f"leg-2 outcomes: {dict(Counter(r['leg2'] for r in rows if r['leg2']))}")
    print(f"ATTACKS FOUND BY BRUTE FORCE: {len(hits)} / "
          f"{sum(1 for r in rows if r['leg2'])} viable candidates")
    if hits:
        print("\n  => this scene IS a positive control. SIREN's search can now be")
        print("     scored against ground truth: does it find these, and how fast?")
    else:
        print("\n  => no attack exists on this grid. The scene needs redesigning,")
        print("     or attacks are genuinely hard to construct here.")
    out = a.out or f"/tmp/scenario_{scn.name}_{a.algo}.json"
    json.dump({"scenario": scn.name, "algo": a.algo, "baseline": base.label,
               "n_candidates": len(cands), "hits": hits, "rows": rows},
              open(out, "w"), indent=2, default=float)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
