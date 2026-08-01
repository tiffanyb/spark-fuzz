"""
Why is the margin objective ANTI-correlated with attack success?

The theory is sound: g < 0 really does certify that no safe control exists. So a
search that drives g down should find the dangerous states. On seed 5 it instead
scored 1/40 against random's 10/40 -- twelve times worse. Something specific must
be wrong, and this measures which.

HYPOTHESIS -- C is DIRECTION-BLIND.

    C = sum_k u_lim,k * |[L_g phi]_k|
                        ^^^^^^^^^^^^ absolute value

C is the arm's leverage on a constraint in EITHER direction. Low C means the
joints can barely move the endangered point at all -- which blocks the filter's
retreat, but equally blocks the reference controller from driving IN. A collision
needs the robot to actually close the remaining gap, and closing also takes
leverage. So minimising C should select poses that are STUCK rather than DOOMED,
and stuck shows up as TIMEOUT, which is not counted as an attack.

Prediction if the hypothesis holds: sort candidates by min_g (the margin
objective's own preference) and the top of that ranking should be dominated by
TIMEOUT, while COLLISION should sit at moderate C -- enough leverage left to
close the gap.

    python -m fuzz.siren.diagnose_why_margin_fails --seed 5 --n 30
"""

import argparse
import json
from collections import Counter

import numpy as np

from .world.run import World
from .world.types import real_filter
from .search.pick import sample_admissible


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--test-case", default="G1FixedBase_D1_AG_SO_v0")
    p.add_argument("--eta", type=float, default=0.02)
    p.add_argument("--d-min", type=float, default=0.02)
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--out", default="/tmp/why_margin_fails.json")
    a = p.parse_args(argv)

    spec = real_filter(algo="ssa", index="distance", d_min=a.d_min,
                       eta=a.eta, k=0.1)
    world = World.build(seed=a.seed, spec=spec, test_case=a.test_case,
                        max_steps=a.max_steps)
    scene = world.scene()
    rng = np.random.RandomState(0)

    rows = []
    tried = 0
    while len(rows) < a.n and tried < a.n * 20:
        tried += 1
        cand = sample_admissible(rng, scene)
        if cand is None:
            continue
        # insertion needs the inserted goal to be reachable on its own
        screen = world.run([cand], spec, max_steps=a.max_steps)
        if not screen.reached:
            continue
        rec = world.run([cand, scene.G1], spec, max_steps=a.max_steps)
        eng = [s for s in rec.steps if s.engaged]
        rows.append({
            "label": rec.label,
            "success": rec.is_attack_success(),
            "min_g": (float(rec.min_g) if np.isfinite(rec.min_g) else None),
            "min_C": (float(rec.min_C_phi) if np.isfinite(rec.min_C_phi) else None),
            "n_engaged": len(eng),
            "handover_clear": (float(rec.handover.clearance)
                               if rec.handover is not None else None),
            "min_clear": float(rec.min_clearance),
            "n_gave_up": rec.n_gave_up,
        })
        if len(rows) % 5 == 0:
            print(f"  {len(rows)}/{a.n}", flush=True)

    json.dump(rows, open(a.out, "w"), indent=2, default=float)
    have_g = [r for r in rows if r["min_g"] is not None]

    print("\n" + "=" * 92)
    print(f"WHY THE MARGIN OBJECTIVE MISLEADS — seed {a.seed}, eta={a.eta}, "
          f"{len(rows)} candidates")
    print("=" * 92)
    print(f"outcomes: {dict(Counter(r['label'] for r in rows))}")

    # --- the margin objective's own ranking, top to bottom ---------------- #
    print("\nRanked by the MARGIN objective's preference (most negative g first):")
    print(f"{'rank':<6}{'label':<11}{'min_g':<11}{'min_C':<10}{'engaged':<9}"
          f"{'handover_clear':<16}{'min_clear'}")
    ranked = sorted(have_g, key=lambda r: r["min_g"])
    for i, r in enumerate(ranked[:12]):
        hc = f"{r['handover_clear']:.4f}" if r["handover_clear"] is not None else "-"
        print(f"{i:<6}{r['label']:<11}{r['min_g']:<11.4f}{r['min_C']:<10.4f}"
              f"{r['n_engaged']:<9}{hc:<16}{r['min_clear']:.4f}")

    # --- does low C mean STUCK rather than DOOMED? ------------------------ #
    print("\n" + "-" * 92)
    print("TEST: does the lowest-authority group collide, or merely stall?")
    print("-" * 92)
    if len(have_g) >= 6:
        by_C = sorted(have_g, key=lambda r: r["min_C"])
        k = max(3, len(by_C) // 3)
        for name, grp in (("lowest-C third  (margin objective's target)", by_C[:k]),
                          ("middle third", by_C[k:2 * k]),
                          ("highest-C third", by_C[2 * k:])):
            cnt = Counter(r["label"] for r in grp)
            n_hit = sum(1 for r in grp if r["success"])
            cmean = np.mean([r["min_C"] for r in grp])
            print(f"  {name:<44} C~{cmean:.4f}  attacks={n_hit}/{len(grp)}  {dict(cnt)}")

    # --- what DO the collisions look like? -------------------------------- #
    print("\n" + "-" * 92)
    print("What distinguishes an actual COLLISION?")
    print("-" * 92)
    hits = [r for r in have_g if r["success"]]
    miss = [r for r in have_g if not r["success"]]
    for key in ("min_g", "min_C", "handover_clear", "n_engaged"):
        h = [r[key] for r in hits if r.get(key) is not None]
        m = [r[key] for r in miss if r.get(key) is not None]
        if h and m:
            print(f"  {key:<18} attacks={np.mean(h):+.4f}   "
                  f"non-attacks={np.mean(m):+.4f}   "
                  f"diff={np.mean(h)-np.mean(m):+.4f}")
    print("=" * 92)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
