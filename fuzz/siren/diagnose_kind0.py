"""
Does "Kind 0" (collision without the filter ever engaging) actually exist?

There is a strong argument that it cannot. At a collision the clearance d is
negative, so the danger number phi = d_min - d exceeds d_min and is therefore
POSITIVE -- which means the constraint is active and the filter IS engaged. Any
collision should be preceded by engagement, and the real question is only whether
the filter was feasible at that moment (Kind 1 vs Kind 2).

The one way Kind 0 could be real is a MEASUREMENT MISMATCH: the safety index
watches one set of robot-obstacle pairs (via phi_mask) while the collision check
uses another (every entry of CollisionVol). Then the robot could collide on a
pair the filter is not watching -- which would be a far more serious finding than
a taxonomy detail, because it means the filter is structurally blind to part of
the robot.

This takes actual COLLIDING runs and reports, at the collision step and just
before it: phi, clearance, engaged, gave_up.

    python -m fuzz.siren.diagnose_kind0 --seed 5 --case G1FixedBase_D2_AG_SO_v0
"""

import argparse

import numpy as np

from .world.run import World
from .world.types import real_filter
from .search.pick import sample_admissible


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--case", default="G1FixedBase_D2_AG_SO_v0")
    p.add_argument("--eta", type=float, default=0.02)
    p.add_argument("--d-min", type=float, default=0.02)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--n-collisions", type=int, default=5)
    a = p.parse_args(argv)

    index = "velocity" if "_D2_" in a.case else "distance"
    spec = real_filter(algo="ssa", index=index, d_min=a.d_min, eta=a.eta, k=0.1)
    world = World.build(seed=a.seed, spec=spec, test_case=a.case,
                        max_steps=a.max_steps)
    scene = world.scene()
    rng = np.random.RandomState(0)

    found = 0
    tried = 0
    print(f"hunting collisions on seed {a.seed} / {a.case} / eta={a.eta}\n")
    while found < a.n_collisions and tried < 60:
        tried += 1
        c = sample_admissible(rng, scene)
        if c is None:
            continue
        rec = world.run([c, scene.G1], spec, max_steps=a.max_steps)
        if rec.label != "COLLISION":
            continue
        found += 1
        steps = rec.steps
        eng = [s for s in steps if s.engaged]
        first_eng = eng[0].step if eng else None

        print(f"--- collision {found} ---------------------------------------")
        print(f"  steps={rec.n_steps}  engaged_steps={len(eng)}  "
              f"gave_up={rec.n_gave_up}  first_engaged_step={first_eng}")
        print(f"  {'step':<7}{'clearance':<13}{'phi':<13}{'engaged':<10}"
              f"{'gave_up':<9}{'g':<12}")
        for s in steps[-4:]:
            g = f"{s.g:.4f}" if np.isfinite(s.g) else "undef"
            print(f"  {s.step:<7}{s.clearance:<13.5f}{s.phi:<13.5f}"
                  f"{str(s.engaged):<10}{str(s.gave_up):<9}{g:<12}")

        last = steps[-1]
        # The decisive consistency check.
        if last.clearance < 0 and last.phi < 0:
            print("  !! MISMATCH: clearance is negative (collision) but phi is")
            print("     negative too -- the safety index is NOT watching the pair")
            print("     that actually collided. The filter is blind to it.")
        elif last.clearance < 0 and not last.engaged:
            print("  !! phi>0 at collision but 'engaged' is False -- check the")
            print("     active-set mask (phi_mask may exclude this pair).")
        elif last.engaged:
            verdict = ("KIND 1 (infeasible when it engaged)" if last.gave_up
                       else "KIND 2 candidate (engaged AND feasible, still collided)")
            print(f"  -> engaged at the collision step. {verdict}")
        print()

    if found == 0:
        print("no collisions found in this sample")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
