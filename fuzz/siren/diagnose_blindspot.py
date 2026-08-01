"""
Which robot-obstacle pair actually collides, and is the filter watching it?

The Kind-0 probe found collisions where the collision checker reports contact
(clearance < 0) while the safety index simultaneously reports safety
(phi ~ -0.065, never engaged). Those two numbers cannot both describe the same
pair, so they must be describing DIFFERENT pairs. This finds out which.

If the colliding pair is absent from the filter's active set, then the "attacks"
are not defeating the filter at all -- they are exploiting a gap in what it
monitors, which is a different claim and needs to be reported as such.

Dumps, at the collision step: the closest robot-obstacle pairs by the collision
checker, the phi vector and its mask from the safety index, and whether the
colliding pair appears in the index at all.

    python -m fuzz.siren.diagnose_blindspot --seed 5 --case G1FixedBase_D2_AG_SO_v0
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
    a = p.parse_args(argv)

    from spark_utils import compute_masked_distance_matrix

    index = "velocity" if "_D2_" in a.case else "distance"
    spec = real_filter(algo="ssa", index=index, d_min=a.d_min, eta=a.eta, k=0.1)
    world = World.build(seed=a.seed, spec=spec, test_case=a.case,
                        max_steps=a.max_steps)
    h = world.harness
    scene = world.scene()
    algo = h.algo.safe_controller.safe_algo

    # --- what does the filter monitor, structurally? --------------------- #
    mask = np.asarray(algo.safety_index.phi_mask, dtype=float).reshape(-1)
    print(f"safety index: {type(algo.safety_index).__name__}")
    print(f"  phi_mask length = {mask.size},  nonzero = {int((mask > 0).sum())}")
    print(f"  d_min(environment) = "
          f"{algo.safety_index.min_distance.get('environment')}")
    vols = list(h.robot_cfg.CollisionVol.keys())
    print(f"  robot collision volumes (collision checker) = {len(vols)}")
    print(f"  first few: {vols[:6]}")

    # --- drive until a collision, then dissect that instant --------------- #
    rng = np.random.RandomState(0)
    for attempt in range(40):
        c = sample_admissible(rng, scene)
        if c is None:
            continue

        agent_feedback, task_info = h.reset()
        h.env.task.set_goal_schedule([c, scene.G1])
        u_safe, action_info = h.algo.act(agent_feedback, task_info)

        for t in range(a.max_steps):
            agent_feedback, task_info = h.env.step(u_safe, action_info)
            u_safe, action_info = h.algo.act(agent_feedback, task_info)
            task = h.env.task
            of = task_info["obstacle"]["frames_world"]
            og = task_info["obstacle"]["geom"]
            if len(of) == 0:
                break
            dmat, _ = compute_masked_distance_matrix(
                frame_list_1=task.robot_frames_world,
                geom_list_1=h.robot_cfg.CollisionVol.values(),
                frame_list_2=of, geom_list_2=og)
            if dmat is None:
                break
            if dmat.min() >= 0:
                continue

            # ---- collision: dissect ---------------------------------- #
            print(f"\nCOLLISION at step {t} (attempt {attempt})")
            flat = np.asarray(dmat)
            print(f"  distance matrix shape = {flat.shape} "
                  f"(robot volumes x obstacles)")
            idx = np.dstack(np.unravel_index(np.argsort(flat, axis=None),
                                             flat.shape))[0]
            print(f"\n  closest pairs by the COLLISION CHECKER:")
            for r, o in idx[:5]:
                name = vols[r] if r < len(vols) else f"vol{r}"
                print(f"    {name:<32} vs obstacle {o}   d = {flat[r, o]:+.5f}")

            last = getattr(algo, "_last_phi", None)
            if last is None:
                print("\n  safety index never evaluated phi")
                return 0
            phi = np.asarray(last[0], dtype=float).reshape(-1)
            print(f"\n  SAFETY INDEX at the same instant:")
            print(f"    phi vector length = {phi.size}  "
                  f"(mask nonzero = {int((mask > 0).sum())})")
            print(f"    max phi over MASKED pairs = "
                  f"{phi[mask > 0].max() if (mask > 0).any() else float('nan'):+.5f}")
            print(f"    max phi over ALL pairs    = {phi.max():+.5f}")
            print(f"    any phi >= 0 (engaged)?   "
                  f"{bool((phi[mask > 0] >= 0).any()) if (mask>0).any() else False}")

            n_rob = flat.shape[0]
            print(f"\n  SIZE CHECK — do the two views even cover the same pairs?")
            print(f"    collision checker: {flat.shape[0]} volumes x "
                  f"{flat.shape[1]} obstacles = {flat.size} pairs")
            print(f"    safety index:      {phi.size} entries, "
                  f"{int((mask > 0).sum())} monitored")
            if phi.size != flat.size:
                print(f"    !! DIFFERENT PAIR COUNTS -- the filter and the")
                print(f"       collision check are not looking at the same set.")
            worst_r = int(idx[0][0])
            print(f"\n  the colliding volume is '{vols[worst_r] if worst_r < len(vols) else worst_r}'"
                  f" (row {worst_r} of {n_rob})")
            return 0

    print("no collision found in this sample")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
