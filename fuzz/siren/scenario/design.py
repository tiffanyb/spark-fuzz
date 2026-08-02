"""
Search for a legitimate, attackable scene instead of hand-designing one.

Three hand-built layouts failed, all for reasons invisible in the coordinates:
the workspace box overlaps the robot's own body, so a plausible-looking obstacle
position can sit inside the forearm. Rather than guess a fourth time, sample
layouts and keep the ones that pass every condition by measurement.

A scene qualifies only if ALL hold:

  1. obstacles disjoint from each other
  2. the robot is NOT in contact at the initial state          clearance > 0
  3. the QP is FEASIBLE at the initial state                   mu <= 0
     (clearance < d_min is fine — the filter may be engaged at t=0 — but it must
      have a solution; a scene that starts infeasible asks the filter to recover
      from a violation it never caused)
  4. G1 respects the same keepout the attacker must respect
  5. SSA completes G0 -> G1 unattacked                          baseline REACHED

Then, and only then, sweep inserted goals to find one that breaks the return
leg. A scene is a positive control only once that sweep has actually found an
attack — not because the layout looks like it should have one.

G0 is not chosen: it is the robot's home pose, identical to every benchmark
scene. G1 and the obstacles are what the search picks.

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 python -m fuzz.siren.scenario.design --layouts 40
"""

import argparse
import json

import numpy as np


def _probe_clear_positions(w, grid=5, min_clear=0.04):
    """Where can a 5 cm obstacle sit without touching the robot at its home pose?

    Measured, not reasoned about: park every obstacle far away, then move a
    single probe obstacle through the workspace and read the real distance
    matrix. This is the map the hand-built layouts needed and did not have.
    """
    h = w.harness
    af, ti = h.reset()
    base = np.asarray(h.env.task.robot_base_frame, float)
    for o in h.env.task.obstacle_task:
        o.frame[:3, 3] = np.array([0.0, 0.0, -10.0])
        o.velocity = 0.0
    sc = w.scene()
    lo = [b[0] for b in sc.bounds]
    hi = [b[1] for b in sc.bounds]
    out = []
    n_ctrl = len(np.asarray(h.algo.safe_controller.safe_algo.control_max).reshape(-1))
    for x in np.linspace(lo[0], hi[0], grid):
        for y in np.linspace(lo[1], hi[1], grid):
            for z in np.linspace(lo[2], hi[2], grid):
                f = np.eye(4)
                f[:3, 3] = [x, y, z]
                h.env.task.obstacle_task[0].frame[:3, 3] = (base @ f)[:3, 3]
                _af, _ti = h.env.step(np.zeros(n_ctrl), {"trigger_safe": False})
                if h.clearance(_ti) > min_clear:
                    out.append(np.array([x, y, z]))
    return out, sc


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--base-case", default="G1FixedBase_D2_AG_SO_v0")
    p.add_argument("--layouts", type=int, default=40)
    p.add_argument("--n-obstacles", type=int, default=4)
    p.add_argument("--grid", type=int, default=5, help="G1' sweep resolution")
    p.add_argument("--max-steps", type=int, default=900)
    p.add_argument("--rng", type=int, default=0)
    p.add_argument("--out", default="/tmp/designed_scene.json")
    a = p.parse_args(argv)

    from ..world.run import World
    from ..world.types import real_filter
    from ..world.sim import probe
    from ..world import derived
    from ..search.pick import is_admissible
    from .scenes import Scenario, apply_scenario

    index = "velocity" if "_D2_" in a.base_case else "distance"
    spec = real_filter(algo="ssa", index=index, d_min=0.02, eta=0.02,
                       lam=10.0, k=0.1)

    w0 = World.build(seed=0, spec=spec, test_case=a.base_case, max_steps=a.max_steps)
    clear_pos, sc0 = _probe_clear_positions(w0)
    print(f"probe: {len(clear_pos)} positions where a 5 cm obstacle leaves the "
          f"robot clear at its home pose", flush=True)
    print(f"G0 (home pose, not chosen) = {np.round(sc0.G0,3)}", flush=True)
    if len(clear_pos) < a.n_obstacles + 2:
        print("not enough clear positions to build a layout")
        return 1

    rng = np.random.RandomState(a.rng)
    n_ctrl = None
    for attempt in range(a.layouts):
        obs = [clear_pos[i] for i in
               rng.choice(len(clear_pos), a.n_obstacles, replace=False)]
        # obstacles must not overlap each other
        if any(np.linalg.norm(obs[i] - obs[j]) < 0.10
               for i in range(len(obs)) for j in range(i + 1, len(obs))):
            continue
        # G1: admissible, and far enough from every obstacle
        G1 = None
        for _ in range(200):
            cand = np.array([rng.uniform(b[0], b[1]) for b in sc0.bounds])
            if all(np.linalg.norm(cand - o) >= sc0.keepout for o in obs):
                G1 = cand
                break
        if G1 is None:
            continue

        scn = Scenario(name=f"designed_{attempt}", base_case=a.base_case,
                       obstacles=[o.tolist() for o in obs], G1=G1.tolist(),
                       note="auto-designed: every condition measured, not assumed")
        w = World.build(seed=0, spec=spec, test_case=a.base_case,
                        max_steps=a.max_steps)
        apply_scenario(w.harness, scn)
        w._scene = None
        h = w.harness
        af, ti = h.reset()
        clear0 = h.clearance(ti)
        if clear0 <= 0.0:
            continue                                  # condition 2
        u, ai = h.algo.act(af, ti)
        raw = probe.read_raw(h)
        if raw:
            d = derived.evaluate(raw["Lg"], raw["Lf"], raw["phi"], raw["phi_mask"],
                                 raw["u_lim"], demand_shape=spec.demand_shape,
                                 eta=spec.eta, lam=spec.lam, exact=True)
            mu0 = d.get("mu", np.nan)
            if d.get("engaged") and np.isfinite(mu0) and mu0 > 0:
                continue                              # condition 3: infeasible
        else:
            mu0 = np.nan
        sc = w.scene()
        base = w.run([np.asarray(sc.G1)], spec, max_steps=a.max_steps)
        if not base.reached:
            continue                                  # condition 5
        print(f"\n[layout {attempt}] LEGITIMATE: clearance@init {clear0:+.4f}, "
              f"mu@init {mu0:+.4f}, baseline REACHED in {base.n_steps} steps",
              flush=True)
        print(f"   G1 = {np.round(G1,3)}   obstacles = "
              f"{[list(np.round(o,3)) for o in obs]}", flush=True)

        # ---- sweep inserted goals for an attack --------------------------- #
        lo = [b[0] for b in sc.bounds]
        hi = [b[1] for b in sc.bounds]
        axes = [np.linspace(lo[i], hi[i], a.grid) for i in range(3)]
        cands = [np.array([x, y, z])
                 for x in axes[0] for y in axes[1] for z in axes[2]]
        cands = [c for c in cands if is_admissible(c, sc)[0]]
        hits = []
        for c in cands:
            one = w.run([c], max_steps=a.max_steps)
            if not one.reached:
                continue
            full = w.run([c, np.asarray(sc.G1)], max_steps=a.max_steps)
            if full.label in ("COLLISION", "DEADLOCK"):
                hits.append({"G1p": c.tolist(), "outcome": full.label,
                             "min_clearance": float(full.min_clearance),
                             "n_steps": int(full.n_steps)})
                print(f"   ATTACK {full.label} at G1'={np.round(c,3)}  "
                      f"clearance {full.min_clearance:+.5f}", flush=True)
        print(f"   swept {len(cands)} inserted goals -> {len(hits)} attacks",
              flush=True)
        if hits:
            out = {"scenario": {"name": scn.name, "base_case": scn.base_case,
                                "obstacles": scn.obstacles, "G1": scn.G1},
                   "G0": [float(x) for x in sc.G0],
                   "clearance_at_init": float(clear0), "mu_at_init": float(mu0),
                   "baseline": base.label, "baseline_steps": int(base.n_steps),
                   "n_swept": len(cands), "attacks": hits}
            json.dump(out, open(a.out, "w"), indent=2, default=float)
            print(f"\n=== POSITIVE CONTROL FOUND — wrote {a.out} ===")
            return 0
    print("\nno layout in this batch was both legitimate and attackable")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
