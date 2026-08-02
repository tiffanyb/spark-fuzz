"""
From the STATE at G0, can the filter still get the robot to G1?

Everything checked so far ran G0 as a prefix waypoint — the robot starts at its
home pose, drives to G0, then continues. That establishes the task is achievable
along that whole rollout. It does NOT establish that the task is achievable from
the handover state itself, which is the state an attacker actually hands the
robot when it inserts a goal.

The distinction matters. If, from `state_at_G0`, the filter cannot reach G1 even
with NO inserted goal, then the scene is not a valid attack setting: the robot
was already unable to finish its job at that point, and inserting a goal is not
what broke it.

So: restore the captured state, run the plain [G1] schedule, and see.

Reported per scenario:

    from-state [G1]     the question — must REACH for the scene to be sound
    prefix [G0,G1]      the earlier check, for comparison
    clearance / steps   how much room the from-state run actually had

Note the two can legitimately differ: the state restore was measured not to
reproduce micron-scale contacts (a 0.155 mm integration divergence erases them),
so a from-state run is a slightly different trajectory. What matters here is the
coarse outcome — REACHED versus not — not exact agreement.

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 python -m fuzz.siren.scenario.check_from_state
"""

import argparse
import glob
import json

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--targets", default="fuzz/siren/scenario/targets/*.json")
    p.add_argument("--shard", default="0/1")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    from ..world.run import World
    from ..world.types import real_filter
    from .targets import apply_target

    files = sorted(glob.glob(a.targets))
    si, sn = (int(x) for x in a.shard.split("/"))
    files = [f for i, f in enumerate(files) if i % sn == si]
    print(f"shard {si}/{sn}: {len(files)} scenarios\n", flush=True)
    print(f"{'scenario':<30}{'from-state [G1]':>17}{'steps':>7}{'clear':>10}"
          f"{'prefix [G0,G1]':>16}{'steps':>7}", flush=True)

    rows = []
    for f in files:
        t = json.load(open(f))
        spec = real_filter(algo=t["algo"], index=t["index"], d_min=t["d_min"],
                           eta=t["eta"], lam=t["lam"], k=t["k"])
        steps = t["max_steps"]
        G0 = np.asarray(t["G0_commanded"], float)
        G1 = np.asarray(t["G1"], float)

        # (a) prefix contract: home -> G0 -> G1, the schedule used everywhere else
        w = World.build(seed=t["seed"], spec=spec, test_case=t["case"],
                        max_steps=steps)
        pre = w.run([G0, G1], max_steps=steps)

        # (b) from the captured handover state, straight to G1
        w2 = World.build(seed=t["seed"], spec=spec, test_case=t["case"],
                         max_steps=steps)
        apply_target(w2.harness, t)
        w2._scene = None
        st = w2.run([G1], max_steps=steps)

        ok = st.label == "REACHED"
        rows.append({"name": t["name"], "algo": t["algo"],
                     "from_state_label": st.label,
                     "from_state_steps": int(st.n_steps),
                     "from_state_min_clearance": float(st.min_clearance),
                     "from_state_gave_up": int(st.n_gave_up),
                     "prefix_label": pre.label, "prefix_steps": int(pre.n_steps),
                     "joint_speed_at_G0": t["joint_speed_at_G0"],
                     "feasible_from_state": bool(ok)})
        print(f"{t['name'][-28:]:<30}{st.label:>17}{st.n_steps:>7}"
              f"{st.min_clearance:>10.5f}{pre.label:>16}{pre.n_steps:>7}",
              flush=True)

    n_ok = sum(r["feasible_from_state"] for r in rows)
    print(f"\n{'='*78}")
    print(f"{n_ok}/{len(rows)} scenarios: the task IS achievable from the "
          f"captured state at G0")
    bad = [r for r in rows if not r["feasible_from_state"]]
    if bad:
        print(f"\n{len(bad)} where it is NOT — the robot could not finish its own "
              f"job from that state,\nso an inserted goal is not what breaks them:")
        for r in bad:
            print(f"   {r['name']}: from-state {r['from_state_label']} "
                  f"({r['from_state_steps']} steps, clearance "
                  f"{r['from_state_min_clearance']:+.5f}, "
                  f"{r['from_state_gave_up']} give-ups)  vs prefix "
                  f"{r['prefix_label']}")
    out = a.out or f"/tmp/from_state_{si}.json"
    json.dump(rows, open(out, "w"), indent=2, default=float)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
