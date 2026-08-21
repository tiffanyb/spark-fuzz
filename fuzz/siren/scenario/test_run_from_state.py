"""
Fidelity test: does resuming from a captured state land where the full run does?

The claim under test, exactly:

    A.  run  home -> g0 -> G1   in one go, capturing s0 the instant wp_idx
        goes 0 -> 1, and the final state when G1 is reached
    B.  run  run_from_state(s0, [G1])
    ->  B's final state must equal A's final state

If it holds, the G0 search can drop the home -> g0 leg from every candidate and
resume from s0 instead. If it does not, the prefix-waypoint contract has to stay.

Reported per (scene, seed, g0), comparing FINAL states:

    d_qpos / d_qvel              MuJoCo generalised coordinates
    d_pos_cmd / d_vel_cmd        the agent's command state -- with
                                 use_sim_dynamics=False this IS what the filter
                                 sees, so it matters more than qpos/qvel
    d_ee                         end-effector position, base frame (metres)
    steps A / steps B            B should be shorter by the length of leg 0

Contacts in this project run 50-500 microns and a 0.155 mm divergence was
measured to flip an outcome, so d_ee is judged against that scale, not against
floating-point equality.

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 python -m fuzz.siren.scenario.test_run_from_state
"""

import argparse
import json

import numpy as np

#: every test case the benchmark generator defines. index follows the D1/D2
#: naming: D1 = velocity control (distance index), D2 = acceleration control
#: (velocity-augmented index).
ALL_CASES = [
    "G1FixedBase_D1_AG_DO_v0", "G1FixedBase_D1_AG_DO_v1",
    "G1FixedBase_D1_AG_SO_v0", "G1FixedBase_D1_AG_SO_v1",
    "G1FixedBase_D2_AG_DO_v0", "G1FixedBase_D2_AG_DO_v1",
    "G1FixedBase_D2_AG_SO_v0", "G1FixedBase_D2_AG_SO_v1",
    "G1MobileBase_D1_WG_DO_v0", "G1MobileBase_D1_WG_DO_v1",
    "G1MobileBase_D1_WG_SO_v0", "G1MobileBase_D1_WG_SO_v1",
    "G1MobileBase_D2_WG_DO_v0", "G1MobileBase_D2_WG_DO_v1",
    "G1MobileBase_D2_WG_SO_v0", "G1MobileBase_D2_WG_SO_v1",
    "G1SportMode_D1_WG_SO_v1",
    "Gen3Single_D1_AG_SO_v0", "Gen3Single_D2_AG_SO_v0",
    "IIWA14Single_D1_AG_SO_v0", "IIWA14Single_D2_AG_SO_v0",
    "LRMate200iD3f_D1_AG_SO_v0", "LRMate200iD3f_D2_AG_SO_v0",
    "R1LiteUpper_D1_AG_SO_v0", "R1LiteUpper_D2_AG_SO_v0",
]
SCENES = [(c, "velocity" if "_D2_" in c else "distance") for c in ALL_CASES]


def step_run(h, schedule, max_steps, capture_wp=None, restore=None):
    """One rollout. Returns (final_state, captured_state, n_steps, reached)."""
    from ..world.sim import probe
    from .state import capture_world, restore_world, refresh_task_cache

    probe.reset_giveups(h)
    af, ti = h.reset()
    if restore is not None:
        restore_world(h, restore)
        af = (h.env.agent.get_feedback()
              if hasattr(h.env.agent, "get_feedback") else af)
        # _update_robot_state BEFORE get_info -- see refresh_task_cache
        refresh_task_cache(h, af)
        ti = h.env.task.get_info(af)
    h.env.task.set_goal_schedule(schedule)
    u, ai = h.algo.act(af, ti)

    cap = None
    t = 0
    for t in range(max_steps):
        af, ti = h.env.step(u, ai)
        u, ai = h.algo.act(af, ti)
        if (capture_wp is not None and cap is None
                and int(getattr(h.env.task, "wp_idx", 0)) >= capture_wp):
            cap = capture_world(h)
        if h.env.task.reached_final:
            break
    return capture_world(h), cap, t + 1, bool(h.env.task.reached_final)


def ee_of(state, h):
    """End-effector position in the base frame, from a captured state.

    `robot_frames_world` is a CACHED attribute, refreshed only inside the task's
    _update_robot_state(); neither mj_forward nor get_info touches it reliably.
    Reading it after a restore returns the PREVIOUS run's value, which made
    every d_ee read exactly 0.0 even where d_pos_cmd was 0.23 -- the column
    measured nothing. Run forward kinematics directly on the restored command
    state instead. forward_kinematics returns frames already in the base frame,
    so no base transform is needed.
    """
    from .state import restore_world
    restore_world(h, state)
    dof_pos = np.asarray(state["robot"]["dof_pos_fbk"], float)
    frames = h.env.task.robot_kinematics.forward_kinematics(dof_pos)
    return np.asarray(frames[h.robot_cfg.Frames.R_ee][:3, 3], float)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="sss")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--n-g0", type=int, default=3)
    p.add_argument("--scenes", default=None)
    p.add_argument("--out", default="fuzz/siren/experiment/run_from_state_test.json")
    a = p.parse_args(argv)

    from ..world.run import World
    from ..world.types import real_filter
    from ..search.pick import is_admissible

    scenes = SCENES
    if a.scenes:
        keep = {x.strip() for x in a.scenes.split(",")}
        scenes = [s for s in SCENES if s[0] in keep]
    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]

    print(f"{'scene':<26}{'sd':>3}{'g0':<22}{'d_qpos':>10}{'d_qvel':>10}"
          f"{'d_poscmd':>10}{'d_velcmd':>10}{'d_ee(m)':>11}"
          f"{'stepsA':>7}{'stepsB':>7}  ok", flush=True)

    rows = []
    for case, index in scenes:
        steps = 900 if "_D2_" in case else 1500
        for sd in seeds:
            spec = real_filter(algo=a.algo, index=index, d_min=0.02, eta=0.02,
                               lam=1.0, k=0.1)
            try:
                w = World.build(seed=sd, spec=spec, test_case=case,
                                max_steps=steps)
                sc = w.scene()
                h = w.harness
            except Exception as e:
                print(f"  {case:<24}{sd:>3}  build failed {type(e).__name__}",
                      flush=True)
                continue
            G1 = np.asarray(sc.G1, float)

            lo = [b[0] for b in sc.bounds]
            hi = [b[1] for b in sc.bounds]
            rng = np.random.RandomState(1234 + sd)
            g0s = []
            while len(g0s) < a.n_g0:
                g = np.array([rng.uniform(l, hh) for l, hh in zip(lo, hi)])
                if is_admissible(g, sc)[0]:
                    g0s.append(g)

            for g0 in g0s:
                try:
                    # A: the full run, capturing s0 at the g0 handover
                    finA, s0, nA, okA = step_run(h, [g0, G1], steps,
                                                 capture_wp=1)
                    if s0 is None:
                        print(f"  {case:<24}{sd:>3}{str(np.round(g0,3)):<22}"
                              f"   never reached g0 — skipped", flush=True)
                        continue
                    # B: resume from s0 and drive to G1
                    finB, _, nB, okB = step_run(h, [G1], steps, restore=s0)
                except Exception as e:
                    print(f"  {case:<24}{sd:>3}{str(np.round(g0,3)):<22}"
                          f"   {type(e).__name__}", flush=True)
                    continue

                def d(k, sub="robot"):
                    x = np.asarray(finA[sub][k], float)
                    y = np.asarray(finB[sub][k], float)
                    return float(np.linalg.norm(x - y))

                d_qpos, d_qvel = d("qpos"), d("qvel")
                d_pc, d_vc = d("dof_pos_cmd"), d("dof_vel_cmd")
                eeA, eeB = ee_of(finA, h), ee_of(finB, h)
                d_ee = float(np.linalg.norm(eeA - eeB))
                # Judge on the COMMAND state as well as the EE. With
                # use_sim_dynamics=False the command arrays are what the filter
                # sees, so they are the authoritative comparison; d_ee alone
                # hid a 0.23 divergence in dof_pos_cmd on R1LiteUpper.
                ok = (d_ee < 1e-4 and d_pc < 1e-6 and d_vc < 1e-6
                      and okA == okB)

                rows.append({"case": case, "seed": sd, "g0": g0.tolist(),
                             "d_qpos": d_qpos, "d_qvel": d_qvel,
                             "d_pos_cmd": d_pc, "d_vel_cmd": d_vc,
                             "d_ee": d_ee, "steps_A": nA, "steps_B": nB,
                             "reached_A": okA, "reached_B": okB, "ok": ok})
                print(f"  {case:<24}{sd:>3}{str(np.round(g0,3)):<22}"
                      f"{d_qpos:>10.2e}{d_qvel:>10.2e}{d_pc:>10.2e}"
                      f"{d_vc:>10.2e}{d_ee:>11.2e}{nA:>7}{nB:>7}"
                      f"  {'OK' if ok else 'MISMATCH'}", flush=True)

    json.dump(rows, open(a.out, "w"), indent=2, default=float)
    n_ok = sum(r["ok"] for r in rows)
    print(f"\n{'='*100}")
    print(f"{n_ok}/{len(rows)} resumed runs land within 0.1 mm of the full run")
    if rows:
        ee = np.array([r["d_ee"] for r in rows])
        print(f"   d_ee  median {np.median(ee):.3e}  max {ee.max():.3e}")
        bad = [r for r in rows if not r["ok"]]
        for r in bad[:10]:
            print(f"   MISMATCH {r['case']} s{r['seed']} g0={np.round(r['g0'],3)}"
                  f"  d_ee={r['d_ee']:.3e}  reached {r['reached_A']}/{r['reached_B']}"
                  f"  steps {r['steps_A']}/{r['steps_B']}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
