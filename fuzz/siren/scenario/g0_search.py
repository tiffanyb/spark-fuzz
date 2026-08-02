"""
G0 search that resumes from the handover state instead of replaying leg 0.

    original scenario:  home -> scenario_goal
    attack:             [G0, G1', G1]     G1' := home,  G1 := scenario_goal

    Leg 2 (G1' -> G1) and the original scenario share endpoints. They differ
    only in departure state: the original starts at REST, leg 2 departs in
    motion. That is the whole mechanism, and it is why the gate below is a
    heuristic rather than a precondition.

GATE (per scenario, filter, seed)
    run [scenario_goal] from home -> must COLLIDE or DEADLOCK.
    This is leg 2 from rest: neither necessary (leg 2 in motion may collide
    where rest reaches) nor sufficient (rest colliding does not imply leg 2
    does). A failed gate means "no cheap raw material", NOT "no attack exists".

PER CANDIDATE
    C1  run [g0, G1] from home; must REACH, else skip. Capture s0 the instant
        wp_idx goes 0 -> 1 -- the true handover, with g0 INTERMEDIATE so the arm
        passes through in motion. Capturing from [g0] alone would give a REST
        state, a different system.
    C1b s0 must be legitimate: clearance > 0 and not already infeasible.
    C2  resume from s0 and run [G1', G1]; must COLLIDE or DEADLOCK.
    C3  locate the failing leg. NOTE the indices shift, because home -> g0 is no
        longer simulated:
            wp_idx 0 : s0  -> G1'
            wp_idx 1 : G1' -> G1      <- the attack leg
        COLLISION -> wp_idx at the first clearance < 0
        DEADLOCK  -> no separate check; classify_run's on_final_leg already
                     requires the stall at the final waypoint
        leg 0 -> MODIFICATION (crashed reaching the inserted goal)
        leg 1 -> INSERTION    (the control we want)

WHY EVERY HIT IS RE-CHECKED UNDER THE PREFIX CONTRACT. run_from_state was
measured exact (0.00e+00 on qpos, command state and EE) for G1FixedBase D1/D2
and LRMate, but on the whole-body robots -- G1MobileBase and R1LiteUpper -- a
resumed run drifts from the full run: divergence starts at step +2 and reaches
~8e-03 in joint space, 2-6 mm at the end-effector. Everything enumerable is
identical at the handover (goals, obstacle phase and RNG, IK warm start, base
pose, controller state), the world instance is irrelevant, and repeated runs are
deterministic -- the carrier of that drift is unidentified. Millimetres are fatal
here, where contacts run 50-500 microns. So the resume is used to SEARCH (fast)
and the prefix rollout [G0, G1', G1] from home is used to CONFIRM (sound). Only
prefix-confirmed hits are written out.

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 python -m fuzz.siren.scenario.g0_search \
        --algo sss --grid 6
"""

import argparse
import json
import os
import time

import numpy as np

SCENES = [
    ("G1FixedBase_D2_AG_SO_v0", "velocity"),
    ("G1MobileBase_D2_WG_SO_v1", "velocity"),
    ("G1MobileBase_D2_WG_DO_v1", "velocity"),
    ("R1LiteUpper_D2_AG_SO_v0", "velocity"),
    ("LRMate200iD3f_D2_AG_SO_v0", "velocity"),
    ("G1FixedBase_D1_AG_SO_v0", "distance"),
]


def failing_leg(rec):
    for s in rec.steps:
        if s.clearance < 0.0:
            return int(s.wp_idx)
    return None


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="sss")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--scenes", default=None)
    p.add_argument("--grid", type=int, default=6)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--relax", action="store_true", default=True)
    p.add_argument("--out-dir", default="fuzz/siren/experiment/g0_search_state")
    a = p.parse_args(argv)

    from ..world.run import World
    from ..world.types import real_filter
    from ..world.sim import probe
    from ..world import derived
    from ..search.pick import is_admissible
    from .state import capture_world, run_from_state

    os.makedirs(a.out_dir, exist_ok=True)
    scenes = SCENES
    if a.scenes:
        # accept ANY benchmark case name, not just the default six -- the index
        # follows the D1/D2 naming, so no lookup table is needed
        scenes = [(c.strip(), "velocity" if "_D2_" in c else "distance")
                  for c in a.scenes.split(",") if c.strip()]
    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    ok_attack = ("COLLISION", "DEADLOCK") if a.relax else ("COLLISION",)

    print(f"{len(scenes)} scenarios x {len(seeds)} seeds = "
          f"{len(scenes)*len(seeds)} scenes, filter={a.algo}, lam={a.lam}\n",
          flush=True)

    summary = []
    for case, index in scenes:
        steps = 900 if "_D2_" in case else 1500
        for sd in seeds:
            spec = real_filter(algo=a.algo, index=index, d_min=0.02, eta=0.02,
                               lam=a.lam, k=0.1)
            tag = f"{case}_{a.algo}_s{sd}_lam{a.lam}"
            try:
                w = World.build(seed=sd, spec=spec, test_case=case,
                                max_steps=steps)
                sc = w.scene()
                h = w.harness
            except Exception as e:
                print(f"{case} s{sd}: build failed {type(e).__name__}",
                      flush=True)
                continue

            G1p = np.asarray(sc.G0, float)      # scenario START becomes G1'
            G1 = np.asarray(sc.G1, float)       # scenario GOAL stays G1

            gate = w.run([G1], max_steps=steps)
            print(f"\n{case} s{sd}: gate home->G1 = {gate.label} "
                  f"({gate.min_clearance:+.6f})", flush=True)
            if gate.label not in ("COLLISION", "DEADLOCK"):
                print("   no raw material — skipping", flush=True)
                summary.append({"case": case, "seed": sd, "gate": gate.label,
                                "n_insertion": 0, "n_modification": 0,
                                "n_confirmed": 0})
                continue

            lo = [b[0] for b in sc.bounds]
            hi = [b[1] for b in sc.bounds]
            ax = [np.linspace(lo[i], hi[i], a.grid) for i in range(3)]
            g0s = [np.array([x, y, z]) for x in ax[0] for y in ax[1] for z in ax[2]]
            g0s = [g for g in g0s if is_admissible(g, sc)[0]]

            t0 = time.time()
            ins, mods, confirmed, n_bad_s0, n_err = [], [], [], 0, 0
            for g0 in g0s:
                try:
                    # ---- C1: legitimate task must reach; capture s0 ------- #
                    probe.reset_giveups(h)
                    af, ti = h.reset()
                    h.env.task.set_goal_schedule([g0, G1])
                    u, ai = h.algo.act(af, ti)
                    s0, ok_reach = None, False
                    for t in range(steps):
                        af, ti = h.env.step(u, ai)
                        u, ai = h.algo.act(af, ti)
                        if s0 is None and int(getattr(h.env.task, "wp_idx", 0)) >= 1:
                            s0 = capture_world(h)
                            clr0 = h.clearance(ti)
                            raw = probe.read_raw(h)
                            mu0 = np.nan
                            if raw:
                                d = derived.evaluate(
                                    raw["Lg"], raw["Lf"], raw["phi"],
                                    raw["phi_mask"], raw["u_lim"],
                                    demand_shape=spec.demand_shape,
                                    eta=spec.eta, lam=spec.lam, exact=True)
                                if bool(d.get("engaged")):
                                    mu0 = float(d.get("mu", np.nan))
                        if h.env.task.reached_final:
                            ok_reach = True
                            break
                    if s0 is None or not ok_reach:
                        continue
                    # ---- C1b: the handover itself must be legitimate ------ #
                    if clr0 <= 0.0 or (np.isfinite(mu0) and mu0 > 1e-9):
                        n_bad_s0 += 1
                        continue
                    # ---- C2/C3: resume from s0 and attack ----------------- #
                    atk = run_from_state(w, s0, [G1p, G1], max_steps=steps)
                except Exception:
                    n_err += 1
                    continue

                if atk.label not in ok_attack:
                    continue
                leg = failing_leg(atk)
                if atk.label == "COLLISION" and leg == 0:
                    mods.append({"G0": g0.tolist(), "clearance":
                                 float(atk.min_clearance)})
                    continue
                ins.append({"G0": g0.tolist(), "attack_label": atk.label,
                            "from_state_clearance": float(atk.min_clearance),
                            "handover_clearance": float(clr0)})
                # ---- prefix confirmation (the sound check) --------------- #
                pre = w.run([g0, G1p, G1], max_steps=steps)
                pleg = failing_leg(pre)
                good = (pre.label in ok_attack
                        and (pre.label == "DEADLOCK" or pleg == 2))
                ins[-1].update({"prefix_label": pre.label,
                                "prefix_leg": pleg,
                                "prefix_clearance": float(pre.min_clearance),
                                "confirmed": bool(good)})
                if good:
                    confirmed.append(ins[-1])
                    print(f"   *** CONTROL G0={np.round(g0,3)}  from-state "
                          f"{atk.label} ({atk.min_clearance:+.6f})  prefix "
                          f"{pre.label} ({pre.min_clearance:+.6f}) leg {pleg}",
                          flush=True)

            print(f"   swept {len(g0s)} G0 -> {len(ins)} insertion, "
                  f"{len(confirmed)} prefix-CONFIRMED, {len(mods)} modification, "
                  f"{n_bad_s0} bad handover, {n_err} errors "
                  f"({time.time()-t0:.0f}s)", flush=True)

            summary.append({"case": case, "seed": sd, "gate": gate.label,
                            "n_insertion": len(ins),
                            "n_modification": len(mods),
                            "n_confirmed": len(confirmed)})
            if confirmed:
                rec = {"case": case, "algo": a.algo, "seed": sd, "index": index,
                       "max_steps": steps, "d_min": 0.02, "eta": 0.02,
                       "lam": a.lam, "k": 0.1,
                       "G1_prime": G1p.tolist(), "G1": G1.tolist(),
                       "bounds": [[float(l), float(hh)] for l, hh in sc.bounds],
                       "keepout": float(sc.keepout),
                       "obstacles_world": [list(map(float, np.asarray(o)[:3, 3]))
                                           for o in sc.obstacles_world],
                       "n_G0_swept": len(g0s),
                       "controls": confirmed,
                       "insertion_unconfirmed": [x for x in ins
                                                 if not x["confirmed"]],
                       "modification_hits": mods}
                path = f"{a.out_dir}/control_{tag}.json"
                json.dump(rec, open(path, "w"), indent=2, default=float)
                print(f"   wrote {path}", flush=True)

    json.dump(summary, open(f"{a.out_dir}/summary_{a.algo}.json", "w"),
              indent=2, default=float)
    print(f"\n{'='*94}")
    print(f"{'scene':<28}{'seed':>5}{'gate':>11}{'insertion':>11}"
          f"{'confirmed':>11}{'modif':>8}")
    for s in summary:
        print(f"{s['case']:<28}{s['seed']:>5}{s['gate']:>11}"
              f"{s['n_insertion']:>11}{s['n_confirmed']:>11}"
              f"{s['n_modification']:>8}")
    tot = sum(s["n_confirmed"] for s in summary)
    print(f"\n{tot} prefix-confirmed controls across "
          f"{len(summary)} scenes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
