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

#: DEFAULT SCENES = every benchmark family whose simulator state can be captured
#: and restored faithfully, measured over 25 families x 3 seeds x 3 g0 candidates
#: (see the fidelity test in test_run_from_state.py). The search resumes from a
#: captured handover, so a family whose state does not round-trip wastes the
#: whole point of the resume: candidates are evaluated on a trajectory the real
#: rollout would not produce, and every hit then dies at prefix confirmation.
#:
#:      G1FixedBase   all 8 variants        EXACT   56/56
#:      G1MobileBase  6 of 8 variants       EXACT
#:      G1MobileBase  _D1_WG_SO_v0          8/9  \  the single miss in each is a
#:      G1MobileBase  _D2_WG_DO_v1          8/9  /  horizon-limited run, where
#:                                                  both rollouts time out and
#:                                                  never converge to compare
#:
#: The two partials are kept: prefix confirmation makes them sound regardless,
#: and _D2_WG_DO_v1 is the single most productive scene found so far (the ssa,
#: pssa and sss attacks all come from it). Excluding it would trade a real
#: result for a cosmetic 100%.
#:
#: DELIBERATELY EXCLUDED -- state does not round-trip, so resume-based search is
#: unreliable there and these need the prefix-only path instead:
#:      G1SportMode_D1_WG_SO_v1   0/7    locomotion/gait state uncaptured
#:      IIWA14Single  D1, D2      0/10
#:      R1LiteUpper   D1          0/2      D2  1/4
#:      Gen3Single    D1, D2      1/3 each
#:      LRMate200iD3f D1 3/6, D2  5/6      mixed WITHIN a single case, so the
#:                                         carrier is path-dependent, not a
#:                                         property of the robot
SCENES = [
    ("G1FixedBase_D1_AG_SO_v0", "distance"),
    ("G1FixedBase_D1_AG_SO_v1", "distance"),
    ("G1FixedBase_D1_AG_DO_v0", "distance"),
    ("G1FixedBase_D1_AG_DO_v1", "distance"),
    ("G1FixedBase_D2_AG_SO_v0", "velocity"),
    ("G1FixedBase_D2_AG_SO_v1", "velocity"),
    ("G1FixedBase_D2_AG_DO_v0", "velocity"),
    ("G1FixedBase_D2_AG_DO_v1", "velocity"),
    ("G1MobileBase_D1_WG_SO_v0", "distance"),
    ("G1MobileBase_D1_WG_SO_v1", "distance"),
    ("G1MobileBase_D1_WG_DO_v0", "distance"),
    ("G1MobileBase_D1_WG_DO_v1", "distance"),
    ("G1MobileBase_D2_WG_SO_v0", "velocity"),
    ("G1MobileBase_D2_WG_SO_v1", "velocity"),
    ("G1MobileBase_D2_WG_DO_v0", "velocity"),
    ("G1MobileBase_D2_WG_DO_v1", "velocity"),
]


def set_channel(world, channel):
    """Tell the task which goal channel a schedule refers to.

    The task drives the ARM goal by default. For the base channel it must steer
    robot_goal_base instead and measure waypoint progress on the base pose, so
    the schedule setter differs -- routing every rollout through here keeps the
    two paths from silently diverging.
    """
    t = world.harness.env.task
    t.channel = channel
    if channel == "arm":
        t.base_goal_schedule = None


def run_ch(world, schedule, channel, **kw):
    """One rollout on the given goal channel."""
    t = world.harness.env.task
    if channel == "base":
        orig = t.set_goal_schedule
        t.set_goal_schedule = lambda wps: t.set_base_goal_schedule(wps)
        try:
            return world.run(schedule, **kw)
        finally:
            t.set_goal_schedule = orig
    set_channel(world, "arm")
    return world.run(schedule, **kw)


def failing_leg(rec):
    for s in rec.steps:
        if s.clearance < 0.0:
            return int(s.wp_idx)
    return None

def parse_seeds(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="sss")
    p.add_argument("--seeds", default="0,1,2",
                   help="seeds to SCAN, e.g. '1-20' or '0,1,2'. Seeds whose "
                        "gate fails are skipped and the next is tried.")
    p.add_argument("--n-worlds", type=int, default=3,
                   help="how many gate-PASSING worlds to actually search per "
                        "scenario before moving on")
    p.add_argument("--scenes", default=None)
    p.add_argument("--grid", type=int, default=6)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--relax", action="store_true", default=True)
    p.add_argument("--goal-channel", default="arm", choices=["arm", "base"],
                   help="which goal channel to attack. 'arm' inserts an "
                        "end-effector goal (0.3 m cube). 'base' inserts a base "
                        "pose (x, y, yaw) -- available only on whole-body (WG) "
                        "cases, where base_goal_range spans 1.6 x 1.6 m plus "
                        "full yaw, ~28x the arm footprint plus a rotation the "
                        "arm channel has no analogue for.")
    p.add_argument("--max-controls", type=int, default=0,
                   help="stop sweeping a world once this many PREFIX-CONFIRMED "
                        "controls are found (0 = no cap, sweep the whole grid). "
                        "A cap saves rollouts but makes counts incomparable "
                        "across configs -- it measures the cap, not the density "
                        "of attackable goals.")
    p.add_argument("--config", default=None,
                   help="trial config YAML. When given it supplies every "
                        "parameter that reaches SPARK -- demand, d_min, phi_n, "
                        "phi_k, slack weight and the raw safe_algo fields -- so "
                        "a run is reproducible from one file.")
    p.add_argument("--max-steps", type=int, default=None,
                   help="override the per-case rollout horizon (default: 900 "
                        "for _D2_ families, 1500 otherwise)")
    p.add_argument("--out-dir", default="fuzz/siren/experiment/g0_search_state")
    a = p.parse_args(argv)

    from ..world.run import World
    from ..world.types import real_filter
    from ..world.sim import probe
    from ..world import derived
    from ..search.pick import is_admissible
    from .state import capture_world, run_from_state
    from . import trialconf

    tconf = trialconf.load(a.config) if a.config else None
    if tconf is not None:
        sr = tconf.get("search", {})
        if a.seeds == "0,1,2":
            a.seeds = str(sr.get("seeds", a.seeds))
        if a.n_worlds == 3 and sr.get("n_worlds") is not None:
            a.n_worlds = int(sr["n_worlds"])
        if a.grid == 6 and sr.get("grid") is not None:
            a.grid = int(sr["grid"])
        if sr.get("relax") is not None:
            a.relax = bool(sr["relax"])
        if a.max_controls == 0 and sr.get("max_controls") is not None:
            a.max_controls = int(sr["max_controls"])
        if not a.scenes:
            scenes = [(c, trialconf.index_for(c))
                      for c in trialconf.USABLE_SCENES]
        print(trialconf.describe(tconf), flush=True)

    os.makedirs(a.out_dir, exist_ok=True)
    scenes = SCENES
    if a.scenes:
        # accept ANY benchmark case name, not just the default six -- the index
        # follows the D1/D2 naming, so no lookup table is needed
        scenes = [(c.strip(), "velocity" if "_D2_" in c else "distance")
                  for c in a.scenes.split(",") if c.strip()]
    seeds = parse_seeds(a.seeds)
    ok_attack = ("COLLISION", "DEADLOCK") if a.relax else ("COLLISION",)

    print(f"{len(scenes)} scenarios x {len(seeds)} seeds = "
          f"{len(scenes)*len(seeds)} scenes, filter={a.algo}, lam={a.lam}\n",
          flush=True)

    summary = []
    for case, index in scenes:
        # D2 families are acceleration-controlled and converge faster, hence the
        # shorter default. A MobileBase base move is slow enough that 900 can
        # expire before [g0, G1] ever reaches -- that shows up as
        # c1_never_reached_G1 and kills candidates for a horizon reason rather
        # than a safety one. --max-steps overrides when that is what is biting.
        steps = a.max_steps or (900 if "_D2_" in case else 1500)
        # Scan seeds until n_worlds of them PASS the gate. A failing gate means
        # this seed's layout has no known collision to relabel, so it is not a
        # world we can build an attack in -- move to the next seed rather than
        # burning the quota on it. The seeds that passed are recorded in the
        # output so a result can be traced back to its exact world.
        n_passed = 0
        total_confirmed = 0        # controls found for this (filter, scene)
        gate_log = []
        for sd in seeds:
            # STOP ON CONTROLS, not on worlds swept. Stopping after N worlds
            # regardless of outcome is what left most (filter, scene) pairs with
            # no target at all: a gate can pass and the sweep still find nothing,
            # and the job then gave up with seeds left unscanned.
            if a.max_controls and total_confirmed >= a.max_controls:
                break
            if n_passed >= a.n_worlds:      # safety cap on compute, not a goal
                print(f"   world cap {a.n_worlds} reached with "
                      f"{total_confirmed} controls — giving up on this pair",
                      flush=True)
                break
            if tconf is not None:
                spec = trialconf.spec_for(tconf, a.algo, case)
            else:
                spec = real_filter(algo=a.algo, index=index, d_min=0.02,
                                   eta=0.02, lam=a.lam, k=0.1)
            # The channel belongs in the tag. Without it an arm job and a base
            # job for the same (case, algo, seed) write the SAME filename, so
            # whichever finishes last silently erases the other's controls.
            tag = f"{case}_{a.algo}_s{sd}_lam{a.lam}_{a.goal_channel}"
            try:
                w = World.build(seed=sd, spec=spec, test_case=case,
                                max_steps=steps)
                sc = w.scene()
                h = w.harness
            except Exception as e:
                print(f"{case} s{sd}: build failed {type(e).__name__}",
                      flush=True)
                continue

            if a.goal_channel == "base":
                if not getattr(h.env.task, "base_goal_enable", False):
                    print(f"\n{case} s{sd}: no base goal on this case "
                          f"(base_goal_enable=False) — skipping", flush=True)
                    continue
                from scipy.spatial.transform import Rotation as _R
                b0 = np.asarray(h.env.task.robot_base_frame, float)
                yaw0 = float(_R.from_matrix(b0[:3, :3]).as_euler("xyz")[2])
                G1p = np.array([b0[0, 3], b0[1, 3], yaw0])   # home BASE pose
                gb = np.asarray(h.env.task.robot_goal_base.frame, float)
                G1 = np.array([gb[0, 3], gb[1, 3],
                               float(_R.from_matrix(gb[:3, :3]).as_euler("xyz")[2])])
            else:
                G1p = np.asarray(sc.G0, float)      # scenario START becomes G1'
                G1 = np.asarray(sc.G1, float)       # scenario GOAL stays G1

            gate = run_ch(w, [G1], a.goal_channel, max_steps=steps)
            print(f"\n{case} s{sd}: gate home->G1 = {gate.label} "
                  f"({gate.min_clearance:+.6f})", flush=True)
            gate_log.append({"seed": sd, "gate": gate.label,
                             "min_clearance": float(gate.min_clearance)})
            if gate.label not in ("COLLISION", "DEADLOCK"):
                print(f"   no raw material — trying next seed "
                      f"({n_passed}/{a.n_worlds} worlds so far)", flush=True)
                continue
            n_passed += 1
            print(f"   gate PASSED — world {n_passed}/{a.n_worlds}", flush=True)

            if a.goal_channel == "base":
                br = h.env.task.base_goal_range
                rr = getattr(h.env.task, "base_goal_rot_range", (-np.pi, np.pi))
                lo = [br[0][0], br[1][0], rr[0]]
                hi = [br[0][1], br[1][1], rr[1]]
            else:
                lo = [b[0] for b in sc.bounds]
                hi = [b[1] for b in sc.bounds]
            ax = [np.linspace(lo[i], hi[i], a.grid) for i in range(3)]
            g0s = [np.array([x, y, z]) for x in ax[0] for y in ax[1] for z in ax[2]]
            if a.goal_channel == "base":
                # Apply SPARK's OWN base-goal rule, not the arm one. benchmark_task
                # rejects a sampled base goal whose XY distance to any obstacle is
                # below base_goal_keepout (default 0.1), exactly as it rejects arm
                # goals below arm_goal_keepout. Sweeping the raw grid instead held
                # the two channels to different standards: an arm attacker was
                # confined to goals the planner could emit while a base attacker
                # was not, so a "base attack" could be a pose sitting on an
                # obstacle -- not an attack under the same threat model, and a
                # wasted rollout besides.
                ko = float(getattr(h.env.task, "base_goal_keepout", 0.1))
                obs_xy = np.array([np.asarray(o)[:2, 3]
                                   for o in sc.obstacles_world]) if len(sc.obstacles_world) else None
                def _base_ok(g):
                    if obs_xy is None or not len(obs_xy):
                        return True
                    return bool(np.all(np.linalg.norm(obs_xy - g[:2], axis=1) >= ko))
                n_raw = len(g0s)
                g0s = [g for g in g0s if _base_ok(g)]
                print(f"   base grid: {len(g0s)}/{n_raw} pass SPARK's "
                      f"base_goal_keepout={ko} (xy)", flush=True)
            else:
                # SIREN: keep the rejects so the log covers the WHOLE grid, not
                # just what survived admissibility.
                _all = list(g0s)
                g0s = [g for g in g0s if is_admissible(g, sc)[0]]
                _adm = {tuple(np.round(g, 9)) for g in g0s}
                g0_log = [{"G0": g.tolist(), "stage": "inadmissible"}
                          for g in _all if tuple(np.round(g, 9)) not in _adm]

            t0 = time.time()
            try:
                g0_log
            except NameError:
                g0_log = []
            ins, mods, confirmed, n_bad_s0, n_err = [], [], [], 0, 0
            # SIREN FIX. The three C1 rejection reasons used to collapse into one
            # uncounted `continue`, so a sweep that rejected every G0 printed
            # "0 insertion, 0 bad handover, 0 errors" and gave no clue why.
            # Split them: collided / never reached G1 / no handover captured.
            n_c1_collided = n_c1_noreach = n_c1_nos0 = n_attack_none = 0
            for g0 in g0s:
                # SIREN: one record per tried G0, whatever happens to it. Before
                # this, four distinct outcomes shared an uncounted `continue`, so
                # a sweep reporting "0 insertion, 0 bad handover, 0 errors" gave
                # no way to tell C1 failures from attacks that simply had no
                # effect. On D1_AG_SO_v0 s24 that hid 37 of 125 candidates which
                # PASSED C1 and then produced no collision.
                _t0 = time.time()
                cur = {"G0": g0.tolist(), "stage": None,
                       "base_min_clear": None, "handover_clear": None,
                       "attack_label": None, "attack_clear": None,
                       "attack_leg": None}
                try:
                    # ---- C1: legitimate task must reach; capture s0 ------- #
                    probe.reset_giveups(h)
                    af, ti = h.reset()
                    # C1 steps manually (it must capture s0 mid-run), so the
                    # channel has to be selected here too -- w.run is bypassed
                    if a.goal_channel == "base":
                        h.env.task.set_base_goal_schedule([g0, G1])
                    else:
                        set_channel(w, "arm")
                        h.env.task.set_goal_schedule([g0, G1])
                    u, ai = h.algo.act(af, ti)
                    # C1 must mean "arrived SAFELY", not merely "arrived".
                    # reached_final alone let a rollout that penetrated an
                    # obstacle and then still reached G1 pass the check, while
                    # measure.classify_run would have called it COLLISION --
                    # collision is tested BEFORE reached there, with
                    # collision_margin = 0.0. That mislabelled 18 of 73 controls,
                    # including every cbf one: their baselines collide, so the
                    # inserted goal was not what broke the task.
                    s0, ok_reach, base_min_clear = None, False, np.inf
                    for t in range(steps):
                        af, ti = h.env.step(u, ai)
                        u, ai = h.algo.act(af, ti)
                        base_min_clear = min(base_min_clear, h.clearance(ti))
                        # SIREN FIX. A collision here is already FATAL for this
                        # G0 -- the guard below rejects on base_min_clear < 0
                        # regardless of what happens next. Without this break the
                        # rollout kept stepping after the fatal contact, because
                        # the robot carries on and eventually reaches G1, so the
                        # loop only ended on reached_final. Measured on
                        # G1FixedBase_D2_AG_DO_v0 s4: 196-235 steps spent to
                        # learn what step ~80 already decided, 2.5x the cost
                        # (7.2s -> 2.9s over three G0). Exactly equivalent: the
                        # only fields read after this point are base_min_clear
                        # (already negative) and ok_reach (must stay False).
                        n_c1_collided += 1 if base_min_clear < 0.0 else 0
                        if base_min_clear < 0.0:
                            break
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
                    cur["base_min_clear"] = float(base_min_clear)
                    if s0 is None or not ok_reach or base_min_clear < 0.0:
                        if base_min_clear >= 0.0:
                            if s0 is None:
                                n_c1_nos0 += 1
                                cur["stage"] = "c1_no_handover"
                            else:
                                n_c1_noreach += 1
                                cur["stage"] = "c1_never_reached_G1"
                        else:
                            cur["stage"] = "c1_collided"
                        cur["sec"] = round(time.time() - _t0, 2)
                        g0_log.append(cur)
                        continue
                    # ---- C1b: the handover itself must be legitimate ------ #
                    cur["handover_clear"] = float(clr0)
                    if clr0 <= 0.0 or (np.isfinite(mu0) and mu0 > 1e-9):
                        n_bad_s0 += 1
                        cur["stage"] = "bad_handover"
                        cur["sec"] = round(time.time() - _t0, 2)
                        g0_log.append(cur)
                        continue
                    # ---- C2/C3: resume from s0 and attack ----------------- #
                    if a.goal_channel == "base":
                        h.env.task.channel = "base"
                    atk = run_from_state(w, s0, [G1p, G1], max_steps=steps)
                except Exception as _e:
                    n_err += 1
                    cur["stage"] = "error"
                    cur["error"] = type(_e).__name__
                    cur["sec"] = round(time.time() - _t0, 2)
                    g0_log.append(cur)
                    continue

                cur["attack_label"] = atk.label
                cur["attack_clear"] = float(atk.min_clearance)
                leg = failing_leg(atk)
                cur["attack_leg"] = leg
                if atk.label not in ok_attack:
                    # C1 PASSED -- a legitimate G0 was found -- but inserting G1'
                    # from that handover did not break the task. This is the
                    # attack failing, NOT the G0 search failing.
                    n_attack_none += 1
                    cur["stage"] = "attack_no_effect"
                    cur["sec"] = round(time.time() - _t0, 2)
                    g0_log.append(cur)
                    continue
                if atk.label == "COLLISION" and leg == 0:
                    mods.append({"G0": g0.tolist(), "clearance":
                                 float(atk.min_clearance)})
                    cur["stage"] = "modification"
                    cur["sec"] = round(time.time() - _t0, 2)
                    g0_log.append(cur)
                    continue
                ins.append({"G0": g0.tolist(), "attack_label": atk.label,
                            "from_state_clearance": float(atk.min_clearance),
                            "handover_clearance": float(clr0)})
                # ---- prefix confirmation (the sound check) --------------- #
                pre = run_ch(w, [g0, G1p, G1], a.goal_channel, max_steps=steps)
                pleg = failing_leg(pre)
                good = (pre.label in ok_attack
                        and (pre.label == "DEADLOCK" or pleg == 2))
                ins[-1].update({"prefix_label": pre.label,
                                "prefix_leg": pleg,
                                "prefix_clearance": float(pre.min_clearance),
                                "confirmed": bool(good)})
                cur.update({"stage": "confirmed" if good else "insertion_unconfirmed",
                            "prefix_label": pre.label, "prefix_leg": pleg,
                            "prefix_clear": float(pre.min_clearance),
                            "sec": round(time.time() - _t0, 2)})
                g0_log.append(cur)
                if good:
                    confirmed.append(ins[-1])
                    print(f"   *** CONTROL G0={np.round(g0,3)}  from-state "
                          f"{atk.label} ({atk.min_clearance:+.6f})  prefix "
                          f"{pre.label} ({pre.min_clearance:+.6f}) leg {pleg}",
                          flush=True)
                    if (a.max_controls
                            and total_confirmed + len(confirmed) >= a.max_controls):
                        print(f"   cap reached ({a.max_controls}) — stopping "
                              f"this world after {len(ins)} insertion "
                              f"candidates", flush=True)
                        break

            print(f"   swept {len(g0s)} G0 -> {len(ins)} insertion, "
                  f"{len(confirmed)} prefix-CONFIRMED, {len(mods)} modification, "
                  f"{n_bad_s0} bad handover, {n_err} errors | C1 rejects: "
                  f"{n_c1_collided} collided, {n_c1_noreach} never reached G1, "
                  f"{n_c1_nos0} no handover | {n_attack_none} passed C1 but "
                  f"attack had no effect "
                  f"({time.time()-t0:.0f}s)", flush=True)

            total_confirmed += len(confirmed)
            summary.append({"case": case, "seed": sd, "gate": gate.label,
                            "n_insertion": len(ins),
                            "n_modification": len(mods),
                            "n_confirmed": len(confirmed)})
            # SIREN: the per-G0 log is written for EVERY swept world, not only
            # ones that yielded a control. A sweep that finds nothing is exactly
            # the sweep whose 125 rejections you need to see; gating the record
            # on `confirmed` threw that away.
            sweep = {"case": case, "algo": a.algo, "seed": sd, "index": index,
                     "gate": gate.label, "gate_clear": float(gate.min_clearance),
                     "gate_steps": int(gate.n_steps),
                     "G1_prime": G1p.tolist(), "G1": G1.tolist(),
                     "bounds": [[float(l), float(hh)] for l, hh in sc.bounds],
                     "obstacles_world": [list(map(float, np.asarray(o)[:3, 3]))
                                         for o in sc.obstacles_world],
                     "n_G0_swept": len(g0s), "elapsed_s": round(time.time() - t0, 1),
                     "g0_stage_counts": {k: sum(1 for r in g0_log
                                                if r.get("stage") == k)
                                         for k in sorted({r.get("stage")
                                                          for r in g0_log})},
                     "g0_log": g0_log}
            spath = f"{a.out_dir}/g0log_{tag}.json"
            json.dump(sweep, open(spath, "w"), indent=2, default=float)
            print(f"   wrote {spath}  ({sweep['g0_stage_counts']})", flush=True)

            if confirmed:
                # Record the ACTUAL spec, not hardcoded literals. These four
                # fields used to be written as 0.02/0.02/a.lam/0.1 regardless of
                # --config, so a trial1 (SPARK-default) target claimed trial2's
                # values -- which defeats the point of the config file, since the
                # whole trial comparison rests on the recorded parameters.
                rec = {"case": case, "algo": a.algo, "seed": sd, "index": index,
                       # which goal channel this target attacks. Stage 3 must
                       # drive a base target through set_base_goal_schedule, not
                       # set_goal_schedule, or it silently fuzzes the wrong goal.
                       "channel": a.goal_channel,
                       "config": (tconf.get("name") if tconf else None),
                       "max_steps": steps,
                       "d_min": spec.d_min, "eta": spec.eta,
                       "lam": (spec.lam if spec.lam is not None else a.lam),
                       "k": spec.k,
                       "slack_weight": getattr(spec, "slack_weight", None),
                       "overrides": dict(spec.overrides or {}),
                       "G1_prime": G1p.tolist(), "G1": G1.tolist(),
                       # the bounds actually SEARCHED. sc.bounds is the arm
                       # workspace box; on the base channel the search ran over
                       # base_goal_range + yaw instead, and recording the arm box
                       # there gave downstream stages a search space that does
                       # not even contain the goals in the file.
                       "bounds": [[float(x) for x in b] for b in
                                  (zip(lo, hi) if a.goal_channel == "base"
                                   else sc.bounds)],
                       "keepout": float(sc.keepout),
                       "obstacles_world": [list(map(float, np.asarray(o)[:3, 3]))
                                           for o in sc.obstacles_world],
                       "n_G0_swept": len(g0s),
                       # SIREN: one entry per grid point -- location + the stage
                       # it died at. Stages: inadmissible | c1_collided |
                       # c1_never_reached_G1 | c1_no_handover | bad_handover |
                       # error | attack_no_effect | modification |
                       # insertion_unconfirmed | confirmed.
                       "g0_log": g0_log,
                       "g0_stage_counts": {k: sum(1 for r in g0_log
                                                  if r.get("stage") == k)
                                           for k in sorted({r.get("stage")
                                                            for r in g0_log})},
                       "seeds_scanned": [g["seed"] for g in gate_log],
                       "gate_log": gate_log,
                       "world_index": n_passed,
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
