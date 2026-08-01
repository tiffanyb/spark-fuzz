"""
Are the 347 collisions infeasibility after all -- in DISCRETE time?

The Kind 1 / Kind 2 taxonomy is defined by infeasibility, and mu says there is
none: over 5817 candidates mu never once became positive, and the collisions sit
FARTHER from the boundary than the non-collisions. Either collisions are not
caused by infeasibility, or mu is answering the wrong question.

It is answering the wrong question. mu is

    mu(x) = min_u max_i ( L_f phi_i + L_g phi_i . u + demand_i )

which asks "does some control make the INSTANTANEOUS derivative acceptable right
now?" -- a continuous-time test. The plant is sampled-data: the control is held
for control_decimation * dt (10 ms as shipped). What decides whether the robot
actually touches anything is

    "is there an admissible control that, HELD FOR ONE INTERVAL, leaves the
     system safe at the next sample?"

Those agree only while the linearisation stays valid across the hold, and it
measurably does not: predicted phi_dot ~ -0.02 against a realised +0.03.

So this measures feasibility by brute force instead of by linearisation. At each
step it snapshots the simulator, tries K admissible controls, and asks what the
best reachable clearance at the next sample actually is. No model, no gradient,
no linearisation -- just the plant.

Two verdicts per step:

  one-step infeasible   no sampled control keeps clearance >= 0 at the next
                        sample. A hard lower bound on infeasibility: if it holds,
                        the state is genuinely doomed within one interval.
  doomed (greedy)       from here, even a controller that spends every step
                        greedily maximising clearance still collides within the
                        horizon. Weaker evidence than the above (greedy is not
                        optimal, so this can over-report), but it is the honest
                        approximation of "was it already too late", and the
                        alternative -- exact backward reachability over a 17-DoF
                        arm -- is not computable here.

Sampling is not exhaustive, so a state reported infeasible IS infeasible (some
control set was searched and none worked), while a state reported feasible might
still be infeasible for controls not sampled. The bound therefore runs in the
safe direction for the claim being tested.

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    python -m fuzz.siren.probe_discrete_feasibility --n-collisions 6
"""

import argparse
import json

import numpy as np


def _task_objects(task):
    """Every TaskObject3D the task owns (obstacles, goals).

    These advance on their OWN dynamics inside env.step, independently of the
    robot, and each carries a private RandomState that is consumed on every move.
    Restoring the robot alone therefore does NOT rewind the world: in the
    dynamic-obstacle cases the first version of this probe let obstacles drift by
    one step for every trial control, i.e. 40x the real motion, and silently
    invalidated every DO result.
    """
    out = []
    for name in dir(task):
        if name.startswith("_"):
            continue
        try:
            v = getattr(task, name)
        except Exception:
            continue
        if hasattr(v, "frame") and hasattr(v, "step_counter"):
            out.append(((name, None), v))
        elif isinstance(v, (list, tuple)):
            for i, x in enumerate(v):
                if hasattr(x, "frame") and hasattr(x, "step_counter"):
                    out.append(((name, i), x))
    return out


def _snap_obj(o):
    d = {"frame": np.array(o.frame, copy=True),
         "step_counter": int(o.step_counter)}
    for k in ("last_direction", "direction", "last_frame"):
        if hasattr(o, k):
            v = getattr(o, k)
            d[k] = np.array(v, copy=True) if v is not None else None
    if hasattr(o, "rs"):
        d["rs_state"] = o.rs.get_state()
    return d


def _restore_obj(o, d):
    o.frame[...] = d["frame"]
    o.step_counter = d["step_counter"]
    for k in ("last_direction", "direction", "last_frame"):
        if k in d and d[k] is not None and hasattr(o, k):
            setattr(o, k, np.array(d[k], copy=True))
    if "rs_state" in d and hasattr(o, "rs"):
        o.rs.set_state(d["rs_state"])


def snapshot(h):
    """Everything env.step advances: MuJoCo state + the agent's held commands +
    every self-propelled task object (obstacles and goals)."""
    ag = h.env.agent
    task = h.env.task
    snap_objs = {k: _snap_obj(o) for k, o in _task_objects(task)}
    return {
        "_objs": snap_objs,
        "qpos": ag.data.qpos.copy(), "qvel": ag.data.qvel.copy(),
        "act": ag.data.act.copy() if ag.data.act.size else None,
        "ctrl": ag.data.ctrl.copy(), "time": float(ag.data.time),
        "dof_pos_cmd": None if ag.dof_pos_cmd is None else ag.dof_pos_cmd.copy(),
        "dof_vel_cmd": None if ag.dof_vel_cmd is None else ag.dof_vel_cmd.copy(),
        "dof_acc_cmd": None if ag.dof_acc_cmd is None else ag.dof_acc_cmd.copy(),
        "dof_pos_fbk": None if ag.dof_pos_fbk is None else ag.dof_pos_fbk.copy(),
        "dof_vel_fbk": None if ag.dof_vel_fbk is None else ag.dof_vel_fbk.copy(),
        "wp_idx": int(getattr(h.env.task, "wp_idx", 0)),
    }


def restore(h, s):
    import mujoco
    ag = h.env.agent
    ag.data.qpos[:] = s["qpos"]
    ag.data.qvel[:] = s["qvel"]
    if s["act"] is not None and ag.data.act.size:
        ag.data.act[:] = s["act"]
    ag.data.ctrl[:] = s["ctrl"]
    ag.data.time = s["time"]
    for k in ("dof_pos_cmd", "dof_vel_cmd", "dof_acc_cmd",
              "dof_pos_fbk", "dof_vel_fbk"):
        if s[k] is not None:
            getattr(ag, k)[:] = s[k]
    try:
        h.env.task.wp_idx = s["wp_idx"]
    except Exception:
        pass
    for k, o in _task_objects(h.env.task):
        if k in s["_objs"]:
            _restore_obj(o, s["_objs"][k])
    mujoco.mj_forward(ag.model, ag.data)


def guarded_clearance(h, task_info, env_mask):
    from spark_utils import compute_masked_distance_matrix
    dm, _ = compute_masked_distance_matrix(
        frame_list_1=h.env.task.robot_frames_world,
        geom_list_1=h.robot_cfg.CollisionVol.values(),
        frame_list_2=task_info["obstacle"]["frames_world"],
        geom_list_2=task_info["obstacle"]["geom"])
    if dm is None:
        return np.inf
    return float(np.where(env_mask, np.asarray(dm, float), np.inf).min())


def greedy_escape(h, snap, ai, u_lim, env_mask, rng, horizon, k):
    """From this exact state, can a BEST-EFFORT controller still avoid contact?

    Each step picks, from k sampled controls, the one maximising next-sample
    clearance, and runs that for `horizon` steps. Returns (escaped, min_clear).

    Greedy is not optimal, so a reported failure is weaker evidence than one-step
    infeasibility -- a cleverer controller might survive where this one does not.
    It is used because exact backward reachability over a 17-DoF arm is not
    computable here, and because one-step infeasibility is nearly tautological at
    the moment of contact: it answers "is this step lost?", never "when was it
    lost?". This answers the second question approximately, which is the one the
    Kind 1 / Kind 2 split actually turns on.
    """
    restore(h, snap)
    worst = np.inf
    for _ in range(horizon):
        inner = snapshot(h)
        best_u, best_c = None, -np.inf
        for uk in control_samples(None, None, u_lim, rng, k):
            restore(h, inner)
            try:
                _af, _ti = h.env.step(uk, ai)
                c = guarded_clearance(h, _ti, env_mask)
            except Exception:
                c = -np.inf
            if c > best_c:
                best_c, best_u = c, uk
        restore(h, inner)
        if best_u is None:
            return False, -np.inf
        try:
            _af, _ti = h.env.step(best_u, ai)
            c = guarded_clearance(h, _ti, env_mask)
        except Exception:
            return False, -np.inf
        worst = min(worst, c)
        if c < 0.0:
            return False, float(worst)
    return True, float(worst)


def control_samples(u_safe, u_ref, u_lim, rng, k):
    """Candidate controls: what the filter chose, what the task wanted, stop,
    full-authority retreats along each axis, then random corners of the box."""
    n = len(u_lim)
    out = []
    if u_safe is not None:
        out.append(np.asarray(u_safe, float).reshape(-1))
    if u_ref is not None:
        out.append(np.asarray(u_ref, float).reshape(-1))
    out.append(np.zeros(n))
    for i in range(n):                       # single-axis full effort, both ways
        for sgn in (+1.0, -1.0):
            u = np.zeros(n)
            u[i] = sgn * u_lim[i]
            out.append(u)
    while len(out) < k:                      # random vertices of the control box
        out.append(rng.choice([-1.0, 1.0], size=n) * u_lim)
    return out[:k]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--from-json", default="/tmp/pipeline_experiment.jsonl")
    p.add_argument("--n-collisions", type=int, default=6,
                   help="ignored when --per-scene is set")
    p.add_argument("--per-scene", type=int, default=0,
                   help="sample this many collisions from EVERY scene")
    p.add_argument("--shard", default="0/1", help="i/n, split scenes across procs")
    p.add_argument("--k-controls", type=int, default=64)
    p.add_argument("--horizon", type=int, default=15, help="greedy escape horizon")
    p.add_argument("--tail", type=int, default=45, help="steps before contact to probe")
    p.add_argument("--out", default="/tmp/discrete_feasibility.json")
    a = p.parse_args(argv)

    from .world.run import World
    from .world.types import real_filter
    from .world.sim import probe

    # pick collided candidates from the sweep, preferring D2 (all 347 are D2)
    cases = {}
    for line in open(a.from_json):
        r = json.loads(line)
        if r.get("status") != "ok":
            continue
        for e in r["evaluations"]:
            if e["stage"] == "evaluated" and e["collided"]:
                cases.setdefault((r["case"], r["seed"], r["max_steps"]), []).append(e)
    keys = sorted(cases)
    si, sn = (int(x) for x in a.shard.split("/"))
    keys = [k for i, k in enumerate(keys) if i % sn == si]

    picks = []
    if a.per_scene:
        # spread WITHIN each scene rather than taking the first few: collisions
        # found early in a search cluster in whatever region the picker locked
        # onto, so the first N are not representative of the scene.
        for k in keys:
            evs = cases[k]
            idx = np.unique(np.linspace(0, len(evs) - 1,
                                        min(a.per_scene, len(evs))).astype(int))
            for i in idx:
                picks.append((k[0], k[1], k[2], evs[i]))
    else:
        for k in keys[:a.n_collisions]:
            picks.append((k[0], k[1], k[2], cases[k][0]))
    print(f"shard {si}/{sn}: probing {len(picks)} collisions across "
          f"{len(keys)} scenes\n", flush=True)

    rng = np.random.RandomState(0)
    rows = []
    for case, seed, ms, ev in picks:
        index = "velocity" if "_D2_" in case else "distance"
        spec = real_filter(algo="ssa", index=index, d_min=0.02, eta=0.02, k=0.1)
        w = World.build(seed=seed, spec=spec, test_case=case, max_steps=ms)
        sc = w.scene()
        h = w.harness
        si = h.algo.safe_controller.safe_algo.safety_index
        env_mask = np.asarray(si.env_collision_mask, bool)
        u_lim = np.abs(np.asarray(h.algo.safe_controller.safe_algo.control_max,
                                  float).reshape(-1))
        cand = np.asarray(ev["candidate"], float)

        # --- REFERENCE run: no probing at all, so nothing can be perturbed ---
        # Compared against the probed run below. If snapshot/restore misses any
        # state, the two trajectories diverge and the candidate is reported
        # UNFAITHFUL rather than silently believed. The DO scenarios failed
        # exactly this check before obstacles were rewound.
        ref = w.run([cand, np.asarray(sc.G1)], max_steps=ms)
        ref_contact = len(ref.steps) - 1
        ref_clear = float(ref.min_clearance)

        probe.reset_giveups(h)
        af, ti = h.reset()
        h.env.task.set_goal_schedule([cand, np.asarray(sc.G1)])
        u, ai = h.algo.act(af, ti)

        traj, snaps = [], {}
        for t in range(ms):
            snap = snapshot(h)
            snaps[t] = snap
            u_ref = ai.get("u_ref", None)
            wp = int(getattr(h.env.task, "wp_idx", 0))
            engaged = bool(ai.get("trigger_safe", False))

            # --- what is the BEST next-sample clearance any control achieves? --
            best = -np.inf
            for uk in control_samples(u, u_ref, u_lim, rng, a.k_controls):
                restore(h, snap)
                try:
                    _af, _ti = h.env.step(uk, ai)
                    c = guarded_clearance(h, _ti, env_mask)
                except Exception:
                    c = -np.inf
                best = max(best, c)
            restore(h, snap)

            # --- advance for real -------------------------------------------- #
            af, ti = h.env.step(u, ai)
            clear = guarded_clearance(h, ti, env_mask)
            traj.append({"t": t, "wp": wp, "engaged": engaged,
                         "clearance": clear, "best_next_clearance": float(best),
                         "one_step_infeasible": bool(best < 0.0)})
            u, ai = h.algo.act(af, ti)
            if h.env.task.reached_final or clear < 0.0:
                break

        contact = traj[-1]["t"]

        # --- point of no return: scan back from contact until escape works --- #
        pnr, escape_log = None, []
        for t in range(contact, max(-1, contact - a.tail), -1):
            if t not in snaps:
                break
            ok, mc = greedy_escape(h, snaps[t], ai, u_lim, env_mask, rng,
                                   a.horizon, max(8, a.k_controls // 4))
            escape_log.append({"t": t, "escaped": bool(ok), "min_clear": mc})
            if ok:
                pnr = t + 1          # earliest step from which it is unavoidable
                break
        if pnr is None and escape_log:
            pnr = escape_log[-1]["t"]   # doomed as far back as we looked
        leg2 = [r for r in traj if r["wp"] >= 1]
        t0 = next((r["t"] for r in leg2 if r["engaged"]), None)
        first_infeas = next((r["t"] for r in traj if r["one_step_infeasible"]), None)
        n_infeas = sum(1 for r in traj if r["one_step_infeasible"])

        # classify on the POINT OF NO RETURN, not on one-step infeasibility:
        # the latter fires at the contact step by construction and so would call
        # everything Kind 2 regardless of what actually happened.
        if pnr is None:
            verdict = "NO_DISCRETE_INFEASIBILITY"
        elif t0 is None:
            verdict = "DOOMED_BUT_NEVER_ENGAGED"
        elif pnr <= t0:
            verdict = "KIND_1 (discrete)"
        else:
            verdict = "KIND_2 (discrete)"

        faithful = (contact == ref_contact)
        rows.append({"case": case, "seed": seed, "candidate": ev["candidate"],
                     "faithful": bool(faithful),
                     "ref_contact_step": ref_contact,
                     "ref_min_clearance": ref_clear,
                     "contact_step": contact, "t0_leg2": t0,
                     "first_one_step_infeasible": first_infeas,
                     "n_one_step_infeasible": n_infeas,
                     "point_of_no_return": pnr,
                     "escape_log": escape_log,
                     "verdict": verdict,
                     "mu_t0_continuous": ev["mu_t0"],
                     "max_mu_continuous": ev["max_mu_leg2"],
                     "tail": traj[-a.tail:]})
        margin = (pnr - t0) if (pnr is not None and t0 is not None) else None
        if not faithful:
            print(f"{case} seed={seed}  UNFAITHFUL: probed contact@{contact} but "
                  f"reference contact@{ref_contact} — snapshot/restore is missing "
                  f"state; verdict NOT trustworthy", flush=True)
        print(f"{case} seed={seed}  contact@{contact}  t0(leg2)={t0}  "
              f"point-of-no-return={pnr}"
              + (f"  ({margin} steps AFTER the filter engaged)" if margin is not None else "")
              + f"  -> {verdict}", flush=True)
        print(f"    continuous-time mu said: mu(t0)={ev['mu_t0']:+.4f}, "
              f"max mu={ev['max_mu_leg2']:+.4f}  (never > 0 => 'feasible')",
              flush=True)

    print("\n" + "=" * 78)
    from collections import Counter
    good = [r for r in rows if r.get("faithful")]
    bad = len(rows) - len(good)
    print(f"  trajectory fidelity: {len(good)}/{len(rows)} faithful"
          + (f"   ({bad} DISCARDED)" if bad else ""))
    c = Counter(r["verdict"] for r in good)
    for k, v in c.most_common():
        print(f"  {k:<34}{v}/{len(good)}")
    print("=" * 78)
    json.dump(rows, open(a.out, "w"), indent=2, default=float)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
