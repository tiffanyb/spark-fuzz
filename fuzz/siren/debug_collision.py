"""
A safe control that still collides: where exactly does the guarantee break?

SSA's promise is a chain, and the collision means one link is broken. This walks
one colliding run step by step and checks every link separately, so the answer is
a measurement and not a story:

  L1  is the colliding pair MONITORED?          phi_mask[i] for the pair that hits
  L2  was it ACTIVE when it mattered?           phi_i >= 0 turns the constraint on
  L3  did the returned control SATISFY it?      r_i = Lf_i + Lg_i.u_safe + eta <= 0
  L4  was the applied control the SOLVED one?   u_applied vs u_safe
  L5  did the state move as the model said?     predicted vs realised change in phi

L3 is the one to watch. The QP is set up with

    prob.setup(..., eps_abs=1e-2, eps_rel=1e-2)

and the demand is eta = 0.02, so OSQP may return status 'solved' while still
violating a constraint by up to half the entire safety demand. If r_i > 0 on the
colliding pair while the solver reported success, the filter never actually
enforced the constraint it believes it enforced, and every downstream conclusion
about feasibility is describing a QP that was not solved to the accuracy its own
guarantee needs.

L5 is the other candidate: even a perfectly satisfied constraint only bounds
phi_dot at the current state. Over a finite timestep the realised phi can exceed
the linear prediction, and if the plant does not track the commanded control
exactly, the guarantee is being enforced on a model the robot is not following.

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    python -m fuzz.siren.debug_collision --seed 1 --tail 12
"""

import argparse
import json

import numpy as np

from .world.run import World
from .world.types import real_filter


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--case", default="G1FixedBase_D2_AG_SO_v0")
    p.add_argument("--eta", type=float, default=0.02)
    p.add_argument("--d-min", type=float, default=0.02)
    p.add_argument("--from-json", default="/tmp/kinds_d2s1.json")
    p.add_argument("--which", type=int, default=0, help="index of collision to dissect")
    p.add_argument("--tail", type=int, default=12, help="steps before contact to print")
    p.add_argument("--max-steps", type=int, default=300)
    a = p.parse_args(argv)

    cand = np.asarray(json.load(open(a.from_json))["rows"][a.which]["candidate"], float)

    index = "velocity" if "_D2_" in a.case else "distance"
    spec = real_filter(algo="ssa", index=index, d_min=a.d_min, eta=a.eta, k=0.1)
    world = World.build(seed=a.seed, spec=spec, test_case=a.case,
                        max_steps=a.max_steps)
    scene = world.scene()
    h = world.harness
    from .world.sim import probe
    from spark_utils import compute_masked_distance_matrix

    algo = h.algo.safe_controller.safe_algo
    si = algo.safety_index
    n_env = si.num_constraint_env
    n_obs = si.num_obstacle_vol
    env_mask = np.asarray(si.env_collision_mask, bool)

    probe.reset_giveups(h)
    agent_feedback, task_info = h.reset()
    h.env.task.set_goal_schedule([cand, np.asarray(scene.G1)])
    u_safe, action_info = h.algo.act(agent_feedback, task_info)

    log = []
    for t in range(a.max_steps):
        # raw ingredients for the control that is ABOUT to be applied
        raw = probe.read_raw(h)
        u_applied = np.asarray(u_safe, float).reshape(-1)

        agent_feedback, task_info = h.env.step(u_safe, action_info)

        dmat, _ = compute_masked_distance_matrix(
            frame_list_1=h.env.task.robot_frames_world,
            geom_list_1=h.robot_cfg.CollisionVol.values(),
            frame_list_2=task_info["obstacle"]["frames_world"],
            geom_list_2=task_info["obstacle"]["geom"])
        dmat = np.asarray(dmat, float)
        guarded = np.where(env_mask, dmat, np.inf)
        flat = int(np.argmin(guarded))
        vi, oi = divmod(flat, n_obs)
        clearance = float(guarded[vi, oi])

        row = {"t": t, "clearance": clearance, "vol": vi, "obs": oi,
               "trigger_safe": bool(action_info.get("trigger_safe", False))}

        if raw:
            phi = raw["phi"]
            Lg = raw["Lg"]
            Lf = raw["Lf"]
            mask = raw["phi_mask"] > 0
            k = vi * n_obs + oi                     # env constraint index of that pair
            active = mask & (phi >= 0.0)
            # the constraint as the QP actually posed it: Lf + Lg.u + eta <= 0
            resid = Lf + Lg @ u_applied + a.eta
            row.update({
                "phi_pair": float(phi[k]),          # L1/L2 for the colliding pair
                "masked_pair": bool(mask[k]),
                "active_pair": bool(active[k]),
                "n_active": int(active.sum()),
                "resid_pair": float(resid[k]),      # L3
                "max_resid_active": (float(resid[active].max())
                                     if active.any() else None),
                "n_violated_active": int((resid[active] > 1e-9).sum())
                                      if active.any() else 0,
            })
        log.append(row)

        u_ref_prev = action_info.get("u_ref", None)
        u_safe, action_info = h.algo.act(agent_feedback, task_info)
        if u_ref_prev is not None:
            log[-1]["dev"] = float(np.linalg.norm(
                u_applied - np.asarray(u_ref_prev, float).reshape(-1)))

        if h.env.task.reached_final or clearance < 0.0:
            break

    hit = log[-1]
    print(f"\n{'='*100}\nCOLLISION at step {hit['t']}: clearance={hit['clearance']:.5f} "
          f"on robot volume {hit['vol']} vs obstacle {hit['obs']}\n{'='*100}")
    print(f"L1  is that pair MONITORED?  masked={hit.get('masked_pair')}")
    print(f"L2  was it ACTIVE at contact?  phi={hit.get('phi_pair'):+.5f} "
          f"-> active={hit.get('active_pair')}")
    print(f"L3  constraint residual on that pair at contact: "
          f"{hit.get('resid_pair'):+.6f}   (must be <= 0)")
    print(f"    worst residual over ALL active constraints: "
          f"{hit.get('max_resid_active')}   violated: {hit.get('n_violated_active')}")

    print(f"\nlast {a.tail} steps before contact:")
    hdr = (f"{'t':>4} {'clear':>9} {'pair':>8} {'phi_pair':>10} {'act':>4} "
           f"{'nact':>5} {'resid_pair':>11} {'worst_resid':>12} {'nviol':>6} {'trig':>5}")
    print(hdr)
    for r in log[-a.tail:]:
        wr = r.get("max_resid_active")
        print(f"{r['t']:>4} {r['clearance']:>9.5f} {str(r['vol'])+'/'+str(r['obs']):>8} "
              f"{r.get('phi_pair', float('nan')):>+10.5f} "
              f"{str(r.get('active_pair', '-')):>4} {r.get('n_active', 0):>5} "
              f"{r.get('resid_pair', float('nan')):>+11.6f} "
              f"{(f'{wr:+.6f}' if wr is not None else '-'):>12} "
              f"{r.get('n_violated_active', 0):>6} {str(r['trigger_safe']):>5}")

    # how often did a 'solved' QP return a control that violated its constraints?
    viol = [r for r in log if r.get("n_violated_active", 0) > 0]
    print(f"\nsteps where the returned control VIOLATED an active constraint: "
          f"{len(viol)}/{len(log)}")
    if viol:
        worst = max(viol, key=lambda r: r["max_resid_active"])
        print(f"   worst violation {worst['max_resid_active']:+.6f} at step "
              f"{worst['t']}  (eta = {a.eta}, OSQP eps_abs = 1e-2)")
    json.dump(log, open("/tmp/debug_collision.json", "w"), indent=2, default=float)
    print("\nwrote /tmp/debug_collision.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
