"""
Why does the filter's model think phi is falling while phi rises?

The QP enforces  L_f phi + L_g phi . u <= -eta.  On the case rendered in
render_case.py the left-hand side reads -0.09 to -0.44 (comfortably satisfied by
the NOMINAL control, so the filter never intervenes) while phi actually climbs at
up to +0.22/s. Two explanations, with different fixes:

  (A) STALE LINEARISATION.  L_f/L_g are exact at the instant they are computed,
      but the control is held for control_decimation * dt = 10 ms, during which
      the state moves and the derivative goes out of date. Fix: raise the control
      rate. Prediction: the mismatch SHRINKS as the measurement interval shrinks,
      and vanishes in the limit.

  (B) WRONG DERIVATIVE.  L_f itself does not equal dphi/dt at the state where it
      is evaluated. Fix: correct the safety index. Prediction: the mismatch
      SURVIVES at arbitrarily small intervals.

These are distinguishable by measurement, not by argument: compare the analytic
prediction against a finite difference of phi taken over ONE physics step (2 ms)
and over a full control interval (10 ms). A stale linearisation must improve by
roughly the ratio of intervals; a wrong derivative will not improve at all.

phi is read by calling the safety index directly at each state, so the comparison
never routes through the filter's own bookkeeping.

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 python -m fuzz.siren.probe_phidot
"""

import argparse
import json

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--from-json", default="/tmp/picked_kind2.json")
    p.add_argument("--first", type=int, default=88, help="first step to report")
    p.add_argument("--last", type=int, default=114)
    a = p.parse_args(argv)

    from .world.run import World
    from .world.types import real_filter
    from .world.sim import probe

    case = json.load(open(a.from_json))
    cand = np.asarray(case["candidate"], float)
    spec = real_filter(algo="ssa", index="velocity", d_min=0.02, eta=0.02, k=0.1)
    w = World.build(seed=case["seed"], spec=spec, test_case=case["case"],
                    max_steps=300)
    sc = w.scene()
    h = w.harness
    ag = h.env.agent
    si = h.algo.safe_controller.safe_algo.safety_index
    dt = float(ag.dt)
    dec = int(ag.control_decimation)

    def phi_now(af, ti):
        """phi straight from the safety index at the CURRENT state."""
        x = np.asarray(af["state"], float)
        out = si.phi(x, ti)
        return np.asarray(out[0], float).reshape(-1)

    def snap():
        return {"qpos": ag.data.qpos.copy(), "qvel": ag.data.qvel.copy(),
                "ctrl": ag.data.ctrl.copy(), "time": float(ag.data.time),
                "pc": None if ag.dof_pos_cmd is None else ag.dof_pos_cmd.copy(),
                "vc": None if ag.dof_vel_cmd is None else ag.dof_vel_cmd.copy(),
                "ac": None if ag.dof_acc_cmd is None else ag.dof_acc_cmd.copy(),
                "pf": None if ag.dof_pos_fbk is None else ag.dof_pos_fbk.copy(),
                "vf": None if ag.dof_vel_fbk is None else ag.dof_vel_fbk.copy(),
                "wp": int(getattr(h.env.task, "wp_idx", 0)),
                "dec": int(ag.control_decimation)}

    def restore(s):
        import mujoco
        ag.data.qpos[:] = s["qpos"]; ag.data.qvel[:] = s["qvel"]
        ag.data.ctrl[:] = s["ctrl"]; ag.data.time = s["time"]
        for k, at in (("pc", "dof_pos_cmd"), ("vc", "dof_vel_cmd"),
                      ("ac", "dof_acc_cmd"), ("pf", "dof_pos_fbk"),
                      ("vf", "dof_vel_fbk")):
            if s[k] is not None:
                getattr(ag, at)[:] = s[k]
        h.env.task.wp_idx = s["wp"]
        ag.control_decimation = s["dec"]
        mujoco.mj_forward(ag.model, ag.data)

    probe.reset_giveups(h)
    af, ti = h.reset()
    h.env.task.set_goal_schedule([cand, np.asarray(sc.G1)])
    u, ai = h.algo.act(af, ti)

    print(f"physics dt = {dt*1000:.1f} ms, control interval = {dt*dec*1000:.1f} ms "
          f"(decimation {dec})\n")
    print(f"{'t':>4} {'phi':>9} | {'ANALYTIC':>10} | {'FD 2ms':>9} {'FD 10ms':>9} | "
          f"{'err@2ms':>9} {'err@10ms':>9} | verdict")
    print("-" * 96)

    rows = []
    for t in range(300):
        raw = probe.read_raw(h)
        u_applied = np.asarray(u, float).reshape(-1)

        if raw and a.first <= t <= a.last:
            phi0 = phi_now(af, ti)
            m = raw["phi_mask"] > 0
            k = int(np.argmax(np.where(m, raw["phi"], -np.inf)))   # binding pair
            analytic = float(raw["Lf"][k] + raw["Lg"][k] @ u_applied)

            s = snap()
            # --- one PHYSICS step, control held ------------------------------ #
            ag.control_decimation = 1
            af1, ti1 = h.env.step(u, ai)
            fd_short = float((phi_now(af1, ti1)[k] - phi0[k]) / dt)
            restore(s)
            # --- one full CONTROL interval, same control held ---------------- #
            af2, ti2 = h.env.step(u, ai)
            fd_long = float((phi_now(af2, ti2)[k] - phi0[k]) / (dt * dec))
            restore(s)

            e_s, e_l = fd_short - analytic, fd_long - analytic
            shrink = abs(e_s) < 0.5 * abs(e_l)
            rows.append({"t": t, "phi": float(phi0[k]), "analytic": analytic,
                         "fd_short": fd_short, "fd_long": fd_long,
                         "err_short": e_s, "err_long": e_l})
            print(f"{t:>4} {phi0[k]:>+9.4f} | {analytic:>+10.4f} | {fd_short:>+9.4f} "
                  f"{fd_long:>+9.4f} | {e_s:>+9.4f} {e_l:>+9.4f} | "
                  f"{'shrinks' if shrink else 'PERSISTS'}")

        af, ti = h.env.step(u, ai)
        u, ai = h.algo.act(af, ti)
        if h.env.task.reached_final:
            break
        if t > a.last:
            break

    if rows:
        es = np.array([r["err_short"] for r in rows])
        el = np.array([r["err_long"] for r in rows])
        print("\n" + "=" * 96)
        print(f"median |error| at 2 ms  : {np.median(np.abs(es)):.4f}")
        print(f"median |error| at 10 ms : {np.median(np.abs(el)):.4f}")
        ratio = np.median(np.abs(es)) / max(1e-12, np.median(np.abs(el)))
        print(f"ratio                   : {ratio:.2f}   "
              f"(a purely STALE linearisation would give ~{1/dec:.2f})")
        print()
        if ratio > 0.5:
            print("VERDICT: the error does NOT shrink with the interval.")
            print("  => (B) the analytic derivative itself disagrees with the plant.")
            print("     Raising the control rate cannot fix this; the index must be.")
        else:
            print("VERDICT: the error shrinks roughly with the interval.")
            print("  => (A) stale linearisation; a faster control rate is the fix.")
        json.dump(rows, open("/tmp/phidot_probe.json", "w"), indent=2, default=float)
        print("\nwrote /tmp/phidot_probe.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
