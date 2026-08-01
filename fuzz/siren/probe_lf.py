"""
Is L_f wrong on D1 too, or only on the second-order (D2) index?

On a D2 scene the drift term L_f disagrees with the plant by ~+0.4 with the
WRONG SIGN, while L_g is exact to <1%. If D1 (velocity control, first-order
distance index) is exact, the defect is specific to the velocity-augmented index
and is a reportable bug rather than a property of safe control. If D1 is wrong
too, the problem is in the shared machinery.

Method, per step where the filter is engaged:

    L_f            what SPARK's safety index reports
    phi_dot(u=0)   measured: hold ZERO control for one physics step and finite-
                   difference phi. With no control input the whole derivative IS
                   the drift, so this is L_f measured directly against the plant.
    L_g . u        what SPARK says the control contributes
    phi_dot(u) - phi_dot(0)   measured contribution of that same control

phi is read by calling the safety index at each state, never through the filter's
own bookkeeping, and the simulator is snapshotted and restored around every
probe so the real trajectory is untouched.

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 python -m fuzz.siren.probe_lf --case G1FixedBase_D1_AG_SO_v0
"""

import argparse
import glob
import json

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="G1FixedBase_D1_AG_SO_v0")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--sweep", default="/tmp/rep_shard*.jsonl")
    p.add_argument("--max-report", type=int, default=25)
    p.add_argument("--out", default="/tmp/lf_probe.json")
    a = p.parse_args(argv)

    import mujoco
    from .world.run import World
    from .world.types import real_filter
    from .world.sim import probe

    # find a candidate on this case that actually ENGAGED the filter -- an
    # unengaged run says nothing about the constraint the QP enforces.
    pick = None
    for f in sorted(glob.glob(a.sweep)):
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("status") != "ok" or r["case"] != a.case:
                continue
            if a.seed is not None and r["seed"] != a.seed:
                continue
            for e in r["evaluations"]:
                if e["stage"] == "evaluated" and e["engaged"] and e["n_engaged_leg2"] > 20:
                    pick = (r["case"], r["seed"], r["max_steps"], e)
                    break
            if pick:
                break
        if pick:
            break
    if pick is None:
        print(f"no engaged candidate found for {a.case}")
        return 1
    case, seed, ms, ev = pick
    cand = np.asarray(ev["candidate"], float)
    print(f"{case} seed={seed}  candidate={np.round(cand,4)}  "
          f"engaged for {ev['n_engaged_leg2']} steps on leg 2\n", flush=True)

    index = "velocity" if "_D2_" in case else "distance"
    spec = real_filter(algo="ssa", index=index, d_min=0.02, eta=0.02, k=0.1)
    w = World.build(seed=seed, spec=spec, test_case=case, max_steps=ms)
    sc = w.scene()
    h = w.harness
    ag = h.env.agent
    si = h.algo.safe_controller.safe_algo.safety_index
    dt = float(ag.dt)

    def phin(af, ti):
        return np.asarray(si.phi(np.asarray(af["state"], float), ti)[0],
                          float).reshape(-1)

    def snap():
        return (ag.data.qpos.copy(), ag.data.qvel.copy(), ag.data.ctrl.copy(),
                float(ag.data.time), ag.dof_pos_cmd.copy(), ag.dof_vel_cmd.copy(),
                ag.dof_acc_cmd.copy(), ag.dof_pos_fbk.copy(), ag.dof_vel_fbk.copy(),
                int(getattr(h.env.task, "wp_idx", 0)), int(ag.control_decimation))

    def rest(s):
        ag.data.qpos[:] = s[0]; ag.data.qvel[:] = s[1]; ag.data.ctrl[:] = s[2]
        ag.data.time = s[3]
        ag.dof_pos_cmd[:] = s[4]; ag.dof_vel_cmd[:] = s[5]; ag.dof_acc_cmd[:] = s[6]
        ag.dof_pos_fbk[:] = s[7]; ag.dof_vel_fbk[:] = s[8]
        h.env.task.wp_idx = s[9]; ag.control_decimation = s[10]
        mujoco.mj_forward(ag.model, ag.data)

    probe.reset_giveups(h)
    af, ti = h.reset()
    h.env.task.set_goal_schedule([cand, np.asarray(sc.G1)])
    u, ai = h.algo.act(af, ti)

    print(f"{'t':>4}{'phi':>9}{'Lf':>11}{'MEAS drift':>12}{'err Lf':>10}"
          f"{'Lg.u':>10}{'MEAS ctrl':>11}{'err Lg':>10}")
    print("-" * 77)
    rows = []
    for t in range(ms):
        raw = probe.read_raw(h)
        ua = np.asarray(u, float).reshape(-1)
        if raw:
            m = raw["phi_mask"] > 0
            if m.any() and np.max(np.where(m, raw["phi"], -np.inf)) >= 0.0:
                k = int(np.argmax(np.where(m, raw["phi"], -np.inf)))
                p0 = phin(af, ti)
                s = snap()
                ag.control_decimation = 1
                af0, ti0 = h.env.step(np.zeros_like(ua), ai)
                fd0 = float((phin(af0, ti0)[k] - p0[k]) / dt)
                rest(s)
                ag.control_decimation = 1
                afu, tiu = h.env.step(u, ai)
                fdu = float((phin(afu, tiu)[k] - p0[k]) / dt)
                rest(s)
                Lf = float(raw["Lf"][k]); Lgu = float(raw["Lg"][k] @ ua)
                rows.append({"t": t, "phi": float(p0[k]), "Lf": Lf,
                             "meas_drift": fd0, "err_Lf": fd0 - Lf,
                             "Lgu": Lgu, "meas_ctrl": fdu - fd0,
                             "err_Lg": (fdu - fd0) - Lgu})
                if len(rows) <= a.max_report:
                    print(f"{t:>4}{p0[k]:>+9.4f}{Lf:>+11.4f}{fd0:>+12.4f}"
                          f"{fd0-Lf:>+10.4f}{Lgu:>+10.4f}{fdu-fd0:>+11.4f}"
                          f"{(fdu-fd0)-Lgu:>+10.4f}")
        af, ti = h.env.step(u, ai)
        u, ai = h.algo.act(af, ti)
        if h.env.task.reached_final:
            break

    if not rows:
        print("filter never engaged on this run — nothing to compare")
        return 1
    eLf = np.array([r["err_Lf"] for r in rows])
    eLg = np.array([r["err_Lg"] for r in rows])
    Lf = np.array([r["Lf"] for r in rows])
    md = np.array([r["meas_drift"] for r in rows])
    print("\n" + "=" * 77)
    print(f"engaged steps compared: {len(rows)}")
    print(f"  L_f   median analytic {np.median(Lf):+.4f}   measured {np.median(md):+.4f}"
          f"   median error {np.median(eLf):+.4f}")
    print(f"  L_g   median error {np.median(eLg):+.5f}")
    sign_flip = float(np.mean(np.sign(Lf) != np.sign(md)))
    print(f"  L_f has the WRONG SIGN on {100*sign_flip:.0f}% of engaged steps")
    rel = np.median(np.abs(eLf)) / max(1e-9, np.median(np.abs(md)))
    print(f"  |error| / |true drift| = {rel:.2f}")
    print()
    if np.median(np.abs(eLf)) < 0.02:
        print("VERDICT: L_f agrees with the plant on this case.")
    else:
        print("VERDICT: L_f disagrees with the plant on this case too.")
    json.dump({"case": case, "seed": seed, "rows": rows}, open(a.out, "w"),
              indent=2, default=float)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
