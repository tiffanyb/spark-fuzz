"""SSA at eta=20: record control authority and the body part it belongs to, per step.

Control authority on constraint i is

    c_i = sum_k u_lim,k * |L_g phi_i,k|,     L_g phi_i = phi_k * normal_i @ J_i @ g(x)

u_lim and phi_k are constant for the whole run, so c changes only through
`normal_i @ J_i` -- i.e. WHICH robot collision volume is nearest the obstacle
(each has its own Jacobian) and how the escape direction lines up with the joints
that can move it. A contact at the hand can be driven by every joint in the arm;
a contact at the elbow can only be driven by the joints upstream of it, because
the wrist joints move the hand without moving the elbow.

The QP is infeasible exactly when c_i - eta - L_f phi_i < 0 on the binding row,
so this file is the record of why eta=20 goes infeasible partway through.

EVERY step of the run is recorded, not only the steps where the filter engaged.
On an idle step (phi < 0 everywhere) there is no binding constraint, so the row
reports the pair CLOSEST to engaging -- the largest phi among the masked rows --
and sets engaged=0. Authority is a property of the pose, not of whether the
filter happened to be active, and omitting the idle steps left 589 holes in a
687-step run.

Env constraints are flattened (robot_vol, obstacle_vol) row-major, then the
self-collision upper triangle, so volume = j // n_obstacle for j < n_env.

Data lives in data/: the attack is read from data/pssa_0_1.json and the trace is
written to data/eta20_authority.csv by default.

Usage (see README.md -- DYLD_INSERT_LIBRARIES is required):
    python -m fuzz.siren.experiment.rq1.rq1_results.casestudy.sweep_eta20_authority
"""

import argparse
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# All data files live in casestudy/data/.
DATA = os.path.join(HERE, "data")
DEFAULT_ATTACK = os.path.join(DATA, "pssa_0_1.json")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--attack", default=DEFAULT_ATTACK)
    p.add_argument("--eta", type=float, default=20.0)
    p.add_argument("--algo", default="ssa")
    p.add_argument("--out", default=os.path.join(DATA, "eta20_authority.csv"))
    a = p.parse_args(argv)

    from fuzz.siren.world.run import World
    from fuzz.siren.world.types import real_filter
    from fuzz.siren.world.sim import probe
    from fuzz.siren.world import derived
    from fuzz.siren.pipeline.stage1_search import set_channel
    from fuzz.siren.world.big_obstacle import set_obstacles

    V = json.load(open(a.attack))
    pos_w = [np.asarray(q, float) for q in V["obstacles_world"]]
    R = float(V["obstacle_radius"])
    steps = int(V["max_steps"])
    G0 = np.asarray(V["controls"][0]["G0"], float)
    G1 = np.asarray(V["G1"], float)
    G1p = np.asarray(V["G1_prime"], float)

    spec = real_filter(algo=a.algo, index=V["index"], d_min=V["d_min"],
                       eta=a.eta, lam=V["lam"], k=V["k"])
    w = World.build(seed=V["seed"], spec=spec, test_case=V["case"],
                    max_steps=steps)
    h = w.harness
    si = h.algo.safe_controller.safe_algo.safety_index
    VOLS = [str(v).split(".")[-1] for v in h.robot_cfg.CollisionVol.keys()]
    DOFS = [str(d).split(".")[-1] for d in h.robot_cfg.DoFs]
    NV = len(VOLS)
    NSELF = NV * (NV - 1) // 2
    n_env = None
    n_obs = None

    orig = h.reset

    def patched(*aa, **kk):
        af_, _ = orig(*aa, **kk)
        set_obstacles(w, pos_w, R)
        return af_, h.env.task.get_info(af_)

    h.reset = patched
    af, ti = h.reset()
    probe.reset_giveups(h)          # count the episode only, not reset()'s warm-up
    set_channel(w, "arm")
    h.env.task.set_goal_schedule([G0, G1p, G1])
    ctrl, ai = h.algo.act(af, ti)

    rows, prev = [], 0
    for t in range(steps):
        af, ti = h.env.step(ctrl, ai)
        try:
            ctrl, ai = h.algo.act(af, ti)
        except Exception:
            ctrl = np.zeros_like(np.asarray(ctrl, float))
        now = probe.giveups(h)
        gave, prev = int(now - prev > 0), now
        clr = float(h.clearance(ti))
        raw = probe.read_raw(h)
        if raw:
            phi, Lg, Lf = raw["phi"], raw["Lg"], raw["Lf"]
            if n_env is None:
                n_env = len(phi) - NSELF
                n_obs = n_env // NV
            act = derived.active_set(phi, raw["phi_mask"], "constant")
            c = derived.control_authority(Lg, raw["u_lim"])
            engaged = bool(act.any())
            if engaged:
                # the constraint the filter is actually fighting
                m = derived.margin(
                    c, derived.demand_vector("constant", phi, eta=a.eta),
                    Lf, act)
                j = int(m["binding"])
                n_active, margin_g = int(m["n_active"]), float(m["g_min"])
            else:
                # Filter idle (phi < 0 everywhere), but authority is still
                # defined -- report it for the pair CLOSEST to engaging, i.e.
                # the largest phi among the masked rows. Without these rows the
                # trace has 589 holes and the wrist->elbow handover looks like a
                # jump rather than the continuous swing it is.
                mask = np.asarray(raw["phi_mask"], float).reshape(-1) > 0
                j = int(np.argmax(np.where(mask, phi, -np.inf)))
                n_active = 0
                margin_g = float(c[j] - a.eta - Lf[j])
            share = np.abs(Lg[j]) * np.abs(raw["u_lim"])
            tot = share.sum()
            share = share / tot if tot else share
            top = int(np.argmax(share))
            rows.append({
                "step": t,
                "leg": int(getattr(h.env.task, "wp_idx", 0)),
                "clearance": round(clr, 9),
                "c_authority": round(float(c[j]), 6),
                "eta": a.eta,
                "phi": round(float(phi[j]), 9),
                "Lf": round(float(Lf[j]), 6),
                "margin_g": round(margin_g, 6),
                "engaged": int(engaged),
                "infeasible": gave,
                "body_part": (VOLS[j // n_obs] if j < n_env
                              else "SELF_COLLISION"),
                "constraint_index": int(j),
                "n_active": n_active,
                "n_joints_1pct": int((share > 0.01).sum()),
                "top_joint": (DOFS[top] if top < len(DOFS) else f"u{top}"),
                "top_joint_share": round(float(share[top]), 4),
            })
        if clr < 0.0 or h.env.task.reached_final:
            break
    h.reset = orig

    cols = ["step", "leg", "clearance", "c_authority", "eta", "phi", "Lf",
            "margin_g", "engaged", "infeasible", "body_part",
            "constraint_index", "n_active", "n_joints_1pct", "top_joint",
            "top_joint_share"]
    with open(a.out, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=cols)
        wtr.writeheader()
        wtr.writerows(rows)

    ne = sum(r["engaged"] for r in rows)
    print(f"{a.algo} eta={a.eta}: {len(rows)} steps -> {a.out}")
    print(f"  filter engaged on {ne}, idle on {len(rows) - ne}")
    print(f"  infeasible on {sum(r['infeasible'] for r in rows)}")
    seen = {}
    for r in rows:
        if r["engaged"]:
            seen.setdefault(r["body_part"], []).append(r["c_authority"])
    for part, cs in sorted(seen.items(), key=lambda kv: -len(kv[1])):
        cs = np.array(cs)
        print(f"  {part:<28} {len(cs):>4} steps   c mean {cs.mean():6.2f}"
              f"  min {cs.min():6.2f}  max {cs.max():6.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
