"""
Verify a claimed positive control, then render the full G0 -> G1' -> G1 run.

Producing a COLLISION label from `run([G0, G1p, G1])` is NOT verification. Two
things have to be checked that the label alone does not tell you:

  1. WHERE the contact happens. The schedule has three legs (home->G0, G0->G1',
     G1'->G1). If contact occurs on an earlier leg the scene is not an insertion
     attack at all — the robot simply crashed on the way, and the inserted goal
     is incidental. The attack claim requires wp_idx == 2 at contact.

  2. WHETHER IT REPRODUCES. These contacts are tens of microns deep, the same
     grazing scale as the pre-fix collisions. A knife-edge that fires once is not
     a control.

Both are checked here, plus the three reference schedules re-run side by side so
the claim is visible in one place:

    [G0, G1]        must REACH     — the task is safe without the attacker
    [G0, G1']       must REACH     — reaching the inserted goal is clean
    [G0, G1', G1]   must COLLIDE   — and on the final leg

Then it renders the whole thing: an orbit-free playback of the actual run with
the three waypoints marked and the filter's state on screen.

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 python -m fuzz.siren.scenario.verify_control \
        --control fuzz/siren/scenario/control_G1FixedBase_D1_AG_SO_v0_cbf_s1.json
"""

import argparse
import glob
import json
import os

import numpy as np

_LEG_NAME = {0: "home -> G0", 1: "G0 -> G1' (inserted)", 2: "G1' -> G1 (attack)"}


def run_and_trace(w, schedule, steps):
    rec = w.run([np.asarray(s, float) for s in schedule], max_steps=steps)
    contact = None
    for s in rec.steps:
        if s.clearance < 0.0:
            contact = {"step": int(s.step), "wp_idx": int(s.wp_idx),
                       "clearance": float(s.clearance)}
            break
    return rec, contact


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--control", default=None,
                   help="a control_*.json; default = all of them")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out-dir", default="fuzz/siren/scenario/verified")
    p.add_argument("--render", action="store_true", default=True)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=20)
    a = p.parse_args(argv)

    from ..world.run import World
    from ..world.types import real_filter

    files = ([a.control] if a.control
             else sorted(glob.glob("fuzz/siren/scenario/control_*.json")))
    if not files:
        print("no control files found")
        return 1
    os.makedirs(a.out_dir, exist_ok=True)

    verified = []
    for path in files:
        ctl = json.load(open(path))
        case, algo, seed = ctl["case"], ctl["algo"], ctl["seed"]
        steps = ctl["max_steps"]
        spec = real_filter(algo=algo, index=ctl["index"], d_min=ctl["d_min"],
                           eta=ctl["eta"], lam=ctl["lam"], k=ctl["k"])
        G1p = np.asarray(ctl["G1_prime"], float)
        G1 = np.asarray(ctl["G1"], float)

        for ci, ctrl in enumerate(ctl["controls"]):
            G0 = np.asarray(ctrl["G0"], float)
            tag = f"{case}_{algo}_s{seed}_G0-{ci}"
            print(f"\n{'='*74}\n{tag}")
            print(f"  G0 {np.round(G0,3)}   G1' {np.round(G1p,3)}   "
                  f"G1 {np.round(G1,3)}")

            w = World.build(seed=seed, spec=spec, test_case=case, max_steps=steps)

            # --- the three reference schedules ---------------------------- #
            results = {}
            for name, sched in (("baseline [G0,G1]", [G0, G1]),
                                ("leg1 [G0,G1']", [G0, G1p]),
                                ("attack [G0,G1',G1]", [G0, G1p, G1])):
                rec, contact = run_and_trace(w, sched, steps)
                results[name] = {"label": rec.label, "steps": int(rec.n_steps),
                                 "min_clearance": float(rec.min_clearance),
                                 "contact": contact}
                c = f"  contact at step {contact['step']} on leg "\
                    f"{contact['wp_idx']} ({_LEG_NAME.get(contact['wp_idx'],'?')})" \
                    if contact else ""
                print(f"  {name:<22}{rec.label:<10}steps={rec.n_steps:<5}"
                      f"clear={rec.min_clearance:+.5f}{c}")

            atk = results["attack [G0,G1',G1]"]
            on_final = bool(atk["contact"] and atk["contact"]["wp_idx"] == 2)

            # --- does it reproduce? --------------------------------------- #
            labels = []
            for _ in range(a.repeats):
                rec, _c = run_and_trace(w, [G0, G1p, G1], steps)
                labels.append(rec.label)
            reproduces = all(l == "COLLISION" for l in labels)
            print(f"  reproduces: {labels} -> {'YES' if reproduces else 'NO'}")

            ok = (results["baseline [G0,G1]"]["label"] == "REACHED"
                  and results["leg1 [G0,G1']"]["label"] == "REACHED"
                  and atk["label"] == "COLLISION" and on_final and reproduces)
            print(f"  VERDICT: {'VERIFIED POSITIVE CONTROL' if ok else 'NOT VERIFIED'}"
                  + ("" if on_final or not atk["contact"]
                     else f"  (contact on leg {atk['contact']['wp_idx']}, "
                          f"not the attack leg)"))
            if not ok:
                continue

            out = {**{k: ctl[k] for k in ("case", "algo", "seed", "index",
                                          "max_steps", "d_min", "eta", "lam", "k",
                                          "obstacles_world", "bounds", "keepout")},
                   "G0": G0.tolist(), "G1_prime": G1p.tolist(), "G1": G1.tolist(),
                   "schedules": results, "contact_on_attack_leg": on_final,
                   "repeat_labels": labels}
            jname = f"{a.out_dir}/{tag}.json"
            json.dump(out, open(jname, "w"), indent=2, default=float)
            print(f"  wrote {jname}")
            verified.append((tag, out))

            if a.render:
                _render(w, G0, G1p, G1, steps, f"{a.out_dir}/{tag}",
                        a.width, a.height, a.fps, tag)

    print(f"\n{'='*74}\n{len(verified)} verified positive control(s) in {a.out_dir}")
    return 0 if verified else 1


def _render(w, G0, G1p, G1, steps, prefix, width, height, fps, tag):
    """Play back the actual G0 -> G1' -> G1 run with the waypoints marked."""
    import mujoco
    import cv2
    from ..world.sim import probe

    h = w.harness
    ag = h.env.agent
    sc = w.scene()
    base = sc.base_frame

    def wf(xyz):
        f = np.eye(4)
        f[:3, 3] = np.asarray(xyz, float)
        return (base @ f)[:3, 3]

    ag.model.vis.global_.offwidth = max(width, ag.model.vis.global_.offwidth)
    ag.model.vis.global_.offheight = max(height, ag.model.vis.global_.offheight)
    renderer = mujoco.Renderer(ag.model, height, width, max_geom=4096)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(ag.model, cam)
    cam.lookat[:] = wf(0.5 * (np.asarray(G1p) + np.asarray(G1)))
    cam.distance, cam.elevation, cam.azimuth = 1.3, -16.0, 140.0

    vw = cv2.VideoWriter(prefix + "_attack.mp4",
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    F = cv2.FONT_HERSHEY_SIMPLEX

    probe.reset_giveups(h)
    af, ti = h.reset()
    h.env.task.set_goal_schedule([np.asarray(G0), np.asarray(G1p), np.asarray(G1)])
    u, ai = h.algo.act(af, ti)
    for t in range(steps):
        af, ti = h.env.step(u, ai)
        u, ai = h.algo.act(af, ti)
        clear = h.clearance(ti)
        wp = int(getattr(h.env.task, "wp_idx", 0))
        engaged = bool(ai.get("trigger_safe", False))

        renderer.update_scene(ag.data, camera=cam)
        s = renderer.scene

        def add(pos, r, rgba):
            if s.ngeom >= s.maxgeom:
                return
            mujoco.mjv_initGeom(s.geoms[s.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([r, r, r], float),
                                np.asarray(pos, float).reshape(3),
                                np.eye(3).flatten(), np.asarray(rgba, np.float32))
            s.ngeom += 1

        for of in ti["obstacle"]["frames_world"]:
            add(np.asarray(of)[:3, 3], 0.05, (0.85, 0.15, 0.15, 0.38))
        add(wf(G0), 0.030, (0.20, 0.45, 1.00, 0.95))
        add(wf(G1p), 0.034, (1.00, 0.85, 0.10, 0.95))
        add(wf(G1), 0.034, (0.10, 0.90, 0.10, 0.97))

        img = np.ascontiguousarray(renderer.render())
        if engaged:
            cv2.rectangle(img, (0, 0), (width - 1, height - 1), (255, 40, 40), 8)

        def txt(y, s_, col=(255, 255, 255), sc_=0.6, th=2):
            cv2.putText(img, s_, (16, y), F, sc_, (0, 0, 0), th + 3, cv2.LINE_AA)
            cv2.putText(img, s_, (16, y), F, sc_, col, th, cv2.LINE_AA)

        txt(32, tag, sc_=0.6)
        txt(60, f"step {t:4d}   leg {wp}: {_LEG_NAME.get(wp,'?')}", sc_=0.66)
        txt(88, "blue = G0    yellow = G1' (inserted)    green = G1")
        txt(116, f"clearance {clear:+.5f} m", (120, 230, 140) if clear > 0
            else (60, 60, 255))
        txt(144, "FILTER ENGAGED" if engaged else "filter idle",
            (60, 90, 255) if engaged else (190, 190, 190), 0.58)
        if clear < 0:
            txt(184, "COLLISION — inserted goal defeated the filter",
                (60, 60, 255), 0.8, 3)

        hold = fps * 2 if clear < 0 else 1
        for _ in range(hold):
            vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if clear < 0:
            cv2.imwrite(prefix + "_contact.png",
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            break
        if h.env.task.reached_final:
            break
    vw.release()
    print(f"  wrote {prefix}_attack.mp4 and {prefix}_contact.png")


if __name__ == "__main__":
    raise SystemExit(main())
