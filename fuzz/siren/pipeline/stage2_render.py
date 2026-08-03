"""
Stage 2b -- render a verified attack by REPLAYING its recorded trace.

No physics and no filter run here. The trace stores MuJoCo's qpos per step and
the obstacle positions per step, so each frame is produced by writing qpos back
and calling mj_forward. Two consequences:

  * the video is provably the rollout that was verified, rather than a fresh
    independent rollout that might differ (which mattered: on some robot
    families a resumed run drifts, and contacts here are 20-500 microns)
  * re-rendering at another camera angle costs no simulation at all

Overlay values (clearance, phi, engaged, leg) come from the trace rather than
being recomputed, except for one deliberate cross-check: on the contact step the
replayed clearance is recomputed from geometry and compared with the stored
value. Every silent-staleness bug in this project came from trusting a derived
value that no longer matched its source, so the check is loud by design.

    python -m fuzz.siren.pipeline.stage2_render --dir /abs/verified
"""

import argparse
import glob
import json
import os

import numpy as np

_LEG = {0: "leg 0: home -> G0 (setup)",
        1: "leg 1: G0 -> G1' (inserted goal)",
        2: "leg 2: G1' -> G1 (back to legitimate)"}


def _frame(xyz):
    f = np.eye(4)
    f[:3, 3] = np.asarray(xyz, float).reshape(3)
    return f


def render_one(meta_path, fps=25, width=1280, height=720, stride=2,
               azimuth=55.0, elevation=-14.0, distance=1.15, check=True):
    import mujoco
    import cv2
    from ..world.run import World
    from ..world.types import real_filter

    tag = os.path.basename(meta_path)[:-5]
    d = os.path.dirname(meta_path)
    trace_path = f"{d}/{tag}_trace.npz"
    if not os.path.exists(trace_path):
        return None, "no trace"
    r = json.load(open(meta_path))
    tr = np.load(trace_path)

    spec = real_filter(algo=r["algo"], index=r["index"], d_min=r["d_min"],
                       eta=r["eta"], lam=r["lam"], k=r["k"])
    w = World.build(seed=r["seed"], spec=spec, test_case=r["case"],
                    max_steps=r["max_steps"])
    h = w.harness
    ag = h.env.agent
    sc = w.scene()
    base = sc.base_frame
    h.reset()

    G0 = np.asarray(r["G0"], float)
    G1p = np.asarray(r["G1_prime"], float)
    G1 = np.asarray(r["G1"], float)

    def world_of(xyz):
        return (base @ _frame(xyz))[:3, 3]

    ag.model.vis.global_.offwidth = max(width, ag.model.vis.global_.offwidth)
    ag.model.vis.global_.offheight = max(height, ag.model.vis.global_.offheight)
    renderer = mujoco.Renderer(ag.model, height, width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(ag.model, cam)
    cam.lookat[:] = np.array([world_of(G0), world_of(G1p),
                              world_of(G1)]).mean(0)
    cam.distance, cam.elevation, cam.azimuth = distance, elevation, azimuth

    qpos = tr["qpos"]
    obst = tr["obstacles"]
    clear = tr["clearance"]
    n = min(len(qpos), len(clear))
    hit = int(np.argmax(clear[:n] < 0.0)) if np.any(clear[:n] < 0.0) else None

    mp4 = f"{d}/{tag}_attack.mp4"
    vw = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                         (width, height))
    F = cv2.FONT_HERSHEY_SIMPLEX
    img = None
    for t in range(n):
        if hit is not None and t > hit:
            break
        if t % stride and (hit is None or t != hit):
            continue
        ag.data.qpos[:] = qpos[t]
        mujoco.mj_forward(ag.model, ag.data)
        renderer.update_scene(ag.data, camera=cam)
        s = renderer.scene

        def add(pos, rad, rgba):
            if s.ngeom >= s.maxgeom:
                return
            mujoco.mjv_initGeom(s.geoms[s.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([rad, rad, rad], float),
                                np.asarray(pos, float).reshape(3),
                                np.eye(3).flatten(),
                                np.asarray(rgba, np.float32))
            s.ngeom += 1

        for j, c in enumerate(obst[t]):
            solid = hit is not None and t == hit
            add(c, 0.05, (0.85, .15, .15, .28) if not solid
                else (1.0, 0.0, 0.0, 1.0))
        add(world_of(G0), 0.024, (0.20, 0.45, 1.00, 0.95))
        add(world_of(G1p), 0.030, (1.00, 0.85, 0.10, 0.95))
        add(world_of(G1), 0.030, (0.10, 0.90, 0.10, 0.95))
        img = np.ascontiguousarray(renderer.render())
        if hit is not None and t == hit:
            cv2.rectangle(img, (0, 0), (width-1, height-1), (0, 0, 255), 12)

        def txt(y, s_, col=(255, 255, 255), sc_=0.58, th=2):
            cv2.putText(img, s_, (16, y), F, sc_, (0, 0, 0), th+3, cv2.LINE_AA)
            cv2.putText(img, s_, (16, y), F, sc_, col, th, cv2.LINE_AA)

        wp = int(tr["wp_idx"][t])
        txt(32, f"{r['kind']}  {r['algo']}  {r['case']}  s{r['seed']} "
                f"lam={r['lam']}", sc_=0.62)
        txt(58, f"step {int(tr['step'][t]):4d}   {_LEG.get(min(wp, 2))}")
        txt(84, f"clearance {clear[t]:+.6f}   phi {tr['phi'][t]:+.4f}   "
                f"{'ENGAGED' if tr['engaged'][t] else 'idle'}")
        txt(110, "blue = G0   yellow = G1' inserted   green = G1  "
                 "(replayed from trace)")
        if hit is not None and t == hit:
            txt(148, f"CONTACT at step {int(tr['step'][t])} on leg {wp}",
                (0, 0, 255), 0.9, 3)
        vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if hit is not None and img is not None:
        cv2.imwrite(f"{d}/{tag}_contact.png",
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        for _ in range(fps * 2):
            vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    vw.release()

    note = "ok"
    if check and hit is not None:
        # replay fidelity: recompute clearance from the replayed state and
        # compare with what the live rollout recorded
        from spark_utils import compute_masked_distance_matrix
        si = h.algo.safe_controller.safe_algo.safety_index
        env_mask = np.asarray(si.env_collision_mask, bool)
        ag.data.qpos[:] = qpos[hit]
        mujoco.mj_forward(ag.model, ag.data)
        af = ag.get_feedback()
        h.env.task._update_robot_state(af)
        ti = h.env.task.get_info(af)
        dmat, _ = compute_masked_distance_matrix(
            frame_list_1=h.env.task.robot_frames_world,
            geom_list_1=h.robot_cfg.CollisionVol.values(),
            frame_list_2=ti["obstacle"]["frames_world"],
            geom_list_2=ti["obstacle"]["geom"])
        got = float(np.where(env_mask, np.asarray(dmat, float), np.inf).min())
        err = abs(got - float(clear[hit]))
        note = f"replay clearance {got:+.6f} vs stored {clear[hit]:+.6f} " \
               f"(|d|={err:.2e})"
        if err > 1e-3:
            note = "REPLAY MISMATCH  " + note
    return mp4, note


def render_dir(d, **kw):
    metas = sorted(f for f in glob.glob(f"{d}/*.json")
                   if "INDEX" not in os.path.basename(f))
    print(f"\nrendering {len(metas)} verified attacks", flush=True)
    for m in metas:
        mp4, note = render_one(m, **kw)
        print(f"  {os.path.basename(m)[:-5]}: "
              f"{'skipped (' + note + ')' if mp4 is None else note}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--azimuth", type=float, default=55.0)
    a = p.parse_args(argv)
    render_dir(a.dir, stride=a.stride, azimuth=a.azimuth)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
