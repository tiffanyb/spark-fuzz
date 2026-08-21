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

#: The BASELINE runs [G0, G1] -- the same setup leg, then straight to the real
#: goal with no inserted waypoint. Its leg indices therefore mean something
#: different from the attack's and need their own labels.
_LEG_BASE = {0: "leg 0: home -> G0 (setup)",
             1: "leg 1: G0 -> G1 (legitimate task, no attack)"}


def _frame(xyz):
    f = np.eye(4)
    f[:3, 3] = np.asarray(xyz, float).reshape(3)
    return f


def render_one(meta_path, fps=25, width=1280, height=720, stride=2,
               azimuth=55.0, elevation=-14.0, distance=1.15, check=True,
               out_dir=None, variant="attack"):
    """Replay one recorded rollout to video.

    variant="attack"   [G0, G1', G1] -- the inserted goal, ends in the contact
    variant="baseline" [G0, G1]      -- the same G0 with NO inserted goal, which
                                        stage 2 already proved REACHES safely.
    The pair is the point: same world, same filter, same G0, and the only
    difference is the inserted waypoint. Rendering only the attack shows a robot
    crashing without establishing that it had any business not crashing.
    """
    import mujoco
    import cv2
    from ..world.run import World
    from ..world.types import real_filter

    tag = os.path.basename(meta_path)[:-5]
    d = os.path.dirname(meta_path)
    suffix = "_trace.npz" if variant == "attack" else "_baseline_trace.npz"
    trace_path = f"{d}/{tag}{suffix}"
    # The trace is read from the verified dir; the mp4/png may be written
    # elsewhere, so that a visualisation folder can be handed around without
    # dragging the multi-megabyte traces along with it.
    vis = out_dir or d
    os.makedirs(vis, exist_ok=True)
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

    # WHICH obstacle collides at the contact step, so only that one is
    # highlighted rather than all of them. It is the obstacle column of the
    # argmin over the guarded robot-obstacle distance matrix (the same
    # computation the fidelity check below uses). Posing the robot here is
    # harmless -- the render loop re-poses qpos every frame. obst[t] enumerates
    # the same obstacle order as this matrix's columns, so the index matches j.
    hit_obs = None
    if hit is not None:
        from spark_utils import compute_masked_distance_matrix
        _si = h.algo.safe_controller.safe_algo.safety_index
        _mask = np.asarray(_si.env_collision_mask, bool)
        ag.data.qpos[:] = qpos[hit]
        mujoco.mj_forward(ag.model, ag.data)
        for _dof in h.robot_cfg.DoFs:
            ag.dof_pos_cmd[_dof] = qpos[hit][h.robot_cfg.DoF_to_MujocoDoF[_dof]]
            ag.dof_vel_cmd[_dof] = 0.0
        _af = ag.get_feedback()
        h.env.task._update_robot_state(_af)
        _ti = h.env.task.get_info(_af)
        _of = [np.array(f, float).copy() for f in _ti["obstacle"]["frames_world"]]
        for _f, _c in zip(_of, obst[hit]):
            _f[:3, 3] = _c
        _dmat, _ = compute_masked_distance_matrix(
            frame_list_1=h.env.task.robot_frames_world,
            geom_list_1=h.robot_cfg.CollisionVol.values(),
            frame_list_2=_of, geom_list_2=_ti["obstacle"]["geom"])
        _dm = np.where(_mask, np.asarray(_dmat, float), np.inf) \
            if np.shape(_mask) == np.asarray(_dmat).shape else np.asarray(_dmat, float)
        hit_obs = int(np.unravel_index(np.argmin(_dm), _dm.shape)[1])

    mp4 = f"{vis}/{tag}_{variant}.mp4"
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
            solid = hit is not None and t == hit and j == hit_obs
            add(c, 0.05, (0.85, .15, .15, .28) if not solid
                else (1.0, 0.0, 0.0, 1.0))
        add(world_of(G0), 0.024, (0.20, 0.45, 1.00, 0.95))
        if variant == "attack":
            add(world_of(G1p), 0.030, (1.00, 0.85, 0.10, 0.95))
        add(world_of(G1), 0.030, (0.10, 0.90, 0.10, 0.95))
        img = np.ascontiguousarray(renderer.render())
        if hit is not None and t == hit:
            cv2.rectangle(img, (0, 0), (width-1, height-1), (0, 0, 255), 12)

        def txt(y, s_, col=(255, 255, 255), sc_=0.58, th=2):
            cv2.putText(img, s_, (16, y), F, sc_, (0, 0, 0), th+3, cv2.LINE_AA)
            cv2.putText(img, s_, (16, y), F, sc_, col, th, cv2.LINE_AA)

        wp = int(tr["wp_idx"][t])
        head = (f"{r['kind']}  " if variant == "attack"
                else "BASELINE (no inserted goal)  ")
        txt(32, f"{head}{r['algo']}  {r['case']}  s{r['seed']} "
                f"lam={r['lam']}", sc_=0.62,
            col=(255, 255, 255) if variant == "attack" else (120, 255, 120))
        legs = _LEG if variant == "attack" else _LEG_BASE
        txt(58, f"step {int(tr['step'][t]):4d}   "
                f"{legs.get(min(wp, max(legs)))}")
        txt(84, f"clearance {clear[t]:+.6f}   phi {tr['phi'][t]:+.4f}   "
                f"{'ENGAGED' if tr['engaged'][t] else 'idle'}")
        txt(110, ("blue = G0   yellow = G1' inserted   green = G1  "
                  "(replayed from trace)") if variant == "attack" else
                 ("blue = G0   green = G1   no inserted goal  "
                  "(replayed from trace)"))
        if hit is not None and t == hit:
            txt(148, f"CONTACT at step {int(tr['step'][t])} on leg {wp}",
                (0, 0, 255), 0.9, 3)
        elif variant == "baseline" and t == n - 1:
            txt(148, f"REACHED G1 safely - min clearance "
                     f"{float(np.min(clear[:n])):+.6f}", (120, 255, 120), 0.8, 3)
        vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if hit is not None and img is not None:
        cv2.imwrite(f"{vis}/{tag}_contact.png",
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        for _ in range(fps * 2):
            vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    vw.release()

    note = "ok"
    probe_at = hit if hit is not None else (int(np.argmin(clear[:n]))
                                            if n else None)
    if check and probe_at is not None:
        hit = probe_at
        # replay fidelity: recompute clearance from the replayed state and
        # compare with what the live rollout recorded
        from spark_utils import compute_masked_distance_matrix
        si = h.algo.safe_controller.safe_algo.safety_index
        env_mask = np.asarray(si.env_collision_mask, bool)
        ag.data.qpos[:] = qpos[hit]
        mujoco.mj_forward(ag.model, ag.data)
        # This world runs with use_sim_dynamics = False, and in that mode
        # get_feedback() ends with `dof_pos_fbk = dof_pos_cmd` -- it reads qpos
        # and then throws the result away. Writing qpos alone therefore leaves
        # the task's frames at whatever the reset left behind, which is why the
        # check reported the SAME replay clearance for four different attacks in
        # one world. The command arrays are the actual state here, so set those.
        for dof in h.robot_cfg.DoFs:
            ag.dof_pos_cmd[dof] = qpos[hit][h.robot_cfg.DoF_to_MujocoDoF[dof]]
            ag.dof_vel_cmd[dof] = 0.0
        af = ag.get_feedback()
        h.env.task._update_robot_state(af)
        ti = h.env.task.get_info(af)
        # The world was rebuilt and reset, so its obstacles sit at their step-0
        # poses. On a dynamic-obstacle scene that is nowhere near where they
        # were at the contact step, and comparing the replayed robot against
        # them reports a mismatch of ~2e-2 on every DO attack -- a bug in the
        # CHECK, not in the replay (the video already draws obst[t]). Put the
        # obstacles back where the trace says they were, keeping each frame's
        # rotation and geom from the live world.
        of = [np.array(f, float).copy() for f in ti["obstacle"]["frames_world"]]
        for f, c in zip(of, obst[hit]):
            f[:3, 3] = c
        dmat, _ = compute_masked_distance_matrix(
            frame_list_1=h.env.task.robot_frames_world,
            geom_list_1=h.robot_cfg.CollisionVol.values(),
            frame_list_2=of,
            geom_list_2=ti["obstacle"]["geom"])
        got = float(np.where(env_mask, np.asarray(dmat, float), np.inf).min())
        err = abs(got - float(clear[hit]))
        note = f"replay clearance {got:+.6f} vs stored {clear[hit]:+.6f} " \
               f"(|d|={err:.2e})"
        if err > 1e-3:
            note = "REPLAY MISMATCH  " + note
    return mp4, note


def render_dir(d, variants=("attack",), **kw):
    metas = sorted(f for f in glob.glob(f"{d}/*.json")
                   if "INDEX" not in os.path.basename(f)
                   and not os.path.basename(f).startswith("SUMMARY"))
    print(f"\nrendering {len(metas)} verified attacks "
          f"x {len(variants)} variant(s): {', '.join(variants)}", flush=True)
    for m in metas:
        for v in variants:
            mp4, note = render_one(m, variant=v, **kw)
            print(f"  [{v:<8}] {os.path.basename(m)[:-5]}: "
                  f"{'skipped (' + note + ')' if mp4 is None else note}",
                  flush=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--azimuth", type=float, default=55.0)
    p.add_argument("--out-dir", default=None,
                   help="where the mp4/png go (default: alongside the trace)")
    p.add_argument("--variants", default="attack",
                   help="comma-separated: attack, baseline, or both. The "
                        "baseline needs stage2_baseline to have written its "
                        "trace first.")
    a = p.parse_args(argv)
    render_dir(a.dir, variants=tuple(v.strip() for v in a.variants.split(",")),
               stride=a.stride, azimuth=a.azimuth, out_dir=a.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
