"""Render the stock-radius static insertions: attack AND legitimate task.

render_constructed derives the filter from the trial config and always rolls out
[G0, G1', G1]. Neither holds here: each of these attacks carries its own eta or
lambda, its own d_min and phi_k, and every attack has to be shown next to the
[G0, G1] run it is being compared against -- a video of a collision proves
nothing on its own, since the claim is that the SAME scene is safe without the
inserted goal. So the spec is rebuilt from the saved attack and the schedule is
a parameter.

Two videos per attack:

    <tag>_attack.mp4     [G0, G1', G1]   contact on the return leg
    <tag>_baseline.mp4   [G0, G1]        same obstacles, same filter, reaches G1

    python -m fuzz.siren.constructed.render_stock
"""

import argparse
import glob
import json
import os

import numpy as np


def rollout(world, schedule, pos_w, radius, steps):
    """Instrumented rollout with the obstacles pinned, mirroring the search.

    The obstacles are pinned inside a patched reset, exactly as run_pinned does.
    Re-pinning every step and recomputing get_info perturbs the run enough to
    lose the contact, so the rollout drawn here would not be the verified one.
    """
    import numpy as np

    from ..pipeline.stage1_search import set_channel
    from ..world.sim import probe
    from .big_obstacle import set_obstacles

    h = world.harness
    ag = h.env.agent
    probe.reset_giveups(h)
    orig = h.reset

    def patched(*aa, **kk):
        af_, ti_ = orig(*aa, **kk)
        set_obstacles(world, pos_w, radius)
        return af_, h.env.task.get_info(af_)

    h.reset = patched
    af, ti = h.reset()
    set_channel(world, "arm")
    h.env.task.set_goal_schedule([np.asarray(x, float) for x in schedule])
    ctrl, ai = h.algo.act(af, ti)
    qpos, obst, clear, wps, vols = [], [], [], [], []
    vol_r = [float(g.attributes["radius"]) if hasattr(g, "attributes")
             else float(g.radius) for g in h.robot_cfg.CollisionVol.values()]
    for _ in range(steps):
        af, ti = h.env.step(ctrl, ai)
        try:
            ctrl, ai = h.algo.act(af, ti)
        except Exception:
            # SPARK answers a failed whole-body IK with a ZERO command; holding
            # the previous one instead makes this rollout diverge from the
            # verified search rollout.
            ctrl = np.zeros_like(np.asarray(ctrl, float))
        qpos.append(np.asarray(ag.data.qpos, float).copy())
        obst.append(np.array([np.asarray(o)[:3, 3]
                              for o in ti["obstacle"]["frames_world"]]))
        clear.append(float(h.clearance(ti)))
        wps.append(int(getattr(h.env.task, "wp_idx", 0)))
        # SPARK checks collision against SPHERES centred on joint frames, not
        # the visual mesh -- 5 cm for every arm link. The mesh sits well inside
        # its own sphere, so a video drawing only the mesh shows a gap at the
        # moment the filter's own geometry is interpenetrating. Record the
        # spheres so the contact can be drawn as the filter sees it.
        fr = h.env.task.robot_frames_world
        vols.append(np.array([np.asarray(fr[i], float)[:3, 3]
                              for i in h.robot_cfg.CollisionVol]))
        if clear[-1] < 0.0 or h.env.task.reached_final:
            break
    h.reset = orig
    clear = np.array(clear)
    hit = int(np.argmax(clear < 0.0)) if np.any(clear < 0.0) else None
    return qpos, obst, clear, wps, hit, ti, vols, vol_r


def which_pair(h, ti, obst_at_hit):
    """(robot volume, obstacle) actually in contact, so only those highlight."""
    from spark_utils import compute_masked_distance_matrix
    si = h.algo.safe_controller.safe_algo.safety_index
    mask = np.asarray(si.env_collision_mask, bool)
    frames = [np.array(f, float).copy() for f in ti["obstacle"]["frames_world"]]
    for f, c in zip(frames, obst_at_hit):
        f[:3, 3] = c
    dm, _ = compute_masked_distance_matrix(
        frame_list_1=h.env.task.robot_frames_world,
        geom_list_1=h.robot_cfg.CollisionVol.values(),
        frame_list_2=frames, geom_list_2=ti["obstacle"]["geom"])
    dm = np.asarray(dm, float)
    if np.shape(mask) == dm.shape:
        dm = np.where(mask, dm, np.inf)
    i, j = np.unravel_index(np.argmin(dm), dm.shape)
    return int(i), int(j)


def draw(world, v, qpos, obst, clear, wps, hit, hit_obs, hit_vol,
         vols, vol_r, marks, out, tag,
         variant, fps=25, width=1280, height=720, stride=2,
         azimuth=125.0, elevation=-14.0, distance=0.95):
    import cv2
    import mujoco
    ag = world.harness.env.agent
    os.makedirs(out, exist_ok=True)
    ag.model.vis.global_.offwidth = max(width, ag.model.vis.global_.offwidth)
    ag.model.vis.global_.offheight = max(height, ag.model.vis.global_.offheight)
    r = mujoco.Renderer(ag.model, height, width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(ag.model, cam)
    cam.lookat[:] = np.mean(list(marks.values()), axis=0)
    cam.distance, cam.elevation, cam.azimuth = distance, elevation, azimuth
    path = f"{out}/{tag}_{variant}.mp4"
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                         (width, height))
    F = cv2.FONT_HERSHEY_SIMPLEX
    R = float(v["obstacle_radius"])
    dem = (f"eta {v['eta']:.4f}" if v["algo"] in ("ssa", "rssa", "pssa")
           else f"lambda {v['lam']:.2f}")
    img = None
    for t in range(len(qpos)):
        if hit is not None and t > hit:
            break
        if t % stride and (hit is None or t != hit):
            continue
        ag.data.qpos[:] = qpos[t]
        mujoco.mj_forward(ag.model, ag.data)
        r.update_scene(ag.data, camera=cam)
        s = r.scene

        def add(pos, rad, rgba):
            if s.ngeom >= s.maxgeom:
                return
            mujoco.mjv_initGeom(s.geoms[s.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([rad, rad, rad], float),
                                np.asarray(pos, float).reshape(3),
                                np.eye(3).flatten(), np.asarray(rgba, np.float32))
            s.ngeom += 1

        for j, c in enumerate(obst[t]):
            # only the obstacle actually in contact goes solid red
            solid = hit is not None and t == hit and j == hit_obs
            add(c, R, (1.0, 0.0, 0.0, 1.0) if solid else (0.85, .15, .15, .30))
        # the filter's own view of the robot: a sphere per joint. Drawing these
        # is what makes the contact legible -- the visual mesh is far slimmer
        # than its collision sphere, so without them the frame looks like a miss.
        for j, c in enumerate(vols[t]):
            touch = hit is not None and t == hit and j == hit_vol
            add(c, vol_r[j], (1.0, 0.55, 0.0, 0.95) if touch
                else (0.35, 0.75, 1.0, 0.13))
        add(marks["G0"], 0.024, (0.20, 0.45, 1.00, 0.95))
        if variant == "attack":
            add(marks["G1p"], 0.030, (1.00, 0.85, 0.10, 0.95))
        add(marks["G1"], 0.030, (0.10, 0.90, 0.10, 0.95))
        img = np.ascontiguousarray(r.render())
        if hit is not None and t == hit:
            cv2.rectangle(img, (0, 0), (width - 1, height - 1), (0, 0, 255), 12)

        def txt(y, s_, col=(255, 255, 255), sc_=0.58, th=2):
            cv2.putText(img, s_, (16, y), F, sc_, (0, 0, 0), th + 3, cv2.LINE_AA)
            cv2.putText(img, s_, (16, y), F, sc_, col, th, cv2.LINE_AA)

        head = ("ATTACK  home -> G0 -> G1' -> G1" if variant == "attack"
                else "LEGITIMATE TASK  home -> G0 -> G1")
        txt(32, f"{head}    {v['algo']}   {v['case']}  s{v['seed']}", sc_=0.62)
        txt(58, f"obstacle radius {R*100:.0f} cm ({R*200:.0f} cm across)  "
                f"x{v['n_obstacles']}   {dem}  d_min {v['d_min']}  "
                f"k {v['k']}   plant unmodified")
        txt(84, f"step {t}   leg {wps[t]}   clearance {clear[t]:+.6f}")
        legend = ("blue = G0   yellow = G1' inserted   green = G1   "
                  "red = colliding obstacle" if variant == "attack"
                  else "blue = G0   green = G1   same obstacles as the attack")
        txt(110, legend)
        txt(136, "pale blue = robot collision spheres SPARK checks (5 cm/joint);"
                 " orange = the one in contact", sc_=0.50)
        if hit is not None and t == hit:
            txt(176, f"CONTACT on leg {wps[t]}  (INSERTION)   "
                     f"surfaces overlap {abs(clear[t])*1000:.3f} mm",
                (0, 0, 255), 0.85, 3)
        elif variant == "baseline" and t == len(qpos) - 1:
            txt(176, "REACHED G1, no contact", (0, 220, 0), 0.9, 3)
        vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if img is not None:
        suffix = "contact" if hit is not None else "final"
        cv2.imwrite(f"{out}/{tag}_{variant}_{suffix}.png",
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        for _ in range(fps * 2):
            vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    vw.release()
    return path


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="fuzz/siren/constructed/stock_attacks/*.json")
    p.add_argument("--out", default="fuzz/siren/constructed/stock_visualizations")
    p.add_argument("--stride", type=int, default=2)
    a = p.parse_args(argv)

    from ..world.run import World
    from ..world.types import real_filter

    files = sorted(glob.glob(a.src))
    print(f"{len(files)} attacks -> {a.out}\n", flush=True)
    for f in files:
        v = json.load(open(f))
        tag = os.path.basename(f).replace(".json", "")
        spec = real_filter(algo=v["algo"], index=v["index"], d_min=v["d_min"],
                           eta=v["eta"], lam=v["lam"], k=v["k"])
        steps = v["max_steps"]
        pos_w = [np.asarray(q, float) for q in v["obstacles_world"]]
        R = float(v["obstacle_radius"])
        G0 = np.asarray(v["controls"][0]["G0"], float)
        G1 = np.asarray(v["G1"], float)
        G1p = np.asarray(v["G1_prime"], float)

        for variant, sched in (("attack", [G0, G1p, G1]), ("baseline", [G0, G1])):
            w = World.build(seed=v["seed"], spec=spec, test_case=v["case"],
                            max_steps=steps)
            bf = np.asarray(w.scene().base_frame, float)

            def tw(q):
                return (bf @ np.append(np.asarray(q, float), 1.0))[:3]

            (qpos, obst, clear, wps, hit, ti,
             vols, vol_r) = rollout(w, sched, pos_w, R, steps)
            hit_vol, hit_obs = (which_pair(w.harness, ti, obst[hit])
                                if hit is not None else (None, None))
            marks = {"G0": tw(G0), "G1p": tw(G1p), "G1": tw(G1)}
            path = draw(w, v, qpos, obst, clear, wps, hit, hit_obs, hit_vol,
                        vols, vol_r, marks, a.out, tag, variant,
                        stride=a.stride)
            leg = wps[hit] if hit is not None else None
            print(f"  {tag:<12} {variant:<8} {len(qpos):4d} steps  "
                  f"min clearance {clear.min():+.6f}  "
                  f"contact leg {leg}  -> {os.path.basename(path)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
