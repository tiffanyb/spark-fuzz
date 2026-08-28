"""Render every RE-VALIDATED attack in rq3/dataset to rq3/visualization.

Only attacks that survived rq3/revalidate are rendered, so a video here is
always of a claim that currently holds -- not of a record that used to.

The rollout is instrumented directly off MuJoCo (ag.data.qpos) rather than
replayed from RunRecord.q: robot_state is family-dependent (a MobileBase run
carries a base pose that a FixedBase one does not), and qpos is the one thing
that is correct for every family in the corpus.

Each attack yields
    <name>.mp4   home -> G0 -> G1' -> G1, stopping at the contact frame
    <name>.png   the contact frame itself

    python -m fuzz.siren.rq3.render --shard 0 --of 8
"""

import argparse
import gc
import glob
import json
import os

import numpy as np

DST = "fuzz/siren/rq3/visualization"


def rollout(world, sched, pin, radius, steps, channel):
    """Instrumented run; obstacles pinned when the record specifies them."""
    import mujoco
    from ..pipeline.stage1_search import set_channel
    from ..world.sim import probe

    h = world.harness
    ag = h.env.agent
    probe.reset_giveups(h)
    orig = h.reset
    if pin is not None:
        from ..constructed.big_obstacle import set_obstacles

        def patched(*aa, **kk):
            af_, ti_ = orig(*aa, **kk)
            set_obstacles(world, pin, radius)
            return af_, h.env.task.get_info(af_)
        h.reset = patched

    af, ti = h.reset()
    set_channel(world, channel)
    h.env.task.set_goal_schedule([np.asarray(x, float) for x in sched])
    ctrl, ai = h.algo.act(af, ti)
    qpos, obst, clear, wps, vols = [], [], [], [], []
    vol_r = [float(g.attributes["radius"]) if hasattr(g, "attributes")
             else float(g.radius) for g in h.robot_cfg.CollisionVol.values()]
    for _ in range(steps):
        af, ti = h.env.step(ctrl, ai)
        try:
            ctrl, ai = h.algo.act(af, ti)
        except Exception:
            ctrl = np.zeros_like(np.asarray(ctrl, float))
        qpos.append(np.asarray(ag.data.qpos, float).copy())
        obst.append(np.array([np.asarray(o)[:3, 3]
                              for o in ti["obstacle"]["frames_world"]]))
        clear.append(float(h.clearance(ti)))
        wps.append(int(getattr(h.env.task, "wp_idx", 0)))
        fr = h.env.task.robot_frames_world
        vols.append(np.array([np.asarray(fr[i], float)[:3, 3]
                              for i in h.robot_cfg.CollisionVol]))
        if clear[-1] < 0.0 or h.env.task.reached_final:
            break
    h.reset = orig
    clear = np.array(clear)
    hit = int(np.argmax(clear < 0.0)) if np.any(clear < 0.0) else None
    return qpos, obst, clear, wps, hit, vols, vol_r


def draw(world, meta, qpos, obst, clear, wps, hit, vols, vol_r, marks,
         path, radius, stride=3, fps=25, w=1120, h_=630):
    import cv2
    import mujoco
    ag = world.harness.env.agent
    ag.model.vis.global_.offwidth = max(w, ag.model.vis.global_.offwidth)
    ag.model.vis.global_.offheight = max(h_, ag.model.vis.global_.offheight)
    r = mujoco.Renderer(ag.model, h_, w)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(ag.model, cam)
    cam.lookat[:] = np.mean(list(marks.values()), axis=0)
    cam.distance, cam.elevation, cam.azimuth = 1.15, -14.0, 125.0
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h_))
    F = cv2.FONT_HERSHEY_SIMPLEX
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

        def add(p, rad, rgba):
            if s.ngeom >= s.maxgeom:
                return
            mujoco.mjv_initGeom(s.geoms[s.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([rad, rad, rad], float),
                                np.asarray(p, float).reshape(3),
                                np.eye(3).flatten(), np.asarray(rgba, np.float32))
            s.ngeom += 1

        for c in obst[t]:
            add(c, radius, (0.85, .15, .15, .45))
        for j, c in enumerate(vols[t]):
            add(c, vol_r[j], (0.35, 0.75, 1.0, 0.12))
        add(marks["G0"], 0.022, (0.20, 0.45, 1.00, 0.95))
        add(marks["G1p"], 0.028, (1.00, 0.85, 0.10, 0.95))
        add(marks["G1"], 0.028, (0.10, 0.90, 0.10, 0.95))
        img = np.ascontiguousarray(r.render())
        if hit is not None and t == hit:
            cv2.rectangle(img, (0, 0), (w - 1, h_ - 1), (0, 0, 255), 10)

        def txt(y, s_, col=(255, 255, 255), sc=0.52, th=2):
            cv2.putText(img, s_, (14, y), F, sc, (0, 0, 0), th + 3, cv2.LINE_AA)
            cv2.putText(img, s_, (14, y), F, sc, col, th, cv2.LINE_AA)

        txt(26, f"{meta['kind']}  {meta['algo']}  {meta['case']}  seed "
                f"{meta['seed']}  ({meta['channel']})", sc=0.56)
        txt(50, "home -> G0 -> G1' -> G1   blue G0   yellow G1' inserted   "
                "green G1   red obstacle")
        txt(74, f"step {t}  leg {wps[t]}  clearance {clear[t]:+.6f}")
        if hit is not None and t == hit:
            txt(104, f"CONTACT on leg {wps[t]}  ->  {meta['kind']}",
                (0, 0, 255), 0.72, 3)
        vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if img is not None:
        cv2.imwrite(path.replace(".mp4", ".png"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        for _ in range(fps * 2):
            vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    vw.release()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--of", type=int, default=1)
    p.add_argument("--limit", type=int)
    p.add_argument("--stride", type=int, default=3)
    a = p.parse_args(argv)

    from .revalidate import norm_record
    from ..world.run import World
    from ..world.types import real_filter

    idx = json.load(open("fuzz/siren/rq3/dataset/INDEX.json"))
    names = sorted(idx)
    names = [n for i, n in enumerate(names) if i % a.of == a.shard]
    if a.limit:
        names = names[:a.limit]
    os.makedirs(DST, exist_ok=True)
    print(f"shard {a.shard}/{a.of}: {len(names)} attacks", flush=True)

    for i, name in enumerate(names):
        stem = name.replace(".json", "")
        out = f"{DST}/{stem}.mp4"
        if os.path.exists(out):
            continue
        e = idx[name]
        try:
            r = norm_record(json.load(open(e["source"])), e["source"])
            spec = real_filter(algo=r["algo"], index=r["index"],
                               d_min=r["d_min"], eta=r["eta"], lam=r["lam"],
                               k=r["k"])
            w = World.build(seed=r["seed"], spec=spec, test_case=r["case"],
                            max_steps=r["max_steps"])
            bf = np.asarray(w.scene().base_frame, float)

            def tw(q):
                return (bf @ np.append(np.asarray(q, float), 1.0))[:3]

            pin = ([np.asarray(q, float) for q in r["obstacles_world"]]
                   if r["obstacles_world"] else None)
            rad = float(r["obstacle_radius"] or 0.05)
            G0, G1, G1p = (np.asarray(r[k], float) for k in ("G0", "G1", "G1p"))
            qpos, obst, clear, wps, hit, vols, vol_r = rollout(
                w, [G0, G1p, G1], pin, rad, r["max_steps"], r["channel"])
            meta = {"kind": e["kind"], "algo": r["algo"], "case": r["case"],
                    "seed": r["seed"], "channel": r["channel"]}
            draw(w, meta, qpos, obst, clear, wps, hit, vols, vol_r,
                 {"G0": tw(G0), "G1p": tw(G1p), "G1": tw(G1)}, out, rad,
                 stride=a.stride)
            w = None
            gc.collect()
            print(f"  [{i+1}/{len(names)}] {stem}  {len(qpos)} steps  "
                  f"hit={hit} leg={wps[hit] if hit is not None else None}",
                  flush=True)
        except Exception as ex:
            print(f"  [{i+1}/{len(names)}] {stem}  FAILED {type(ex).__name__}: "
                  f"{str(ex)[:90]}", flush=True)
            gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
