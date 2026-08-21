"""
Stage 4b -- render every location the search tried, coloured by iteration.

One MuJoCo scene per target: the robot posed at the G0 handover, the obstacles,
G0 / G1 / the planted G1', and EVERY evaluated location -- misses included, not
just the hits. Orbit mp4 plus a still.

    tried locations   rainbow by ask/tell ROUND, violet (first) -> red (last)
    attacks           WHITE and larger, deliberately outside the rainbow: if
                      attacks were red they would be indistinguishable from
                      late-round misses
    inadmissible      small grey, if --show-inadmissible

Colour is by round rather than by evaluation index because a run of budget 60 /
batch 10 has 6 rounds -- a readable number of hues, where 60 is not -- and the
round is the unit at which CEM updates its mean/std and BO refits. Stage 3b
records the round explicitly rather than deriving it as eval // batch, which is
wrong whenever the final batch is short.

    python -m fuzz.siren.pipeline.stage4_render \
        --fuzz /abs/fuzz_<target>.json --targets-dir /abs/targets \
        --out-dir /abs/analysis
"""

import argparse
import glob
import json
import os

import numpy as np


def _frame(xyz):
    f = np.eye(4)
    f[:3, 3] = np.asarray(xyz, float).reshape(3)
    return f


def rainbow(frac):
    """violet -> blue -> green -> yellow -> red, as RGB in 0..1."""
    import colorsys
    # hue 0.75 (violet) down to 0.0 (red)
    h = 0.75 * (1.0 - float(np.clip(frac, 0.0, 1.0)))
    r, g, b = colorsys.hsv_to_rgb(h, 0.95, 0.95)
    return (r, g, b, 0.95)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--fuzz", required=True)
    p.add_argument("--targets-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--picker", default=None,
                   help="restrict to one strategy; default draws all")
    p.add_argument("--show-inadmissible", action="store_true", default=False)
    p.add_argument("--frames", type=int, default=180)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--elevation", type=float, default=-18.0)
    p.add_argument("--azimuth", type=float, default=135.0)
    p.add_argument("--distance", type=float, default=1.15)
    p.add_argument("--grid", type=float, default=0.0,
                   help="draw a measuring lattice over the goal box at this "
                        "spacing in metres, e.g. 0.05. The box is the actual "
                        "sampling bounds, so the lattice gives a physical "
                        "scale for how far apart the searched points are.")
    a = p.parse_args(argv)

    import mujoco
    import cv2
    from ..scenario.targets import load_target

    os.makedirs(a.out_dir, exist_ok=True)
    res = json.load(open(a.fuzz))
    tpath = f"{a.targets_dir}/{res['target']}.json"
    w, tgt = load_target(tpath)
    sc = w.scene()
    h = w.harness
    ag = h.env.agent
    base = sc.base_frame

    G0 = np.asarray(tgt["G0_commanded"], float)
    G1 = np.asarray(tgt["G1"], float)
    # Open-search targets (fuzz/siren/test) have no planted goal.
    _t = tgt.get("G1_prime_truth")
    truth = None if _t is None else np.asarray(_t, float)

    pts = []
    max_round = 0
    for row in res["results"]:
        if a.picker and row["picker"] != a.picker:
            continue
        for e in row["evaluations"]:
            if e.get("stage") == "inadmissible" and not a.show_inadmissible:
                continue
            rnd = int(e.get("round", 0))
            max_round = max(max_round, rnd)
            pts.append((np.asarray(e["cand"], float), rnd,
                        bool(e.get("is_attack")),
                        e.get("stage") == "inadmissible"))
    n_atk = sum(1 for _c, _r, atk, _i in pts if atk)
    print(f"{res['target']}: {len(pts)} searched locations, {n_atk} attacks, "
          f"{max_round+1} rounds", flush=True)

    # pose the robot at the handover, as the searched goals are relative to it
    h.reset()
    # Benchmark targets carry the captured handover state; open-search targets
    # set G0 to the home end-effector pose, so the reset pose IS the handover
    # and there is nothing to restore.
    if tgt.get("state_at_G0") is not None:
        from ..scenario.targets import restore_state
        restore_state(h, tgt["state_at_G0"])
    h.env.task._update_robot_state(ag.get_feedback())
    mujoco.mj_forward(ag.model, ag.data)

    ag.model.vis.global_.offwidth = max(a.width, ag.model.vis.global_.offwidth)
    ag.model.vis.global_.offheight = max(a.height, ag.model.vis.global_.offheight)
    renderer = mujoco.Renderer(ag.model, a.height, a.width,
                               max_geom=len(pts) + 4096)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(ag.model, cam)
    obs_w = np.asarray(sc.obstacles_world)

    def world_of(xyz):
        return (base @ _frame(xyz))[:3, 3]

    aim = [world_of(G0), world_of(G1)] + (
        [world_of(truth)] if truth is not None else [])
    cam.lookat[:] = np.array(aim).mean(0)
    cam.distance, cam.elevation = a.distance, a.elevation

    tag = res["target"]
    png = f"{a.out_dir}/searched_{tag}.png"
    mp4 = f"{a.out_dir}/searched_{tag}_orbit.mp4"
    vw = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*"mp4v"), a.fps,
                         (a.width, a.height))
    F = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(a.frames):
        cam.azimuth = a.azimuth + 360.0 * i / a.frames
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

        def line(p0, p1, width, rgba):
            if s.ngeom >= s.maxgeom:
                return
            g = s.geoms[s.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                np.zeros(3), np.zeros(3), np.eye(3).flatten(),
                                np.asarray(rgba, np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
                                 np.asarray(p0, float), np.asarray(p1, float))
            s.ngeom += 1

        if a.grid > 0:
            blo = np.array([b[0] for b in sc.bounds], float)
            bhi = np.array([b[1] for b in sc.bounds], float)
            # box edges, bright; then a lattice on the z-min face at `grid`
            # spacing so in-plane distances can be read off directly
            crn = [(x, y, z) for x in (blo[0], bhi[0]) for y in (blo[1], bhi[1])
                   for z in (blo[2], bhi[2])]
            # NB: not `i` -- that is the frame index of the enclosing loop,
            # and shadowing it silently skipped the `if i == 0` still capture.
            for ci, c0 in enumerate(crn):
                for c1 in crn[ci+1:]:
                    if sum(abs(np.array(c0) - np.array(c1)) > 1e-9) == 1:
                        line(world_of(c0), world_of(c1), 0.0016,
                             (0.15, 0.85, 0.95, 0.85))
            nx = int(round((bhi[0] - blo[0]) / a.grid))
            ny = int(round((bhi[1] - blo[1]) / a.grid))
            for gi in range(nx + 1):
                x = blo[0] + gi * a.grid
                line(world_of((x, blo[1], blo[2])), world_of((x, bhi[1], blo[2])),
                     0.0008, (0.15, 0.85, 0.95, 0.40))
            for gj in range(ny + 1):
                y = blo[1] + gj * a.grid
                line(world_of((blo[0], y, blo[2])), world_of((bhi[0], y, blo[2])),
                     0.0008, (0.15, 0.85, 0.95, 0.40))
            # vertical posts at the corners, ticked every `grid`
            nz = int(round((bhi[2] - blo[2]) / a.grid))
            for (x, y) in ((blo[0], blo[1]), (bhi[0], blo[1]),
                           (blo[0], bhi[1]), (bhi[0], bhi[1])):
                for k in range(nz + 1):
                    z = blo[2] + k * a.grid
                    add(world_of((x, y, z)), 0.0045,
                        (0.15, 0.85, 0.95, 0.75))

        for of in obs_w:
            add(np.asarray(of)[:3, 3], 0.05, (0.85, 0.15, 0.15, 0.25))
            # the KEEP-OUT shell is a goal-sampling rule, not the obstacle's
            # physical size: no candidate may be sampled within sc.keepout of an
            # obstacle centre. Drawing both shows how much of the box is
            # actually reachable by the search.
            if a.grid > 0 and sc.keepout > 0.05:
                add(np.asarray(of)[:3, 3], float(sc.keepout),
                    (0.95, 0.55, 0.10, 0.10))
        for c, rnd, atk, inadm in pts:
            if atk:
                continue
            col = ((0.5, 0.5, 0.5, 0.35) if inadm
                   else rainbow(rnd / max(1, max_round)))
            add(world_of(c), 0.007, col)
        for c, rnd, atk, _inadm in pts:      # attacks last, so they draw on top
            if atk:
                add(world_of(c), 0.013, (1.0, 1.0, 1.0, 1.0))
        add(world_of(G0), 0.026, (0.20, 0.45, 1.00, 0.95))
        add(world_of(G1), 0.030, (0.10, 0.90, 0.10, 0.97))
        if truth is not None:
            add(world_of(truth), 0.020, (1.00, 0.85, 0.10, 0.98))

        img = np.ascontiguousarray(renderer.render())

        def txt(y, t_, col=(255, 255, 255), sc_=0.55, th=2):
            cv2.putText(img, t_, (16, y), F, sc_, (0, 0, 0), th+3, cv2.LINE_AA)
            cv2.putText(img, t_, (16, y), F, sc_, col, th, cv2.LINE_AA)

        txt(32, f"{res['case']}  {res['algo']}  s{res['seed']}  "
                f"kind={res.get('kind')}", sc_=0.62)
        txt(58, f"{len(pts)} searched locations over {max_round+1} rounds, "
                f"{n_atk} attacks")
        txt(84, "rainbow = search round (violet=first -> red=last)")
        txt(108, "WHITE = attack    blue = G0   green = G1"
                 + ("   yellow = planted G1'" if truth is not None
                    else "   (no planted G1' - open search)"))
        if a.grid > 0:
            _lo = np.array([b[0] for b in sc.bounds], float)
            _hi = np.array([b[1] for b in sc.bounds], float)
            _d = _hi - _lo
            _vol = float(np.prod(_d))
            txt(132, f"CYAN LATTICE = goal box, {a.grid*100:.0f} cm spacing "
                     f"| box {_d[0]:.2f} x {_d[1]:.2f} x {_d[2]:.2f} m "
                     f"= {_vol*1000:.0f} L",
                (255, 235, 60), 0.55)
            txt(156, f"orange shell = {sc.keepout*100:.0f} cm goal keep-out "
                     f"| red = obstacle", (255, 190, 60), 0.55)
            txt(180, f"{len(pts)} samples in {_vol*1000:.0f} L "
                     f"-> {1e6*_vol/max(1,len(pts)):.0f} cm^3 each, "
                     f"typical spacing {100*(_vol/max(1,len(pts)))**(1/3):.1f} cm",
                (255, 235, 60), 0.55)

        # a rainbow legend strip, so the round order is readable
        x0, y0 = 16, (204 if a.grid > 0 else 128)
        for k in range(max_round + 1):
            c = rainbow(k / max(1, max_round))
            bgr = (int(255*c[2]), int(255*c[1]), int(255*c[0]))
            cv2.rectangle(img, (x0 + k*26, y0), (x0 + k*26 + 22, y0 + 14),
                          bgr, -1)
            cv2.putText(img, str(k), (x0 + k*26 + 6, y0 + 30), F, 0.4,
                        (255, 255, 255), 1, cv2.LINE_AA)

        if i == 0:
            cv2.imwrite(png, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    vw.release()
    print(f"wrote {png}\nwrote {mp4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
