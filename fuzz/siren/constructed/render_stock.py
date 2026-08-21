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

RE-SIMULATION
-------------
The saved attack fixes the scene (seed, obstacle positions, G0/G1'/G1) AND the
filter it was found against. Everything about the filter can be overridden on
the command line to re-run the SAME scene under different settings, which is how
you ask whether an attack survives a retune rather than whether it exists:

    # does it still land at SPARK's shipped demand?
    ... --src .../pssa_0_1.json --eta 0.5

    # widen the keep-out shell and restore the shipped lookahead
    ... --src .../pssa_0_1.json --d-min 0.05 --k 1.0

    # same scene, a different filter entirely
    ... --src .../pssa_0_1.json --algo cbf --lam 10.0

    # 5x faster control loop (the outer-loop rate the attack depends on)
    ... --src .../pssa_0_1.json --decimation 1

Overrides are appended to the output filename, so a sweep does not overwrite the
recorded run. --no-video writes the still and the numbers only, which is what a
sweep usually wants -- the mp4 is the expensive part.

PRESENTATION
------------
--no-text drops every overlay (captions, legend, engagement banner and the
frame borders) and renders the geometry alone, for use as a figure.

--orbit additionally holds the decisive frame -- the contact frame, or the last
frame if the run never made contact -- and walks the camera once around it:

    ... --src .../ssa_0_1.json --variants attack --no-text --orbit

A single still cannot settle a sub-millimetre interpenetration, because whether
two spheres overlap is not decidable from one viewpoint. The orbit is what makes
the geometry checkable.

--obstacles solid draws the obstacle opaque on every frame instead of only on
the colliding one. The translucent default reads better in a trajectory video,
but a translucent sphere does not show you where its surface is, so pair
--obstacles solid with --orbit whenever the question is geometric.
"""

import argparse
import gc
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
    # Observe how hard the filter deflects the command. Pure observation --
    # the wrapper forwards the call untouched, so the rollout is still the
    # verified one.
    algo = h.algo.safe_controller.safe_algo
    _qp = algo.qp_solver
    _seen = {}

    def qp_watch(u_ref, Q_u, Lg, Lf, eps_=1.00e-2, abs_=1.00e-2):
        u_sol, viol = _qp(u_ref, Q_u, Lg, Lf, eps_, abs_)
        _seen["du"] = float(np.linalg.norm(
            np.asarray(u_sol, float).reshape(-1)
            - np.asarray(u_ref, float).reshape(-1)))
        return u_sol, viol
    algo.qp_solver = qp_watch

    af, ti = h.reset()
    set_channel(world, "arm")
    h.env.task.set_goal_schedule([np.asarray(x, float) for x in schedule])
    ctrl, ai = h.algo.act(af, ti)
    qpos, obst, clear, wps, vols = [], [], [], [], []
    phis, nact, dus = [], [], []
    vol_r = [float(g.attributes["radius"]) if hasattr(g, "attributes")
             else float(g.radius) for g in h.robot_cfg.CollisionVol.values()]
    for _ in range(steps):
        _seen.pop("du", None)
        af, ti = h.env.step(ctrl, ai)
        try:
            ctrl, ai = h.algo.act(af, ti)
        except Exception:
            # SPARK answers a failed whole-body IK with a ZERO command; holding
            # the previous one instead makes this rollout diverge from the
            # verified search rollout.
            ctrl = np.zeros_like(np.asarray(ctrl, float))
        # SPARK's own trigger test (basic_safe_set_algorithm.py:83): the filter
        # acts iff some masked constraint has phi > 0. d_min and phi_k decide
        # WHERE that happens; the demand (eta/lambda) decides how hard it pushes
        # once it does.
        raw = probe.read_raw(h)
        if raw:
            m = np.asarray(raw["phi_mask"], float).reshape(-1) > 0
            ph = np.asarray(raw["phi"], float).reshape(-1)
            phis.append(float(ph[m].max()) if m.any() else float("nan"))
            nact.append(int((m & (ph >= 0)).sum()))
        else:
            phis.append(float("nan"))
            nact.append(0)
        dus.append(_seen.get("du", 0.0))
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
    algo.qp_solver = _qp
    clear = np.array(clear)
    hit = int(np.argmax(clear < 0.0)) if np.any(clear < 0.0) else None
    return (qpos, obst, clear, wps, hit, ti, vols, vol_r,
            np.array(phis), np.array(nact), np.array(dus))


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


def paint(scene, obst_t, vols_t, vol_r, R, marks, variant,
          hit_obs=None, hit_vol=None, mark_contact=False,
          obstacle_style="contact"):
    """Add the spheres SPARK actually reasons about to a rendered scene.

    Shared by the trajectory renderer and the orbit renderer so the two cannot
    drift apart. The robot's visual mesh is far slimmer than its collision
    sphere, so without these a contact frame looks like a clear miss.

    obstacle_style decides how the obstacle is drawn:

      "contact"  translucent until the frame where it is the colliding pair,
                 then opaque. Reads well in a trajectory video -- the moment of
                 contact announces itself -- but it means the obstacle's own
                 surface is only truthfully placed on one frame, so it is the
                 wrong choice when the geometry itself is the subject.
      "solid"    opaque throughout. The obstacle's surface is where it actually
                 is on every frame, which is what an orbit is for; the
                 colliding one stays a brighter red so it is still identifiable.
    """
    import mujoco

    def add(pos, rad, rgba):
        if scene.ngeom >= scene.maxgeom:
            return
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom],
                            mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([rad, rad, rad], float),
                            np.asarray(pos, float).reshape(3),
                            np.eye(3).flatten(), np.asarray(rgba, np.float32))
        scene.ngeom += 1

    always = (obstacle_style == "solid")
    for j, c in enumerate(obst_t):
        hot = (mark_contact and j == hit_obs)
        if hot:
            rgba = (1.0, 0.0, 0.0, 1.0)
        elif always:
            rgba = (0.85, 0.15, 0.15, 1.0)
        else:
            rgba = (0.85, 0.15, 0.15, 0.30)
        add(c, R, rgba)
    for j, c in enumerate(vols_t):
        touch = mark_contact and j == hit_vol
        add(c, vol_r[j], (1.0, 0.55, 0.0, 0.95) if touch
            else (0.35, 0.75, 1.0, 0.13))
    add(marks["G0"], 0.024, (0.20, 0.45, 1.00, 0.95))
    if variant == "attack":
        add(marks["G1p"], 0.030, (1.00, 0.85, 0.10, 0.95))
    add(marks["G1"], 0.030, (0.10, 0.90, 0.10, 0.95))


def orbit(world, v, qpos, obst, vols, vol_r, marks, out, tag, variant,
          frame, hit_obs, hit_vol, mark_contact, fps=30, width=1280,
          height=720, frames=180, elevation=-14.0, distance=0.95,
          azimuth0=125.0, obstacle_style="contact"):
    """Hold one pose and walk the camera once around it.

    A trajectory video answers "what happened"; an orbit answers "what is the
    geometry", which is the question a still of a sub-millimetre contact cannot
    settle from a single viewpoint.
    """
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
    cam.distance, cam.elevation = distance, elevation

    ag.data.qpos[:] = qpos[frame]
    mujoco.mj_forward(ag.model, ag.data)

    path = f"{out}/{tag}_{variant}_orbit.mp4"
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                         (width, height))
    R = float(v["obstacle_radius"])
    for i in range(frames):
        cam.azimuth = azimuth0 + 360.0 * i / frames
        r.update_scene(ag.data, camera=cam)
        paint(r.scene, obst[frame], vols[frame], vol_r, R, marks, variant,
              hit_obs, hit_vol, mark_contact,
              obstacle_style=obstacle_style)
        img = np.ascontiguousarray(r.render())
        vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    vw.release()
    return path


def draw(world, v, qpos, obst, clear, wps, hit, hit_obs, hit_vol,
         vols, vol_r, marks, out, tag,
         variant, fps=25, width=1280, height=720, stride=2,
         azimuth=125.0, elevation=-14.0, distance=0.95, video=True,
         phis=None, nact=None, dus=None, text=True,
         obstacle_style="contact"):
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
    # A parameter sweep wants the numbers and one still, not 24 mp4s; encoding
    # is the expensive half of a render, so --no-video skips the writer
    # entirely rather than writing a file and deleting it.
    vw = (cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                          (width, height)) if video else None)
    F = cv2.FONT_HERSHEY_SIMPLEX
    R = float(v["obstacle_radius"])
    dem = (f"eta {v['eta']:.4f}" if v["algo"] in ("ssa", "rssa", "pssa")
           else f"lambda {v['lam']:.2f}")
    img = None
    keep_last = (hit if hit is not None else len(qpos) - 1)
    for t in range(len(qpos)):
        if hit is not None and t > hit:
            break
        if not video:
            # stills-only: render exactly the decisive frame. This has to skip
            # the stride filter -- keep_last is odd half the time, and with
            # stride=2 an odd index was silently dropped, leaving no still at all.
            if t != keep_last:
                continue
        elif t % stride and (hit is None or t != hit):
            continue
        ag.data.qpos[:] = qpos[t]
        mujoco.mj_forward(ag.model, ag.data)
        r.update_scene(ag.data, camera=cam)
        paint(r.scene, obst[t], vols[t], vol_r, R, marks, variant,
              hit_obs, hit_vol, mark_contact=(hit is not None and t == hit),
              obstacle_style=obstacle_style)
        img = np.ascontiguousarray(r.render())
        eng = bool(nact is not None and t < len(nact) and nact[t] > 0)
        if not text:
            if vw is not None:
                vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            continue
        if eng:
            cv2.rectangle(img, (0, 0), (width - 1, height - 1), (0, 165, 255), 8)
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
        # The filter's own on/off state. phi > 0 is the trigger test; |du| is
        # how much the command was actually changed, which is the part a large
        # demand blows up. Both are needed: phi says WHETHER, |du| says HOW HARD.
        if phis is not None and t < len(phis):
            ph = phis[t]
            du = dus[t] if dus is not None and t < len(dus) else 0.0
            if eng:
                txt(206, f"FILTER ENGAGED  phi {ph:+.5f} > 0   {int(nact[t])} "
                         f"constraint(s)   deflection |u_safe-u_ref| = {du:.1f}",
                    (0, 200, 255), 0.66, 2)
            else:
                txt(206, f"filter idle  phi {ph:+.5f} <= 0   (command passes "
                         f"through unchanged)", (190, 190, 190), 0.60, 2)
        if hit is not None and t == hit:
            txt(176, f"CONTACT on leg {wps[t]}  (INSERTION)   "
                     f"surfaces overlap {abs(clear[t])*1000:.3f} mm",
                (0, 0, 255), 0.85, 3)
        elif variant == "baseline" and t == len(qpos) - 1:
            txt(176, "REACHED G1, no contact", (0, 220, 0), 0.9, 3)
        if vw is not None:
            vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if img is not None:
        suffix = "contact" if hit is not None else "final"
        cv2.imwrite(f"{out}/{tag}_{variant}_{suffix}.png",
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if vw is not None:
            for _ in range(fps * 2):
                vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if vw is None:
        return f"{out}/{tag}_{variant}_{'contact' if hit is not None else 'final'}.png"
    vw.release()
    return path


#: CLI flag -> key in the saved attack record. Everything here describes the
#: FILTER; the scene (seed, obstacles, G0/G1'/G1) is deliberately not overridable
#: -- re-running a different scene would not be the same attack any more.
_OVERRIDABLE = {"algo": "algo", "d_min": "d_min", "eta": "eta",
                "lam": "lam", "k": "k", "steps": "max_steps"}

#: short names for the filename suffix, so a sweep is self-identifying on disk
_SHORT = {"algo": "", "d_min": "dmin", "eta": "eta", "lam": "lam",
          "k": "k", "steps": "steps", "decimation": "dec"}


def apply_overrides(v, args):
    """Return (record with the CLI overrides written in, {field: (old, new)}).

    The record drives BOTH the rollout and the on-screen caption, so overriding
    it here is what keeps the video honest about what was actually simulated.
    """
    v = dict(v)
    changed = {}
    for flag, key in _OVERRIDABLE.items():
        val = getattr(args, flag, None)
        if val is None:
            continue
        if v.get(key) != val:
            changed[flag] = (v.get(key), val)
        v[key] = val
    if args.decimation is not None:
        changed["decimation"] = (None, args.decimation)
    return v, changed


def suffix_for(changed):
    """`__eta0.5_dmin0.05` -- empty when the run is unmodified, so the recorded
    render keeps its original filename and a sweep never overwrites it."""
    if not changed:
        return ""
    parts = []
    for flag, (_, new) in changed.items():
        name = _SHORT[flag]
        parts.append(f"{name}{new:g}" if isinstance(new, (int, float))
                     else str(new))
    return "__" + "_".join(parts)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="fuzz/siren/constructed/stock_attacks/*.json")
    p.add_argument("--out", default="fuzz/siren/constructed/stock_visualizations")
    p.add_argument("--stride", type=int, default=2)
    g = p.add_argument_group(
        "re-simulation", "override the filter the attack was found against and "
                         "re-run the SAME scene; output names carry the change")
    g.add_argument("--algo", help="ssa rssa pssa cbf rcbf sss rsss pfm sma")
    g.add_argument("--d-min", dest="d_min", type=float,
                   help="keep-out shell; with phi_k it sets WHERE the filter engages")
    g.add_argument("--eta", type=float,
                   help="constant demand, m/s (ssa/rssa/pssa). SPARK ships 0.5")
    g.add_argument("--lam", type=float,
                   help="proportional demand (cbf/rcbf/sss/rsss). SPARK ships 10.0")
    g.add_argument("--k", type=float,
                   help="phi_k, the lookahead time in seconds AND the whole "
                        "control gain of the second-order index. SPARK ships 1.0")
    g.add_argument("--steps", type=int, help="override max_steps")
    g.add_argument("--decimation", type=int,
                   help="agent control_decimation: the outer-loop rate. Lower = "
                        "the filter acts more often. max_steps is rescaled to "
                        "hold the simulated horizon fixed unless --steps is given")
    g.add_argument("--variants", default="attack,baseline",
                   help="which rollouts to render (default both)")
    g.add_argument("--no-video", action="store_true",
                   help="stills and numbers only; skips mp4 encoding")
    g.add_argument("--suffix", help="override the auto-generated filename suffix")
    o = p.add_argument_group("presentation")
    o.add_argument("--no-text", action="store_true",
                   help="draw no captions, legends or frame borders — the "
                        "rendered geometry only, for use as a figure")
    o.add_argument("--orbit", action="store_true",
                   help="also walk the camera once around the decisive frame "
                        "(the contact frame, or the last frame if none) and "
                        "write <tag>_<variant>_orbit.mp4")
    o.add_argument("--orbit-frames", type=int, default=180)
    o.add_argument("--orbit-fps", type=int, default=30)
    o.add_argument("--orbit-distance", type=float, default=0.95)
    o.add_argument("--orbit-elevation", type=float, default=-14.0)
    o.add_argument("--obstacles", choices=("contact", "solid"),
                   default="contact",
                   help="obstacle opacity: 'contact' (default) draws it "
                        "translucent and turns it opaque only on the colliding "
                        "frame; 'solid' draws it opaque throughout, which is "
                        "what you want when the geometry is the subject")
    o.add_argument("--orbit-mark-contact", action="store_true",
                   help="keep the contacting pair highlighted red/orange "
                        "through the orbit (off by default: every frame of an "
                        "orbit IS the contact frame, so the highlight stops "
                        "distinguishing anything)")
    a = p.parse_args(argv)

    from ..world.run import World
    from ..world.types import real_filter

    files = sorted(glob.glob(a.src))
    want = [x.strip() for x in a.variants.split(",") if x.strip()]
    print(f"{len(files)} attacks -> {a.out}\n", flush=True)
    for f in files:
        rec = json.load(open(f))
        v, changed = apply_overrides(rec, a)
        base = os.path.basename(f).replace(".json", "")
        tag = base + (a.suffix if a.suffix is not None else suffix_for(changed))
        if changed:
            print("  " + base + "  RE-SIMULATED  " + "  ".join(
                f"{fl}: {old_!r} -> {new_!r}"
                for fl, (old_, new_) in changed.items()), flush=True)
        spec = real_filter(algo=v["algo"], index=v["index"], d_min=v["d_min"],
                           eta=v["eta"], lam=v["lam"], k=v["k"])
        steps = v["max_steps"]
        pos_w = [np.asarray(q, float) for q in v["obstacles_world"]]
        R = float(v["obstacle_radius"])
        G0 = np.asarray(v["controls"][0]["G0"], float)
        G1 = np.asarray(v["G1"], float)
        G1p = np.asarray(v["G1_prime"], float)

        for variant, sched in (("attack", [G0, G1p, G1]), ("baseline", [G0, G1])):
            if variant not in want:
                continue
            w = World.build(seed=v["seed"], spec=spec, test_case=v["case"],
                            max_steps=steps)
            run_steps = steps
            if a.decimation is not None:
                ag = w.harness.env.agent
                # control_decimation is how many physics ticks one command is
                # held for, so halving it halves the time each step covers.
                # Rescale the budget or the rollout silently ends early and a
                # "no contact" would just mean "ran out of steps".
                if a.steps is None:
                    run_steps = int(round(steps * ag.control_decimation
                                          / a.decimation))
                ag.control_decimation = a.decimation
            bf = np.asarray(w.scene().base_frame, float)

            def tw(q):
                return (bf @ np.append(np.asarray(q, float), 1.0))[:3]

            (qpos, obst, clear, wps, hit, ti, vols, vol_r,
             phis, nact, dus) = rollout(w, sched, pos_w, R, run_steps)
            hit_vol, hit_obs = (which_pair(w.harness, ti, obst[hit])
                                if hit is not None else (None, None))
            marks = {"G0": tw(G0), "G1p": tw(G1p), "G1": tw(G1)}
            path = draw(w, v, qpos, obst, clear, wps, hit, hit_obs, hit_vol,
                        vols, vol_r, marks, a.out, tag, variant,
                        stride=a.stride, video=not a.no_video,
                        phis=phis, nact=nact, dus=dus, text=not a.no_text,
                        obstacle_style=a.obstacles)
            if a.orbit:
                fr = hit if hit is not None else len(qpos) - 1
                opath = orbit(w, v, qpos, obst, vols, vol_r, marks, a.out, tag,
                              variant, fr, hit_obs, hit_vol,
                              mark_contact=a.orbit_mark_contact,
                              fps=a.orbit_fps, frames=a.orbit_frames,
                              distance=a.orbit_distance,
                              elevation=a.orbit_elevation,
                              obstacle_style=a.obstacles)
                print(f"  {tag:<12} {variant:<8} orbit {a.orbit_frames} frames "
                      f"around step {fr}  -> {os.path.basename(opath)}",
                      flush=True)
            leg = wps[hit] if hit is not None else None
            # The recorded outcome is what a re-simulation is being compared
            # against, so print the verdict rather than making the reader
            # diff two clearance numbers by eye.
            ref = rec.get("leg2_min" if variant == "attack" else "baseline_min")
            if changed and ref is not None:
                was = "CONTACT" if ref < 0 else "clear"
                now = "CONTACT" if clear.min() < 0 else "clear"
                verdict = (f"  [{was} -> {now}"
                           + ("" if was == now else "  CHANGED") + "]")
            else:
                verdict = ""
            e = nact > 0
            trans = int((e[1:] != e[:-1]).sum()) if len(e) > 1 else 0
            print(f"  {tag:<12} {variant:<8} {len(qpos):4d} steps  "
                  f"min clearance {clear.min():+.6f}  "
                  f"contact leg {leg}{verdict}  | filter engaged {int(e.sum())}"
                  f"/{len(e)} steps, {trans} on-off transitions, max deflection "
                  f"{float(np.max(dus)) if len(dus) else 0:.1f}"
                  f"  -> {os.path.basename(path)}", flush=True)
            # One World holds ~1 GB of MuJoCo/SPARK state and nothing drops it
            # until the process exits; a sweep builds two per attack, so
            # without this an 8-point sweep is 16 GB.
            w = None
            gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
