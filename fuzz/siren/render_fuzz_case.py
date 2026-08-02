"""
Render one fuzz target: the robot at the handover, and every goal SIREN found.

Companion to render_scene3d.py, but driven by a target + a run_fuzzer result
rather than a pipeline sweep. Two differences that matter:

  * the robot is posed at `state_at_G0`, not at its home pose. This scene is
    about what happens when an inserted goal arrives mid-motion, so the arm is
    drawn where it actually is at the handover — the pose the attack acts on.
  * discovered goals are coloured by the self-check verdict (insertion vs
    modification) when a self-check is supplied, so what is drawn is the
    verified classification rather than the raw search output.

Only ATTACKING candidates appear. run_fuzzer records `hits`, not misses, so the
goals that were tried and came back clean are not recoverable from its output —
the cloud shown is the attack set, not the search trace.

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 python -m fuzz.siren.render_fuzz_case \
        --fuzz-result fuzz/siren/experiment/case/fuzz_<name>.json
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np

_KIND_RGBA = {"INSERTION":    (1.00, 0.16, 0.16, 0.95),
              "MODIFICATION": (0.85, 0.20, 0.95, 0.95),
              "REJECT":       (0.55, 0.55, 0.55, 0.70),
              None:           (1.00, 0.35, 0.10, 0.90)}


def _frame(xyz):
    f = np.eye(4)
    f[:3, 3] = np.asarray(xyz, float).reshape(3)
    return f


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--fuzz-result", required=True)
    p.add_argument("--targets-dir", default="fuzz/siren/scenario/targets")
    p.add_argument("--selfcheck", default="/tmp/selfcheck_*.json",
                   help="glob of self_check shards; used to colour by verdict")
    p.add_argument("--out-dir", default=None,
                   help="defaults to the fuzz result's own directory")
    p.add_argument("--frames", type=int, default=180)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--elevation", type=float, default=-18.0)
    p.add_argument("--azimuth", type=float, default=135.0)
    p.add_argument("--distance", type=float, default=1.15)
    p.add_argument("--no-video", action="store_true")
    a = p.parse_args(argv)

    import mujoco
    import cv2
    from .scenario.targets import load_target, restore_state

    res = json.loads(Path(a.fuzz_result).read_text())
    tpath = f"{a.targets_dir}/{res['target']}.json"
    w, tgt = load_target(tpath)
    sc = w.scene()
    h = w.harness
    ag = h.env.agent

    # ---- discovered goals, de-duplicated, tagged with who found them ------- #
    found = {}
    for row in res["results"]:
        for hit in row["hits"]:
            key = tuple(np.round(hit["cand"], 9))
            e = found.setdefault(key, {"pickers": set(), "kind": None,
                                       "dist": hit["dist_to_truth"]})
            e["pickers"].add(row["picker"])

    kinds = {}
    for f in sorted(glob.glob(a.selfcheck)):
        d = json.loads(Path(f).read_text())
        if d.get("target") != res["target"]:
            continue
        for c in d["candidates"]:
            kinds[tuple(np.round(c["cand"], 9))] = c["kind"]
    for key, e in found.items():
        e["kind"] = kinds.get(key)
    n_checked = sum(1 for e in found.values() if e["kind"] is not None)

    counts = {}
    for e in found.values():
        counts[e["kind"] or "unchecked"] = counts.get(e["kind"] or "unchecked", 0) + 1

    # ---- pose the robot at the handover state ----------------------------- #
    h.reset()
    restore_state(h, tgt["state_at_G0"])
    mujoco.mj_forward(ag.model, ag.data)

    base = sc.base_frame
    G0 = np.asarray(tgt["G0_commanded"], float)
    G1 = np.asarray(tgt["G1"], float)
    truth = np.asarray(tgt["G1_prime_truth"], float)

    ag.model.vis.global_.offwidth = max(a.width, ag.model.vis.global_.offwidth)
    ag.model.vis.global_.offheight = max(a.height, ag.model.vis.global_.offheight)
    renderer = mujoco.Renderer(ag.model, a.height, a.width,
                               max_geom=len(found) + 4096)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(ag.model, cam)
    obs_w = np.asarray(sc.obstacles_world)

    def world_of(xyz_base):
        return (base @ _frame(xyz_base))[:3, 3]

    # centre on the goals, not the obstacle centroid: several obstacles sit
    # well outside the reachable region and drag the view off the action
    anchors = np.array([world_of(G0), world_of(G1), world_of(truth)])
    cam.lookat[:] = anchors.mean(0)
    cam.distance, cam.elevation = a.distance, a.elevation

    out_dir = Path(a.out_dir or Path(a.fuzz_result).parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"scene_{res['target']}"
    png = out_dir / f"{stem}.png"
    mp4 = out_dir / f"{stem}_orbit.mp4"

    vw = None if a.no_video else cv2.VideoWriter(
        str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (a.width, a.height))
    F = cv2.FONT_HERSHEY_SIMPLEX
    n_frames = 1 if a.no_video else a.frames

    for i in range(n_frames):
        cam.azimuth = a.azimuth if a.no_video else (a.azimuth + 360.0 * i / a.frames)
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

        for of in obs_w:
            add(np.asarray(of)[:3, 3], 0.05, (0.85, 0.15, 0.15, 0.30))
        for key, e in found.items():
            add(world_of(np.asarray(key, float)), 0.009,
                _KIND_RGBA.get(e["kind"], _KIND_RGBA[None]))
        add(world_of(G0), 0.026, (0.20, 0.45, 1.00, 0.95))      # handover
        add(world_of(G1), 0.030, (0.10, 0.90, 0.10, 0.97))      # real goal
        add(world_of(truth), 0.022, (1.00, 0.85, 0.10, 0.98))   # planted G1'

        img = np.ascontiguousarray(renderer.render())

        def txt(y, t, col=(255, 255, 255), sc_=0.6, th=2):
            cv2.putText(img, t, (16, y), F, sc_, (0, 0, 0), th + 3, cv2.LINE_AA)
            cv2.putText(img, t, (16, y), F, sc_, col, th, cv2.LINE_AA)

        txt(34, f"{tgt['case']}   filter={tgt['algo']}   seed={tgt['seed']}", sc_=0.66)
        txt(60, f"robot posed at the G0 handover  (|v| = "
                f"{tgt['joint_speed_at_G0']:.3f})", sc_=0.55)
        txt(88, "blue = G0 handover   green = G1 real goal   "
                "yellow = planted G1'", sc_=0.55)
        txt(112, f"red spheres = obstacles", sc_=0.55)
        y = 146
        txt(y, f"{len(found)} distinct attack goals found by SIREN", sc_=0.62)
        y += 28
        for lab in ("INSERTION", "MODIFICATION", "REJECT", "unchecked"):
            if counts.get(lab):
                col = tuple(int(255 * v) for v in
                            _KIND_RGBA.get(lab, _KIND_RGBA[None])[:3])
                txt(y, f"   {lab}: {counts[lab]}", col, 0.56)
                y += 25
        by_picker = {r["picker"]: r["n_attacks"] for r in res["results"]}
        txt(y + 6, "  ".join(f"{k} {v}/{res['budget']}"
                             for k, v in by_picker.items()), sc_=0.55)

        if a.no_video or i == 0:
            cv2.imwrite(str(png), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if vw is not None:
            vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    if vw is not None:
        vw.release()

    print(f"target {res['target']}")
    print(f"  {len(found)} distinct attack goals, {n_checked} self-checked")
    print(f"  verdicts: {counts}")
    print(f"  per picker: {by_picker} of {res['budget']} evals each")
    print(f"wrote {png}")
    if vw is not None:
        print(f"wrote {mp4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
