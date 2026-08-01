"""
Render a SIREN scene with every inserted goal G1' the search tried, coloured by
what happened.

    G0        blue      where the hand starts
    G1        green     the legitimate goal
    obstacles translucent red
    G1' tried REACHED grey / TIMEOUT orange / COLLISION red / DEADLOCK magenta

This is the SIREN-world counterpart of fuzz/plot_candidates.py, which predates
world/ and drives the old single-arm harness directly. Same rendering approach,
but it goes through World so the picture reflects exactly the scene the search
sees -- including the guarded-pairs clearance fix (BUG-1) and the FilterSpec.

Two views are written: the MuJoCo render, and a 2-D projection panel that is
usually the more readable of the two, because the render collapses depth and the
interesting structure is where the failures sit RELATIVE to the obstacles.

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    python -m fuzz.siren.plot_scene --seed 1 --case G1FixedBase_D2_AG_SO_v0 \
        --eta 0.02 --n 80 --out /tmp/scene_seed1
"""

import argparse

import numpy as np

from .search.pick import sample_admissible
from .world.run import World
from .world.types import real_filter

_OUTCOME_RGBA = {
    "REACHED":   (0.62, 0.62, 0.62, 0.85),
    "TIMEOUT":   (1.00, 0.60, 0.10, 0.95),
    "COLLISION": (1.00, 0.10, 0.10, 0.98),
    "DEADLOCK":  (0.80, 0.10, 0.90, 0.98),
}


def _xyz_to_frame(xyz):
    f = np.eye(4)
    f[:3, 3] = np.asarray(xyz, dtype=float).reshape(3)
    return f


def collect(world, scene, n, max_steps, search_seed=0, verbose=True):
    """Sample admissible G1', keep the ones that are individually reachable, and
    record what the full G0 -> G1' -> G1 schedule does.

    The reachability screen is the same gate the real search uses: a G1' the arm
    cannot reach on its own is not a usable inserted goal, so it should not
    appear in the picture as though it had been fairly tried.
    """
    rng = np.random.RandomState(search_seed)
    out = []
    attempts = 0
    while len(out) < n and attempts < n * 20:
        attempts += 1
        cand = sample_admissible(rng, scene)
        if cand is None:
            continue
        screen = world.run([cand], max_steps=max_steps)
        if not screen.reached:
            continue
        rec = world.run([cand, np.asarray(scene.G1)], max_steps=max_steps)
        out.append((np.asarray(cand, float), rec.label,
                    float(rec.min_clearance)))
        if verbose and len(out) % 10 == 0:
            print(f"  collected {len(out)}/{n}", flush=True)
    return out


def render_mujoco(world, scene, cands, out_path, width=1280, height=720,
                  azimuth=135.0, elevation=-20.0, distance=1.3):
    import mujoco
    import cv2

    h = world.harness
    agent = h.env.agent
    base = scene.base_frame
    h.reset()
    mujoco.mj_forward(agent.model, agent.data)

    agent.model.vis.global_.offwidth = max(width, agent.model.vis.global_.offwidth)
    agent.model.vis.global_.offheight = max(height, agent.model.vis.global_.offheight)
    renderer = mujoco.Renderer(agent.model, height, width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(agent.model, cam)
    obs_f = np.asarray(scene.obstacles_world)
    cam.lookat[:] = (obs_f[:, :3, 3].mean(0) if len(obs_f)
                     else (base @ _xyz_to_frame(scene.G1))[:3, 3])
    cam.distance, cam.elevation, cam.azimuth = distance, elevation, azimuth
    renderer.update_scene(agent.data, camera=cam)
    sc = renderer.scene

    def add(pos, r, rgba):
        if sc.ngeom >= sc.maxgeom:
            return
        g = sc.geoms[sc.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([r, r, r], float),
                            np.asarray(pos, float).reshape(3),
                            np.eye(3).flatten(),
                            np.asarray(rgba, np.float32))
        sc.ngeom += 1

    def w(xyz_base):
        return (base @ _xyz_to_frame(xyz_base))[:3, 3]

    for of in obs_f:
        add(of[:3, 3], 0.05, (0.85, 0.15, 0.15, 0.45))
    counts = {}
    for cand, label, _c in cands:
        counts[label] = counts.get(label, 0) + 1
        add(w(cand), 0.012, _OUTCOME_RGBA.get(label, (0.5, 0.5, 0.5, 0.8)))
    add(w(scene.G0), 0.030, (0.20, 0.45, 1.00, 0.95))
    add(w(scene.G1), 0.040, (0.10, 0.90, 0.10, 0.97))

    img = renderer.render()
    cv2.putText(img, f"tried G1' = {len(cands)}   (blue=G0  green=G1)", (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
    y = 58
    for label in ("REACHED", "TIMEOUT", "COLLISION", "DEADLOCK"):
        col = tuple(int(255 * c) for c in _OUTCOME_RGBA[label][:3])
        cv2.putText(img, f"{label}: {counts.get(label, 0)}", (16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        y += 26
    cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"[plot] wrote {out_path}  outcomes={counts}")
    return counts


def render_projections(scene, cands, out_path, title=""):
    """Three 2-D slices through the goal box, obstacles drawn to scale.

    Obstacles are spheres in WORLD coordinates and the goals live in BASE
    coordinates, so the obstacle centres are converted into the base frame first
    -- otherwise the two are drawn in different frames and the picture silently
    lies about which goals are near which obstacle.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    inv = np.linalg.inv(scene.base_frame)
    obs = np.asarray(scene.obstacles_world)
    obs_base = np.array([(inv @ np.append(o[:3, 3], 1.0))[:3] for o in obs]) \
        if len(obs) else np.zeros((0, 3))
    r_obs = 0.05

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    planes = [(0, 1, "x", "y"), (0, 2, "x", "z"), (1, 2, "y", "z")]
    for ax, (i, j, li, lj) in zip(axes, planes):
        for c in obs_base:
            ax.add_patch(Circle((c[i], c[j]), r_obs, color="red", alpha=0.22, zorder=1))
            ax.plot(c[i], c[j], "x", color="darkred", ms=6, zorder=2)
        for label in ("REACHED", "TIMEOUT", "COLLISION", "DEADLOCK"):
            pts = np.array([c for c, l, _ in cands if l == label])
            if not len(pts):
                continue
            rgba = _OUTCOME_RGBA[label]
            ax.scatter(pts[:, i], pts[:, j], s=34, color=rgba[:3],
                       edgecolor="k", linewidth=0.35, label=label, zorder=3)
        ax.plot(*[scene.G0[k] for k in (i, j)], "o", ms=13, color="royalblue",
                markeredgecolor="k", label="G0 (start)", zorder=5)
        ax.plot(*[scene.G1[k] for k in (i, j)], "*", ms=20, color="limegreen",
                markeredgecolor="k", label="G1 (real goal)", zorder=5)
        lo = [b[0] for b in scene.bounds]
        hi = [b[1] for b in scene.bounds]
        ax.add_patch(plt.Rectangle((lo[i], lo[j]), hi[i] - lo[i], hi[j] - lo[j],
                                   fill=False, ls="--", ec="gray", lw=1.0,
                                   label="admissible box"))
        ax.set_xlabel(f"{li} (m, base frame)")
        ax.set_ylabel(f"{lj} (m, base frame)")
        ax.set_title(f"{li}-{lj}")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.25)
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=7, frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(out_path, dpi=140)
    print(f"[plot] wrote {out_path}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--case", default="G1FixedBase_D2_AG_SO_v0")
    p.add_argument("--eta", type=float, default=0.02)
    p.add_argument("--d-min", type=float, default=0.02)
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--search-seed", type=int, default=0)
    p.add_argument("--out", default="/tmp/scene")
    p.add_argument("--no-mujoco", action="store_true")
    a = p.parse_args(argv)

    index = "velocity" if "_D2_" in a.case else "distance"
    spec = real_filter(algo="ssa", index=index, d_min=a.d_min, eta=a.eta, k=0.1)
    world = World.build(seed=a.seed, spec=spec, test_case=a.case,
                        max_steps=a.max_steps)
    scene = world.scene()
    print(f"[scene] {a.case} seed={a.seed}  G0={np.round(scene.G0,3)}  "
          f"G1={np.round(scene.G1,3)}  obstacles={len(scene.obstacles_world)}  "
          f"bounds={[tuple(np.round(b,2)) for b in scene.bounds]}", flush=True)

    cands = collect(world, scene, a.n, a.max_steps, a.search_seed)
    counts = {}
    for _c, l, _m in cands:
        counts[l] = counts.get(l, 0) + 1
    print(f"[collect] {len(cands)} reachable G1' -> {counts}", flush=True)

    title = (f"{a.case}  seed={a.seed}  eta={a.eta}   "
             f"tried {len(cands)} inserted goals -> {counts}")
    render_projections(scene, cands, f"{a.out}_proj.png", title=title)
    if not a.no_mujoco:
        try:
            render_mujoco(world, scene, cands, f"{a.out}_3d.png")
        except Exception as e:
            print(f"[plot] mujoco render skipped: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
