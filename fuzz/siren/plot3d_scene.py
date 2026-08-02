"""
3-D view of one scene and every goal already tried on it.

Candidates and outcomes come from the sweep record — no re-simulation. The world
is built once, and only to read the base frame, because the sweep stores obstacle
centres in WORLD coordinates while goals are in the robot BASE frame; drawing
them together without that transform would put the obstacles in the wrong place.

    python -m fuzz.siren.plot3d_scene --case G1FixedBase_D1_AG_SO_v0 --seed 0
"""

import argparse
import glob
import json

import numpy as np

_RGB = {"REACHED": "#9aa5ad", "TIMEOUT": "#e08a1e",
        "COLLISION": "#d81c1c", "DEADLOCK": "#c020d0"}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="G1FixedBase_D1_AG_SO_v0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sweep", default="/tmp/rep_shard*.jsonl")
    p.add_argument("--obstacle-radius", type=float, default=0.05)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    scene = None
    pts = []
    for f in sorted(glob.glob(a.sweep)):
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("status") != "ok" or r["case"] != a.case or r["seed"] != a.seed:
                continue
            scene = r["scene"]
            for e in r["evaluations"]:
                if e["stage"] == "inadmissible":
                    continue
                lab = (e["leg2_label"] if e["stage"] == "evaluated"
                       else (e["leg1_label"] or "TIMEOUT"))
                pts.append((e["candidate"], lab or "TIMEOUT"))
    if scene is None:
        print(f"no records for {a.case} seed {a.seed}")
        return 1

    # one build, no rollouts — purely to get the base frame for the transform
    from .world.run import World
    from .world.types import real_filter
    index = "velocity" if "_D2_" in a.case else "distance"
    spec = real_filter(algo="ssa", index=index, d_min=0.02, eta=0.02, k=0.1)
    w = World.build(seed=a.seed, spec=spec, test_case=a.case, max_steps=10)
    sc = w.scene()
    inv = np.linalg.inv(sc.base_frame)
    obs_base = np.array([(inv @ np.append(np.asarray(o)[:3, 3], 1.0))[:3]
                         for o in sc.obstacles_world])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    G0 = np.array(scene["G0"]); G1 = np.array(scene["G1"])
    bounds = scene["bounds"]
    P = np.array([c for c, _ in pts]); L = [l for _, l in pts]
    counts = {}
    for l in L:
        counts[l] = counts.get(l, 0) + 1

    fig = plt.figure(figsize=(15, 6.8))
    for panel, (elev, azim) in enumerate([(22, -60), (14, 20), (78, -90)]):
        ax = fig.add_subplot(1, 3, panel + 1, projection="3d")

        # obstacles, drawn to scale as translucent spheres
        uu, vv = np.mgrid[0:2*np.pi:22j, 0:np.pi:12j]
        for c in obs_base:
            ax.plot_surface(c[0] + a.obstacle_radius*np.cos(uu)*np.sin(vv),
                            c[1] + a.obstacle_radius*np.sin(uu)*np.sin(vv),
                            c[2] + a.obstacle_radius*np.cos(vv),
                            color="#d81c1c", alpha=0.16, linewidth=0, shade=False)
            ax.scatter(*c, color="#8b0000", marker="x", s=34, depthshade=False)

        for lab in ("REACHED", "TIMEOUT", "DEADLOCK", "COLLISION"):
            m = [k for k, x in enumerate(L) if x == lab]
            if m:
                ax.scatter(P[m, 0], P[m, 1], P[m, 2], s=13, c=_RGB[lab],
                           edgecolors="k", linewidths=0.15, alpha=0.8,
                           depthshade=False, label=f"{lab} ({len(m)})")

        ax.scatter(*G0, s=150, c="#2f6fe0", marker="o", edgecolors="k",
                   linewidths=0.8, depthshade=False, label="G0 (start)")
        ax.scatter(*G1, s=280, c="#22c55e", marker="*", edgecolors="k",
                   linewidths=0.8, depthshade=False, label="G1 (real goal)")

        # admissible box, wireframe
        lo = [b[0] for b in bounds]; hi = [b[1] for b in bounds]
        for s, e in [((0,0,0),(1,0,0)),((0,0,0),(0,1,0)),((0,0,0),(0,0,1)),
                     ((1,1,1),(0,1,1)),((1,1,1),(1,0,1)),((1,1,1),(1,1,0)),
                     ((1,0,0),(1,1,0)),((1,0,0),(1,0,1)),((0,1,0),(1,1,0)),
                     ((0,1,0),(0,1,1)),((0,0,1),(1,0,1)),((0,0,1),(0,1,1))]:
            ax.plot(*[[ (lo[d],hi[d])[s[d]], (lo[d],hi[d])[e[d]] ] for d in range(3)],
                    color="#888", lw=0.7, ls="--", alpha=0.65)

        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("x (m)", fontsize=8); ax.set_ylabel("y (m)", fontsize=8)
        ax.set_zlabel("z (m)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(["oblique", "side", "top-down"][panel], fontsize=10)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass

    h_, l_ = fig.axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=6, frameon=False, fontsize=9)
    fig.suptitle(f"{a.case}  seed={a.seed}   {len(pts)} inserted goals tried  ->  {counts}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    out = a.out or f"/tmp/scene3d_{a.case}_s{a.seed}.png"
    fig.savefig(out, dpi=145)
    print(f"{len(pts)} goals, {len(obs_base)} obstacles  outcomes={counts}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
