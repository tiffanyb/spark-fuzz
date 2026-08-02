"""
Plot a scene and every goal already tried on it — straight from the sweep record.

No simulation. The sweep already stored every candidate's position and outcome,
so re-running them (as plot_scene.py does) is minutes of compute to redraw data
that is sitting on disk. This reads it instead.

    python -m fuzz.siren.plot_from_sweep --case G1FixedBase_D1_AG_SO_v0 --seed 0
"""

import argparse
import glob
import json

import numpy as np

_RGB = {"REACHED": (0.62, 0.62, 0.62), "TIMEOUT": (1.00, 0.60, 0.10),
        "COLLISION": (1.00, 0.10, 0.10), "DEADLOCK": (0.80, 0.10, 0.90)}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="G1FixedBase_D1_AG_SO_v0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sweep", default="/tmp/rep_shard*.jsonl")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    scene = None
    pts = []          # (xyz, label)
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

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    G0 = np.array(scene["G0"]); G1 = np.array(scene["G1"])
    bounds = scene["bounds"]; keep = scene["keepout"]
    base = None
    # obstacles are stored in WORLD coords; the goals are in BASE coords. The
    # sweep kept only obstacle centres, so recover the base frame from the run
    # that produced them: G0/G1 are base-frame, obstacles world-frame, and the
    # benchmark's base frame is a pure translation for the fixed-base G1.
    obs_w = np.array(scene["obstacles_world"]) if scene["obstacles_world"] else np.zeros((0, 3))

    counts = {}
    for _c, l in pts:
        counts[l] = counts.get(l, 0) + 1

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    planes = [(0, 1, "x", "y"), (0, 2, "x", "z"), (1, 2, "y", "z")]
    P = np.array([c for c, _ in pts])
    L = [l for _, l in pts]
    for ax, (i, j, li, lj) in zip(axes, planes):
        for lab in ("REACHED", "TIMEOUT", "DEADLOCK", "COLLISION"):
            m = [k for k, x in enumerate(L) if x == lab]
            if not m:
                continue
            ax.scatter(P[m, i], P[m, j], s=18, color=_RGB.get(lab, (.5, .5, .5)),
                       edgecolor="k", linewidth=0.2, label=f"{lab} ({len(m)})",
                       zorder=3, alpha=0.75)
        ax.plot(G0[i], G0[j], "o", ms=13, color="royalblue",
                markeredgecolor="k", label="G0 (start)", zorder=5)
        ax.plot(G1[i], G1[j], "*", ms=20, color="limegreen",
                markeredgecolor="k", label="G1 (real goal)", zorder=5)
        lo = [b[0] for b in bounds]; hi = [b[1] for b in bounds]
        ax.add_patch(plt.Rectangle((lo[i], lo[j]), hi[i]-lo[i], hi[j]-lo[j],
                                   fill=False, ls="--", ec="gray", lw=1.0,
                                   label="admissible box"))
        ax.set_xlabel(f"{li} (m, base frame)"); ax.set_ylabel(f"{lj} (m, base frame)")
        ax.set_title(f"{li}-{lj}"); ax.grid(alpha=0.25)
        ax.set_aspect("equal", adjustable="datalim")
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=7, frameon=False, fontsize=9)
    fig.suptitle(f"{a.case}  seed={a.seed}   {len(pts)} goals tried "
                 f"(from the sweep, no re-simulation)   ->  {counts}", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    out = a.out or f"/tmp/sweep_{a.case}_s{a.seed}.png"
    fig.savefig(out, dpi=140)
    print(f"{len(pts)} goals  outcomes={counts}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
