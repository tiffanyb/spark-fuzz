"""Figure 1: SSA's demand parameter vs the safety margin it actually delivers.

Reads eta_sweep.csv (SSA rows), writes casestudy_eta.pdf.

Design notes:
  * ONE y axis. Clearance is the measure; whether the robot finished the
    schedule rides on the marker fill, not a second scale.
  * Log x -- eta spans 0.005..100, four decades; linear x would crush the whole
    feasible region into the left margin.
  * y = 0 is the outcome boundary, not a gridline: below it the robot has
    penetrated the obstacle, i.e. THE ATTACK SUCCEEDED. The two half-planes are
    washed in the status palette (critical = attack succeeds, good = attack
    fails) and labelled inside the plot, so the sign is readable without
    tracing tick labels.
  * Marker fill is a SECOND encoding of a different fact: filled = the robot
    reached the final goal, hollow = it never did. A run that avoids collision
    by stalling the task is not a defence, and colour alone would hide it.
Palette: dataviz reference status steps (critical #d03b3b, good #0ca30c) for the
outcome regions; categorical slot 1 (#2a78d6, validated on the light surface)
for the series.
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))

SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
MUTED, GRID, AXIS = "#898781", "#e1e0d9", "#c3c2b7"
CRITICAL, GOOD = "#d03b3b", "#0ca30c"
SERIES = "#2a78d6"


def load(path, algo):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["algo"] != algo:
                continue
            rows.append((float(r["eta"]), float(r["min_clearance"]),
                         int(r["reached_final"]), int(r["collision"])))
    rows.sort()
    return rows


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=os.path.join(HERE, "eta_sweep.csv"))
    p.add_argument("--algo", default="ssa")
    p.add_argument("--out", default=os.path.join(HERE, "casestudy_eta.pdf"))
    a = p.parse_args(argv)

    rows = load(a.csv, a.algo)
    if not rows:
        raise SystemExit(f"no rows for algo={a.algo} in {a.csv}")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial",
                            "DejaVu Sans"],
        "pdf.fonttype": 42,          # embed TrueType, not Type-3
        "axes.linewidth": 0.8,
    })
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    xs = [r[0] for r in rows]
    ys = [r[1] for r in rows]
    lo, hi = min(ys), max(ys)
    span = hi - lo
    ylo, yhi = lo - 0.14 * span, hi + 0.10 * span
    ax.set_ylim(ylo, yhi)
    ax.set_xlim(min(xs) * 0.6, max(xs) * 1.7)

    # outcome regions: below zero the obstacle was penetrated
    ax.axhspan(ylo, 0.0, color=CRITICAL, alpha=0.10, zorder=0, lw=0)
    ax.axhspan(0.0, yhi, color=GOOD, alpha=0.07, zorder=0, lw=0)
    ax.axhline(0.0, color=CRITICAL, lw=1.2, zorder=3)

    ax.text(min(xs) * 0.72, yhi - 0.03 * span,
            "ATTACK FAILS   no contact", color="#006300", fontsize=9,
            fontweight="bold", va="top", ha="left", zorder=6)
    ax.text(min(xs) * 0.72, ylo + 0.02 * span,
            "ATTACK SUCCEEDS   obstacle penetrated", color=CRITICAL,
            fontsize=9, fontweight="bold", va="bottom", ha="left", zorder=6)

    ax.grid(axis="y", color=GRID, lw=0.7, zorder=1)
    ax.set_axisbelow(True)

    ax.plot(xs, ys, color=SERIES, lw=2.0, zorder=4, solid_capstyle="round")
    for x, y, reached, _ in rows:
        ax.plot([x], [y], marker="o", ms=6.8, zorder=5, color=SERIES,
                markerfacecolor=(SERIES if reached else SURFACE),
                markeredgecolor=SERIES, markeredgewidth=1.7)

    ax.set_xscale("log")
    ax.set_xlabel("filter demand  $\\eta$   (rate the safety index must fall, 1/s)",
                  color=INK_2, fontsize=10)
    ax.set_ylabel("minimum clearance over the run  (m)", color=INK_2,
                  fontsize=10)
    ax.set_title("SSA: raising the demand stops helping, then stops working",
                 color=INK, fontsize=12.5, fontweight="bold", loc="left",
                 pad=12)
    ax.tick_params(colors=MUTED, labelsize=9, which="both")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:g}" for x in xs], fontsize=8, rotation=45,
                       ha="right", rotation_mode="anchor")
    ax.minorticks_off()
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)

    handles = [
        Line2D([], [], color=SERIES, lw=2.0, marker="o", ms=6.8,
               markerfacecolor=SERIES, markeredgecolor=SERIES,
               label="reached the final goal"),
        Line2D([], [], color=SERIES, lw=2.0, marker="o", ms=6.8,
               markerfacecolor=SURFACE, markeredgecolor=SERIES,
               markeredgewidth=1.7, label="never reached it (task stalled)"),
        Patch(facecolor=CRITICAL, alpha=0.10, label="attack succeeds"),
        Patch(facecolor=GOOD, alpha=0.07, label="attack fails"),
    ]
    leg = ax.legend(handles=handles, loc="upper left", frameon=False,
                    fontsize=8.6, handletextpad=0.7, borderaxespad=0.0,
                    labelspacing=0.5, bbox_to_anchor=(0.015, 0.90))
    for t in leg.get_texts():
        t.set_color(INK_2)

    fig.tight_layout()
    fig.savefig(a.out, format="pdf", facecolor=SURFACE, bbox_inches="tight")
    png = os.path.splitext(a.out)[0] + ".png"
    fig.savefig(png, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {a.out}\n      {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
