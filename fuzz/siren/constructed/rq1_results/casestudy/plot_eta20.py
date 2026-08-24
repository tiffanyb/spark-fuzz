"""Figure 2: why SSA at eta=20 goes infeasible -- authority follows the body part.

Reads eta20_authority.csv, writes casestudy_eta20.pdf.

The story the chart has to carry: c is NOT a constant of the robot. u_lim and
phi_k never change during the run, so every move in c comes from WHICH collision
volume is nearest the obstacle. A contact at the hand can be driven by the whole
arm; a contact at the elbow can only be driven by the joints upstream of it,
because the wrist joints move the hand without moving the elbow. When the nearest
point hands over from wrist to elbow, c halves, falls under eta, and the QP has
no solution -- SSA then returns the unfiltered reference command.

Design notes:
  * ONE y axis: authority. eta is a reference line on the SAME scale (it is the
    same kind of quantity, a rate), never a second axis.
  * The filter is only engaged on 98 of 687 steps, in clusters. Segments break on
    a step gap as well as on a body-part change, so the line never draws a
    connection across steps where nothing was measured.
  * Colour is assigned by how much of the run each body part owns, so the two
    segments carrying the story get the two strongest categorical slots. Every
    segment worth reading is directly labelled, so identity never rests on
    colour alone.
  * Infeasible steps are a rug along the bottom -- the consequence -- rather than
    a restyling of the authority line, which would overload one mark with two
    meanings.
Palette: dataviz reference categorical slots (validated on the light surface)
ordered by segment size; status critical for eta and for the infeasible rug.
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))

SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
MUTED, GRID, AXIS = "#898781", "#e1e0d9", "#c3c2b7"
CRITICAL = "#d03b3b"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#898781"]
GAP = 6          # a break of more than this many steps is not a continuous run

PRETTY = {"right_wrist_roll_joint": "right wrist roll",
          "right_wrist_pitch_joint": "right wrist pitch",
          "right_elbow_joint": "right ELBOW",
          "torso_link_1": "torso",
          "R_ee": "right hand"}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=os.path.join(HERE, "eta20_authority.csv"))
    p.add_argument("--out", default=os.path.join(HERE, "casestudy_eta20.pdf"))
    a = p.parse_args(argv)

    rows = list(csv.DictReader(open(a.csv)))
    step = [int(r["step"]) for r in rows]
    c = [float(r["c_authority"]) for r in rows]
    part = [r["body_part"] for r in rows]
    infeas = [int(r["infeasible"]) for r in rows]
    eta = float(rows[0]["eta"])

    # colour by share of the run, so the dominant parts get the strongest slots
    counts = {}
    for q in part:
        counts[q] = counts.get(q, 0) + 1
    ranked = sorted(counts, key=lambda q: -counts[q])
    colour = {q: SLOTS[min(i, len(SLOTS) - 1)] for i, q in enumerate(ranked)}

    # split on body-part change OR a gap in the engaged steps
    segs, s0 = [], 0
    for i in range(1, len(rows) + 1):
        brk = (i == len(rows) or part[i] != part[s0]
               or step[i] - step[i - 1] > GAP)
        if brk:
            segs.append((s0, i - 1))
            s0 = i

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial",
                            "DejaVu Sans"],
        "pdf.fonttype": 42,
        "axes.linewidth": 0.8,
    })
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    top = max(c) * 1.16
    ax.set_ylim(-top * 0.055, top)
    ax.set_xlim(min(step) - 25, max(step) + 30)
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=1)
    ax.set_axisbelow(True)

    ax.axhline(eta, color=CRITICAL, lw=1.3, ls=(0, (5, 3)), zorder=3)
    ax.text(min(step) - 18, eta + top * 0.012,
            f"demand $\\eta$ = {eta:g}", color=CRITICAL, fontsize=9.5,
            fontweight="bold", va="bottom", ha="left", zorder=6)
    ax.text(min(step) - 18, eta - top * 0.018,
            "no command exists below", color=CRITICAL, fontsize=7.8,
            va="top", ha="left", zorder=6)

    for s0, s1 in segs:
        xs, ys = step[s0:s1 + 1], c[s0:s1 + 1]
        col = colour[part[s0]]
        if len(xs) == 1:
            ax.plot(xs, ys, marker="o", ms=6.0, color=col, zorder=5)
        else:
            ax.plot(xs, ys, color=col, lw=2.2, zorder=4,
                    solid_capstyle="round")

    # direct-label only the segments long enough to read, above the segment
    for s0, s1 in segs:
        if (s1 - s0) < 6:
            continue
        mid = (s0 + s1) // 2
        ax.annotate(PRETTY.get(part[s0], part[s0]),
                    xy=(step[mid], c[mid]),
                    xytext=(step[mid], c[mid] + top * 0.055),
                    color=colour[part[s0]], fontsize=10, fontweight="bold",
                    ha="center", va="bottom", zorder=6)

    # the handover that matters: last major wrist segment -> first elbow segment
    major = [(s0, s1) for s0, s1 in segs if (s1 - s0) >= 6]
    for (p0, p1), (q0, q1) in zip(major, major[1:]):
        if part[p0] == part[q0]:
            continue
        x = step[q0]
        ax.axvline(x, color=AXIS, lw=0.9, ls=(0, (2, 3)), zorder=2)
        ax.annotate(f"nearest point hands over at step {x}\n"
                    f"{PRETTY.get(part[p0], part[p0])}  to  "
                    f"{PRETTY.get(part[q0], part[q0])}",
                    xy=(x, top * 0.38), xytext=(x - 22, top * 0.38),
                    color=INK_2, fontsize=8.6, ha="right", va="center",
                    zorder=6)

    y0 = -top * 0.030
    xs_bad = [x for x, b in zip(step, infeas) if b]
    ax.plot(xs_bad, [y0] * len(xs_bad), marker="|", ms=8, lw=0,
            color=CRITICAL, zorder=5)
    ax.text(min(step) - 18, y0, "QP infeasible", color=CRITICAL, fontsize=8.2,
            va="center", ha="left", zorder=6)

    ax.set_xlabel("control step   (only the 98 steps where the filter engaged)",
                  color=INK_2, fontsize=10)
    ax.set_ylabel("control authority  $c$  on the binding constraint  (1/s)",
                  color=INK_2, fontsize=10)
    ax.set_title("SSA $\\eta$ = 20: authority collapses when the contact "
                 "moves from wrist to elbow",
                 color=INK, fontsize=12.5, fontweight="bold", loc="left",
                 pad=10)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_yticks([0, 10, 20, 30, 40])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)

    handles = [Line2D([], [], color=colour[q], lw=2.2,
                      label=f"{PRETTY.get(q, q)}  ({counts[q]} step"
                            f"{'s' if counts[q] != 1 else ''})")
               for q in ranked]
    handles += [
        Line2D([], [], color=CRITICAL, lw=1.3, ls=(0, (5, 3)),
               label=f"demand $\\eta$ = {eta:g}"),
        Line2D([], [], color=CRITICAL, lw=0, marker="|", ms=8,
               label="QP infeasible: runs unfiltered"),
    ]
    leg = ax.legend(handles=handles, title="nearest point to the obstacle",
                    loc="upper center", frameon=False, fontsize=8.3, ncol=3,
                    handletextpad=0.6, columnspacing=1.6, borderaxespad=0.2,
                    bbox_to_anchor=(0.52, 1.005))
    leg.get_title().set_color(INK_2)
    leg.get_title().set_fontsize(8.3)
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
