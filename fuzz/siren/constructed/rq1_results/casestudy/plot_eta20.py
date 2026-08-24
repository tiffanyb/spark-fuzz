"""Figure 2: why SSA at eta=20 goes infeasible -- authority follows the body part.

Reads eta20_authority.csv (every step of the run), writes casestudy_eta20.pdf.

What the chart has to carry: c is NOT a constant of the robot. u_lim and phi_k
never change during the run, so every move in c comes from WHICH collision volume
is nearest the obstacle. A contact at the wrist can be driven by the whole arm; a
contact at the elbow can only be driven by the joints upstream of it, because the
wrist joints move the hand without moving the elbow. Wrist contact gives
c ~ 17-21.5, elbow contact c ~ 7.7-11 -- and the demand is 20.

The nearest point alternates between wrist and elbow throughout the run as the
arm swings, so this is NOT a single handover event. What ends the run is the
last transition: from step 632 the elbow stays nearest, c sits at half the
demand, every step is infeasible, and the robot runs unfiltered into the
obstacle.

Design notes:
  * ONE y axis: authority. eta is a reference line on the SAME scale (it is the
    same kind of quantity, a rate), never a second axis.
  * The full-run trace is drawn recessive; the steps where the filter actually
    ENGAGED are drawn prominently on top. Authority is defined at every pose,
    but it only has consequences where the filter is trying to act.
  * Colour separates the two regimes that matter (wrist-family vs elbow), not
    every volume the run touches -- eight hues for eight body parts would be a
    palette, not an argument. Everything else is one muted "other".
  * Infeasible steps are a rug along the bottom -- the consequence -- rather
    than a restyling of the authority line, which would overload one mark.
Palette: dataviz reference categorical slots 1-2 (validated adjacent CVD dE 24.7
protan, normal-vision 33.6, both clear of the floors) on the light surface;
status critical for eta and the rug.
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
ELBOW, WRIST, OTHER = "#2a78d6", "#eb6834", "#898781"


def group(part):
    if "elbow" in part:
        return "elbow"
    if "wrist" in part or part == "R_ee":
        return "wrist"
    return "other"


COLOUR = {"elbow": ELBOW, "wrist": WRIST, "other": OTHER}
LABEL = {"elbow": "elbow nearest", "wrist": "wrist / hand nearest",
         "other": "shoulder or torso nearest"}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=os.path.join(HERE, "eta20_authority.csv"))
    p.add_argument("--out", default=os.path.join(HERE, "casestudy_eta20.pdf"))
    a = p.parse_args(argv)

    rows = list(csv.DictReader(open(a.csv)))
    step = [int(r["step"]) for r in rows]
    c = [float(r["c_authority"]) for r in rows]
    grp = [group(r["body_part"]) for r in rows]
    eng = [int(r["engaged"]) for r in rows]
    infeas = [int(r["infeasible"]) for r in rows]
    eta = float(rows[0]["eta"])

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

    top = max(c) * 1.10
    ax.set_ylim(-top * 0.06, top)
    ax.set_xlim(min(step) - 12, max(step) + 12)
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=1)
    ax.set_axisbelow(True)

    # the demand the authority has to clear
    ax.axhline(eta, color=CRITICAL, lw=1.3, ls=(0, (5, 3)), zorder=3)
    ax.text(min(step) - 6, eta + top * 0.012, f"demand $\\eta$ = {eta:g}",
            color=CRITICAL, fontsize=9.5, fontweight="bold", va="bottom",
            ha="left", zorder=6)
    ax.text(min(step) - 6, eta - top * 0.016, "no command exists below",
            color=CRITICAL, fontsize=7.8, va="top", ha="left", zorder=6)

    # full run, recessive: authority is defined at every pose
    ax.plot(step, c, color=MUTED, lw=0.8, alpha=0.55, zorder=2)

    # the steps where the filter actually engaged, in the fore
    s0 = 0
    for i in range(1, len(rows) + 1):
        brk = (i == len(rows) or eng[i] != eng[s0] or grp[i] != grp[s0]
               or step[i] - step[i - 1] > 1)
        if not brk:
            continue
        if eng[s0]:
            xs, ys = step[s0:i], c[s0:i]
            col = COLOUR[grp[s0]]
            if len(xs) == 1:
                ax.plot(xs, ys, marker="o", ms=5.0, color=col, zorder=5)
            else:
                ax.plot(xs, ys, color=col, lw=2.6, zorder=5,
                        solid_capstyle="round")
        s0 = i

    # the transition that ends the run
    last0 = next(i for i in range(len(rows) - 1, -1, -1)
                 if not (eng[i] and grp[i] == "elbow"))
    x = step[last0 + 1]
    ax.axvline(x, color=AXIS, lw=0.9, ls=(0, (2, 3)), zorder=2)
    ax.annotate(f"from step {x} the elbow stays nearest:\n"
                f"authority sits at half the demand,\n"
                f"every step infeasible, then contact",
                xy=(x, top * 0.60), xytext=(x - 18, top * 0.60),
                color=INK_2, fontsize=8.6, ha="right", va="center", zorder=6)

    y0 = -top * 0.035
    xs_bad = [xx for xx, b in zip(step, infeas) if b]
    ax.plot(xs_bad, [y0] * len(xs_bad), marker="|", ms=8, lw=0,
            color=CRITICAL, zorder=5)
    ax.text(min(step) - 6, y0, "QP infeasible", color=CRITICAL, fontsize=8.2,
            va="center", ha="left", zorder=6)

    n_eng = sum(eng)
    ax.set_xlabel(f"control step   (all {len(rows)}; the filter engaged on "
                  f"{n_eng})", color=INK_2, fontsize=10)
    ax.set_ylabel("control authority  $c$  on the binding constraint  (1/s)",
                  color=INK_2, fontsize=10)
    ax.set_title("SSA $\\eta$ = 20: the arm can only meet the demand while the "
                 "wrist is nearest", color=INK, fontsize=12.5,
                 fontweight="bold", loc="left", pad=10)
    ax.tick_params(colors=MUTED, labelsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)

    seen = [g for g in ("wrist", "elbow", "other") if g in grp]
    handles = [Line2D([], [], color=COLOUR[g], lw=2.6,
                      label=f"filter engaged, {LABEL[g]}") for g in seen]
    handles += [
        Line2D([], [], color=MUTED, lw=0.8, alpha=0.55,
               label="filter idle (authority still defined)"),
        Line2D([], [], color=CRITICAL, lw=1.3, ls=(0, (5, 3)),
               label=f"demand $\\eta$ = {eta:g}"),
        Line2D([], [], color=CRITICAL, lw=0, marker="|", ms=8,
               label="QP infeasible: runs unfiltered"),
    ]
    leg = ax.legend(handles=handles, loc="upper center", frameon=False,
                    fontsize=8.3, ncol=2, handletextpad=0.6,
                    columnspacing=1.8, borderaxespad=0.2,
                    bbox_to_anchor=(0.60, 1.005))
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
