"""Plot control authority over time for the eta=20 case study.

Reads data/eta20_authority.csv and writes data/eta20_authority_steps.pdf/png.
"""

import argparse
import csv
import os

import matplotlib

# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))

SURFACE = "#ffffff"
INK = "#0b0b0b"
GRID = "#d9d9d9"
FONTSIZE = 18
BODY_PART_COLORS = {
    "right_elbow_joint": "#0072B2",
    "right_wrist_roll_joint": "#D55E00",
    "right_wrist_pitch_joint": "#009E73",
    "right_shoulder_yaw_joint": "#CC79A7",
    "R_ee": "#E69F00",
    "right_shoulder_roll_joint": "#56B4E9",
    "torso_link_1": "#666666",
}


def pretty_body_part(name):
    return {
        "right_elbow_joint": "Right elbow",
        "right_wrist_roll_joint": "Right wrist roll",
        "right_wrist_pitch_joint": "Right wrist pitch",
        "right_shoulder_yaw_joint": "Right shoulder yaw",
        "right_shoulder_roll_joint": "Right shoulder roll",
        "R_ee": "Right end effector",
        "torso_link_1": "Torso",
    }.get(name, name.replace("_", " "))


def load_rows(path):
    rows = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            rows.append({
                "step": int(row["step"]),
                "authority": float(row["c_authority"]),
                "body_part": row["body_part"],
            })
    rows.sort(key=lambda row: row["step"])
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=os.path.join(HERE, "data", "eta20_authority.csv"))
    parser.add_argument("--out", default=os.path.join(HERE, "data", "eta20_authority_steps.pdf"))
    args = parser.parse_args(argv)

    rows = load_rows(args.csv)
    if not rows:
        raise SystemExit(f"no rows in {args.csv}")

    xs = [row["step"] for row in rows]
    ys = [row["authority"] for row in rows]
    body_parts = sorted({row["body_part"] for row in rows})

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.8,
    })

    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot(xs, ys, color=INK, lw=1, zorder=2)
    # ax.plot(xs, ys, color=INK, lw=1, linestyle="--", zorder=2)
    for body_part in body_parts:
        part_rows = [row for row in rows if row["body_part"] == body_part]
        ax.scatter(
            [row["step"] for row in part_rows],
            [row["authority"] for row in part_rows],
            color=BODY_PART_COLORS.get(body_part, INK),
            s=2,
            zorder=3,
        )

    ax.set_xlim(min(xs), max(xs))
    ax.set_ylim(0, max(ys) * 1.12)
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=1)
    ax.set_axisbelow(True)

    ax.set_xlabel("Step", color=INK, fontsize=FONTSIZE, fontname="Times New Roman")
    ax.set_ylabel("Control Authority of the\nNearest Body Part",
                  color=INK, fontsize=FONTSIZE,
                  fontname="Times New Roman")
    ax.tick_params(colors=INK, labelsize=FONTSIZE, which="both")

    handles = [
        Line2D([], [], linestyle="", marker="o", markersize=5,
               color=BODY_PART_COLORS.get(body_part, INK),
               label=pretty_body_part(body_part))
        for body_part in body_parts
    ]
    legend = ax.legend(handles=handles, loc="upper right",
                       ncol=2, frameon=False,
                       fontsize=14, handletextpad=0.4, borderaxespad=0.2,
                       columnspacing=1.1, labelspacing=0.3)
    for text in legend.get_texts():
        text.set_color(INK)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK)

    fig.tight_layout()
    fig.savefig(args.out, format="pdf", facecolor=SURFACE, bbox_inches="tight")
    png = os.path.splitext(args.out)[0] + ".png"
    fig.savefig(png, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {args.out}\n      {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
