"""Plot eta sweep clearance with task success/failure ranges.

Reads eta_sweep.csv and writes eta_clearance_task.pdf/png. The main series is
minimum clearance over the run. Two status lines at the bottom mark eta samples
where the full task succeeds or fails.
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = "#2a78d6"
COLLISION = "#d03b3b"
SUCCESS = "#0b8f34"
FAILURE = "#b33b2e"


def load_rows(path, algo):
    rows = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            if row["algo"] != algo:
                continue
            rows.append({
                "eta": float(row["eta"]),
                "clearance": float(row["min_clearance"]),
                "collision": bool(int(row["collision"])),
                "reached_final": bool(int(row["reached_final"])),
            })
    rows.sort(key=lambda row: row["eta"])
    return rows


def contiguous_segments(xs, flags, want):
    """Return x/y-ready segments for consecutive samples with flags == want."""
    segments = []
    start = None
    prev = None
    for x, flag in zip(xs, flags):
        if flag == want:
            if start is None:
                start = x
            prev = x
        elif start is not None:
            segments.append((start, prev))
            start = prev = None
    if start is not None:
        segments.append((start, prev))
    return segments


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=os.path.join(HERE, "eta_sweep.csv"))
    parser.add_argument("--algo", default="ssa")
    parser.add_argument("--out", default=os.path.join(HERE, "eta_clearance_task.pdf"))
    args = parser.parse_args(argv)

    rows = load_rows(args.csv, args.algo)
    if not rows:
        raise SystemExit(f"no rows for algo={args.algo} in {args.csv}")

    xs = [row["eta"] for row in rows]
    ys = [row["clearance"] for row in rows]
    reached = [row["reached_final"] for row in rows]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "axes.linewidth": 0.8,
    })

    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    lo = min(ys)
    hi = max(ys)
    span = hi - lo
    ylo = lo - 0.35 * span
    yhi = hi + 0.10 * span
    ax.set_ylim(ylo, yhi)
    ax.set_xlim(min(xs) * 0.6, max(xs) * 1.7)

    ax.axhline(0.0, color=COLLISION, lw=1.1, zorder=2)
    ax.text(max(xs) * 1.45, 0.0, "collision boundary", color=COLLISION,
            fontsize=8.5, va="bottom", ha="right")

    ax.plot(xs, ys, color=SERIES, lw=2.0, marker="o", ms=6.5,
            markerfacecolor=SURFACE, markeredgewidth=1.7, zorder=4)

    success_y = ylo + 0.20 * (yhi - ylo)
    failure_y = ylo + 0.10 * (yhi - ylo)
    for start, end in contiguous_segments(xs, reached, True):
        ax.plot([start, end], [success_y, success_y], color=SUCCESS, lw=5.0,
                solid_capstyle="round", zorder=3)
    for start, end in contiguous_segments(xs, reached, False):
        ax.plot([start, end], [failure_y, failure_y], color=FAILURE, lw=5.0,
                solid_capstyle="round", zorder=3)

    for x, ok in zip(xs, reached):
        ax.plot([x], [success_y if ok else failure_y], marker="|", ms=12,
                color=SUCCESS if ok else FAILURE, markeredgewidth=2.0, zorder=5)

    ax.text(min(xs) * 0.72, success_y, "task succeeds", color=SUCCESS,
            fontsize=9, fontweight="bold", va="center", ha="left")
    ax.text(min(xs) * 0.72, failure_y, "task fails", color=FAILURE,
            fontsize=9, fontweight="bold", va="center", ha="left")

    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:g}" for x in xs], fontsize=8, rotation=45,
                       ha="right", rotation_mode="anchor")
    ax.minorticks_off()
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=1)
    ax.set_axisbelow(True)

    ax.set_xlabel(r"SSA safety demand $\eta$", color=INK_2, fontsize=10)
    ax.set_ylabel("minimum clearance over the run (m)", color=INK_2, fontsize=10)
    ax.set_title(r"Clearance under the fixed insertion as $\eta$ changes",
                 color=INK, fontsize=12.5, fontweight="bold", loc="left", pad=12)
    ax.tick_params(colors=MUTED, labelsize=9, which="both")

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)

    handles = [
        Line2D([], [], color=SERIES, lw=2.0, marker="o", ms=6.5,
               markerfacecolor=SURFACE, markeredgewidth=1.7,
               label="minimum clearance"),
        Line2D([], [], color=SUCCESS, lw=5.0, solid_capstyle="round",
               label="eta samples where task succeeds"),
        Line2D([], [], color=FAILURE, lw=5.0, solid_capstyle="round",
               label="eta samples where task fails"),
        Line2D([], [], color=COLLISION, lw=1.1, label="collision boundary"),
    ]
    legend = ax.legend(handles=handles, loc="upper left", frameon=False,
                       fontsize=8.5, handletextpad=0.7)
    for text in legend.get_texts():
        text.set_color(INK_2)

    fig.tight_layout()
    fig.savefig(args.out, format="pdf", facecolor=SURFACE, bbox_inches="tight")
    png = os.path.splitext(args.out)[0] + ".png"
    fig.savefig(png, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {args.out}\n      {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
