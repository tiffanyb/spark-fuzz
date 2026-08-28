"""Plot eta sweep episode length with task success/failure regions.

Reads data/eta_sweep.csv and writes data/eta_steps_task.pdf/png. The main series is the
number of simulation steps. Red/green vertical regions show where the task fails
or succeeds for the sampled eta values.
"""

import argparse
import csv
import math
import os

import matplotlib

# matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

SURFACE = "#ffffff"
INK = "#0b0b0b"
GRID = "#d9d9d9"
SUCCESS = "#006400"
FAILURE = "#8B0000"

FONTSIZE = 18

def load_rows(path, algo):
    rows = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            if row["algo"] != algo:
                continue
            rows.append({
                "eta": float(row["eta"]),
                "steps": int(row["steps"]),
                "reached_final": bool(int(row["reached_final"])),
            })
    rows.sort(key=lambda row: row["eta"])
    return rows


def log_boundaries(xs):
    """Build region boundaries halfway between adjacent x values on a log axis."""
    logs = [math.log10(x) for x in xs]
    mids = [(logs[i] + logs[i + 1]) / 2.0 for i in range(len(logs) - 1)]
    left = logs[0] - (mids[0] - logs[0])
    right = logs[-1] + (logs[-1] - mids[-1])
    return [10 ** left] + [10 ** mid for mid in mids] + [10 ** right]


def log_midpoint(left, right):
    return 10 ** ((math.log10(left) + math.log10(right)) / 2.0)


def region_spans(bounds, flags):
    spans = []
    start = 0
    for i in range(1, len(flags)):
        if flags[i] != flags[start]:
            spans.append((bounds[start], bounds[i], flags[start]))
            start = i
    spans.append((bounds[start], bounds[len(flags)], flags[start]))
    return spans


def exp_label(x):
    exponent = int(math.floor(math.log10(x)))
    mantissa = x / (10 ** exponent)
    if abs(mantissa - 1.0) < 1e-9:
        return rf"$10^{{{exponent}}}$"
    return rf"${mantissa:g}\times10^{{{exponent}}}$"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=os.path.join(HERE, "data", "eta_sweep.csv"))
    parser.add_argument("--algo", default="ssa")
    parser.add_argument("--out", default=os.path.join(HERE, "data", "eta_steps_task.pdf"))
    args = parser.parse_args(argv)

    rows = load_rows(args.csv, args.algo)
    if not rows:
        raise SystemExit(f"no rows for algo={args.algo} in {args.csv}")

    xs = [row["eta"] for row in rows]
    ys = [row["steps"] for row in rows]
    reached = [row["reached_final"] for row in rows]
    bounds = log_boundaries(xs)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.8,
    })

    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for i, ok in enumerate(reached):
        ax.axvspan(bounds[i], bounds[i + 1],
                   color=SUCCESS if ok else FAILURE,
                   alpha=0.09 if ok else 0.12,
                   lw=0, zorder=0)

    ax.plot(xs, ys, color=INK, lw=1, marker="o", ms=3.5,
            markerfacecolor=INK, markeredgecolor=INK, markeredgewidth=1)

    ymax = max(ys)
    ax.set_ylim(0, ymax * 1.22)
    ax.set_xlim(bounds[0], bounds[-1])
    ax.set_xscale("log")
    max_exp = math.ceil(math.log10(max(xs)))
    decade_ticks = [10 ** exp for exp in range(-2, max_exp + 1)]
    ax.set_xticks(decade_ticks)
    ax.set_xticklabels([rf"$10^{{{exp}}}$" for exp in range(-2, max_exp + 1)],
                       fontsize=FONTSIZE)
    ax.minorticks_off()

    spans = region_spans(bounds, reached)
    for left, right, ok in spans:
        if ok:
            ax.text(log_midpoint(left, right), ymax * 1.04, "Task Succeeds",
                    color=SUCCESS, fontsize=FONTSIZE, va="top", ha="center")

    failure_spans = [(left, right) for left, right, ok in spans if not ok]
    if failure_spans:
        label_x = log_midpoint(failure_spans[0][0], failure_spans[-1][1])
        label_y = ymax * 1.2
        arrow_start_y = ymax * 1.10
        ax.text(label_x, label_y, "Task Fails", color=FAILURE,
                fontsize=FONTSIZE, va="top", ha="center")
        for left, right in failure_spans:
            ax.annotate(
                "",
                xy=(log_midpoint(left, right), ymax * 1.05),
                xytext=(label_x, arrow_start_y),
                arrowprops={
                    "arrowstyle": "->",
                    "color": FAILURE,
                    "lw": 1.0,
                    "shrinkA": 2,
                    "shrinkB": 2,
                },
            )

    ax.grid(axis="y", color=GRID, lw=0.7, zorder=1)
    ax.set_axisbelow(True)
    ax.set_xlabel(r"SSA Safety Parameter $\eta_{SSA}$",
                  color=INK, fontsize=FONTSIZE, fontname="Times New Roman")
    ax.set_ylabel("Simulation Steps",
                  color=INK, fontsize=FONTSIZE, fontname="Times New Roman")
    ax.tick_params(colors=INK, labelsize=FONTSIZE, which="both")

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
