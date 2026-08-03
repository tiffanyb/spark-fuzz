"""
Stage 4a -- discovery curves: attacks found vs trials, and vs wall-clock time.

Reads stage-3b output. No re-simulation: every evaluation was already logged.

    attacks vs TRIALS   the primary axis. One evaluation is one full rollout no
                        matter which strategy proposed it, so this is invariant
                        to machine load and to how many searches ran at once.
    attacks vs TIME     secondary, and only honest because stage 3b now stamps
                        each evaluation. Note it is NOT proportional to trials:
                        a rollout aborts at contact but a deadlock runs the full
                        horizon, so attacks are systematically cheaper than
                        misses.

Across search seeds report MEDIAN and inter-quartile band, not mean -- with a
handful of replicates one unlucky seed drags a mean badly.

The first `n_init` evaluations are a shared initial design identical across
strategies, so every curve is identical below that by construction. It is marked
in the output rather than left to look like agreement.

    python -m fuzz.siren.pipeline.stage4_curves --src '/abs/fuzz_*.json' \
        --out-dir /abs/analysis
"""

import argparse
import glob
import json
import os

import numpy as np

N_INIT = 8          # fuzz.siren.search.pick._N_INIT, the shared initial design


def curve_by_trials(evals, budget):
    """Cumulative attacks after each evaluation, length = budget."""
    ev = sorted((e for e in evals if e.get("stage") == "evaluated"),
                key=lambda e: e["eval"])
    out = np.zeros(budget, dtype=int)
    n = 0
    for e in ev:
        n += int(bool(e.get("is_attack")))
        i = int(e["eval"]) - 1
        if 0 <= i < budget:
            out[i] = n
    # forward-fill so the curve is a step function, not zeros where an
    # evaluation index happened to be skipped
    for i in range(1, budget):
        if out[i] == 0 and out[i-1] > 0:
            out[i] = out[i-1]
    return out


def curve_by_time(evals):
    """(t, cumulative attacks) pairs, from the per-evaluation timestamps."""
    ev = sorted((e for e in evals if e.get("stage") == "evaluated"),
                key=lambda e: e.get("t", 0.0))
    ts, ys, n = [], [], 0
    for e in ev:
        n += int(bool(e.get("is_attack")))
        ts.append(float(e.get("t", 0.0)))
        ys.append(n)
    return np.array(ts), np.array(ys, dtype=int)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--plot", action="store_true", default=True)
    a = p.parse_args(argv)

    os.makedirs(a.out_dir, exist_ok=True)
    files = sorted(glob.glob(a.src))
    print(f"{len(files)} fuzz results\n", flush=True)

    summary = []
    for f in files:
        r = json.load(open(f))
        budget = int(r["budget"])
        target = r["target"]
        per_strategy = {}
        for row in r["results"]:
            key = row["picker"]
            per_strategy.setdefault(key, []).append(row)

        print(f"{target}   kind={r.get('kind')}  hit_leg={r.get('hit_leg')}")
        print(f"  {'strategy':<16}{'attacks':>9}{'rate':>7}{'1st':>6}"
              f"{'inadm':>7}{'closest':>10}")
        entry = {"target": target, "case": r["case"], "algo": r["algo"],
                 "seed": r["seed"], "kind": r.get("kind"), "budget": budget,
                 "n_init": N_INIT, "strategies": {}}
        for name, rows in sorted(per_strategy.items()):
            curves = np.array([curve_by_trials(x["evaluations"], budget)
                               for x in rows])
            med = np.median(curves, axis=0)
            q1 = np.percentile(curves, 25, axis=0)
            q3 = np.percentile(curves, 75, axis=0)
            firsts = [x["first_hit"] for x in rows]
            n_att = [x["n_attacks"] for x in rows]
            inadm = [x.get("n_inadmissible", 0) for x in rows]
            best = [x["best_dist_to_truth"] for x in rows
                    if x["best_dist_to_truth"] is not None]
            tcs = [curve_by_time(x["evaluations"]) for x in rows]
            entry["strategies"][name] = {
                "n_seeds": len(rows),
                "attacks_per_seed": n_att,
                "median_attacks": float(np.median(n_att)),
                "first_hit_per_seed": firsts,
                "n_inadmissible": inadm,
                "closest_to_truth": (float(np.min(best)) if best else None),
                "curve_trials_median": med.tolist(),
                "curve_trials_q1": q1.tolist(),
                "curve_trials_q3": q3.tolist(),
                "curve_time": [{"t": t.tolist(), "y": y.tolist()}
                               for t, y in tcs]}
            fh = [x for x in firsts if x is not None]
            print(f"  {name:<16}{int(np.median(n_att)):>9}"
                  f"{100*np.median(n_att)/budget:>6.0f}%"
                  f"{(str(int(np.median(fh))) if fh else '-'):>6}"
                  f"{int(np.median(inadm)):>7}"
                  f"{(f'{min(best):.3f}' if best else '-'):>10}")
        summary.append(entry)
        print(flush=True)

    out = f"{a.out_dir}/curves.json"
    json.dump(summary, open(out, "w"), indent=2, default=float)
    print(f"wrote {out}")

    if a.plot:
        try:
            _plot(summary, a.out_dir)
        except Exception as e:
            print(f"plot skipped: {type(e).__name__}: {e}")
    return 0


def _plot(summary, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for entry in summary:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        cmap = plt.get_cmap("tab10")
        for i, (name, s) in enumerate(sorted(entry["strategies"].items())):
            med = np.array(s["curve_trials_median"])
            x = np.arange(1, len(med) + 1)
            c = cmap(i % 10)
            axes[0].step(x, med, where="post", label=name, color=c)
            axes[0].fill_between(x, s["curve_trials_q1"], s["curve_trials_q3"],
                                 step="post", alpha=0.15, color=c)
            for tc in s["curve_time"]:
                axes[1].step(tc["t"], tc["y"], where="post", alpha=0.7, color=c)
        # the shared initial design: identical for every strategy by
        # construction, so shade it rather than let it read as agreement
        axes[0].axvspan(1, entry["n_init"], color="0.85", zorder=0)
        axes[0].text(entry["n_init"] / 2, axes[0].get_ylim()[1] * 0.95,
                     "shared\ninit design", ha="center", va="top", fontsize=7,
                     color="0.35")
        axes[0].set_xlabel("trials (evaluations)")
        axes[0].set_ylabel("cumulative attacks found")
        axes[0].set_title(f"{entry['target']}\nattacks vs trials "
                          f"(median, IQR over {max(s['n_seeds'] for s in entry['strategies'].values())} seeds)",
                          fontsize=9)
        axes[0].legend(fontsize=7)
        axes[1].set_xlabel("wall-clock seconds")
        axes[1].set_ylabel("cumulative attacks found")
        axes[1].set_title("attacks vs time (per seed)", fontsize=9)
        for ax in axes:
            ax.grid(alpha=0.3)
        fig.tight_layout()
        path = f"{out_dir}/curves_{entry['target']}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        print(f"wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
