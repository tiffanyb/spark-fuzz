"""
CEM vs Bayesian optimisation vs random, on the same scene and the same budget.

The question, reduced. Both optimisers are given the SAME obstacle-anchored
initial design (see pick.obstacle_seed_points) and the same evaluation budget, so
this measures the optimiser and NOT the starting points. What is left to compare
is how each one spends the remaining budget after seeing the same first batch:

    CEM  fit a Gaussian to the elites and resample around them
    BO   fit a GP to everything and maximise Expected Improvement

Random keeps its usual role as the floor.

Why this is worth measuring rather than arguing. BO is the textbook fit for the
regime (3-D, expensive evaluations, tens of samples), but its model assumes a
smooth objective and this one demonstrably is not: the engagement gate is a step,
a collision truncates the run, and IK redundancy means nearby goals can resolve
to different arm branches. Whether the GP's global view beats CEM's mode-chasing
on a landscape that rough is an empirical question.

Two metrics, because they answer different things:

    hits / budget           how much of the region the search finds  (coverage)
    evaluations to 1st hit  how fast it finds anything               (efficiency)

Sample efficiency is the metric BO actually claims to improve, so it is the one
that decides this. Reported per search seed and paired across seeds, since scene
difficulty varies far more than the optimiser gap.

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    python -m fuzz.siren.compare_pickers --seed 1 --case G1FixedBase_D2_AG_SO_v0 \
        --eta 0.02 --budget 32 --reps 5
"""

import argparse
import json
import time

import numpy as np

from .search.attacks import make_attack
from .search.loop import search
from .search.pick import make_picker
from .search.threat import threat_model
from .world.run import World
from .world.types import real_filter

ARMS = ["random", "cem_seeded", "bo"]


def first_hit_index(report):
    """1-based evaluation number of the first success, or None if never."""
    for i, r in enumerate(report.results):
        if r.success:
            return i + 1
    return None


def run_arm(world, scene, spec, arm, rep_seed, budget, batch, max_steps, index):
    tm = threat_model("random", d_min=spec.d_min, index=index, real=spec)
    t0 = time.time()
    rep = search(world, make_attack("insertion"),
                 make_picker(arm, scene, seed=rep_seed), tm,
                 budget=budget, batch=batch, max_steps=max_steps, verbose=False)
    return {
        "arm": arm, "rep_seed": rep_seed,
        "aborted": rep.aborted,
        "screened": rep.n_screened,
        "hits": rep.n_success,
        "rate": (rep.n_success / rep.n_screened) if rep.n_screened else None,
        "first_hit": first_hit_index(rep),
        "elapsed_s": round(time.time() - t0, 1),
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1, help="SCENE seed")
    p.add_argument("--case", default="G1FixedBase_D2_AG_SO_v0")
    p.add_argument("--eta", type=float, default=0.02)
    p.add_argument("--d-min", type=float, default=0.02)
    p.add_argument("--budget", type=int, default=32)
    p.add_argument("--batch", type=int, default=8, help="must match picker n_init")
    p.add_argument("--reps", type=int, default=5, help="independent SEARCH seeds")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--out", default="/tmp/picker_comparison.json")
    a = p.parse_args(argv)

    index = "velocity" if "_D2_" in a.case else "distance"
    spec = real_filter(algo="ssa", index=index, d_min=a.d_min, eta=a.eta, k=0.1)
    world = World.build(seed=a.seed, spec=spec, test_case=a.case,
                        max_steps=a.max_steps)
    scene = world.scene()

    print(f"scene seed={a.seed} case={a.case} eta={a.eta} "
          f"budget={a.budget} batch={a.batch} reps={a.reps}", flush=True)
    print(f"{'arm':<12}{'rep':<6}{'hits':<8}{'rate':<9}{'1st hit':<10}{'secs':<8}",
          flush=True)

    rows = []
    for rep_seed in range(a.reps):
        for arm in ARMS:
            r = run_arm(world, scene, spec, arm, rep_seed,
                        a.budget, a.batch, a.max_steps, index)
            rows.append(r)
            fh = r["first_hit"]
            rate = "-" if r["rate"] is None else "{:.0f}%".format(100 * r["rate"])
            print(f"{arm:<12}{rep_seed:<6}"
                  f"{str(r['hits'])+'/'+str(r['screened']):<8}{rate:<9}"
                  f"{(str(fh) if fh else 'never'):<10}{r['elapsed_s']:<8}"
                  + (f"  ABORTED {r['aborted']}" if r["aborted"] else ""), flush=True)

    # ---------------- summary ------------------------------------------- #
    print("\n" + "=" * 72)
    print(f"{'arm':<12}{'mean hits':<12}{'mean rate':<12}{'median 1st hit':<16}"
          f"{'never found':<12}")
    print("-" * 72)
    summary = {}
    for arm in ARMS:
        rs = [r for r in rows if r["arm"] == arm and not r["aborted"]]
        if not rs:
            continue
        hits = [r["hits"] for r in rs]
        fhs = [r["first_hit"] for r in rs if r["first_hit"] is not None]
        never = sum(1 for r in rs if r["first_hit"] is None)
        # censored runs: report the median over the runs that DID find something
        # and the never-count alongside, rather than silently dropping failures.
        med = float(np.median(fhs)) if fhs else float("nan")
        summary[arm] = {"mean_hits": float(np.mean(hits)),
                        "mean_rate": float(np.mean([r["rate"] for r in rs])),
                        "median_first_hit": med, "never": never, "n": len(rs)}
        print(f"{arm:<12}{np.mean(hits):<12.1f}"
              f"{100*np.mean([r['rate'] for r in rs]):<11.0f}%"
              f"{(f'{med:.1f}' if fhs else 'n/a'):<16}{never}/{len(rs):<12}")
    print("=" * 72)

    # paired differences: scene difficulty varies far more than the optimiser gap,
    # so compare arms WITHIN a search seed rather than across pooled means.
    print("\npaired per-seed differences (positive = first arm better)")
    for a1, a2 in (("bo", "cem_seeded"), ("cem_seeded", "random"), ("bo", "random")):
        d_hits, d_fh = [], []
        for s in range(a.reps):
            r1 = next((r for r in rows if r["arm"] == a1 and r["rep_seed"] == s), None)
            r2 = next((r for r in rows if r["arm"] == a2 and r["rep_seed"] == s), None)
            if not r1 or not r2 or r1["aborted"] or r2["aborted"]:
                continue
            d_hits.append(r1["hits"] - r2["hits"])
            if r1["first_hit"] and r2["first_hit"]:
                d_fh.append(r2["first_hit"] - r1["first_hit"])   # fewer evals = better
        if d_hits:
            w = sum(1 for d in d_hits if d > 0)
            print(f"  {a1:<11} vs {a2:<11} hits {np.mean(d_hits):+.1f} "
                  f"(wins {w}/{len(d_hits)})"
                  + (f"   evals-to-first-hit {np.mean(d_fh):+.1f}" if d_fh else ""))

    json.dump({"scene_seed": a.seed, "case": a.case, "eta": a.eta,
               "budget": a.budget, "rows": rows, "summary": summary},
              open(a.out, "w"), indent=2, default=float)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
