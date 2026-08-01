"""
Read the experiment and answer the two questions it was run to answer.

    step 5  which search strategy should we use?
    step 6  is obstacle-anchored seeding worth anything?

Both are answered with PAIRED comparisons across scenes. Scene difficulty varies
enormously -- some scenes yield 20 collisions in 30 tries and most yield none --
so pooling arms and comparing means would mostly measure which arm happened to
land on the easy scenes. Every arm ran on every scene, so the difference WITHIN a
scene is the estimate that carries information, and the scene-to-scene spread
becomes a blocking factor rather than noise.

Significance is a two-sided sign test on the per-scene differences: it needs no
distributional assumption, and with counts this small and this skewed a t-test
would be leaning on a normality that plainly does not hold. Scenes where the two
arms tie contribute nothing, which is the standard (and conservative) treatment.

    python -m fuzz.siren.analyze_pipeline_experiment --in /tmp/pipeline_experiment.jsonl
"""

import argparse
import json
import math
from collections import defaultdict

import numpy as np

KINDS = ("KIND_1", "KIND_2", "NO_INFEASIBILITY", "NEVER_ENGAGED")


def sign_test(diffs):
    """Two-sided sign test. Returns (n_pos, n_neg, n_tie, p)."""
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    tie = sum(1 for d in diffs if d == 0)
    n = pos + neg
    if n == 0:
        return pos, neg, tie, 1.0
    k = min(pos, neg)
    # P(X <= k) under Binomial(n, 1/2), doubled
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return pos, neg, tie, min(1.0, 2.0 * tail)


def load(path):
    recs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="path", default="/tmp/pipeline_experiment.jsonl")
    p.add_argument("--metric", default="n_collisions_leg2",
                   help="summary key to rank arms by")
    a = p.parse_args(argv)

    recs = load(a.path)
    runs = [r for r in recs if r.get("status") == "ok"]
    gated = [r for r in recs if r.get("status") == "gated_out"]
    errs = [r for r in recs if r.get("status") == "error"]

    print("=" * 78)
    print("COVERAGE")
    print("=" * 78)
    scenes_run = {(r["case"], r["seed"]) for r in runs}
    scenes_gated = {(r["case"], r["seed"]) for r in gated}
    print(f"arm-runs ok: {len(runs)}   scenes used: {len(scenes_run)}   "
          f"scenes gated out: {len(scenes_gated)}   errors: {len(errs)}")
    if errs:
        seen = set()
        for r in errs:
            k = r.get("error", "?").split(":")[0]
            if k not in seen:
                seen.add(k)
                print(f"   error sample: {r.get('error')[:120]}")

    by_case = defaultdict(lambda: [0, 0])
    for c, s in scenes_run:
        by_case[c][0] += 1
    for c, s in scenes_gated:
        by_case[c][1] += 1
    print(f"\n{'case':<28}{'used':>6}{'gated':>7}   gate reasons")
    reasons = defaultdict(lambda: defaultdict(int))
    for r in gated:
        reasons[r["case"]][r["reason"].split("(")[0]] += 1
    for c in sorted(by_case):
        rs = ", ".join(f"{k}x{v}" for k, v in sorted(reasons[c].items()))
        print(f"{c:<28}{by_case[c][0]:>6}{by_case[c][1]:>7}   {rs}")

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("WHAT THE ATTACKS ACTUALLY ARE  (all arms pooled)")
    print("=" * 78)
    tot = defaultdict(int)
    for r in runs:
        for k, v in r["summary"].items():
            if isinstance(v, (int, float)) and v is not None and k != "attack_rate_leg2":
                tot[k] += v
    n_full = tot["n_full"]
    print(f"candidates fully evaluated (reached G1', continued to G1): {n_full}")
    print(f"  leg-2 collisions  (the INSERTION attack)   {tot['n_collisions_leg2']:>6}"
          f"  ({100*tot['n_collisions_leg2']/max(1,n_full):.1f}%)")
    print(f"  leg-1 collisions  (MODIFICATION hits)      {tot['n_collisions_leg1']:>6}")
    print(f"  leg-1 timeouts    (discarded)              {tot['n_leg1_timeout']:>6}")
    print(f"  runs where the filter gave up              {tot['n_gave_up_runs']:>6}")
    print(f"\n  kind of infeasibility, over evaluated candidates:")
    for k in KINDS:
        print(f"    {k:<20}{tot['kind_'+k]:>6}   of which collided: "
              f"{tot['kind_'+k+'_collided']:>5}")

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 78)
    print(f"STEP 5 — WHICH SEARCH STRATEGY?   (metric: {a.metric})")
    print("=" * 78)
    per = defaultdict(dict)                       # scene -> arm -> value
    for r in runs:
        per[(r["case"], r["seed"])][r["arm"]] = r["summary"].get(a.metric, 0)
    arms = sorted({r["arm"] for r in runs})

    print(f"{'arm':<16}{'total':>8}{'mean/scene':>12}{'scenes>0':>10}")
    totals = {}
    for arm in arms:
        vals = [v[arm] for v in per.values() if arm in v]
        totals[arm] = sum(vals)
        print(f"{arm:<16}{sum(vals):>8}{np.mean(vals):>12.2f}"
              f"{sum(1 for x in vals if x > 0):>10}")

    print(f"\npaired, scene by scene (positive => row arm better):")
    print(f"{'comparison':<34}{'mean diff':>11}{'W-L-T':>12}{'p':>9}")
    order = sorted(arms, key=lambda x: -totals[x])
    for i, a1 in enumerate(order):
        for a2 in order[i + 1:]:
            d = [v[a1] - v[a2] for v in per.values() if a1 in v and a2 in v]
            if not d:
                continue
            pos, neg, tie, pv = sign_test(d)
            star = "  *" if pv < 0.05 else ""
            print(f"{a1+' vs '+a2:<34}{np.mean(d):>+11.2f}"
                  f"{f'{pos}-{neg}-{tie}':>12}{pv:>9.3f}{star}")

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("STEP 6 — IS OBSTACLE SEEDING WORTH IT?")
    print("=" * 78)
    print(f"{'strategy':<14}{'unseeded':>10}{'seeded':>9}{'mean diff':>11}"
          f"{'W-L-T':>12}{'p':>9}")
    for base in ("random", "cem", "bo"):
        s, u = f"{base}_seeded", base
        d, su, uu = [], [], []
        for v in per.values():
            if s in v and u in v:
                d.append(v[s] - v[u])
                su.append(v[s])
                uu.append(v[u])
        if not d:
            continue
        pos, neg, tie, pv = sign_test(d)
        star = "  *" if pv < 0.05 else ""
        print(f"{base:<14}{sum(uu):>10}{sum(su):>9}{np.mean(d):>+11.2f}"
              f"{f'{pos}-{neg}-{tie}':>12}{pv:>9.3f}{star}")

    # pooled across the three strategies: does anchoring help AT ALL?
    dall = []
    for v in per.values():
        for base in ("random", "cem", "bo"):
            s, u = f"{base}_seeded", base
            if s in v and u in v:
                dall.append(v[s] - v[u])
    if dall:
        pos, neg, tie, pv = sign_test(dall)
        print(f"\npooled over all three strategies: mean diff {np.mean(dall):+.2f}, "
              f"W-L-T {pos}-{neg}-{tie}, p={pv:.4f}"
              f"{'  * seeding helps' if pv < 0.05 and np.mean(dall) > 0 else ''}")
    print("* = p < 0.05, two-sided sign test on per-scene differences")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
