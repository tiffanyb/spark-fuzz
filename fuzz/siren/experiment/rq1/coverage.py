"""Coverage of the attack corpus: does every (seed, filter) pair have an attack?

Attack files are named {algo}_{hits}_{seed}.json, so coverage is read straight
off the directory. Reports the matrix, the gaps, and -- for each gap -- whether
a placement corpus even exists for that seed, since "no corpus" and "corpus but
no attack" call for different follow-up.

    python -m fuzz.siren.constructed.coverage
    python -m fuzz.siren.constructed.coverage --seeds 1-10 --gaps-only
"""

import argparse
import glob
import json
import os
import re

FILTERS = ["ssa", "rssa", "pssa", "cbf", "rcbf", "sss", "rsss"]
RQ1 = "fuzz/siren/constructed/rq1_results"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="1-10")
    p.add_argument("--rq1", default=RQ1)
    p.add_argument("--filters", default=",".join(FILTERS))
    p.add_argument("--gaps-only", action="store_true")
    p.add_argument("--exclude", default="",
                   help="seeds to leave out of the coverage denominator, e.g. "
                        "a scene whose legitimate task finishes in one step and "
                        "so has no return leg to attack")
    a = p.parse_args(argv)
    lo, hi = (int(x) for x in a.seeds.split("-")) if "-" in a.seeds \
        else (int(a.seeds), int(a.seeds))
    seeds = list(range(lo, hi + 1))
    if a.exclude:
        drop = {int(x) for x in a.exclude.split(",") if x.strip()}
        seeds = [s for s in seeds if s not in drop]
    filters = a.filters.split(",")

    found = {}
    for f in glob.glob(f"{a.rq1}/stock_attacks/*.json"):
        m = re.match(r"([a-z]+)_(\d+)_(\d+)\.json$", os.path.basename(f))
        if m:
            found.setdefault((int(m.group(3)), m.group(1)), []).append(f)

    if not a.gaps_only:
        print("      " + "".join(f"{c:>6}" for c in filters))
        for s in seeds:
            row = "".join(f"{len(found.get((s, c), [])):>6}" for c in filters)
            corpus = os.path.exists(f"{a.rq1}/stock_spots_{s}.json")
            print(f"  s{s:<3d}{row}   {'' if corpus else '(no corpus)'}")

    gaps = [(s, c) for s in seeds for c in filters if not found.get((s, c))]
    cov = 1 - len(gaps) / (len(seeds) * len(filters))
    print(f"\ncoverage {len(seeds)*len(filters)-len(gaps)}/"
          f"{len(seeds)*len(filters)} = {cov:.1%}")
    if gaps:
        no_corpus = sorted({s for s, _ in gaps
                            if not os.path.exists(f"{a.rq1}/stock_spots_{s}.json")})
        print(f"{len(gaps)} gaps")
        by_seed = {}
        for s, c in gaps:
            by_seed.setdefault(s, []).append(c)
        for s in sorted(by_seed):
            tag = " [no corpus]" if s in no_corpus else ""
            print(f"   seed {s}: {' '.join(by_seed[s])}{tag}")
        print("\nre-hunt the gaps with:")
        for s in sorted(by_seed):
            if s in no_corpus:
                continue
            print(f"   SEED={s} ALLOW_PARALLEL=1 python -m "
                  f"fuzz.siren.constructed.run_rq1 --phase hunt --want 1 "
                  f"--seed {s} --algos {','.join(by_seed[s])} ...")
    return 0 if not gaps else 1


if __name__ == "__main__":
    raise SystemExit(main())
