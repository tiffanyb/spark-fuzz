"""Link every attack that SURVIVED re-simulation into rq3/dataset.

Symlinks, not copies: the record in its original directory stays the single
source of truth, so a link can never drift from what it points at. The link name
carries the identity that matters when reading a directory listing -- observed
kind, scenario case, filter, seed -- and the short content hash disambiguates
records that agree on all of those.

An attack appears here only if, re-run from scratch:
    the baseline [G0, G1] REACHED with clearance > 0 throughout, and
    [G0, G1', G1] was unsafe on EVERY repeat, on a consistent contact leg.
"""

import json, glob, os, collections

VER = "fuzz/siren/rq3/logs/verdicts"
DST = "fuzz/siren/rq3/dataset"


def main():
    os.makedirs(DST, exist_ok=True)
    for old in glob.glob(f"{DST}/*"):
        if os.path.islink(old):
            os.unlink(old)
    V = [json.load(open(f)) for f in glob.glob(f"{VER}/*.json")]
    valid = [d for d in V if d["verdict"] == "VALID"]
    n, by = 0, collections.Counter()
    index = {}
    for d in valid:
        kind = d.get("observed_kind", "UNKNOWN")
        name = (f"{kind}_{d['case']}_{d['algo']}_s{d['seed']}"
                f"_{d.get('channel','arm')}_{d['id'][:8]}.json")
        src = os.path.abspath(d["src"])
        link = os.path.join(DST, name)
        rel = os.path.relpath(src, DST)
        os.symlink(rel, link)
        index[name] = {
            "id": d["id"], "kind": kind, "claimed_kind": d.get("claimed_kind"),
            "case": d["case"], "algo": d["algo"], "seed": d["seed"],
            "channel": d.get("channel"), "source": d["src"],
            "duplicate_sources": d.get("all_sources", []),
            "baseline_label": d.get("baseline_label"),
            "baseline_min": d.get("baseline_min"),
            "attack_labels": d.get("attack_labels"),
            "attack_mins": d.get("attack_mins"),
            "kind_matches_claim": d.get("kind_matches_claim", True),
        }
        by[kind] += 1
        n += 1
    json.dump(index, open(f"{DST}/INDEX.json", "w"), indent=1, sort_keys=True)
    print(f"  linked {n} validated attacks -> {DST}")
    for k, c in by.most_common():
        print(f"    {c:>4}  {k}")
    algos = collections.Counter(v["algo"] for v in index.values())
    cases = collections.Counter(v["case"] for v in index.values())
    print(f"  filters: " + "  ".join(f"{a}={c}" for a, c in algos.most_common()))
    print(f"  cases  : {len(cases)} distinct")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
