"""
Aggregate the experiment matrix into a readable report.

    python -m fuzz.siren.summarize_experiment /tmp/siren_exp/*.json
"""

import glob
import json
import sys
from collections import defaultdict


def load(paths):
    cells = []
    for pat in paths:
        for p in sorted(glob.glob(pat)):
            try:
                cells.extend(json.load(open(p)).get("cells", []))
            except Exception as e:
                print(f"  (skipped {p}: {e})")
    return cells


def main(argv):
    paths = argv or ["/tmp/siren_exp/*.json"]
    cells = load(paths)
    if not cells:
        print("no cells found")
        return 1

    ok = [c for c in cells if c.get("ok")]
    err = [c for c in cells if not c.get("ok")]
    print("=" * 78)
    print(f"SIREN EXPERIMENT — {len(cells)} cells  ({len(ok)} ran, {len(err)} errored)")
    print("=" * 78)

    # ---- validity: did the scene even admit an attack? -------------------- #
    aborted = [c for c in ok if c.get("aborted")]
    valid = [c for c in ok if not c.get("aborted")]
    if aborted:
        print(f"\n{len(aborted)} cells skipped as INVALID scenes "
              f"(the legitimate goal was not reachable, so no attack is meaningful):")
        for key, n in sorted(defaultdict(int, {
                (c["test_case"], c["seed"]): 1 for c in aborted}).items()):
            pass
        seen = sorted({(c["test_case"], c["seed"], c.get("aborted")) for c in aborted})
        for tc, sd, why in seen:
            print(f"    {tc} seed={sd}: {why}")

    # ---- headline: any attacks at all? ------------------------------------ #
    hits = [c for c in valid if c.get("n_success", 0) > 0]
    print(f"\nHEADLINE: {len(hits)}/{len(valid)} valid cells produced at least one "
          f"confirmed goal attack.")
    total_screened = sum(c.get("n_screened", 0) for c in valid)
    total_success = sum(c.get("n_success", 0) for c in valid)
    print(f"          {total_success} attacks found across {total_screened} "
          f"screened candidates "
          f"({100.0*total_success/total_screened:.1f}%)" if total_screened else "")

    # ---- matrix: threat x attack ------------------------------------------ #
    print("\n" + "-" * 78)
    print("SUCCESS RATE BY THREAT MODEL x ATTACK  (attacks found / candidates screened)")
    print("-" * 78)
    threats = ["white", "gray", "gray-proportional", "weak-black", "strict-black", "random"]
    attacks = ["insertion", "modification"]
    print(f"{'threat':<20}", end="")
    for a in attacks:
        print(f"{a:>26}", end="")
    print()
    for t in threats:
        print(f"{t:<20}", end="")
        for a in attacks:
            sel = [c for c in valid if c["threat"] == t and c["attack"] == a]
            s = sum(c.get("n_success", 0) for c in sel)
            n = sum(c.get("n_screened", 0) for c in sel)
            cellstr = f"{s}/{n}" + (f" ({100.0*s/n:.0f}%)" if n else "")
            print(f"{cellstr:>26}", end="")
        print()

    # ---- matrix: test case ------------------------------------------------ #
    print("\n" + "-" * 78)
    print("BY SPARK TEST CASE")
    print("-" * 78)
    cases = sorted({c["test_case"] for c in valid})
    print(f"{'test case':<28}{'index':<10}{'baseline':<12}{'attacks':<12}{'screened':<10}")
    for tc in cases:
        sel = [c for c in valid if c["test_case"] == tc]
        s = sum(c.get("n_success", 0) for c in sel)
        n = sum(c.get("n_screened", 0) for c in sel)
        idx = sel[0].get("index", "?")
        base = sorted({str(c.get("baseline_label")) for c in sel})
        print(f"{tc:<28}{idx:<10}{','.join(base):<12}{s:<12}{n:<10}")

    # ---- guided vs random: is the SEARCH doing anything? ------------------ #
    #
    # A high success rate is not by itself evidence of a clever attack. If
    # unguided random sampling succeeds just as often, then almost every
    # admissible goal breaks that scene -- the scene is FRAGILE, and the search
    # deserves no credit. The random column is the control, and the only honest
    # way to claim the guidance works is to beat it.
    print("\n" + "-" * 78)
    print("GUIDED vs RANDOM  (per scene: is the search beating the control?)")
    print("-" * 78)
    print(f"{'test case':<26}{'attack':<14}{'random':<12}{'best guided':<14}{'verdict'}")
    scenes = sorted({(c["test_case"], c["seed"], c["attack"]) for c in valid})
    for tc, sd, at in scenes:
        sel = [c for c in valid if c["test_case"] == tc and c["seed"] == sd
               and c["attack"] == at]
        rnd = [c for c in sel if c["threat"] == "random"]
        gui = [c for c in sel if c["threat"] != "random"]
        if not rnd or not gui:
            continue
        rn = sum(c.get("n_screened", 0) for c in rnd)
        rs = sum(c.get("n_success", 0) for c in rnd)
        r_rate = (rs / rn) if rn else None
        best, best_t = None, None
        for c in gui:
            n, s = c.get("n_screened", 0), c.get("n_success", 0)
            if n and (best is None or s / n > best):
                best, best_t = s / n, c["threat"]
        if r_rate is None or best is None:
            continue
        if r_rate >= 0.9:
            verdict = "SCENE FRAGILE (random already ~always wins)"
        elif r_rate == 0 and best > 0:
            verdict = f"guidance REQUIRED ({best_t})"
        elif best > r_rate * 1.5:
            verdict = f"guidance helps ({best_t})"
        elif best <= r_rate:
            verdict = "no better than random"
        else:
            verdict = "marginal"
        print(f"{tc:<26}{at:<14}{r_rate*100:>5.0f}%      "
              f"{best*100:>5.0f}% {str(best_t or ''):<8} {verdict}")

    # ---- outcome kinds ---------------------------------------------------- #
    labels = defaultdict(int)
    for c in valid:
        for l in c.get("hit_labels", []):
            labels[l] += 1
    if labels:
        print("\nATTACK OUTCOME KINDS (cells reporting each):")
        for l, n in sorted(labels.items(), key=lambda kv: -kv[1]):
            print(f"    {l:<12} {n}")

    # ---- white box: mechanism -------------------------------------------- #
    print("\n" + "-" * 78)
    print("WHITE BOX — WHICH MECHANISM?")
    print("-" * 78)
    mechs = []
    for c in valid:
        if c["threat"] != "white":
            continue
        for m in c.get("mechanisms", []) or []:
            if "error" not in m:
                mechs.append((c, m))
    if not mechs:
        print("  no white-box hits to analyse")
    else:
        induced = [(c, m) for c, m in mechs if m.get("filter_induced")]
        intrinsic = [(c, m) for c, m in mechs if not m.get("filter_induced")]
        print(f"  analysed {len(mechs)} white-box attacks:")
        print(f"    FILTER-INDUCED  {len(induced):<4} "
              f"(every leg individually feasible; only the ORDER broke the filter)")
        print(f"    intrinsic       {len(intrinsic):<4} "
              f"(the inserted goal alone already strained the filter)")
        if induced:
            print("\n  Confirmed filter-induced infeasibility:")
            for c, m in induced[:12]:
                print(f"    {c['test_case']} seed={c['seed']} {c['attack']}: "
                      f"baseline={m['baseline_label']}/{m['baseline_gave_up']} "
                      f"onehop={m.get('onehop_label')}/{m.get('onehop_gave_up')} "
                      f"-> attack={m['attack_label']} "
                      f"gave_up={m['attack_gave_up']} min_g={m.get('attack_min_g')}")

    # ---- errors ---------------------------------------------------------- #
    if err:
        print("\n" + "-" * 78)
        print("ERRORS")
        print("-" * 78)
        seen = {}
        for c in err:
            seen.setdefault(c.get("error", "?"), []).append(
                f"{c['test_case']}/{c['attack']}/{c['threat']}")
        for e, where in seen.items():
            print(f"  {e}\n      {len(where)} cells, e.g. {where[:3]}")

    print("\n" + "=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
