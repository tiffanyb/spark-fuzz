"""
Validation of the guidance fixes.

Answers, on the same scene, with everything else held constant:

  P1  does the demand guard fire where it should, and pass where it should?
  P2  do goal attacks still exist once the demand is FEASIBLE -- i.e. against a
      filter that actually works, not one that is broken by construction?
      This must be a fresh SEARCH: the old candidates were found at eta=0.5 and
      are tuned to that regime, so replaying them proves nothing.
  P6  does the new proximity guidance beat the old margin guidance, and does
      either beat unguided random sampling?

The comparison that matters is guided-vs-random at equal budget. Anything that
does not beat random is not guidance.

    python -m fuzz.siren.validate_fixes --budget 40
"""

import argparse
import json
import time

import numpy as np

from .world.run import World
from .world.types import real_filter
from .search.attacks import make_attack
from .search.loop import search
from .search.pick import make_picker
from .search.threat import objective_for, threat_model, PRESETS


def index_for(test_case: str) -> str:
    """The safety index is dictated by the scene's control mode, not chosen.

    SPARK asserts that the distance index needs a Dynamic1 (velocity-controlled)
    robot and the velocity-augmented index needs a Dynamic2 (acceleration-
    controlled) one, so a D2 case hard-coded to "distance" aborts every cell with
    no_compatible_surrogate. Derive it from the case name instead.
    """
    return "velocity" if "_D2_" in test_case else "distance"


def run_cell(seed, test_case, eta, d_min, threat, guidance, picker, attack,
             budget, batch, max_steps, allow_infeasible):
    index = index_for(test_case)
    spec = real_filter(algo="ssa", index=index, d_min=d_min, eta=eta, k=0.1)
    world = World.build(seed=seed, spec=spec, test_case=test_case,
                        max_steps=max_steps)
    tm = threat_model(threat, d_min=d_min, index=index, real=spec)
    if guidance is not None:
        tm.objective = objective_for(PRESETS[threat], guidance=guidance)
    t0 = time.time()
    rep = search(world, make_attack(attack),
                 make_picker(picker, world.scene(), seed=0), tm,
                 budget=budget, batch=batch, max_steps=max_steps, verbose=False,
                 allow_infeasible_demand=allow_infeasible)
    return {
        "eta": eta, "threat": threat, "guidance": guidance, "picker": picker,
        "attack": attack,
        "aborted": rep.aborted,
        "demand_verdict": (rep.demand_check or {}).get("verdict"),
        "frac_feasible": (rep.demand_check or {}).get("frac_feasible_when_engaged"),
        "n_screened": rep.n_screened,
        "n_success": rep.n_success,
        "rate": (rep.n_success / rep.n_screened) if rep.n_screened else None,
        "elapsed": round(time.time() - t0, 1),
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=20)
    p.add_argument("--test-case", default="G1FixedBase_D1_AG_SO_v0")
    p.add_argument("--d-min", type=float, default=0.02)
    p.add_argument("--eta-broken", type=float, default=0.5)
    p.add_argument("--eta-working", type=float, default=0.02)
    p.add_argument("--budget", type=int, default=40)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--out", default="/tmp/validate_fixes.json")
    a = p.parse_args(argv)
    batch = max(4, a.budget // 5)
    rows = []

    # ---- P1: the guard --------------------------------------------------- #
    print("[P1] demand guard", flush=True)
    for eta in (a.eta_broken, a.eta_working):
        r = run_cell(a.seed, a.test_case, eta, a.d_min, "white", None,
                     "random", "insertion", 4, 4, a.max_steps, False)
        r["phase"] = "P1"
        rows.append(r)
        print(f"   eta={eta}: verdict={r['demand_verdict']} "
              f"feasible={100*(r['frac_feasible'] or 0):.0f}% "
              f"aborted={'YES' if r['aborted'] else 'no'}", flush=True)

    # ---- P2 + P6: fresh search at a FEASIBLE demand ---------------------- #
    print(f"\n[P2/P6] fresh search at eta={a.eta_working} (working filter), "
          f"budget={a.budget}", flush=True)
    plans = [
        ("white", "proximity", "cem"),
        ("white", "margin",    "cem"),
        ("white", None,        "random"),      # the control
        ("strict-black", "proximity", "cem"),
        ("strict-black", None,        "random"),
    ]
    for attack in ("insertion", "modification"):
        for threat, guidance, picker in plans:
            r = run_cell(a.seed, a.test_case, a.eta_working, a.d_min, threat,
                         guidance, picker, attack, a.budget, batch,
                         a.max_steps, False)
            r["phase"] = "P2/P6"
            rows.append(r)
            tag = f"{threat}/{guidance or 'none'}/{picker}"
            rate = f"{100*r['rate']:.0f}%" if r["rate"] is not None else "-"
            print(f"   {attack:<13}{tag:<32}"
                  f"{r['n_success']}/{r['n_screened']} = {rate:<7}"
                  f"({r['elapsed']}s)"
                  + (f"  ABORTED: {str(r['aborted'])[:40]}" if r["aborted"] else ""),
                  flush=True)

    # ---- report ---------------------------------------------------------- #
    print("\n" + "=" * 86)
    print("VALIDATION SUMMARY")
    print("=" * 86)

    p1 = [r for r in rows if r["phase"] == "P1"]
    broke = next((r for r in p1 if r["eta"] == a.eta_broken), None)
    work = next((r for r in p1 if r["eta"] == a.eta_working), None)
    print("P1  demand guard  (whether a demand is achievable is SCENE-SPECIFIC:")
    print("     the guard should track the measured feasibility, not the eta value)")
    for r in (broke, work):
        if not r:
            continue
        f = r["frac_feasible"] or 0.0
        blocked = bool(r["aborted"])
        # correct behaviour = block exactly when the filter mostly cannot operate
        should_block = f < 0.5
        ok = "correct" if blocked == should_block else "MISMATCH"
        print(f"      eta={r['eta']}: feasible at {100*f:.0f}% of engaged steps "
              f"-> {'blocked' if blocked else 'allowed'} ({ok})")

    print("\nP2  do attacks survive against a WORKING filter?")
    for attack in ("insertion", "modification"):
        sel = [r for r in rows if r["phase"] == "P2/P6" and r["attack"] == attack
               and not r["aborted"]]
        tot = sum(r["n_success"] for r in sel)
        scr = sum(r["n_screened"] for r in sel)
        print(f"      {attack:<14}{tot} attacks / {scr} screened"
              + ("   -> attacks DO survive" if tot else
                 "   -> NONE found at this budget"))

    print("\nP6  does guidance beat the random control? (same budget, same scene)")
    print(f"      {'attack':<14}{'random':<12}{'margin':<12}{'proximity':<12}verdict")
    for attack in ("insertion", "modification"):
        def rate_of(threat, guid, pick):
            r = next((x for x in rows if x["phase"] == "P2/P6"
                      and x["attack"] == attack and x["threat"] == threat
                      and x["guidance"] == guid and x["picker"] == pick), None)
            return (r["rate"] if r and r["rate"] is not None else None)
        rnd = rate_of("white", None, "random")
        mar = rate_of("white", "margin", "cem")
        pro = rate_of("white", "proximity", "cem")
        def f(x):
            return f"{100*x:.0f}%" if x is not None else "-"
        if rnd is None or pro is None:
            verdict = "inconclusive"
        elif pro > rnd and (mar is None or pro > mar):
            verdict = "proximity WINS"
        elif mar is not None and mar > rnd and mar > pro:
            verdict = "margin wins"
        elif pro <= rnd:
            verdict = "no better than random"
        else:
            verdict = "mixed"
        print(f"      {attack:<14}{f(rnd):<12}{f(mar):<12}{f(pro):<12}{verdict}")

    json.dump(rows, open(a.out, "w"), indent=2, default=float)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
