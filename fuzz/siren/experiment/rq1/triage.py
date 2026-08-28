"""Classify each confirmed attack by the MECHANISM that produced the contact.

The paper's thesis is safe-control INFEASIBILITY, so it matters whether a
collision happened because the QP had no solution or for some other reason. Three
mechanisms are distinguishable from what the harness already records:

    INFEASIBILITY     the QP could not be solved. SPARK's own `gave_up` flag
                      fires and it falls back to the unfiltered reference
                      control; equivalently the exact multi-constraint margin
                      mu > 0, meaning no single control satisfies every active
                      constraint at once.
    SLACK-VIOLATION   the QP stayed solvable, but a relaxed filter bought a
                      constraint violation with its slack variable. The
                      guarantee was sold, not broken.
    WEAK-TUNING       the QP stayed solvable, no slack was taken, and the filter
                      honoured its own constraint all the way to contact. The
                      demand was simply too weak to stop the arm in time --
                      a guarantee too weak to matter, not a violated one.

`brake_margin` is reported alongside because mu cannot see the second-order
effect: mu asks "can some control make phi fall at the demanded rate NOW", which
is nearly always yes, while the arm may already be inside its own stopping
distance. `phi` at the start of the return leg says whether the inserted goal
handed the arm over already inside the keep-out shell.

Filter parameters are read from each record, never from the config: sss and rsss
were found at lam=0.5 and do not reproduce at the config default of 10.

    python -m fuzz.siren.constructed.triage
"""

import argparse
import glob
import json
import os

import numpy as np

SLACK_EPS = 1e-6


def run_instrumented(world, schedule, pos_w, radius, steps):
    """Pinned-obstacle rollout with the exact feasibility margin computed."""
    from ..pipeline.stage1_search import set_channel
    from .big_obstacle import set_obstacles
    h = world.harness
    orig = h.reset

    def patched(*a, **k):
        af, ti = orig(*a, **k)
        set_obstacles(world, pos_w, radius)
        return af, h.env.task.get_info(af)

    h.reset = patched
    try:
        set_channel(world, "arm")
        return world.run([np.asarray(x, float) for x in schedule],
                         max_steps=steps, exact_margin=True)
    finally:
        h.reset = orig


def classify(rec):
    """Mechanism, plus the evidence for it."""
    steps = rec.steps
    contact = next((i for i, s in enumerate(steps) if s.clearance < 0.0), None)
    upto = steps[:contact + 1] if contact is not None else steps
    eng = [s for s in upto if getattr(s, "engaged", False)]
    gave = [i for i, s in enumerate(upto) if getattr(s, "gave_up", False)]
    mus = [float(s.mu) for s in eng if np.isfinite(getattr(s, "mu", np.nan))]
    slacks = [float(getattr(s, "slack", 0.0) or 0.0) for s in upto]
    gs = [float(s.g) for s in eng if np.isfinite(getattr(s, "g", np.nan))]
    leg2 = [s for s in upto if int(s.wp_idx) == 2]
    ev = {
        "contact_step": contact,
        "n_steps": len(steps),
        "n_engaged": len(eng),
        "n_gave_up": len(gave),
        "contact_on_gave_up": bool(contact is not None and contact in set(gave)),
        "max_mu": (max(mus) if mus else float("nan")),
        "min_g": (min(gs) if gs else float("nan")),
        "max_slack": (max(slacks) if slacks else 0.0),
        "brake_margin_at_contact": (float(steps[contact].brake_margin)
                                    if contact is not None else float("nan")),
        "phi_at_leg2_start": (float(leg2[0].phi) if leg2 else float("nan")),
        "leg2_min_clearance": (min((float(s.clearance) for s in leg2))
                               if leg2 else float("nan")),
        "label": rec.label,
    }
    if ev["n_gave_up"] > 0 or (mus and ev["max_mu"] > 0):
        ev["mechanism"] = "INFEASIBILITY"
    elif ev["max_slack"] > SLACK_EPS:
        ev["mechanism"] = "SLACK-VIOLATION"
    elif contact is not None:
        ev["mechanism"] = "WEAK-TUNING"
    else:
        ev["mechanism"] = "DID-NOT-REPRODUCE"
    return ev


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--src",
                   default="fuzz/siren/constructed/rq1_results/stock_attacks/*.json")
    p.add_argument("--out", default="fuzz/siren/constructed/TRIAGE.md")
    a = p.parse_args(argv)

    import gc

    from ..world.run import World
    from ..world.types import real_filter

    rows = []
    for f in sorted(glob.glob(a.src)):
        v = json.load(open(f))
        spec = real_filter(algo=v["algo"], index=v["index"], d_min=v["d_min"],
                           eta=v["eta"], lam=v["lam"], k=v["k"])
        w = World.build(seed=v["seed"], spec=spec, test_case=v["case"],
                        max_steps=v["max_steps"])
        rec = run_instrumented(
            w, [np.asarray(v["controls"][0]["G0"], float),
                np.asarray(v["G1_prime"], float), np.asarray(v["G1"], float)],
            [np.asarray(q, float) for q in v["obstacles_world"]],
            float(v["obstacle_radius"]), v["max_steps"])
        ev = classify(rec)
        ev.update(name=os.path.basename(f).replace(".json", ""),
                  algo=v["algo"], seed=v["seed"], eta=v["eta"], lam=v["lam"],
                  d_min=v["d_min"], k=v["k"])
        rows.append(ev)
        print(f"  {ev['name']:<16} {ev['mechanism']:<18} "
              f"gave_up={ev['n_gave_up']:<3} max_mu={ev['max_mu']:+.3f} "
              f"max_slack={ev['max_slack']:.4f} "
              f"brake@contact={ev['brake_margin_at_contact']:+.4f}", flush=True)
        w = None
        gc.collect()

    tally = {}
    for r in rows:
        tally[r["mechanism"]] = tally.get(r["mechanism"], 0) + 1
    lines = ["# Attack mechanism triage", "",
             f"{len(rows)} attacks in `{a.src}`.", "",
             "| attack | filter | seed | demand | d_min | k | gave_up | max mu |"
             " max slack | brake@contact | phi at leg-2 start | mechanism |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: (x["mechanism"], x["algo"])):
        dem = (f"eta={r['eta']:.4f}" if r["algo"] in ("ssa", "rssa", "pssa")
               else f"lam={r['lam']:.2f}")
        lines.append(
            f"| {r['name']} | {r['algo']} | {r['seed']} | {dem} | {r['d_min']} |"
            f" {r['k']} | {r['n_gave_up']} | {r['max_mu']:+.3f} |"
            f" {r['max_slack']:.4f} | {r['brake_margin_at_contact']:+.4f} |"
            f" {r['phi_at_leg2_start']:+.4f} | **{r['mechanism']}** |")
    lines += ["", "## Tally", ""]
    for k, n in sorted(tally.items()):
        lines.append(f"* **{k}**: {n}")
    open(a.out, "w").write("\n".join(lines) + "\n")
    print(f"\n  tally: {tally}")
    print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
