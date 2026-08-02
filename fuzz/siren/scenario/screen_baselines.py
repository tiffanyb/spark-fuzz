"""
Feasibility screen for candidate (scenario, filter, seed) triples.

The D1 fixed-base work was invalidated because its scene ran with lambda*phi > C
at EVERY step: the filter could not satisfy its own constraints anywhere, so the
collisions were guaranteed before any attacker existed. Nothing caught that until
after 50 attacks had been verified.

This runs the check first. For each triple it reports:

    mu@reset        infeasible at t=0?  (the L2 criterion)
    % infeasible    fraction of the baseline trajectory with mu > 0
    degenerate      did the run end at step <= 1 (start in contact / at goal)?
    committed@t0    was the state already past braking distance when the filter
                    first engaged?

A triple is usable target material when the collision is NOT explained by any of
these -- i.e. the filter was feasible, the run was real, and it had authority
when it first engaged and lost it anyway.

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 python -m fuzz.siren.scenario.screen_baselines \
        --cases R1LiteUpper_D2_AG_SO_v0 --algos cbf,rcbf,sss,rsss
"""

import argparse
import json

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", default="/tmp/colliding_baselines.json")
    p.add_argument("--cases", default=None, help="comma-separated; default all")
    p.add_argument("--algos", default="cbf,rcbf,sss,rsss")
    p.add_argument("--out", default="fuzz/siren/experiment/baseline_screen.json")
    a = p.parse_args(argv)

    from ..world.run import World
    from ..world.types import real_filter
    from ..world.sim import probe
    from ..world import derived

    cands = json.load(open(a.candidates))
    keep_algo = {x.strip() for x in a.algos.split(",") if x.strip()}
    cands = [c for c in cands if c["algo"] in keep_algo]
    if a.cases:
        keep_case = {x.strip() for x in a.cases.split(",") if x.strip()}
        cands = [c for c in cands if c["case"] in keep_case]
    cands.sort(key=lambda r: (r["case"], r["algo"], r["seed"]))
    print(f"{len(cands)} triples to screen\n", flush=True)
    print(f"{'case':<28}{'algo':<6}{'sd':>3}{'outcome':>10}{'steps':>7}"
          f"{'mu@reset':>10}{'%infeas':>9}{'m@t0':>9}{'verdict':>12}", flush=True)

    rows = []
    for c in cands:
        case, algo, seed = c["case"], c["algo"], c["seed"]
        index = "velocity" if "_D2_" in case else "distance"
        steps = c.get("max_steps") or (900 if "_D2_" in case else 1500)
        spec = real_filter(algo=algo, index=index, d_min=0.02, eta=0.02,
                           lam=10.0, k=0.1)
        try:
            w = World.build(seed=seed, spec=spec, test_case=case,
                            max_steps=steps)
            sc = w.scene()
            h = w.harness
        except Exception as e:
            print(f"  {case:<26}{algo:<6}{seed:>3}   build failed "
                  f"{type(e).__name__}", flush=True)
            continue

        af, ti = h.reset()
        u, ai = h.algo.act(af, ti)
        raw = probe.read_raw(h)
        mu0 = np.nan
        if raw:
            d = derived.evaluate(raw["Lg"], raw["Lf"], raw["phi"],
                                 raw["phi_mask"], raw["u_lim"],
                                 demand_shape=spec.demand_shape, eta=spec.eta,
                                 lam=spec.lam, exact=True)
            mu0 = float(d.get("mu", np.nan))
            if not bool(d.get("engaged")):
                mu0 = -np.inf          # no active constraint => trivially feasible

        try:
            # exact_margin=True is REQUIRED: mu is only computed on the exact LP
            # path. Without it every step reports mu=nan, the infeasibility
            # check silently passes, and a 100%-infeasible scene scores USABLE.
            rec = w.run([np.asarray(sc.G1, float)], max_steps=steps,
                        exact_margin=True)
        except Exception as e:
            print(f"  {case:<26}{algo:<6}{seed:>3}   run failed "
                  f"{type(e).__name__}", flush=True)
            continue

        mus = np.array([s.mu for s in rec.steps], float)
        fin = np.isfinite(mus)
        pct = 100.0 * (mus[fin] > 0).mean() if fin.any() else float("nan")
        bm = np.array([s.brake_margin for s in rec.steps], float)
        eng = np.array([bool(s.engaged) for s in rec.steps])
        t0 = int(np.argmax(eng)) if eng.any() else None
        m_t0 = float(bm[t0]) if t0 is not None else float("nan")
        degenerate = rec.n_steps <= 1

        if degenerate:
            verdict = "DEGENERATE"
        elif np.isfinite(pct) and pct > 50.0:
            verdict = "INFEASIBLE"
        elif np.isfinite(mu0) and mu0 > 1e-9:
            # fall back to the reset probe if the trajectory mu is unavailable,
            # rather than letting a missing measurement read as a pass
            verdict = "INFEASIBLE@t0"
        elif t0 is not None and m_t0 < 0:
            verdict = "PRE-DOOMED"
        elif rec.label in ("COLLISION", "DEADLOCK"):
            verdict = "USABLE"
        else:
            verdict = "no-failure"

        rows.append({"case": case, "algo": algo, "seed": seed,
                     "label": rec.label, "n_steps": int(rec.n_steps),
                     "mu_at_reset": mu0, "pct_infeasible": pct,
                     "margin_at_t0": m_t0, "degenerate": degenerate,
                     "min_clearance": float(rec.min_clearance),
                     "verdict": verdict})
        s_mu = "  idle" if np.isneginf(mu0) else f"{mu0:+.4f}"
        s_m0 = ("  none" if t0 is None else
                ("  +inf" if np.isinf(m_t0) else f"{m_t0:+.4f}"))
        print(f"  {case:<26}{algo:<6}{seed:>3}{rec.label:>10}{rec.n_steps:>7}"
              f"{s_mu:>10}{pct:>9.1f}{s_m0:>9}{verdict:>12}", flush=True)

    json.dump(rows, open(a.out, "w"), indent=2, default=float)
    usable = [r for r in rows if r["verdict"] == "USABLE"]
    print(f"\n{'='*92}")
    print(f"{len(usable)}/{len(rows)} triples USABLE as target material")
    for r in usable:
        print(f"   {r['case']:<28}{r['algo']:<6} seed {r['seed']}  "
              f"{r['label']}  clearance {r['min_clearance']:+.6f}")
    bad = {}
    for r in rows:
        if r["verdict"] != "USABLE":
            bad.setdefault(r["verdict"], []).append(
                f"{r['case']}/{r['algo']}/s{r['seed']}")
    for k, v in bad.items():
        print(f"\n   {k}: {len(v)}")
        for x in v[:6]:
            print(f"      {x}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
