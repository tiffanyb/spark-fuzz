"""
Is the D1 attack real, or only an artifact of a CBF asking for more than it has?

All five D1 targets fail the initial-state feasibility check (L2 in self_check):
at reset, mu = +0.17212 > 0, so no control satisfies all 95 active constraints.
At rest L_f = 0, so mu reduces to demand - authority, and the numbers reconstruct
exactly:

    D1/cbf   lambda*phi = 10 x 0.01989 = 0.19890,  C = 0.02677  ->  mu = +0.17213
    D2/pssa  eta        =         0.02          ,  C = 0.26772  ->  mu = -0.24772

So the D1 scene is infeasible because lambda = 10 demands phi decay ~7.4x faster
than the arm can deliver -- a property of the GAIN, not of the geometry. The same
scene is feasible under the SSA family's flat eta demand.

That makes the finding ambiguous in a way that matters: "D1/CBF is attackable"
and "we pointed an over-tuned CBF at a robot 0.11 mm from an obstacle" predict
the same 50 attacks. This separates them.

Feasibility at reset needs lambda*phi <= C, i.e. lambda <= 0.02677/0.01989 =
1.346. So sweep lambda below that and ask, for each target:

    mu@reset < 0        is the scene now L2-legitimate?
    [G0,G1]  REACHED    does the legitimate task still work?
    [G0,G1',G1] COLLIDE does the known attack still land?

If the attacks survive at a feasible lambda, D1/CBF is genuinely attackable and
the targets just need re-parameterising. If they vanish, the D1 result was about
the gain and should be retracted.

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 python -m fuzz.siren.scenario.retune_lambda
"""

import argparse
import glob
import json

import numpy as np


def probe_reset(w):
    """mu and engagement at the reset state, exactly as self_check's L2 does."""
    from ..world.sim import probe
    from ..world import derived
    h = w.harness
    af, ti = h.reset()
    clear = h.clearance(ti)
    u, ai = h.algo.act(af, ti)
    raw = probe.read_raw(h)
    if not raw:
        return dict(mu=np.nan, engaged=False, C=np.nan, phi=np.nan, clear=clear)
    d = derived.evaluate(raw["Lg"], raw["Lf"], raw["phi"], raw["phi_mask"],
                         raw["u_lim"], demand_shape=h.spec.demand_shape,
                         eta=h.spec.eta, lam=h.spec.lam, exact=True)
    phi = np.asarray(raw["phi"], float)
    mask = np.asarray(raw["phi_mask"], bool)
    act = phi[mask] if mask.any() else phi
    return dict(mu=float(d.get("mu", np.nan)), engaged=bool(d.get("engaged")),
                C=float(d.get("C_phi", np.nan)), phi=float(act.max()),
                clear=float(clear))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--targets", default="fuzz/siren/scenario/targets/*_D1_*.json")
    p.add_argument("--lambdas", default="10.0,1.3,1.0,0.5,0.2")
    p.add_argument("--out", default="fuzz/siren/experiment/cbf/retune_lambda.json")
    a = p.parse_args(argv)

    from ..world.run import World
    from ..world.types import real_filter

    lams = [float(x) for x in a.lambdas.split(",")]
    files = sorted(glob.glob(a.targets))
    print(f"{len(files)} D1 targets x {len(lams)} lambda values\n", flush=True)
    print(f"{'target':<38}{'lam':>6}{'mu@reset':>10}{'L2':>6}"
          f"{'[G0,G1]':>10}{'[G0,G1*,G1]':>13}{'clear':>11}", flush=True)

    rows = []
    for f in files:
        t = json.loads(open(f).read())
        tag = t["name"][-18:]
        G0 = np.asarray(t["G0_commanded"], float)
        G1 = np.asarray(t["G1"], float)
        G1p = np.asarray(t["G1_prime_truth"], float)
        steps = t["max_steps"]
        for lam in lams:
            spec = real_filter(algo=t["algo"], index=t["index"],
                               d_min=t["d_min"], eta=t["eta"], lam=lam,
                               k=t["k"])
            w = World.build(seed=t["seed"], spec=spec, test_case=t["case"],
                            max_steps=steps)
            r = probe_reset(w)
            l2 = not (r["engaged"] and np.isfinite(r["mu"]) and r["mu"] > 1e-9)

            base = w.run([G0, G1], max_steps=steps)
            atk = w.run([G0, G1p, G1], max_steps=steps)
            leg = None
            for s in atk.steps:
                if s.clearance < 0.0:
                    leg = int(s.wp_idx)
                    break

            rows.append({"target": t["name"], "algo": t["algo"], "lam": lam,
                         "mu": r["mu"], "C": r["C"], "phi": r["phi"],
                         "L2_pass": bool(l2),
                         "baseline": base.label, "baseline_steps": base.n_steps,
                         "attack": atk.label, "attack_leg": leg,
                         "attack_clearance": float(atk.min_clearance),
                         "baseline_gave_up": int(base.n_gave_up)})
            print(f"{tag:<38}{lam:>6.1f}{r['mu']:>+10.4f}"
                  f"{'pass' if l2 else 'FAIL':>6}{base.label:>10}"
                  f"{atk.label:>13}{atk.min_clearance:>+11.6f}", flush=True)

    json.dump(rows, open(a.out, "w"), indent=2, default=float)

    print(f"\n{'='*80}")
    print("Targets that are BOTH L2-legitimate AND still reproduce the attack:")
    good = [r for r in rows if r["L2_pass"] and r["baseline"] == "REACHED"
            and r["attack"] == "COLLISION" and r["attack_leg"] == 2]
    if good:
        for r in good:
            print(f"   lam={r['lam']:<5} {r['target']}  "
                  f"mu {r['mu']:+.4f}  clearance {r['attack_clearance']:+.6f}")
    else:
        print("   NONE -- at every feasible lambda the known attack stops "
              "reproducing.\n   The D1 result was about the gain, not the scene.")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
