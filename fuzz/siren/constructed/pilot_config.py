"""Pilot: does a single fixed configuration discriminate between filters?

The corpus design tests a filter-independent set of candidate (E, G1') pairs
against every filter and classifies each pair three ways. The open question is
whether one configuration suffices, or whether the evaluation needs an
algorithm x configuration matrix. That turns on whether the per-filter attack
rates SEPARATE at a fixed configuration: if every filter scores 0, or all score
alike, a single configuration cannot rank them and the matrix is required.

Both configurations already exist with provenance:

    trial1_spark_default   SPARK's shipped values (eta 0.1, d_min 0.1, phi_k 1.0)
    trial2_ours            this project's values (eta 0.02, d_min 0.02, phi_k 0.1)

Classification per (candidate, filter, configuration), in this order:

    INELIGIBLE   the legitimate task [G0, G1] does not reach, or loses
                 clearance, in this scene under this filter. There is no
                 insertion to observe -- the filter already fails the plain
                 task -- so the pair is excluded from the denominator.
    ATTACK       [G0, G1] is safe and [G0, G1', G1] makes contact on leg 2.
    MODIFICATION contact on leg 1: the detour itself collides, which is a
                 different attack class and not counted as an insertion.
    DEFENDED     [G0, G1] safe and no contact.

Counting INELIGIBLE as "not vulnerable" would flatter the worst filters, so the
rate reported is ATTACK / (eligible), with the ineligible count kept alongside.

    python -m fuzz.siren.constructed.pilot_config --n 24
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np

from .separated_search import CASE, PARK
from .stock_radius import R_STOCK, SPOTS, legwise, run

OUT = "fuzz/siren/constructed/pilot"


def sha(path):
    try:
        return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
    except OSError:
        return None


def classify(world_fn, G0, G1, G1p, pos_w, steps):
    """Three-way outcome for one (candidate, filter, configuration)."""
    b = run(world_fn(), [G0, G1], pos_w, R_STOCK, steps)
    if b is None:
        return "SOLVER_FAIL", {}
    bper, bhit, bn = legwise(b)
    bmin = min(bper.values()) if bper else np.inf
    if b.label != "REACHED" or bmin <= 0:
        return "INELIGIBLE", {"baseline_label": b.label,
                              "baseline_min": float(bmin)}
    a = run(world_fn(), [G0, G1p, G1], pos_w, R_STOCK, steps)
    if a is None:
        return "SOLVER_FAIL", {"baseline_min": float(bmin)}
    aper, ahit, an = legwise(a)
    info = {"baseline_label": b.label, "baseline_min": float(bmin),
            "baseline_steps": bn, "attack_label": a.label,
            "leg1_min": float(aper.get(1, np.inf)),
            "leg2_min": float(aper.get(2, np.inf)),
            "attack_steps": an}
    if ahit == 2:
        return "ATTACK", info
    if ahit == 1:
        return "MODIFICATION", info
    return "DEFENDED", info


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--configs",
                   default="trial1_spark_default,trial2_ours")
    p.add_argument("--algos", default="ssa,rssa,pssa,cbf,rcbf,sss,rsss")
    p.add_argument("--n", type=int, default=24,
                   help="candidates sampled from the corpus, spread over gap")
    p.add_argument("--steps", type=int, default=900)
    p.add_argument("--out", default=OUT)
    p.add_argument("--seed", type=int, default=1,
                   help="which per-seed corpus to read; SPOTS is a prefix")
    a = p.parse_args(argv)

    from ..pipeline import trialconf
    from ..world.run import World

    os.makedirs(a.out, exist_ok=True)
    spot_file = f"{SPOTS}_{a.seed}.json"
    s = json.load(open(spot_file))
    G0 = np.asarray(s["G0"], float)
    G1 = np.asarray(s["G1"], float)
    seed = s["seed"]
    spots = sorted(s["spots"], key=lambda x: x["gap"])
    idx = np.linspace(0, len(spots) - 1, min(a.n, len(spots))).astype(int)
    cand = [spots[i] for i in sorted(set(idx.tolist()))]
    print(f"corpus {len(s['spots'])} candidates, {len(cand)} sampled "
          f"(gap {cand[0]['gap']:.4f} .. {cand[-1]['gap']:.4f})", flush=True)

    algos = a.algos.split(",")
    configs = a.configs.split(",")
    w0 = World.build(seed=seed,
                     spec=trialconf.spec_for(
                         trialconf.load("fuzz/siren/pipeline/configs/"
                                        "trial2_ours.yaml"), "ssa", CASE),
                     test_case=CASE, max_steps=a.steps)
    n_obs = len(w0.scene().obstacles_world)

    t0 = time.time()
    records = []
    for cname in configs:
        cfg = trialconf.load(f"fuzz/siren/pipeline/configs/{cname}.yaml")
        for algo in algos:
            spec = trialconf.spec_for(cfg, algo, CASE)
            tally = {}
            for ci, sp in enumerate(cand):
                G1p = np.asarray(sp["G1_prime"], float)
                pos_w = ([np.asarray(sp["spot"], float)]
                         + [PARK] * (n_obs - 1))

                def wf(spec=spec):
                    return World.build(seed=seed, spec=spec, test_case=CASE,
                                       max_steps=a.steps)

                out, info = classify(wf, G0, G1, G1p, pos_w, a.steps)
                tally[out] = tally.get(out, 0) + 1
                records.append({"config": cname, "algo": algo,
                                "candidate": ci, "gap": sp["gap"],
                                "G1_prime": [float(x) for x in G1p],
                                "spot": [float(x) for x in sp["spot"]],
                                "outcome": out, **info})
            elig = sum(v for k, v in tally.items()
                       if k in ("ATTACK", "MODIFICATION", "DEFENDED"))
            atk = tally.get("ATTACK", 0)
            print(f"  {cname:<22} {algo:<5} "
                  f"ATTACK {atk:3d}  MOD {tally.get('MODIFICATION',0):3d}  "
                  f"DEF {tally.get('DEFENDED',0):3d}  "
                  f"INELIG {tally.get('INELIGIBLE',0):3d}  "
                  f"FAIL {tally.get('SOLVER_FAIL',0):3d}  "
                  f"rate {atk/elig if elig else float('nan'):.3f}", flush=True)

    meta = {
        "experiment": "pilot_config",
        "purpose": "decide single fixed configuration vs algorithm x "
                   "configuration matrix",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.time() - t0, 1),
        "case": CASE, "seed": seed, "steps": a.steps,
        "obstacle_radius": R_STOCK, "n_obstacles_active": 1,
        "G0": G0.tolist(), "G1": G1.tolist(),
        "configs": configs, "algos": algos,
        "n_candidates_corpus": len(s["spots"]),
        "n_candidates_sampled": len(cand),
        "candidates": cand,
        "argv": sys.argv,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "hashes": {
            "corpus": sha(spot_file),
            "stock_radius.py": sha("fuzz/siren/constructed/stock_radius.py"),
            "separated_search.py":
                sha("fuzz/siren/constructed/separated_search.py"),
            "pilot_config.py": sha("fuzz/siren/constructed/pilot_config.py"),
            **{f"{c}.yaml": sha(f"fuzz/siren/pipeline/configs/{c}.yaml")
               for c in configs}},
    }
    with open(f"{a.out}/records.jsonl", "w") as fh:
        for r in records:
            fh.write(json.dumps(r, default=float) + "\n")
    json.dump(meta, open(f"{a.out}/meta.json", "w"), indent=2, default=float)
    print(f"\n{len(records)} records -> {a.out}/records.jsonl", flush=True)
    print(f"metadata -> {a.out}/meta.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
