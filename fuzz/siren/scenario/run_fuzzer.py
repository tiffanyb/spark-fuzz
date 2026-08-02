"""
Point SIREN at a target and score it against ground truth.

Every negative result in this project has the same ambiguity: an empty search
could mean no attack exists, or that the search cannot find one. A target
removes it — an attack is known to exist, at known coordinates, and the search
is never told where.

The schedule carries G0 as a fixed prefix waypoint, so the legs are

    leg 0   home -> G0        the setup, identical every run
    leg 1   G0 -> G1'         reaching the inserted goal
    leg 2   G1' -> G1         the attack leg

A candidate counts as an attack only if contact lands on leg 2. Contact on an
earlier leg means the robot crashed on the way and the inserted goal is
incidental — the same distinction that mattered when verifying the controls.

Reported per picker:

    attacks found        how many of the budget broke the task
    evals to first hit   the metric guidance is supposed to improve
    distance to truth    how close the best hit is to the known G1'

`G1_prime_truth` is loaded but never passed to the search; it is used only to
score afterwards.

EVERY evaluation is logged, not just the attacks -- see eval_record(). The
output carries an `evaluations` list per picker holding the candidate, its
score, outcome label, contact leg, clearance, and distance to truth, plus the
candidates rejected as inadmissible before costing a rollout. That is what makes
the search trace reconstructable afterwards rather than only its hit rate.

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 python -m fuzz.siren.scenario.run_fuzzer \
        --target fuzz/siren/scenario/targets/<name>.json --budget 60
"""

import argparse
import json
import time

import numpy as np


def evaluate_candidate(world, G0, cand, G1, steps):
    """Run [G0, cand, G1] and decide whether it is an attack.

    Returns (is_attack, record, contact_leg). The score mirrors the pipeline's
    lexicographic rule: engaged candidates outrank unengaged ones, and within
    each group the braking margin (or clearance) decides.
    """
    rec = world.run([np.asarray(G0, float), np.asarray(cand, float),
                     np.asarray(G1, float)], max_steps=steps)
    leg2 = [s for s in rec.steps if s.wp_idx >= 2]
    contact_leg = None
    for s in rec.steps:
        if s.clearance < 0.0:
            contact_leg = int(s.wp_idx)
            break
    is_attack = (rec.label == "COLLISION" and contact_leg == 2)

    engaged = [s for s in leg2 if s.engaged]
    if engaged:
        bm = [float(s.brake_margin) for s in leg2
              if np.isfinite(s.brake_margin)]
        score = 1000.0 - (min(bm) if bm else 0.0)
    else:
        cl = [s.clearance for s in leg2 if np.isfinite(s.clearance)]
        score = -float(min(cl)) if cl else 0.0
    return is_attack, rec, contact_leg, score, bool(engaged)


def eval_record(cand, n_eval, is_attack, rec, contact_leg, score, engaged,
                truth):
    """One row per candidate EVALUATED, attack or not.

    Recording only the attacks -- what this script did originally -- throws away
    the majority of every run (98 of 180 on the first target) and keeps no score
    at all, so the objective landscape cannot be reconstructed afterwards. That
    makes it impossible to ask whether CEM's mean actually migrated toward the
    attack region or whether BO's surrogate fit anything real; only the hit rate
    survives, which is the one number that does not distinguish the optimisers.
    """
    return {
        "eval": int(n_eval),
        "cand": [float(x) for x in np.asarray(cand, float).reshape(-1)],
        "score": float(score),
        "is_attack": bool(is_attack),
        "label": rec.label,
        "contact_leg": None if contact_leg is None else int(contact_leg),
        "min_clearance": float(rec.min_clearance),
        "n_steps": int(rec.n_steps),
        "n_gave_up": int(getattr(rec, "n_gave_up", 0)),
        "engaged": bool(engaged),
        "dist_to_truth": float(np.linalg.norm(
            np.asarray(cand, float).reshape(-1) - np.asarray(truth, float))),
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--target", default=None)
    p.add_argument("--pickers", default="random,cem,bo")
    p.add_argument("--budget", type=int, default=60)
    p.add_argument("--batch", type=int, default=10)
    p.add_argument("--search-seeds", default="0")
    p.add_argument("--lam", type=float, default=None,
                   help="override the target's lambda (CBF demand gain)")
    p.add_argument("--open-search", action="store_true",
                   help="proceed even if the planted attack does not reproduce")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    from .targets import load_target, list_targets
    from ..search.pick import make_picker, is_admissible

    tpath = a.target or (list_targets() or [None])[0]
    if tpath is None:
        print("no targets found")
        return 1
    w, tgt = load_target(tpath)
    if a.lam is not None and a.lam != tgt["lam"]:
        # rebuild under a different demand gain. The target's G0 was searched at
        # its ORIGINAL lambda, so at a new gain it is just an admissible
        # handover, not a tuned one -- and G1_prime_truth is no longer a known
        # answer. Both facts are recorded in the output.
        from ..world.run import World
        from ..world.types import real_filter
        spec = real_filter(algo=tgt["algo"], index=tgt["index"],
                           d_min=tgt["d_min"], eta=tgt["eta"], lam=a.lam,
                           k=tgt["k"])
        w = World.build(seed=tgt["seed"], spec=spec, test_case=tgt["case"],
                        max_steps=tgt["max_steps"])
        print(f"  lambda overridden: {tgt['lam']} -> {a.lam}")
    sc = w.scene()
    steps = tgt["max_steps"]
    G0 = np.asarray(tgt["G0_commanded"], float)
    G1 = np.asarray(tgt["G1"], float)
    truth = np.asarray(tgt["G1_prime_truth"], float)

    print(f"target {tgt['name']}")
    print(f"  {tgt['case']}  filter={tgt['algo']}  seed={tgt['seed']}")
    print(f"  G0 {np.round(G0,3)}   G1 {np.round(G1,3)}   "
          f"|v| at G0 = {tgt['joint_speed_at_G0']:.3f}")

    base = w.run([G0, G1], max_steps=steps)
    known = w.run([G0, truth, G1], max_steps=steps)
    print(f"  baseline [G0,G1]        {base.label}  ({base.n_steps} steps)")
    print(f"  known attack [G0,G1*,G1] {known.label}  "
          f"clearance {known.min_clearance:+.6f}")
    # The baseline MUST reach: without that the scene is broken and a collision
    # says nothing about the inserted goal. The planted attack reproducing is a
    # separate matter -- required when rediscovery is being scored, meaningless
    # when the question is whether ANY attack exists under a changed filter.
    if base.label != "REACHED":
        print("  baseline does not reach — aborting (scene is broken)")
        return 1
    open_search = known.label != "COLLISION"
    if open_search and not a.open_search:
        print("  planted attack does not reproduce — aborting "
              "(pass --open-search to search anyway)")
        return 1
    if open_search:
        print("  OPEN SEARCH: no planted attack under this filter, so "
              "'distance to truth'\n               is not a rediscovery "
              "metric here — it only says where a hit sits\n               "
              "relative to the goal that worked at the original gain.")
    print(f"  (the search is NOT told G1* = {np.round(truth,3)})\n", flush=True)

    rows = []
    for pname in [x.strip() for x in a.pickers.split(",") if x.strip()]:
        for ss in [int(x) for x in a.search_seeds.split(",") if x.strip()]:
            picker = make_picker(pname, sc, seed=ss)
            t0 = time.time()
            n_eval, first_hit, hits, evals, n_inadm = 0, None, [], [], 0
            while n_eval < a.budget:
                want = min(a.batch, a.budget - n_eval)
                cands = picker.ask(want)
                if not cands:
                    break
                scored = []
                for c in cands:
                    if not is_admissible(c, sc)[0]:
                        # does not consume budget, but IS part of the search
                        # trace: an arm that proposes many out-of-bounds points
                        # is behaving differently even at equal hit rate
                        n_inadm += 1
                        evals.append({"eval": None, "stage": "inadmissible",
                                      "cand": [float(x) for x in
                                               np.asarray(c, float).reshape(-1)]})
                        continue
                    atk, rec, leg, score, eng = evaluate_candidate(
                        w, G0, c, G1, steps)
                    n_eval += 1
                    scored.append((c, score))
                    evals.append(dict(stage="evaluated",
                                      **eval_record(c, n_eval, atk, rec, leg,
                                                    score, eng, truth)))
                    if atk:
                        d = float(np.linalg.norm(np.asarray(c) - truth))
                        hits.append({"cand": list(map(float, c)),
                                     "eval": n_eval,
                                     "dist_to_truth": d,
                                     "min_clearance": float(rec.min_clearance)})
                        if first_hit is None:
                            first_hit = n_eval
                            print(f"    {pname}/s{ss}: FIRST HIT at eval "
                                  f"{n_eval}, {np.round(c,3)}, "
                                  f"{d:.3f} m from truth", flush=True)
                    if n_eval >= a.budget:
                        break
                if scored:
                    picker.tell(scored)
            best = min((h["dist_to_truth"] for h in hits), default=None)
            row = {"picker": pname, "search_seed": ss, "n_eval": n_eval,
                   "n_attacks": len(hits), "first_hit": first_hit,
                   "best_dist_to_truth": best, "n_inadmissible": n_inadm,
                   "elapsed_s": round(time.time() - t0, 1), "hits": hits,
                   "evaluations": evals}
            rows.append(row)
            print(f"  {pname:<14} s{ss}  attacks {len(hits):>3}/{n_eval}  "
                  f"first hit {str(first_hit):>6}  closest to truth "
                  f"{'%.3f' % best if best is not None else '   -  '} m  "
                  f"({row['elapsed_s']}s, {n_inadm} inadmissible)", flush=True)

    print(f"\n{'='*78}")
    print(f"{'picker':<16}{'attacks':>9}{'rate':>8}{'1st hit':>9}"
          f"{'closest to truth':>19}{'logged':>9}")
    for r in rows:
        rate = f"{100*r['n_attacks']/max(1,r['n_eval']):.0f}%"
        print(f"{r['picker']:<16}{r['n_attacks']:>9}{rate:>8}"
              f"{str(r['first_hit']):>9}"
              f"{('%.3f m' % r['best_dist_to_truth']) if r['best_dist_to_truth'] is not None else '-':>19}"
              f"{len(r['evaluations']):>9}")
    out = a.out or f"/tmp/fuzz_{tgt['name']}.json"
    json.dump({"target": tgt["name"], "case": tgt["case"], "algo": tgt["algo"],
               "budget": a.budget, "G1_prime_truth": truth.tolist(),
               "lam": a.lam if a.lam is not None else tgt["lam"],
               "lam_original": tgt["lam"],
               "open_search": bool(open_search),
               "baseline_label": base.label, "baseline_steps": int(base.n_steps),
               "planted_attack_label": known.label,
               "results": rows}, open(out, "w"), indent=2, default=float)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
