"""
Two-stage authority-guided attack sweep over benchmark seeds.

Per seed:
  STAGE 1  c(x) SCREEN  -- does a control-authority violation even exist in this
           scene?  Probe reachable contacts (baseline + a few admissible one-hop
           goals) and take the lowest c(x) at any active constraint. If
           min_c < eta, a state that defeats SSA is reachable -> the seed is
           "vulnerable" and worth attacking. Otherwise skip it (cheap).

  STAGE 2  FUZZ (only if vulnerable) -- CEM over an admissible inserted goal G1',
           scoring each candidate by how LOW it drives the trajectory's min c
           (objective = -min_c over the two-hop rollout G0->G1'->G1). This is the
           dense, white-box c(x) objective derived in cx_derivation.pdf, replacing
           the old outcome-label fitness. Report the best G1' and whether it
           actually realizes a failure (collision / infeasible give-up).

Run (SPARK conda env, single-thread):
    python -m fuzz.authority_sweep --seeds 0-19 --eta 0.5 --out /tmp/authsweep.json
"""

import argparse
import json
import os

import numpy as np

from .config import build_single_arm_config
from .harness import SingleArmHarness
from .admissibility import sample_admissible, is_admissible
from . import authority as A


def _harness(seed, safe_algo, d_min, max_steps):
    cfg = build_single_arm_config(seed=seed, safe_algo=safe_algo,
                                  max_steps=max_steps, d_min_env=d_min)
    return SingleArmHarness(cfg)


# ----------------------------- Stage 1: screen ----------------------------- #
def _near_obstacle_goals(rng, sc, k):
    """Admissible goals placed just OUTSIDE each obstacle's keep-out shell, so
    reaching them drives a robot body up against the obstacle -> measures the
    low-c (near-contact) authority that random goals miss."""
    obs = sc["obstacles_world"]; keep = sc["keepout"]; base = sc["base_frame"]
    inv = np.linalg.inv(base)
    out = []
    if len(obs) == 0:
        return out
    for _ in range(k * 6):
        if len(out) >= k:
            break
        o = np.asarray(obs[rng.randint(len(obs))])[:3, 3]
        d = rng.normal(size=3); d /= (np.linalg.norm(d) + 1e-9)
        p_world = o + (keep + rng.uniform(0.005, 0.06)) * d
        p_base = (inv @ np.append(p_world, 1.0))[:3]
        ok, _ = is_admissible(p_base, sc["bounds"], sc["base_frame"], sc["obstacles_world"], sc["keepout"])
        if ok:
            out.append(p_base)
    return out


def screen_seed(seed, eta, n_probe=16, max_steps=200, safe_algo="ssa", d_min=0.02, seed_rng=0):
    """Return whether a c(x)<eta state is reachable in this seed's scene. Probes
    baseline + obstacle-biased + random admissible one-hop goals and tracks the
    lowest c(x) seen at any active (boundary) contact."""
    rng = np.random.RandomState(1000 + seed + seed_rng)
    h = _harness(seed, safe_algo, d_min, max_steps); sc = h.scene_info()
    G1 = sc["G1_base"]

    _, b = A.rollout(h, [G1], max_steps=max_steps, eta=eta, compute_mu=False)
    if not b["reached_final"]:
        return {"seed": seed, "valid": False, "reason": "baseline_unreached",
                "vulnerable": False, "min_c": float(b["min_c_over_traj"])}

    min_c = b["min_c_over_traj"]; where = "baseline"; probed = 0
    cands = _near_obstacle_goals(rng, sc, n_probe)            # bias toward obstacles
    while len(cands) < 2 * n_probe:                            # plus random admissible
        c = sample_admissible(rng, bounds=sc["bounds"], base_frame=sc["base_frame"],
                              obstacles_world=sc["obstacles_world"], keepout=sc["keepout"])
        if c is None:
            break
        cands.append(c)
    for cand in cands:
        _, s = A.rollout(h, [cand], max_steps=max_steps, eta=eta, compute_mu=False)
        probed += 1
        if s["min_c_over_traj"] < min_c:
            min_c = s["min_c_over_traj"]; where = f"onehop:{np.round(cand,3).tolist()}"
    return {"seed": seed, "valid": True, "baseline_reached": True,
            "min_c": float(min_c), "eta": float(eta),
            "vulnerable": bool(min_c < eta), "n_probed": probed, "where": where}


# ------------------------------ Stage 2: fuzz ------------------------------ #
def _admissible_gaussian(rng, mean, std, lo, hi, sc, want, cap=None):
    cap = cap if cap is not None else want * 50
    out = []; tries = 0
    while len(out) < want and tries < cap:
        tries += 1
        c = np.clip(rng.normal(mean, std), lo, hi)
        ok, _ = is_admissible(c, sc["bounds"], sc["base_frame"], sc["obstacles_world"], sc["keepout"])
        if ok:
            out.append(c)
    return out


def fuzz_seed(seed, eta, iters=6, pop=8, elite=3, init_std=0.06,
              max_steps=300, safe_algo="ssa", d_min=0.02, seed_rng=0):
    """CEM over G1' maximizing (-min_c): drive the trajectory's lowest control
    authority as far below eta as possible. Returns the best realized candidate."""
    rng = np.random.RandomState(7000 + seed + seed_rng)
    h = _harness(seed, safe_algo, d_min, max_steps); sc = h.scene_info()
    G1 = sc["G1_base"]; bounds = sc["bounds"]
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])

    _, b = A.rollout(h, [G1], max_steps=max_steps, eta=eta, compute_mu=False)
    if not b["reached_final"]:
        return {"seed": seed, "aborted": "baseline_unreached", "best": None}

    m0 = sample_admissible(rng, sc["bounds"], sc["base_frame"], sc["obstacles_world"], sc["keepout"])
    mean = m0 if m0 is not None else (lo + hi) / 2.0
    std = (hi - lo) * init_std
    best = None
    n_eval = 0
    for it in range(iters):
        cands = _admissible_gaussian(rng, mean, std, lo, hi, sc, pop)
        scored = []
        for c in cands:
            _, s1 = A.rollout(h, [c], max_steps=max_steps, eta=eta, compute_mu=False)
            if not s1["reached_final"]:
                scored.append((c, -1e9)); continue          # not an admissible (reachable) insert
            _, s = A.rollout(h, [c, G1], max_steps=max_steps, eta=eta, compute_mu=False)
            n_eval += 1
            score = -s["min_c_over_traj"]                    # maximize => minimize min_c
            scored.append((c, score))
            rec = {"candidate": [round(float(x), 4) for x in c],
                   "min_c": round(float(s["min_c_over_traj"]), 4),
                   "violation": bool(s["min_c_over_traj"] < eta),
                   "collided": bool(s["collided"]), "reached": bool(s["reached_final"]),
                   "n_infeasible": int(s["n_infeasible"]),
                   "first_infeasible": s["first_infeasible_step"]}
            if best is None or s["min_c_over_traj"] < best["min_c"]:
                best = rec
        scored.sort(key=lambda x: x[1], reverse=True)
        elites = np.array([c for c, sc_ in scored[:elite] if sc_ > -1e8])
        if len(elites) >= 2:
            mean = elites.mean(0); std = elites.std(0) + 1e-3
    return {"seed": seed, "best": best, "n_eval": n_eval}


# ------------------------------- the sweep -------------------------------- #
def sweep(seeds, eta, n_probe=16, iters=6, pop=8, safe_algo="ssa", d_min=0.02,
          screen_steps=200, fuzz_steps=300, verbose=True):
    results = []
    for sd in seeds:
        scr = screen_seed(sd, eta, n_probe=n_probe, max_steps=screen_steps,
                          safe_algo=safe_algo, d_min=d_min)
        rec = {"seed": sd, "screen": scr, "fuzz": None}
        if scr.get("valid") and scr.get("vulnerable"):
            fz = fuzz_seed(sd, eta, iters=iters, pop=pop, max_steps=fuzz_steps,
                           safe_algo=safe_algo, d_min=d_min)
            rec["fuzz"] = fz
        if verbose:
            b = rec["fuzz"]["best"] if (rec["fuzz"] and rec["fuzz"].get("best")) else None
            tag = "SKIP (not vulnerable)" if not scr.get("vulnerable") else (
                "no-realize" if not b else
                f"G1'={b['candidate']} min_c={b['min_c']} viol={b['violation']} collide={b['collided']} infeas@{b['first_infeasible']}")
            print(f"[seed {sd:2d}] valid={scr.get('valid')} min_c={scr.get('min_c'):.4f} "
                  f"vuln={scr.get('vulnerable')}  ->  {tag}", flush=True)
        results.append(rec)
    return results


def _parse_seeds(s):
    out = []
    for tok in s.split(","):
        if "-" in tok:
            a, b = tok.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return out


def main():
    ap = argparse.ArgumentParser(description="Authority-guided screen+fuzz over seeds")
    ap.add_argument("--seeds", default="0-19")
    ap.add_argument("--eta", type=float, default=0.5)
    ap.add_argument("--safe-algo", default="ssa")
    ap.add_argument("--d-min", type=float, default=0.02)
    ap.add_argument("--n-probe", type=int, default=16)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--pop", type=int, default=8)
    ap.add_argument("--screen-steps", type=int, default=200)
    ap.add_argument("--fuzz-steps", type=int, default=300)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    seeds = _parse_seeds(a.seeds)
    print(f"[authority-sweep] seeds={seeds} eta={a.eta} algo={a.safe_algo} d_min={a.d_min}", flush=True)
    res = sweep(seeds, a.eta, n_probe=a.n_probe, iters=a.iters, pop=a.pop,
                safe_algo=a.safe_algo, d_min=a.d_min,
                screen_steps=a.screen_steps, fuzz_steps=a.fuzz_steps)

    n_valid = sum(1 for r in res if r["screen"].get("valid"))
    n_vuln = sum(1 for r in res if r["screen"].get("vulnerable"))
    n_real = sum(1 for r in res if r["fuzz"] and r["fuzz"].get("best")
                 and r["fuzz"]["best"]["violation"])
    n_coll = sum(1 for r in res if r["fuzz"] and r["fuzz"].get("best")
                 and r["fuzz"]["best"]["collided"])
    print("\n==================== SWEEP SUMMARY ====================")
    print(f"seeds={len(seeds)}  valid={n_valid}  vulnerable(c<eta)={n_vuln}  "
          f"realized-violation={n_real}  realized-collision={n_coll}")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"args": vars(a), "seeds": seeds, "results": res}, f, indent=2)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
