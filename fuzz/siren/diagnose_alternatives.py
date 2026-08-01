"""
Which statistic COULD guide the search?

The first diagnostic showed the current objective (min over the trajectory of
g) carries no signal. This one measures a set of candidate replacements on the
same runs and reports which ones actually separate attacks from non-attacks.

Candidates tested, and the reasoning behind each:

  A  min_g over ENGAGED steps only        removes the inactive-constraint
                                          fallback (which is meaningless and,
                                          worse, inverted)
  B  n_engaged / n_gave_up                EXTENT of infeasibility rather than
                                          its depth -- how long the filter is
                                          stuck, not how far
  C  integral of (-g)+ over engaged       "infeasibility exposure": depth x time
  D  C, g, phi at the HANDOVER state      the configuration when the robot
                                          switches from G1' to G1. This is the
                                          ONE state the inserted goal actually
                                          controls -- every candidate then flies
                                          the same final leg to the same fixed
                                          G1, so anything dominated by that
                                          shared approach cannot discriminate
  E  min clearance on the final leg       geometric proximity, filter-free
  F  max_phi over engaged steps           how far inside the keep-out shell

Separation is reported as a standardised effect size (difference of means over
pooled standard deviation), so the statistics are comparable to one another.
"""

import argparse
import json

import numpy as np

from .world.run import World
from .world.types import real_filter


def stats_for(world, scene, cand, spec, max_steps):
    rec = world.run([np.asarray(cand), np.asarray(scene.G1)], spec,
                    max_steps=max_steps)
    steps = rec.steps
    if not steps:
        return None

    engaged = [s for s in steps if s.phi >= 0]
    final_leg = [s for s in steps if s.wp_idx >= 1]

    # the handover: first step on the final leg
    handover = final_leg[0] if final_leg else None

    def fin(v, default=np.nan):
        return float(v) if (v is not None and np.isfinite(v)) else default

    out = {
        "outcome": rec.label,
        "success": rec.is_attack_success(),
        # --- A: engaged-only depth ---
        "A_min_g_engaged": fin(min((s.g for s in engaged
                                    if np.isfinite(s.g)), default=np.nan)),
        # --- B: extent ---
        "B_n_engaged": len(engaged),
        "B_frac_engaged": len(engaged) / len(steps),
        "B_n_gave_up": rec.n_gave_up,
        # --- C: exposure = depth x time ---
        "C_exposure": float(sum(max(0.0, -s.g) for s in engaged
                                if np.isfinite(s.g))),
        # --- D: the handover state (what the inserted goal controls) ---
        "D_C_handover": fin(handover.C_phi if handover else None),
        "D_g_handover": fin(handover.g if handover else None),
        "D_phi_handover": fin(handover.phi if handover else None),
        "D_clear_handover": fin(handover.clearance if handover else None),
        # --- E: geometric proximity on the attack leg ---
        "E_min_clear_final": fin(min((s.clearance for s in final_leg
                                      if np.isfinite(s.clearance)), default=np.nan)),
        # --- F: penetration depth while engaged ---
        "F_max_phi_engaged": fin(max((s.phi for s in engaged
                                      if np.isfinite(s.phi)), default=np.nan)),
        "n_steps": len(steps),
    }
    return out


def effect_size(hits, misses, key):
    h = np.array([r[key] for r in hits if np.isfinite(r.get(key, np.nan))], dtype=float)
    m = np.array([r[key] for r in misses if np.isfinite(r.get(key, np.nan))], dtype=float)
    if len(h) < 2 or len(m) < 2:
        return None
    pooled = np.sqrt((h.var(ddof=1) + m.var(ddof=1)) / 2.0)
    if pooled < 1e-12:
        return {"h": h.mean(), "m": m.mean(), "d": 0.0, "sep": h.mean() - m.mean()}
    return {"h": h.mean(), "m": m.mean(),
            "d": (h.mean() - m.mean()) / pooled,
            "sep": h.mean() - m.mean()}


def ranking_power(rows, key, higher_is_better):
    """If we ranked candidates by this statistic alone, how many of the true
    attacks land in the top-K (K = number of attacks)? Chance = K*K/N."""
    vals = [(r[key], r["success"]) for r in rows if np.isfinite(r.get(key, np.nan))]
    if not vals:
        return None
    vals.sort(key=lambda t: t[0], reverse=higher_is_better)
    k = sum(1 for _, s in vals if s)
    if k == 0:
        return None
    top = sum(1 for _, s in vals[:k] if s)
    return top, k, len(vals), (k * k) / len(vals)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--from-json", default="/tmp/siren_w2.json")
    p.add_argument("--seed", type=int, default=20)
    p.add_argument("--test-case", default="G1FixedBase_D1_AG_SO_v0")
    p.add_argument("--n-miss", type=int, default=14)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--eta", type=float, default=0.5)
    p.add_argument("--d-min", type=float, default=0.02)
    p.add_argument("--out", default="/tmp/guidance_alternatives.json")
    a = p.parse_args(argv)

    data = json.load(open(a.from_json))["results"]
    hits = [r["candidate"] for r in data if r["success"]]
    miss = [r["candidate"] for r in data if not r["success"]][:a.n_miss]
    print(f"[setup] {len(hits)} attacks, {len(miss)} non-attacks, "
          f"eta={a.eta} d_min={a.d_min}", flush=True)

    spec = real_filter(algo="ssa", index="distance", d_min=a.d_min,
                       eta=a.eta, k=0.1)
    world = World.build(seed=a.seed, spec=spec, test_case=a.test_case,
                        max_steps=a.max_steps)
    scene = world.scene()

    rows = []
    for i, c in enumerate(hits + miss):
        s = stats_for(world, scene, c, spec, a.max_steps)
        if s:
            rows.append(s)
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(hits)+len(miss)}", flush=True)

    H = [r for r in rows if r["success"]]
    M = [r for r in rows if not r["success"]]
    print(f"\nran {len(rows)}: {len(H)} attacks, {len(M)} non-attacks")

    KEYS = [
        ("A_min_g_engaged", False, "min g over engaged steps"),
        ("B_n_engaged", True, "# engaged steps"),
        ("B_frac_engaged", True, "fraction of steps engaged"),
        ("B_n_gave_up", True, "# give-ups (observed)"),
        ("C_exposure", True, "infeasibility exposure (depth x time)"),
        ("D_C_handover", False, "authority AT HANDOVER"),
        ("D_g_handover", False, "margin AT HANDOVER"),
        ("D_phi_handover", True, "danger AT HANDOVER"),
        ("D_clear_handover", False, "clearance AT HANDOVER"),
        ("E_min_clear_final", False, "min clearance on final leg"),
        ("F_max_phi_engaged", True, "max danger while engaged"),
    ]

    print("\n" + "=" * 104)
    print("CANDIDATE GUIDANCE STATISTICS")
    print("=" * 104)
    print(f"{'statistic':<40}{'attacks':<13}{'non-attacks':<14}"
          f"{'effect':<10}{'top-K':<12}{'verdict'}")
    print("-" * 104)
    scored = []
    for key, hib, label in KEYS:
        e = effect_size(H, M, key)
        rp = ranking_power(rows, key, hib)
        if e is None:
            print(f"{label:<40}{'-':<13}{'-':<14}{'-':<10}{'-':<12}unusable")
            continue
        topk = f"{rp[0]}/{rp[1]}" if rp else "-"
        chance = f"(ch {rp[3]:.1f})" if rp else ""
        d = abs(e["d"])
        verdict = ("STRONG" if d > 1.5 else "moderate" if d > 0.8
                   else "weak" if d > 0.3 else "none")
        scored.append((d, label, key, verdict, rp))
        print(f"{label:<40}{e['h']:<13.4f}{e['m']:<14.4f}"
              f"{e['d']:<+10.2f}{topk+' '+chance:<12}{verdict}")

    print("\n" + "-" * 104)
    print("RANKED BY DISCRIMINATIVE POWER")
    print("-" * 104)
    for d, label, key, verdict, rp in sorted(scored, reverse=True):
        note = ""
        if rp:
            note = f"  ranks {rp[0]}/{rp[1]} attacks in top-{rp[1]} (chance {rp[3]:.1f})"
        print(f"  |effect| = {d:5.2f}  {verdict:<9} {label}{note}")

    json.dump(rows, open(a.out, "w"), indent=2, default=float)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
