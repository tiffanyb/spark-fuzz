"""
Braking-distance screen: are the D2 dynamic-obstacle scenes winnable at all?

Two scenes have already been ruled out as target material because their failures
are explained before an attacker exists:

    D1 fixed-base SO   lambda*phi > C at every step -- filter mathematically
                       unable to satisfy its constraints (100% infeasible)
    D1 any DO          obstacle closes at 0.400 m/s against C = 0.026 m/s of
                       retreat authority, and Cartesian_Lf is hardcoded to zero
                       so the filter cannot see the obstacle move at all

The D2 DO scenes survive both objections: authority is 100-300x larger and L_f is
computed. But D2's phi is velocity-augmented, so C is not in m/s and cannot be
compared to obstacle speed by arithmetic. The braking-distance test is the right
instrument -- it is the integrated second-order quantity, and it was validated at
83% against the brute-force escape search.

    brake_margin = clearance - v_close^2 / (2 * BRAKE_SCALE * C_phi)

Negative means the pair is already committed: no braking effort arrests the
closing before contact.

The screen asks, on the LEGITIMATE task (no inserted goal), for each scene x
filter x seed:

    outcome              does the baseline reach?
    margin at t0         was the state already committed at the FIRST step the
                         filter intervened?  negative => pre-doomed, like the
                         cases already rejected
    min margin           how close it came overall
    n_gave_up            did the filter ever declare failure

A scene is usable target material when its collisions are NOT already committed
at t0 -- that is the difference between "the filter was handed a lost state" and
"the filter had authority and lost it anyway".

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 python -m fuzz.siren.scenario.screen_do_scenes
"""

import argparse
import json

import numpy as np

SCENES = [
    ("G1FixedBase_D2_AG_DO_v0", "velocity"),
    ("G1FixedBase_D2_AG_DO_v1", "velocity"),
    ("G1MobileBase_D2_WG_DO_v1", "velocity"),
]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--algos", default="ssa,pssa,cbf,sss")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--max-steps", type=int, default=900)
    p.add_argument("--out",
                   default="fuzz/siren/experiment/do_screen.json")
    a = p.parse_args(argv)

    from ..world.run import World
    from ..world.types import real_filter

    algos = [x.strip() for x in a.algos.split(",") if x.strip()]
    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]

    print(f"{'scene':<28}{'algo':<6}{'seed':>5}{'outcome':>10}{'steps':>7}"
          f"{'t0':>6}{'m@t0':>10}{'t_commit':>9}{'min m':>11}"
          f"{'min clear':>11}{'gaveup':>7}", flush=True)
    rows = []
    for case, index in SCENES:
        for algo in algos:
            for sd in seeds:
                spec = real_filter(algo=algo, index=index, d_min=0.02,
                                   eta=0.02, lam=10.0, k=0.1)
                try:
                    w = World.build(seed=sd, spec=spec, test_case=case,
                                    max_steps=a.max_steps)
                    # The legitimate task, as a single fixed waypoint. This
                    # freezes the goal, where the stock benchmark lets it drift
                    # -- deliberate: the question is whether the SCENE can be
                    # survived, and a wandering goal adds a second moving part.
                    rec = w.run([np.asarray(w.scene().G1, float)],
                                max_steps=a.max_steps)
                except Exception as e:
                    print(f"  {case:<24}{algo:<6}{sd:>5}   ERROR "
                          f"{type(e).__name__}: {str(e)[:40]}", flush=True)
                    continue

                bm = np.array([s.brake_margin for s in rec.steps], float)
                eng = np.array([bool(s.engaged) for s in rec.steps])
                cl = np.array([s.clearance for s in rec.steps], float)
                finite = np.isfinite(bm)
                # t0 = first step the filter intervened at all
                t0 = int(np.argmax(eng)) if eng.any() else None
                # brake_margin is +inf when the pair is not closing (v_close<=0).
                # That is NOT missing data -- it means "nothing to arrest", i.e.
                # comfortably uncommitted. Treating it as nan would silently drop
                # every healthy step from the classification.
                m_t0 = float(bm[t0]) if t0 is not None else float("nan")
                m_min = float(bm[finite].min()) if finite.any() else float("nan")
                # when did it first become committed, and was that after t0?
                neg = np.where(finite & (bm < 0))[0]
                t_commit = int(neg[0]) if len(neg) else None
                committed_at_t0 = bool(t0 is not None and m_t0 < 0)

                rows.append({"case": case, "algo": algo, "seed": sd,
                             "label": rec.label, "n_steps": int(rec.n_steps),
                             "t0": t0, "margin_at_t0": m_t0,
                             "committed_at_t0": committed_at_t0,
                             "t_commit": t_commit, "min_margin": m_min,
                             "min_clearance": float(rec.min_clearance),
                             "n_gave_up": int(rec.n_gave_up)})
                s_t0 = ("  none" if t0 is None else
                        ("  +inf" if np.isinf(m_t0) else f"{m_t0:+.4f}"))
                print(f"  {case:<26}{algo:<6}{sd:>5}{rec.label:>10}"
                      f"{rec.n_steps:>7}{str(t0):>6}{s_t0:>10}"
                      f"{str(t_commit):>9}{m_min:>11.4f}"
                      f"{rec.min_clearance:>+11.5f}{rec.n_gave_up:>7}",
                      flush=True)

    json.dump(rows, open(a.out, "w"), indent=2, default=float)

    print(f"\n{'='*94}")
    for case, _ in SCENES:
        sub = [r for r in rows if r["case"] == case]
        if not sub:
            continue
        coll = [r for r in sub if r["label"] == "COLLISION"]
        doomed = [r for r in coll if r["committed_at_t0"]]
        winnable = [r for r in coll if not r["committed_at_t0"]]
        print(f"{case}")
        print(f"   {len(sub)} runs: {sum(r['label']=='REACHED' for r in sub)} reached, "
              f"{len(coll)} collided")
        print(f"   of the collisions: {len(doomed)} already committed at t0 "
              f"(pre-doomed), {len(winnable)} had margin at t0")
        if winnable:
            print(f"   -> USABLE: {len(winnable)} collisions the filter had "
                  f"authority to avoid")
        elif coll:
            print(f"   -> NOT usable: every collision was already lost when the "
                  f"filter first engaged")
        else:
            print(f"   -> no collisions on the legitimate task at these seeds")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
