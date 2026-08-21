"""
Tests for the pipeline. No SPARK, no MuJoCo — fake runs, so they are fast and
deterministic and can assert things a live scene would only show by luck
(a Kind 1 that is not a Kind 2, a leg-1 deadlock, an unengaged candidate).

    KMP_DUPLICATE_LIB_OK=TRUE python -m fuzz.siren.test_pipeline
"""

import numpy as np

from .pipeline_spec import (Evaluation, classify, evaluate, run_search, summarize,
                       _ENGAGED_OFFSET)
from .world.measure import StepMeasurement
from .world.types import Scene

_PASS, _FAIL = [], []


def check(name, fn):
    try:
        fn()
        _PASS.append(name)
        print(f"  PASS  {name}")
    except AssertionError as e:
        _FAIL.append((name, e))
        print(f"  FAILED  {name}\n          {type(e).__name__}: {e}")


# --------------------------------------------------------------------------- #
#  fakes
# --------------------------------------------------------------------------- #
class FakeRecord:
    def __init__(self, steps, label, n_gave_up=0, min_g=np.inf):
        self.steps = steps
        self.label = label
        self.n_gave_up = n_gave_up
        self.min_g = min_g


class FakeWorld:
    """Returns a scripted record; records the schedules it was asked to run."""
    def __init__(self, record):
        self.record = record
        self.calls = []

    def run(self, schedule, *a, **kw):
        self.calls.append((schedule, kw))
        return self.record


def step(i, wp, clearance=0.10, phi=-1.0, mu=np.nan, engaged=False, gave_up=False,
         brake=np.inf):
    return StepMeasurement(step=i, wp_idx=wp, dist_final=0.5, reached_final=False,
                           clearance=clearance, phi=phi, mu=mu, engaged=engaged,
                           gave_up=gave_up, brake_margin=brake)


def scene():
    return Scene(G0=np.zeros(3), G1=np.array([0.3, -0.2, 0.1]),
                 base_frame=np.eye(4),
                 obstacles_world=np.zeros((0, 4, 4)),
                 bounds=[(0.0, 0.5), (-0.4, 0.1), (-0.1, 0.3)],
                 keepout=0.0, seed=0, test_case="fake")


def main():
    print("\n[pipeline] step 5 — classification")

    check("mu > 0 at t0 is Kind 1", lambda: (
        _eq(classify(True, 5, 0.4, 0.9), "KIND_1")))
    check("feasible at t0, infeasible later is Kind 2", lambda: (
        _eq(classify(True, 5, -0.2, 0.4), "KIND_2")))
    check("feasible throughout is NO_INFEASIBILITY", lambda: (
        _eq(classify(True, 5, -0.2, -0.05), "NO_INFEASIBILITY")))
    check("never engaged is its own category, not Kind 1", lambda: (
        _eq(classify(False, None, None, None), "NEVER_ENGAGED")))

    def t_sign():
        # the convention that had to be reconciled: mu > 0 means INFEASIBLE, so a
        # very negative mu must never be mistaken for an attack
        assert classify(True, 3, -5.0, -4.0) == "NO_INFEASIBILITY"
        assert classify(True, 3, +0.001, None) == "KIND_1"
    check("sign convention: mu > 0 is infeasible, mu < 0 is comfortable", t_sign)

    print("\n[pipeline] steps 0-2 — staging")

    def t_inadmissible():
        sc = scene()
        w = FakeWorld(FakeRecord([], "REACHED"))
        ev = evaluate(w, sc, np.array([99.0, 99.0, 99.0]))
        assert ev.stage == "inadmissible", ev.stage
        assert ev.score is None
        assert not w.calls, "an inadmissible candidate must not be simulated"
    check("out-of-bounds candidate is rejected on geometry alone", t_inadmissible)

    def t_leg1_collision_is_modification_hit():
        steps = [step(i, 0) for i in range(6)]          # never reaches wp_idx 1
        w = FakeWorld(FakeRecord(steps, "COLLISION"))
        ev = evaluate(w, scene(), np.array([0.2, -0.2, 0.1]))
        assert ev.stage == "modification_hit", ev.stage
        assert ev.leg1_label == "COLLISION"
        assert ev.score == 0.0, "spec: score 0 for the insertion search"
        assert ev.collided
    check("leg-1 collision logs a modification hit, score 0", t_leg1_collision_is_modification_hit)

    def t_leg1_deadlock():
        w = FakeWorld(FakeRecord([step(i, 0) for i in range(4)], "DEADLOCK"))
        ev = evaluate(w, scene(), np.array([0.2, -0.2, 0.1]))
        assert ev.stage == "modification_hit"
        assert ev.score == 0.0
        assert not ev.collided, "a deadlock is not a collision"
    check("leg-1 deadlock is a modification hit but not a collision", t_leg1_deadlock)

    def t_leg1_timeout_discarded():
        w = FakeWorld(FakeRecord([step(i, 0) for i in range(4)], "TIMEOUT"))
        ev = evaluate(w, scene(), np.array([0.2, -0.2, 0.1]))
        assert ev.stage == "leg1_timeout"
        assert ev.score is None, "spec: a leg-1 timeout is DISCARDED, not scored"
    check("leg-1 timeout is discarded with no score", t_leg1_timeout_discarded)

    def t_one_continuous_run():
        steps = [step(0, 0), step(1, 1)]
        w = FakeWorld(FakeRecord(steps, "REACHED"))
        evaluate(w, scene(), np.array([0.2, -0.2, 0.1]))
        assert len(w.calls) == 1, "legs 1 and 2 must be ONE run (departing in motion)"
        sched, kw = w.calls[0]
        assert len(sched) == 2, "schedule must be [G1', G1]"
        assert kw.get("exact_margin") is True, "step 3 needs the exact LP margin"
    check("legs 1+2 are a single continuous run with exact margin", t_one_continuous_run)

    print("\n[pipeline] steps 3-4 — handover and scoring")

    def t_t0_is_on_leg2():
        # engaged during leg 1 AND later on leg 2: t0 must be the LEG-2 one, or a
        # Kind 2 would be misread as a Kind 1
        steps = [step(0, 0, phi=0.1, mu=+0.5, engaged=True),
                 step(1, 1, phi=0.1, mu=-0.3, engaged=True),
                 step(2, 1, phi=0.1, mu=+0.2, engaged=True)]
        w = FakeWorld(FakeRecord(steps, "COLLISION"))
        ev = evaluate(w, scene(), np.array([0.2, -0.2, 0.1]))
        assert ev.t0 == 1, f"t0 must be first engagement on leg 2, got {ev.t0}"
        assert ev.kind == "KIND_2", ev.kind
    check("t0 is the first intervention on LEG 2, not leg 1", t_t0_is_on_leg2)

    def t_handover_recorded():
        steps = [step(0, 0, clearance=0.30), step(1, 1, clearance=0.11, phi=-0.4),
                 step(2, 1, clearance=0.05)]
        w = FakeWorld(FakeRecord(steps, "REACHED"))
        ev = evaluate(w, scene(), np.array([0.2, -0.2, 0.1]))
        assert ev.handover_step == 1
        assert abs(ev.handover_clearance - 0.11) < 1e-9
        assert abs(ev.min_clearance_leg2 - 0.05) < 1e-9
    check("handover state is the first leg-2 step", t_handover_recorded)

    def t_engaged_outranks_unengaged():
        eng = [step(0, 0), step(1, 1, phi=0.1, mu=-9.0, engaged=True)]
        un = [step(0, 0), step(1, 1, clearance=1e-6)]
        a = evaluate(FakeWorld(FakeRecord(eng, "REACHED")), scene(),
                     np.array([0.2, -0.2, 0.1]))
        b = evaluate(FakeWorld(FakeRecord(un, "REACHED")), scene(),
                     np.array([0.2, -0.2, 0.1]))
        assert a.engaged and not b.engaged
        assert a.score > b.score, (
            "lexicographic: ANY engaged candidate must outrank EVERY unengaged "
            f"one, got engaged={a.score} vs unengaged={b.score}")
    check("lexicographic: engaged beats unengaged even at worst mu", t_engaged_outranks_unengaged)

    def t_score_uses_min_brake_margin():
        steps = [step(0, 0), step(1, 1, phi=0.1, mu=-0.5, engaged=True, brake=0.08),
                 step(2, 1, phi=0.1, mu=+0.7, engaged=True, brake=-0.03)]
        ev = evaluate(FakeWorld(FakeRecord(steps, "COLLISION")), scene(),
                      np.array([0.2, -0.2, 0.1]))
        assert abs(ev.score - (_ENGAGED_OFFSET + 0.03)) < 1e-9, ev.score
        assert abs(ev.min_brake_margin + 0.03) < 1e-9
        assert ev.kind == "KIND_2"
    check("engaged score is -min braking margin (deeper past braking = better)",
          t_score_uses_min_brake_margin)

    def t_score_is_not_mu_regression():
        """BUG-4 guard. mu is ANTI-correlated with collisions here (the quartile
        closest to infeasibility had a 0.0% collision rate), so a score that
        rises with mu steers the search AWAY from attacks. Two candidates with
        identical braking margins must score identically no matter what mu did."""
        mk = lambda mu_hi: evaluate(FakeWorld(FakeRecord(
            [step(0, 0), step(1, 1, phi=0.1, mu=mu_hi, engaged=True, brake=0.05)],
            "REACHED")), scene(), np.array([0.2, -0.2, 0.1]))
        assert abs(mk(-9.0).score - mk(-0.001).score) < 1e-12, (
            "score still depends on mu — BUG-4 has regressed")
    check("score does NOT depend on mu (BUG-4 regression guard)",
          t_score_is_not_mu_regression)

    def t_unengaged_score_is_neg_clearance():
        steps = [step(0, 0), step(1, 1, clearance=0.04)]
        ev = evaluate(FakeWorld(FakeRecord(steps, "REACHED")), scene(),
                      np.array([0.2, -0.2, 0.1]))
        assert abs(ev.score + 0.04) < 1e-9, ev.score
    check("unengaged score is -min clearance (closer is better)", t_unengaged_score_is_neg_clearance)

    print("\n[pipeline] step 6 — feedback and bookkeeping")

    class Picker:
        name = "fake"
        def __init__(self, pts):
            self.pts, self.told = list(pts), []
        def ask(self, n):
            out, self.pts = self.pts[:n], self.pts[n:]
            return out
        def tell(self, scored):
            self.told.extend(scored)

    def t_discards_not_fed():
        w = FakeWorld(FakeRecord([step(i, 0) for i in range(3)], "TIMEOUT"))
        p = Picker([np.array([0.2, -0.2, 0.1])] * 4)
        evs = run_search(w, scene(), p, budget=4, batch=4)
        assert len(evs) == 4
        assert p.told == [], "discarded candidates must not be fed to the picker"
    check("leg-1 timeouts are never fed back to the search", t_discards_not_fed)

    def t_budget_respected():
        w = FakeWorld(FakeRecord([step(0, 0), step(1, 1)], "REACHED"))
        p = Picker([np.array([0.2, -0.2, 0.1])] * 50)
        evs = run_search(w, scene(), p, budget=7, batch=3)
        assert len(evs) == 7, len(evs)
        assert len(p.told) == 7
    check("budget is honoured exactly", t_budget_respected)

    def t_summary_splits_collisions_by_leg():
        sc, pt = scene(), np.array([0.2, -0.2, 0.1])
        leg1 = evaluate(FakeWorld(FakeRecord([step(0, 0)], "COLLISION")), sc, pt)
        leg2 = evaluate(FakeWorld(FakeRecord(
            [step(0, 0), step(1, 1, phi=0.1, mu=-0.2, engaged=True)],
            "COLLISION")), sc, pt)
        s = summarize([leg1, leg2])
        assert s["n_collisions_leg1"] == 1, s
        assert s["n_collisions_leg2"] == 1, s
        assert s["n_modification_hits"] == 1
        assert s["kind_NO_INFEASIBILITY"] == 1
        assert abs(s["attack_rate_leg2"] - 1.0) < 1e-9, s["attack_rate_leg2"]
    check("summary separates leg-1 (modification) from leg-2 (insertion) hits",
          t_summary_splits_collisions_by_leg)

    def t_kind_counts_are_disjoint():
        sc, pt = scene(), np.array([0.2, -0.2, 0.1])
        mk = lambda mu0, mu1, lbl: evaluate(FakeWorld(FakeRecord(
            [step(0, 0), step(1, 1, phi=0.1, mu=mu0, engaged=True),
             step(2, 1, phi=0.1, mu=mu1, engaged=True)], lbl)), sc, pt)
        evs = [mk(+0.5, +0.5, "COLLISION"), mk(-0.5, +0.5, "COLLISION"),
               mk(-0.5, -0.5, "REACHED")]
        s = summarize(evs)
        assert (s["kind_KIND_1"], s["kind_KIND_2"], s["kind_NO_INFEASIBILITY"]) == (1, 1, 1), s
        assert sum(s[f"kind_{k}"] for k in
                   ("KIND_1", "KIND_2", "NO_INFEASIBILITY", "NEVER_ENGAGED")) == 3
    check("every evaluated candidate lands in exactly one kind", t_kind_counts_are_disjoint)

    print("\n" + "=" * 70)
    print(f"{len(_PASS)}/{len(_PASS) + len(_FAIL)} passed")
    print("=" * 70)
    return 1 if _FAIL else 0


def _eq(a, b):
    assert a == b, f"{a!r} != {b!r}"


if __name__ == "__main__":
    raise SystemExit(main())
