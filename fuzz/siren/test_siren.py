"""
SIREN test suite.

Three layers, cheapest first, so a failure is localised immediately:

  L1  pure math      derived.py and measure.py against hand-made arrays.
                     No SPARK, no MuJoCo — runs in milliseconds.
  L2  wiring         attacks, pickers, the knowledge budget, threat presets,
                     against a FAKE world. Still no MuJoCo.
  L3  SPARK          every SPARK benchmark test case is built and driven for a
                     few steps, plus every safety filter, plus a real end-to-end
                     search. This is the slow one.

Run:
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 python -m fuzz.siren.test_siren            # L1+L2
    ... python -m fuzz.siren.test_siren --spark                       # + L3
    ... python -m fuzz.siren.test_siren --spark --all-cases           # + all 25 cases
"""

import argparse
import sys
import traceback

import numpy as np

# ---------------------------------------------------------------------------- #
#  Tiny test harness (no pytest dependency)
# ---------------------------------------------------------------------------- #
_RESULTS = []


def check(name, fn):
    try:
        fn()
        _RESULTS.append((name, True, ""))
        print(f"  PASS  {name}", flush=True)
    except Exception as e:
        detail = f"{type(e).__name__}: {e}"
        _RESULTS.append((name, False, detail))
        print(f"  FAIL  {name}\n        {detail}", flush=True)
        if "-v" in sys.argv:
            traceback.print_exc()


def approx(a, b, tol=1e-9):
    assert abs(float(a) - float(b)) < tol, f"{a} != {b}"


# ============================================================================ #
#  L1 — pure math
# ============================================================================ #
def test_layer1():
    from fuzz.siren.world import derived
    from fuzz.siren.world.measure import StepMeasurement, classify_run, RunRecord
    from fuzz.siren.world.types import (FilterSpec, strict_black_ensemble,
                                        weak_black_ensemble)

    print("\n[L1] pure math — authority, margin, classification")

    def t_authority_basic():
        # c = sum_k u_lim_k * |Lg_k| : one constraint, two joints
        c = derived.control_authority(np.array([[0.5, -2.0]]), np.array([2.0, 1.0]))
        approx(c[0], 0.5 * 2.0 + 2.0 * 1.0)          # = 3.0
    check("authority = sum(u_lim * |Lg|)", t_authority_basic)

    def t_authority_zero_Lg():
        # No control effect on this constraint -> zero authority. The extreme
        # low-authority corner: infeasible no matter what the filter is.
        c = derived.control_authority(np.zeros((1, 3)), np.ones(3))
        approx(c[0], 0.0)
    check("zero Lg -> zero authority", t_authority_zero_Lg)

    def t_authority_sign_invariant():
        a = derived.control_authority(np.array([[1.0, -1.0]]), np.ones(2))
        b = derived.control_authority(np.array([[-1.0, 1.0]]), np.ones(2))
        approx(a[0], b[0])                            # box is symmetric
    check("authority is sign-invariant", t_authority_sign_invariant)

    def t_demand_shapes():
        phi = np.array([0.0, 0.5])
        approx(derived.demand_vector("constant", phi, eta=0.3)[1], 0.3)
        # proportional demand VANISHES at the boundary -> grazing cannot defeat it
        approx(derived.demand_vector("proportional", phi, lam=10.0)[0], 0.0)
        approx(derived.demand_vector("proportional", phi, lam=10.0)[1], 5.0)
    check("demand: constant bites at boundary, proportional does not", t_demand_shapes)

    def t_active_set():
        phi = np.array([-0.1, 0.2])
        mask = np.array([1.0, 1.0])
        act = derived.active_set(phi, mask, "constant")
        assert not act[0] and act[1], act
    check("active set = masked AND phi>=0", t_active_set)

    def t_margin_identity():
        # g = c - demand - drift
        c = np.array([1.0]); d = np.array([0.4]); Lf = np.array([0.1])
        m = derived.margin(c, d, Lf, np.array([True]))
        approx(m["g_min"], 0.5)
        assert not (m["g_min"] < 0)
    check("margin g = C - demand - drift", t_margin_identity)

    def t_margin_negative_is_void():
        m = derived.margin(np.array([0.2]), np.array([0.5]), np.array([0.0]),
                           np.array([True]))
        assert m["g_min"] < 0, m
        approx(m["pressure"], 0.3)         # slack a soft filter is forced to take
    check("g<0 flags void guarantee, pressure = |g|", t_margin_negative_is_void)

    def t_margin_picks_tightest():
        m = derived.margin(np.array([1.0, 0.1]), np.array([0.2, 0.2]),
                           np.zeros(2), np.array([True, True]))
        assert m["binding"] == 1, m
        approx(m["g_min"], -0.1)
    check("margin reports the tightest constraint", t_margin_picks_tightest)

    def t_margin_no_active():
        m = derived.margin(np.array([1.0]), np.array([0.1]), np.array([0.0]),
                           np.array([False]))
        assert m["g_min"] == np.inf and m["n_active"] == 0
    check("no active constraint -> infinite margin", t_margin_no_active)

    def t_exact_lp_agrees_single():
        # With one constraint the LP must reproduce the closed form.
        Lg = np.array([[1.0, 1.0]]); u_lim = np.array([1.0, 1.0])
        Lf = np.array([0.0]); demand = np.array([0.5])
        mu = derived.exact_margin_lp(Lg, demand, Lf, u_lim, np.array([True]))
        c = derived.control_authority(Lg, u_lim)[0]        # = 2.0
        approx(mu, demand[0] + Lf[0] - c)                  # = -1.5  (feasible)
    check("exact LP agrees with closed form (1 constraint)", t_exact_lp_agrees_single)

    def t_exact_lp_conflict():
        # A pincer: two constraints that individually are fine but conflict.
        Lg = np.array([[1.0], [-1.0]]); u_lim = np.array([1.0])
        Lf = np.array([0.0, 0.0]); demand = np.array([2.0, 2.0])
        mu = derived.exact_margin_lp(Lg, demand, Lf, u_lim, np.array([True, True]))
        assert mu > 0, mu                                   # infeasible together
    check("exact LP detects a conflicting pair", t_exact_lp_conflict)

    def t_guidance_when_not_engaged():
        # Nothing active (phi < 0), but the attacker still needs a gradient:
        # authority must be reported, while "infeasible" must stay False.
        out = derived.evaluate(Lg=np.array([[1.0, 0.0]]), Lf=np.array([0.0]),
                               phi=np.array([-0.5]), phi_mask=np.array([1.0]),
                               u_lim=np.array([0.3, 0.3]),
                               demand_shape="constant", eta=0.5)
        assert np.isfinite(out["C_phi"]), "no guidance signal on a safe candidate"
        approx(out["C_phi"], 0.3)
        assert not out["engaged"]
        assert not out["predicted_infeasible"], "inactive constraint reported violated"
    check("guidance is finite even when the filter never engages",
          t_guidance_when_not_engaged)

    def t_guidance_uses_nearest_not_weakest():
        # Two obstacles, neither active. Pair 0 is far away and the arm barely
        # affects it (tiny authority); pair 1 is nearly touching. The guidance
        # must describe pair 1 -- reporting the far pair's tiny authority would
        # be a near-constant, uninformative score.
        out = derived.evaluate(
            Lg=np.array([[0.01, 0.0],     # far obstacle, almost no leverage
                         [1.00, 0.0]]),   # near obstacle, good leverage
            Lf=np.zeros(2),
            phi=np.array([-5.0, -0.01]),  # pair 1 is the one nearly engaged
            phi_mask=np.ones(2), u_lim=np.array([1.0, 1.0]),
            demand_shape="constant", eta=0.5)
        approx(out["C_phi"], 1.0)         # the NEAR pair, not the weak far one
        approx(out["phi"], -0.01)
    check("guidance describes the nearest constraint, not the weakest",
          t_guidance_uses_nearest_not_weakest)

    def t_engaged_flag():
        out = derived.evaluate(Lg=np.array([[1.0, 0.0]]), Lf=np.array([0.0]),
                               phi=np.array([0.1]), phi_mask=np.array([1.0]),
                               u_lim=np.array([0.3, 0.3]),
                               demand_shape="constant", eta=0.5)
        assert out["engaged"] and out["predicted_infeasible"]
    check("an active, under-powered constraint IS flagged infeasible", t_engaged_flag)

    def t_evaluate_roundtrip():
        out = derived.evaluate(Lg=np.array([[1.0, 0.0]]), Lf=np.array([0.0]),
                               phi=np.array([0.1]), phi_mask=np.array([1.0]),
                               u_lim=np.array([0.3, 0.3]),
                               demand_shape="constant", eta=0.5)
        approx(out["C_phi"], 0.3)
        approx(out["g"], 0.3 - 0.5)
        assert out["predicted_infeasible"]
    check("evaluate() end to end", t_evaluate_roundtrip)

    # ---- classification --------------------------------------------------- #
    def mk(n, dist=1.0, clearance=0.05, wp=0, reached=False, jitter=0.0):
        return [StepMeasurement(step=i, wp_idx=wp,
                                dist_final=dist + jitter * (i % 2),
                                reached_final=reached, clearance=clearance)
                for i in range(n)]

    def t_classify_collision():
        s = mk(50); s[10].clearance = -1e-4
        assert classify_run(s) == "COLLISION"
    check("classify: penetration -> COLLISION", t_classify_collision)

    def t_classify_reached():
        s = mk(50); s[40].reached_final = True
        assert classify_run(s) == "REACHED"
    check("classify: reached -> REACHED", t_classify_reached)

    def t_classify_deadlock():
        s = mk(150, dist=1.0, wp=0)                  # far, frozen, full horizon
        assert classify_run(s, schedule=[0]) == "DEADLOCK"
    check("classify: far+frozen+full+final-leg -> DEADLOCK", t_classify_deadlock)

    def t_classify_short_is_timeout():
        # The seed-4 false positive: a brief stall on a SHORT run is not a trap.
        assert classify_run(mk(40, dist=1.0), schedule=[0]) == "TIMEOUT"
    check("classify: short horizon is TIMEOUT not DEADLOCK", t_classify_short_is_timeout)

    def t_classify_near_goal_is_timeout():
        assert classify_run(mk(150, dist=0.06), schedule=[0]) == "TIMEOUT"
    check("classify: frozen but NEAR goal is TIMEOUT", t_classify_near_goal_is_timeout)

    def t_classify_wrong_leg_is_timeout():
        # Frozen on the INSERTED leg = horizon starvation, not a trap.
        s = mk(150, dist=1.0, wp=0)
        assert classify_run(s, schedule=[0, 1]) == "TIMEOUT"
    check("classify: frozen on the inserted leg is TIMEOUT", t_classify_wrong_leg_is_timeout)

    def t_classify_moving_is_timeout():
        s = [StepMeasurement(step=i, wp_idx=0, dist_final=2.0 - 0.01 * i,
                             reached_final=False, clearance=0.05) for i in range(150)]
        assert classify_run(s, schedule=[0]) == "TIMEOUT"
    check("classify: still approaching is TIMEOUT", t_classify_moving_is_timeout)

    def t_record_summary():
        s = mk(120, dist=1.0)
        s[5].C_d = 0.2; s[5].g = -0.3; s[5].gave_up = True; s[5].clearance = -0.002
        rec = RunRecord.summarize(s, [0], FilterSpec())
        approx(rec.min_C_d, 0.2)
        approx(rec.min_g, -0.3)
        assert rec.n_gave_up == 1
        approx(rec.max_penetration, 0.002)
        assert rec.label == "COLLISION" and rec.is_attack_success()
    check("RunRecord.summarize rolls up correctly", t_record_summary)

    # ---- ensembles -------------------------------------------------------- #
    def t_ensembles():
        assert len(strict_black_ensemble()) == 6     # resistance x index
        assert len(weak_black_ensemble()) == 3       # resistance only
        idx = {s.index for s in strict_black_ensemble()}
        assert idx == {"distance", "velocity"}, idx
    check("ensembles: strict=6, weak=3", t_ensembles)

    def t_filterspec_families():
        assert FilterSpec(algo="ssa").demand_shape == "constant"
        assert FilterSpec(algo="cbf").demand_shape == "proportional"
        assert FilterSpec(algo="ssa").is_hard            # gives up -> collision
        assert not FilterSpec(algo="rssa").is_hard       # slacks   -> stall
    check("FilterSpec knows demand shape and hard/soft", t_filterspec_families)


# ============================================================================ #
#  L2 — wiring, against a fake world
# ============================================================================ #
class FakeWorld:
    """A world with no physics: it just reports whatever the script says. Lets us
    test every piece of search/ without booting MuJoCo."""

    def __init__(self, scene, script=None):
        self._scene = scene
        self.script = script or (lambda schedule, spec: ("REACHED", {}))
        self.calls = []

    def scene(self):
        return self._scene

    def accepts(self, spec):
        return spec.index == "distance"

    @property
    def supports_index(self):
        return "distance"

    def run(self, schedule, spec=None, max_steps=None):
        from fuzz.siren.world.measure import RunRecord, StepMeasurement
        self.calls.append((len(schedule), getattr(spec, "label", "")))
        label, over = self.script(schedule, spec)
        steps = [StepMeasurement(step=i, wp_idx=len(schedule) - 1, dist_final=1.0,
                                 reached_final=(label == "REACHED" and i == 9),
                                 clearance=(-1e-4 if label == "COLLISION" and i == 5 else 0.05),
                                 C_d=over.get("C_d", 0.5), C_phi=over.get("C_phi", 0.5),
                                 g=over.get("g", 0.2), phi=0.0,
                                 deviation=over.get("deviation", 0.0))
                 for i in range(10)]
        rec = RunRecord.summarize(steps, schedule, spec)
        rec.label = label
        rec.reached = (label == "REACHED")
        return rec


def test_layer2():
    from fuzz.siren.world.types import Scene, FilterSpec
    from fuzz.siren.search.attacks import make_attack
    from fuzz.siren.search.pick import (is_admissible, sample_admissible,
                                        make_picker, CEMPicker)
    from fuzz.siren.search.threat import (threat_model, PRESETS, KnowledgeError,
                                          WhiteView, GrayView, BlackView,
                                          CoarseMotion, AttackerCapability,
                                          IndexKnowledge, DemandShape, Observability)
    from fuzz.siren.search.loop import search

    print("\n[L2] wiring — attacks, pickers, knowledge budget, threat presets")

    scene = Scene(G0=np.zeros(3), G1=np.array([0.3, -0.2, 0.2]),
                  base_frame=np.eye(4),
                  obstacles_world=np.array([np.eye(4)]),
                  bounds=((0.0, 0.4), (-0.4, 0.0), (0.0, 0.3)),
                  keepout=0.05, seed=1, test_case="fake")

    # ---- attacks ---------------------------------------------------------- #
    def t_insertion_schedule():
        a = make_attack("insertion")
        s = a.build_schedule(np.array([0.1, -0.1, 0.1]), scene)
        assert len(s) == 2 and np.allclose(s[1], scene.G1)
    check("insertion schedule = [G1', G1]", t_insertion_schedule)

    def t_modification_schedule():
        a = make_attack("modification")
        s = a.build_schedule(np.array([0.1, -0.1, 0.1]), scene)
        assert len(s) == 1
    check("modification schedule = [G1']", t_modification_schedule)

    def t_insertion_gate_rejects_unreachable():
        a = make_attack("insertion")
        w = FakeWorld(scene, lambda sch, sp: ("TIMEOUT", {}))
        ok, why = a.gate(np.array([0.1, -0.1, 0.1]), scene, lambda s: w.run(s))
        assert not ok and "one_hop" in why
    check("insertion gate rejects a one-hop trap", t_insertion_gate_rejects_unreachable)

    def t_modification_gate_passes():
        a = make_attack("modification")
        ok, _ = a.gate(np.array([0.1, -0.1, 0.1]), scene, None)
        assert ok
    check("modification gate needs no rollout", t_modification_gate_passes)

    # ---- admissibility ---------------------------------------------------- #
    def t_admissible_bounds():
        assert not is_admissible(np.array([9.0, 0.0, 0.0]), scene)[0]
        assert is_admissible(np.array([0.3, -0.3, 0.2]), scene)[0]
    check("admissibility rejects out-of-bounds", t_admissible_bounds)

    def t_admissible_keepout():
        ok, why = is_admissible(np.array([0.01, -0.01, 0.01]), scene)
        assert not ok and "too_close" in why
    check("admissibility enforces obstacle keepout", t_admissible_keepout)

    def t_sample_admissible():
        rng = np.random.RandomState(0)
        for _ in range(20):
            c = sample_admissible(rng, scene)
            assert c is not None and is_admissible(c, scene)[0]
    check("sampler only returns admissible goals", t_sample_admissible)

    # ---- pickers ---------------------------------------------------------- #
    def t_random_picker():
        p = make_picker("random", scene, seed=0)
        c = p.ask(8)
        assert len(c) == 8 and all(is_admissible(x, scene)[0] for x in c)
        p.tell([(x, 1.0) for x in c])          # must be a no-op, not an error
    check("RandomPicker asks and ignores tell", t_random_picker)

    def t_cem_concentrates():
        p = CEMPicker(scene, seed=0)
        target = np.array([0.35, -0.35, 0.25])
        start_err = float(np.linalg.norm(p.mean - target))
        start_spread = float(np.mean(p.std))
        for _ in range(10):
            c = p.ask(12)
            p.tell([(x, -float(np.linalg.norm(x - target))) for x in c])
        end_err = float(np.linalg.norm(p.mean - target))
        end_spread = float(np.mean(p.std))
        # it must move TOWARD the good region and TIGHTEN around it
        assert end_err < start_err, f"mean did not improve: {start_err} -> {end_err}"
        assert end_spread < start_spread, f"did not concentrate: {start_spread} -> {end_spread}"
        assert end_err < 0.15, f"converged too loosely: {end_err}"
    check("CEM concentrates on the high-scoring region", t_cem_concentrates)

    def t_cem_std_floor():
        p = CEMPicker(scene, seed=0)
        same = np.array([0.2, -0.2, 0.15])
        p.tell([(same, 1.0), (same, 1.0), (same, 1.0)])
        assert np.all(p.std > 0), p.std          # never collapses to a point
    check("CEM std never collapses to zero", t_cem_std_floor)

    # ---- the knowledge budget -------------------------------------------- #
    def t_white_reads_g():
        w = FakeWorld(scene)
        rec = w.run([scene.G1], FilterSpec(label="x"))
        assert np.isfinite(WhiteView(rec).min_g)
    check("white-box may read the exact margin g", t_white_reads_g)

    def t_gray_cannot_read_g():
        rec = FakeWorld(scene).run([scene.G1], FilterSpec())
        try:
            _ = GrayView(rec).min_g
        except KnowledgeError:
            return
        raise AssertionError("gray-box read min_g — knowledge budget not enforced")
    check("gray-box CANNOT read g (budget enforced)", t_gray_cannot_read_g)

    def t_black_cannot_read_C_phi():
        rec = FakeWorld(scene).run([scene.G1], FilterSpec())
        for field in ("min_C_phi", "phi", "min_g", "demand"):
            try:
                getattr(BlackView(rec), field)
            except KnowledgeError:
                continue
            raise AssertionError(f"strict black-box read '{field}'")
    check("strict black CANNOT read index-dependent fields", t_black_cannot_read_C_phi)

    def t_everyone_reads_observables():
        rec = FakeWorld(scene).run([scene.G1], FilterSpec())
        for V in (WhiteView, GrayView, BlackView):
            v = V(rec)
            _ = v.label, v.n_gave_up, v.max_penetration, v.min_clearance
    check("every tier may read observable behaviour", t_everyone_reads_observables)

    # ---- capabilities derive everything ----------------------------------- #
    def t_capability_derivation():
        assert PRESETS["white"].can_compute_g
        assert not PRESETS["gray"].can_compute_g
        assert PRESETS["gray"].can_compute_C_phi
        assert not PRESETS["gray"].needs_penetration_hedge      # constant demand
        assert PRESETS["gray-proportional"].needs_penetration_hedge
        assert PRESETS["weak-black"].can_compute_phi
        assert not PRESETS["strict-black"].can_compute_phi
        assert PRESETS["strict-black"].can_compute_C_d          # URDF is public
        assert PRESETS["strict-black"].ruler == "min_C_d"
        assert PRESETS["gray"].ruler == "min_C_phi"
    check("capability flags derive correctly for all tiers", t_capability_derivation)

    def t_passive_identification():
        assert PRESETS["white"].can_identify_passively          # full state
        assert not PRESETS["strict-black"].can_identify_passively
    check("passive identification requires full state", t_passive_identification)

    def t_ensemble_sizes_by_tier():
        assert len(threat_model("white").specs) == 1
        assert len(threat_model("gray").specs) == 1
        assert len(threat_model("weak-black").specs) == 3
        assert len(threat_model("strict-black").specs) == 6
    check("ensemble size follows the tier", t_ensemble_sizes_by_tier)

    def t_objective_choice():
        from fuzz.siren.search.threat import CertificateObjective, MarginProxyObjective
        assert isinstance(threat_model("white").objective, CertificateObjective)
        o = threat_model("strict-black").objective
        assert isinstance(o, MarginProxyObjective) and o.ruler == "min_C_d" and o.hedge
        g = threat_model("gray").objective
        assert g.ruler == "min_C_phi" and not g.hedge
    check("objective is derived, not hand-picked", t_objective_choice)

    def t_scores_prefer_failure():
        w = FakeWorld(scene)
        obj = threat_model("strict-black").objective
        good = obj.score(w.run([scene.G1], FilterSpec()))          # REACHED
        w2 = FakeWorld(scene, lambda s, sp: ("COLLISION", {}))
        bad = obj.score(w2.run([scene.G1], FilterSpec()))          # COLLISION
        assert bad > good, (bad, good)
    check("a collision scores above a clean reach", t_scores_prefer_failure)

    def t_lower_authority_scores_higher():
        obj = threat_model("strict-black").objective
        hi = FakeWorld(scene, lambda s, sp: ("REACHED", {"C_d": 2.0}))
        lo = FakeWorld(scene, lambda s, sp: ("REACHED", {"C_d": 0.1}))
        assert (obj.score(lo.run([scene.G1], FilterSpec()))
                > obj.score(hi.run([scene.G1], FilterSpec())))
    check("lower authority scores higher (the guidance gradient)", t_lower_authority_scores_higher)

    def t_coarse_motion_delays_onset():
        from fuzz.siren.world.measure import StepMeasurement, RunRecord
        steps = [StepMeasurement(step=i, wp_idx=0, dist_final=1.0, reached_final=False,
                                 clearance=0.10 - 0.01 * i, deviation=0.001 * i)
                 for i in range(10)]
        rec = RunRecord.summarize(steps, [0], FilterSpec())
        coarse = CoarseMotion(deviation_threshold=0.005).observe(rec)
        # full state would see step 1; coarse only notices once it is visible
        assert coarse.first_deviation.step >= 5, coarse.first_deviation
    check("coarse motion detects intervention LATE (biased inward)", t_coarse_motion_delays_onset)

    # ---- the loop --------------------------------------------------------- #
    def t_search_end_to_end():
        w = FakeWorld(scene, lambda sch, sp: ("COLLISION" if len(sch) == 2 else "REACHED", {}))
        r = search(w, make_attack("insertion"), make_picker("random", scene, 0),
                   threat_model("white"), budget=6, batch=3, verbose=False)
        assert r.n_screened == 6 and r.n_success == 6, r.summary()
        assert r.ranked()[0].score >= r.ranked()[-1].score
    check("search runs end to end and ranks results", t_search_end_to_end)

    def t_search_aborts_on_bad_baseline():
        w = FakeWorld(scene, lambda sch, sp: ("TIMEOUT", {}))
        r = search(w, make_attack("insertion"), make_picker("random", scene, 0),
                   threat_model("white"), budget=4, verbose=False)
        assert r.aborted and "baseline" in r.aborted
    check("search aborts when the legitimate goal is unreachable", t_search_aborts_on_bad_baseline)

    def t_gate_runs_even_if_first_spec_incompatible():
        # strict-black's first ensemble member is a distance-index spec. Against a
        # world that only runs the velocity index, a naive gate would pick that
        # incompatible spec, skip the baseline check, and happily report attacks
        # on an already-broken scene. The gate must instead pick a COMPATIBLE
        # spec and still abort.
        class VelocityOnlyWorld(FakeWorld):
            def accepts(self, spec):
                return spec.index == "velocity"
            @property
            def supports_index(self):
                return "velocity"

        w = VelocityOnlyWorld(scene, lambda sch, sp: ("COLLISION", {}))
        r = search(w, make_attack("insertion"), make_picker("random", scene, 0),
                   threat_model("strict-black"), budget=4, verbose=False)
        assert r.aborted, "invalid scene slipped past the baseline gate"
        assert "baseline" in r.aborted, r.aborted
        assert r.n_success == 0, "banked attacks on an unvalidated scene"
    check("baseline gate uses a COMPATIBLE spec, never skipped",
          t_gate_runs_even_if_first_spec_incompatible)

    def t_abort_when_no_compatible_surrogate():
        class NothingWorld(FakeWorld):
            def accepts(self, spec):
                return False
            @property
            def supports_index(self):
                return "velocity"
        r = search(NothingWorld(scene), make_attack("modification"),
                   make_picker("random", scene, 0), threat_model("gray"),
                   budget=3, verbose=False)
        assert r.aborted and "no_compatible_surrogate" in r.aborted, r.aborted
    check("aborts clearly when no surrogate fits the world",
          t_abort_when_no_compatible_surrogate)

    def t_inert_filter_scene_rejected():
        # The filter giving up on the BASELINE means the obstacle-blind reference
        # controller is driving. Attacks there are against an inert filter and
        # must not be counted.
        from fuzz.siren.world.measure import RunRecord, StepMeasurement

        class InertWorld(FakeWorld):
            def run(self, schedule, spec=None, max_steps=None):
                rec = super().run(schedule, spec, max_steps)
                if len(schedule) == 1 and np.allclose(schedule[0], scene.G1):
                    for s in rec.steps:
                        s.gave_up = True
                    rec.n_gave_up = len(rec.steps)
                return rec

        r = search(InertWorld(scene, lambda s, sp: ("REACHED", {})),
                   make_attack("insertion"), make_picker("random", scene, 0),
                   threat_model("white"), budget=4, verbose=False)
        assert r.aborted and "inert" in r.aborted, r.aborted
    check("rejects scenes where the filter is already inert",
          t_inert_filter_scene_rejected)

    def t_search_skips_unsupported_index():
        # strict-black asks for velocity-index members; the fake world only does
        # distance. They must be SKIPPED AND REPORTED, never silently dropped.
        w = FakeWorld(scene, lambda sch, sp: ("REACHED", {}))
        r = search(w, make_attack("modification"), make_picker("random", scene, 0),
                   threat_model("strict-black"), budget=3, batch=3, verbose=False)
        assert r.n_screened == 3
        assert all(len(res.records) == 3 for res in r.results)   # 3 of 6 ran
    check("unsupported ensemble members are skipped, not hidden", t_search_skips_unsupported_index)

    def t_worst_case_aggregation():
        # score = min over members, so one easy member cannot carry a candidate
        tm = threat_model("weak-black")
        assert tm.aggregate is min
    check("aggregation is worst-case across the ensemble", t_worst_case_aggregation)


# ============================================================================ #
#  L3 — against real SPARK
# ============================================================================ #
# Every benchmark case SPARK ships (from
# pipeline/spark_pipeline/autonomy/benchmark_test_case_generator.py).
#
# IN SCOPE: the G1FixedBase family. SIREN's schedule-driven task steers the G1's
#   RIGHT wrist goal, so it needs a G1 with arm goals.
# OUT OF SCOPE, deliberately and by name — never silently skipped:
#   G1MobileBase / G1SportMode  drive a BASE goal (WG) rather than an arm goal,
#                               so the arm-goal schedule does not apply.
#   Gen3 / IIWA14 / LRMate / R1Lite  are different robots entirely (no G1 frames,
#                               no dual-wrist IK), so the task class cannot load.
SPARK_CASES_IN_SCOPE = [
    "G1FixedBase_D1_AG_SO_v0", "G1FixedBase_D1_AG_SO_v1",
    "G1FixedBase_D1_AG_DO_v0", "G1FixedBase_D1_AG_DO_v1",
    "G1FixedBase_D2_AG_SO_v0", "G1FixedBase_D2_AG_SO_v1",
    "G1FixedBase_D2_AG_DO_v0", "G1FixedBase_D2_AG_DO_v1",
]

SPARK_CASES_OUT_OF_SCOPE = [
    "G1MobileBase_D1_WG_SO_v0", "G1MobileBase_D1_WG_SO_v1",
    "G1MobileBase_D1_WG_DO_v0", "G1MobileBase_D1_WG_DO_v1",
    "G1MobileBase_D2_WG_SO_v0", "G1MobileBase_D2_WG_SO_v1",
    "G1MobileBase_D2_WG_DO_v0", "G1MobileBase_D2_WG_DO_v1",
    "G1SportMode_D1_WG_SO_v1",
    "Gen3Single_D1_AG_SO_v0", "Gen3Single_D2_AG_SO_v0",
    "IIWA14Single_D1_AG_SO_v0", "IIWA14Single_D2_AG_SO_v0",
    "LRMate200iD3f_D1_AG_SO_v0", "LRMate200iD3f_D2_AG_SO_v0",
    "R1LiteUpper_D1_AG_SO_v0", "R1LiteUpper_D2_AG_SO_v0",
]

SPARK_TEST_CASES = SPARK_CASES_IN_SCOPE

SPARK_ALGOS = ["ssa", "rssa", "pssa", "cbf", "rcbf", "sss", "rsss"]


def test_layer3(all_cases=False, budget=6):
    from fuzz.siren.world.run import World
    from fuzz.siren.world.types import FilterSpec, real_filter
    from fuzz.siren.search.attacks import make_attack
    from fuzz.siren.search.pick import make_picker
    from fuzz.siren.search.threat import threat_model
    from fuzz.siren.search.loop import search

    print("\n[L3] SPARK — real worlds, real filters, real search")

    cases = SPARK_CASES_IN_SCOPE if all_cases else SPARK_CASES_IN_SCOPE[:2]

    # ---- the case list matches what SPARK actually ships ------------------ #
    def t_case_list_complete():
        import inspect, re
        import spark_pipeline
        src = inspect.getsource(spark_pipeline.generate_benchmark_test_case)
        shipped = set(re.findall(r'case\s+"([A-Za-z0-9_]+)"', src))
        covered = set(SPARK_CASES_IN_SCOPE) | set(SPARK_CASES_OUT_OF_SCOPE)
        missing = shipped - covered
        assert not missing, f"SPARK ships cases we never account for: {sorted(missing)}"
        stale = covered - shipped
        assert not stale, f"we list cases SPARK no longer ships: {sorted(stale)}"
    check(f"every SPARK benchmark case is accounted for "
          f"({len(SPARK_CASES_IN_SCOPE)} in scope, "
          f"{len(SPARK_CASES_OUT_OF_SCOPE)} out)", t_case_list_complete)

    # ---- every benchmark case builds and runs ----------------------------- #
    for case in cases:
        def t_case(case=case):
            # The index is dictated by the robot's control mode: D1 (velocity
            # control) -> distance index; D2 (acceleration control) -> velocity-
            # augmented index. SPARK asserts on a mismatch.
            index = "velocity" if "_D2_" in case else "distance"
            spec = FilterSpec(algo="ssa", index=index, d_min=0.02, eta=0.5, k=0.1)
            w = World.build(seed=0, spec=spec, test_case=case, max_steps=30)
            assert w.supports_index == index, \
                f"{case}: expected {index}, world says {w.supports_index}"
            sc = w.scene()
            assert sc.G1.shape == (3,) and sc.n_obstacles >= 0
            rec = w.run([sc.G1], spec, max_steps=20)
            assert rec.n_steps > 0
            assert rec.label in ("REACHED", "TIMEOUT", "COLLISION", "DEADLOCK")
        check(f"SPARK case builds+runs: {case}", t_case)

    def t_index_is_dictated_by_control_mode():
        from fuzz.siren.world.sim.config import index_required_by
        assert index_required_by("G1FixedBaseDynamic1Config") == "distance"
        assert index_required_by("G1FixedBaseDynamic2Config") == "velocity"
    check("safety index is dictated by the robot's control mode",
          t_index_is_dictated_by_control_mode)

    def t_d2_world_runs_velocity_index():
        spec = FilterSpec(algo="ssa", index="velocity", d_min=0.02, eta=0.5, k=0.1)
        w = World.build(seed=0, spec=spec, test_case="G1FixedBase_D2_AG_SO_v0",
                        max_steps=25)
        assert w.supports_index == "velocity"
        assert not w.accepts(FilterSpec(index="distance"))
        rec = w.run([w.scene().G1], spec, max_steps=20)
        assert rec.n_steps > 0 and np.isfinite(rec.min_C_phi)
    check("D2 world runs the velocity-augmented index end to end",
          t_d2_world_runs_velocity_index)

    # ---- every safety filter runs ----------------------------------------- #
    for algo in SPARK_ALGOS:
        def t_algo(algo=algo):
            spec = FilterSpec(algo=algo, index="distance", d_min=0.02,
                              eta=0.5, lam=10.0)
            w = World.build(seed=0, spec=spec,
                            test_case="G1FixedBase_D1_AG_SO_v0", max_steps=25)
            rec = w.run([w.scene().G1], spec, max_steps=20)
            assert rec.n_steps > 0, "no steps recorded"
            assert np.isfinite(rec.min_C_phi), "authority never measured"
        check(f"SPARK filter runs + authority measured: {algo}", t_algo)

    # ---- the probe actually measures something ---------------------------- #
    def t_probe_measures():
        spec = real_filter(algo="ssa", d_min=0.02, eta=0.5)
        w = World.build(seed=20, spec=spec, max_steps=60)
        rec = w.run([w.scene().G1], spec, max_steps=50)
        assert np.isfinite(rec.min_C_phi), "C never computed"
        assert np.isfinite(rec.min_g), "margin never computed"
        assert any(np.isfinite(s.clearance) for s in rec.steps)
    check("probe yields finite authority and margin", t_probe_measures)

    def t_giveup_counter():
        # d_min=0.10 puts benchmark goals INSIDE the keep-out shell, so the QP is
        # infeasible constantly and the filter gives up — the counter must see it.
        spec = FilterSpec(algo="ssa", index="distance", d_min=0.10, eta=0.5)
        w = World.build(seed=20, spec=spec, max_steps=60)
        rec = w.run([w.scene().G1], spec, max_steps=50)
        assert rec.n_gave_up > 0, "give-up counter never fired at d_min=0.10"
    check("give-up counter fires when the filter is inert", t_giveup_counter)

    def t_feasible_regime_has_no_giveups():
        spec = FilterSpec(algo="ssa", index="distance", d_min=0.02, eta=0.5)
        w = World.build(seed=20, spec=spec, max_steps=120)
        rec = w.run([w.scene().G1], spec, max_steps=100)
        assert rec.label == "REACHED", f"seed-20 baseline should reach, got {rec.label}"
    check("seed-20 baseline reaches at d_min=0.02 (filter active+feasible)",
          t_feasible_regime_has_no_giveups)

    # ---- scene is stable across runs (the experiment depends on it) ------- #
    def t_scene_is_fixed():
        spec = real_filter(algo="ssa", d_min=0.02, eta=0.5)
        w = World.build(seed=7, spec=spec, max_steps=20)
        a = w.scene()
        w.run([a.G1], spec, max_steps=10)
        b = w.harness.scene()
        assert np.allclose(a.G1, b.G1) and np.allclose(a.G0, b.G0), "scene drifted"
    check("scene identical across runs (same seed)", t_scene_is_fixed)

    # ---- retuning changes behaviour --------------------------------------- #
    def t_retune_changes_demand():
        w = World.build(seed=20, spec=real_filter(algo="ssa", d_min=0.02, eta=0.5),
                        max_steps=40)
        lo = w.run([w.scene().G1], FilterSpec(algo="ssa", d_min=0.02, eta=0.05),
                   max_steps=30)
        hi = w.run([w.scene().G1], FilterSpec(algo="ssa", d_min=0.02, eta=2.0),
                   max_steps=30)
        assert lo.min_g != hi.min_g, "retune had no effect on the margin"
    check("retuning the demand changes the margin", t_retune_changes_demand)

    # ---- the velocity index needs acceleration control -------------------- #
    def t_velocity_index_needs_D2():
        w = World.build(seed=0, spec=FilterSpec(algo="ssa", index="distance"),
                        test_case="G1FixedBase_D1_AG_SO_v0", max_steps=20)
        assert w.supports_index == "distance"
        assert not w.accepts(FilterSpec(index="velocity"))
    check("D1 world declares distance-only index support", t_velocity_index_needs_D2)

    # ---- end-to-end searches ---------------------------------------------- #
    for threat in ("white", "gray", "weak-black", "strict-black", "random"):
        def t_search(threat=threat):
            spec = real_filter(algo="ssa", index="distance", d_min=0.02, eta=0.5)
            w = World.build(seed=20, spec=spec, max_steps=200)
            tm = threat_model(threat, d_min=0.02, index="distance", real=spec)
            r = search(w, make_attack("insertion"),
                       make_picker("random", w.scene(), seed=0), tm,
                       budget=budget, batch=budget, max_steps=200, verbose=False)
            assert r.aborted is None, r.aborted
            assert r.n_screened > 0, "nothing screened"
            for res in r.results:
                assert np.isfinite(res.score), "non-finite score"
        check(f"end-to-end insertion search: threat={threat}", t_search)

    def t_modification_search():
        spec = real_filter(algo="ssa", index="distance", d_min=0.02, eta=0.5)
        w = World.build(seed=20, spec=spec, max_steps=200)
        r = search(w, make_attack("modification"),
                   make_picker("cem", w.scene(), seed=0),
                   threat_model("white", d_min=0.02, real=spec),
                   budget=budget, batch=budget, max_steps=200, verbose=False)
        assert r.n_screened > 0
    check("end-to-end modification search (CEM picker)", t_modification_search)


# ============================================================================ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spark", action="store_true", help="run the slow SPARK layer")
    ap.add_argument("--all-cases", action="store_true", help="every benchmark case")
    ap.add_argument("--budget", type=int, default=6)
    args, _ = ap.parse_known_args()

    print("=" * 70)
    print("SIREN test suite")
    print("=" * 70)

    test_layer1()
    test_layer2()
    if args.spark:
        test_layer3(all_cases=args.all_cases, budget=args.budget)
    else:
        print("\n[L3] skipped (pass --spark to run against real SPARK)")

    n = len(_RESULTS)
    bad = [r for r in _RESULTS if not r[1]]
    print("\n" + "=" * 70)
    print(f"{n - len(bad)}/{n} passed")
    for name, _, detail in bad:
        print(f"  FAILED  {name}\n          {detail}")
    print("=" * 70)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
