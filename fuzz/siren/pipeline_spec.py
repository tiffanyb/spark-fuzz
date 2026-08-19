"""
THE PIPELINE — one inserted goal, evaluated exactly as specified.

    0. admissible?                                   geometry only
    1. run G0 -> G1'
         COLLIDES / DEADLOCKS -> modification hit; score 0 for the insertion search
         TIMES OUT            -> discard (not fed to the picker at all)
         REACHES              -> continue
    2. continue from that state to G1  (departing IN MOTION)
    3. record t0, mu on leg 2, engagement, handover state
    4. score, lexicographically:  engaged -> -min braking margin
                                  not engaged -> -min clearance
    5. classify:  mu(x_t0) > 0 -> Kind 1
                  else mu > 0 later on leg 2 -> Kind 2
                  else no infeasibility
    6. feed the score to the picker

The ONLY thing that varies between experimental arms is which picker supplies
step 6's next candidate. Everything above is identical, so a difference between
arms is a difference between search strategies and nothing else.

--- two places where the spec had to be made sign-consistent ----------------

`mu` is the exact multi-constraint LP margin

    mu(x) = min_u max_i ( L_f phi_i + L_g phi_i . u + demand_i )

so mu <= 0 means SOME control satisfies every active constraint, and mu > 0
means none does. Infeasible is mu > 0 (derived.evaluate uses exactly this test).

  * Step 5's second clause was written "mu < 0 later -> Kind 2", which under this
    convention is the condition for staying FEASIBLE. Kind 2 is late-arriving
    infeasibility, so it is implemented as mu > 0 later.
  * Step 4 was written "-min mu". Under this convention that rewards the most
    comfortably feasible candidate. It was first implemented as `max mu` (same
    intent, sign matching the code) -- and that turned out to be wrong too, for a
    reason neither sign convention could fix: mu simply does not predict contact
    here. Measured over 9936 engaged candidates, the mu quartile CLOSEST to
    infeasibility collided 0.0% of the time and the farthest 18.1%
    (corr = -0.14), so ANY monotone function of mu steers the search the wrong
    way. Step 4's guidance is therefore the BRAKING MARGIN instead; see
    StepMeasurement.brake_margin. The lexicographic first level is unchanged and
    was the only part earning its keep: every collision observed was engaged.

--- one run, not two -------------------------------------------------------

Steps 1 and 2 are a SINGLE continuous run of the schedule [G1', G1]. Leg 1's
outcome is read off the waypoint index rather than by running [G1'] separately,
because leg 2 must depart from the state leg 1 actually ended in, in motion. Two
separate runs would settle the arm at G1' first and measure a different system.
It is also half the simulation cost.
"""

import os
from dataclasses import dataclass, field, asdict
from typing import Optional, List

import numpy as np

from .search.pick import is_admissible
from .world.types import Scene

#: any engaged candidate outranks any unengaged one; the offset just encodes
#: that ordering in the single float the pickers consume.
_ENGAGED_OFFSET = 1000.0

#: Which quantity the ENGAGED branch of the lexicographic score minimises.
#: See SCORING_HISTORY.md for every objective tried and why.
#:
#:   "brake"  clearance - v^2/(2 s C)   relative-degree-2 momentum commitment.
#:            L_f-free by construction. Monotone with collisions (corr -0.59),
#:            but blind to D1, which has no braking distance.
#:   "g"      C - demand - L_f phi      the EXACT feasibility test: g < 0 iff the
#:            QP has no solution. Rearranged, C >= eta + L_f phi -- the authority
#:            condition, in both families. The only lever in D1.
#:
#: `g` was refuted early and that refutation is void: L_f carried a sign error
#: that inflated g by 2x the closing speed, largest exactly when approaching
#: fastest, which is why it "never went negative" across ~18k candidates.
SCORE_MODE = os.environ.get("SIREN_SCORE", "brake").strip().lower()

KINDS = ("KIND_1", "KIND_2", "NO_INFEASIBILITY", "NEVER_ENGAGED")
STAGES = ("inadmissible", "leg1_timeout", "modification_hit", "evaluated")


@dataclass
class Evaluation:
    """Everything observed about one candidate. Kept whole for the record."""
    candidate: List[float]
    stage: str
    score: Optional[float] = None          # None => not fed to the picker

    # ---- leg 1 -------------------------------------------------------- #
    leg1_label: Optional[str] = None
    leg1_reached: bool = False

    # ---- leg 2 -------------------------------------------------------- #
    leg2_label: Optional[str] = None
    collided: bool = False
    engaged: bool = False
    t0: Optional[int] = None               # first intervention ON LEG 2
    mu_t0: Optional[float] = None
    max_mu_leg2: Optional[float] = None
    min_brake_margin: Optional[float] = None
    min_g_leg2: Optional[float] = None
    min_clearance_leg2: Optional[float] = None
    n_engaged_leg2: int = 0
    n_steps_leg2: int = 0
    n_gave_up: int = 0
    min_g: Optional[float] = None

    # ---- handover (step 3) -------------------------------------------- #
    handover_step: Optional[int] = None
    handover_clearance: Optional[float] = None
    handover_phi: Optional[float] = None
    handover_mu: Optional[float] = None

    # ---- verdict ------------------------------------------------------ #
    kind: Optional[str] = None
    n_steps_total: int = 0

    def as_row(self):
        return asdict(self)


def _f(x):
    return float(x) if (x is not None and np.isfinite(x)) else None


def classify(engaged, t0, mu_t0, mu_after_t0_max):
    """Step 5. Kind is defined by WHEN feasibility was lost, on leg 2."""
    if not engaged or t0 is None:
        return "NEVER_ENGAGED"
    if mu_t0 is not None and mu_t0 > 1e-9:
        return "KIND_1"                       # already infeasible when it acted
    if mu_after_t0_max is not None and mu_after_t0_max > 1e-9:
        return "KIND_2"                       # lost feasibility only afterwards
    return "NO_INFEASIBILITY"


def evaluate(world, scene: Scene, cand, max_steps=None) -> Evaluation:
    """Steps 0-5 for one candidate."""
    cand = np.asarray(cand, dtype=float)
    ev = Evaluation(candidate=[float(x) for x in cand], stage="evaluated")

    # -- step 0 ---------------------------------------------------------- #
    if not is_admissible(cand, scene)[0]:
        ev.stage = "inadmissible"
        return ev

    # -- steps 1+2: ONE continuous run ----------------------------------- #
    rec = world.run([cand, np.asarray(scene.G1)], max_steps=max_steps,
                    exact_margin=True)
    ev.n_steps_total = len(rec.steps)
    leg2 = [s for s in rec.steps if s.wp_idx >= 1]
    ev.leg1_reached = len(leg2) > 0

    if not ev.leg1_reached:
        # leg 1 never handed over: its label IS the whole run's label
        ev.leg1_label = rec.label
        if rec.label in ("COLLISION", "DEADLOCK"):
            ev.stage = "modification_hit"
            ev.score = 0.0                    # neutral for the INSERTION search
            ev.collided = (rec.label == "COLLISION")
        else:
            ev.stage = "leg1_timeout"         # discarded: score stays None
        return ev

    ev.leg1_label = "REACHED"
    ev.leg2_label = rec.label
    ev.collided = (rec.label == "COLLISION")
    ev.n_steps_leg2 = len(leg2)
    ev.n_gave_up = int(rec.n_gave_up)
    ev.min_g = _f(rec.min_g)

    # -- step 3: handover + leg-2 timeline ------------------------------- #
    h = leg2[0]
    ev.handover_step = int(h.step)
    ev.handover_clearance = _f(h.clearance)
    ev.handover_phi = _f(h.phi)
    ev.handover_mu = _f(h.mu)

    engaged_steps = [s for s in leg2 if s.engaged]
    ev.n_engaged_leg2 = len(engaged_steps)
    ev.engaged = bool(engaged_steps)
    clears = [s.clearance for s in leg2 if np.isfinite(s.clearance)]
    ev.min_clearance_leg2 = float(min(clears)) if clears else None

    # BUG-4: the engaged score used to be max mu, which is ANTI-correlated with
    # collisions -- measured on 9936 engaged candidates, the quartile of mu
    # CLOSEST to infeasibility had a 0.0% collision rate while the farthest had
    # 18.1% (corr = -0.14). The search was being steered away from attacks at its
    # second level; only the lexicographic first level (engaged beats unengaged)
    # was pulling the right way. Guidance is now the BRAKING MARGIN, the
    # quantity that actually predicts contact here.
    min_brake = None
    bm = [float(s.brake_margin) for s in leg2 if np.isfinite(s.brake_margin)]
    if bm:
        min_brake = min(bm)
    ev.min_brake_margin = min_brake

    # g is only defined where a constraint is actually enforced (the P3 rule in
    # derived.evaluate returns inf otherwise), so restrict to engaged steps.
    gs = [float(s.g) for s in leg2 if s.engaged and np.isfinite(s.g)]
    min_g_l2 = min(gs) if gs else None
    ev.min_g_leg2 = min_g_l2

    mu_t0 = None
    max_mu = None
    max_mu_after = None
    if ev.engaged:
        t0_step = engaged_steps[0]
        ev.t0 = int(t0_step.step)
        mu_t0 = _f(t0_step.mu)
        mus = [float(s.mu) for s in engaged_steps if np.isfinite(s.mu)]
        max_mu = max(mus) if mus else None
        after = [float(s.mu) for s in engaged_steps
                 if s.step > t0_step.step and np.isfinite(s.mu)]
        max_mu_after = max(after) if after else None
    ev.mu_t0 = mu_t0
    ev.max_mu_leg2 = max_mu

    # -- step 4: lexicographic score ------------------------------------- #
    if ev.engaged:
        # Both branches MINIMISE their quantity: smaller braking margin = deeper
        # inside the stopping distance; smaller g = closer to the QP having no
        # solution at all. Falling back to 0 when the quantity is never finite
        # still leaves an engaged candidate above every unengaged one, which is
        # the lexicographic rule.
        key = min_g_l2 if SCORE_MODE == "g" else min_brake
        ev.score = _ENGAGED_OFFSET - (key if key is not None else 0.0)
    else:
        mc = ev.min_clearance_leg2
        ev.score = -float(mc) if mc is not None else 0.0

    # -- step 5 ----------------------------------------------------------- #
    ev.kind = classify(ev.engaged, ev.t0, mu_t0, max_mu_after)
    return ev


def run_search(world, scene: Scene, picker, budget=30, batch=10, max_steps=None,
               verbose=False):
    """Steps 0-6 over a budget of candidates. Returns every Evaluation."""
    evals: List[Evaluation] = []
    n_fed = 0
    while len(evals) < budget:
        want = min(batch, budget - len(evals))
        cands = picker.ask(want)
        if not cands:
            break
        scored = []
        for c in cands:
            ev = evaluate(world, scene, c, max_steps=max_steps)
            evals.append(ev)
            if ev.score is not None:          # discards are NOT fed back
                scored.append((c, ev.score))
            if len(evals) >= budget:
                break
        if scored:
            picker.tell(scored)               # step 6
            n_fed += len(scored)
        if verbose:
            print(f"   {len(evals)}/{budget} evaluated, "
                  f"{sum(1 for e in evals if e.collided)} collisions", flush=True)
    return evals


def summarize(evals: List[Evaluation]) -> dict:
    """Counts that the experiment reports. Collisions are split by WHICH leg and
    by kind, because 'a collision' on leg 1 is a modification hit and says
    nothing about the inserted-goal attack."""
    out = {
        "n_evaluated": len(evals),
        "n_inadmissible": sum(1 for e in evals if e.stage == "inadmissible"),
        "n_leg1_timeout": sum(1 for e in evals if e.stage == "leg1_timeout"),
        "n_modification_hits": sum(1 for e in evals if e.stage == "modification_hit"),
        "n_full": sum(1 for e in evals if e.stage == "evaluated"),
        "n_collisions_leg1": sum(1 for e in evals
                                 if e.stage == "modification_hit" and e.collided),
        "n_collisions_leg2": sum(1 for e in evals
                                 if e.stage == "evaluated" and e.collided),
        "n_engaged": sum(1 for e in evals if e.engaged),
        "n_gave_up_runs": sum(1 for e in evals if e.n_gave_up > 0),
    }
    for k in KINDS:
        out[f"kind_{k}"] = sum(1 for e in evals if e.kind == k)
        out[f"kind_{k}_collided"] = sum(1 for e in evals
                                        if e.kind == k and e.collided)
    full = [e for e in evals if e.stage == "evaluated"]
    out["attack_rate_leg2"] = (out["n_collisions_leg2"] / len(full)) if full else None
    mus = [e.max_mu_leg2 for e in evals if e.max_mu_leg2 is not None]
    out["max_mu_seen"] = float(max(mus)) if mus else None
    return out
