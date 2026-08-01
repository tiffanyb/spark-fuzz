"""
The flight recorder — what goes OUT of a run.

StepMeasurement   one snapshot per timestep. Holds BOTH things observed
                  directly (clearance, configuration, whether the robot visibly
                  gave up) and things computed from them (authority, margin).
                  They live together because they describe the same instant;
                  the FORMULAS live in derived.py, only the NUMBERS live here.

RunRecord         the whole run: every step, the summary scalars the attacker
                  scores on, and the verdict label.

classify_run      steps -> REACHED | COLLISION | DEADLOCK | TIMEOUT.

Pure data + one classifier. No SPARK, no MuJoCo, no scipy — so the DEADLOCK
criteria below can be unit-tested against hand-made step lists.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------- #
#  One timestep
# ---------------------------------------------------------------------------- #
@dataclass
class StepMeasurement:
    step: int
    wp_idx: int                       # which scheduled waypoint is active
    dist_final: float                 # end-effector -> the legitimate goal G1
    reached_final: bool
    clearance: float                  # min robot-obstacle distance (<0 = penetration)

    # --- authority family (formulas in derived.py) ---------------------- #
    C_d: float = np.inf               # retreat capacity  — index-FREE ruler
    C_phi: float = np.inf             # authority w.r.t. the deployed index
    phi: float = -np.inf              # the danger number (>=0 means engaged)
    demand: float = np.nan            # what the filter insists on
    g: float = np.inf                 # C_phi - demand - L_f phi ; <0 => void

    # The EXACT multi-constraint LP margin, populated only when a run is made
    # with exact_margin=True (it costs an LP per step). Sign is the OPPOSITE of
    # g by construction: mu = min_u max_i (L_f phi_i + L_g phi_i . u + demand_i),
    # so mu <= 0 means some u satisfies EVERY active constraint, and mu > 0 means
    # none does. g only ever looks at the single binding constraint, so it can
    # read "safe" while several constraints are jointly unsatisfiable -- which is
    # exactly the pincer case the Kind-1/Kind-2 split has to get right.
    mu: float = np.nan

    #: BRAKING MARGIN: clearance - v_closing^2 / (2 * BRAKE_SCALE * C_phi).
    #: Negative means the pair is already inside its stopping distance, i.e. no
    #: control can arrest the approach in time. This is the second-order,
    #: INTEGRATED quantity that mu (a first-order rate condition) cannot express,
    #: and it is what actually predicts collisions here: validated at 83% kind
    #: agreement against the brute-force escape search on 77 labelled runs.
    brake_margin: float = np.inf

    # --- what the filter visibly did ----------------------------------- #
    engaged: bool = False             # a constraint is actually being enforced
    trigger_safe: bool = False        # the filter intervened this step
    deviation: float = 0.0            # ||u_safe - u_ref||  (the observable signal)
    gave_up: bool = False             # QP infeasible -> u_ref   OBSERVED
    predicted_infeasible: bool = False  # g < 0                  ANALYTIC
    slack: float = 0.0

    q: Optional[np.ndarray] = None    # configuration (full state observability)

    @property
    def penetration(self) -> float:
        """How far inside the keep-out shell, >=0. The demand hedge rewards this."""
        return max(0.0, -self.clearance) if np.isfinite(self.clearance) else 0.0


# ---------------------------------------------------------------------------- #
#  Classification
# ---------------------------------------------------------------------------- #
def classify_run(steps: List[StepMeasurement],
                 schedule=None,
                 collision_margin: float = 0.0,
                 reach_radius: float = 0.05,
                 stall_window: int = 100,
                 stall_std_eps: float = 3e-3,
                 progress_eps: float = 1e-2,
                 far_factor: float = 2.0) -> str:
    """Reduce per-step measurements to one verdict.

    DEADLOCK is deliberately strict — it requires ALL of:
      * not reached and not collided,
      * a full-length attempt (>= stall_window steps),
      * the end-effector still FAR from the goal,
      * a persistent FREEZE over the trailing window (tiny spread AND tiny net
        progress) — so "slow but arriving" is TIMEOUT, not a trap,
      * and the freeze happens on the FINAL leg. Otherwise a slow-to-reach
        inserted goal eats the horizon and the run freezes on the *inserted*
        leg with no budget left for G1 — a horizon artifact, not a trap.

    Only DEADLOCK and COLLISION count as attack successes; TIMEOUT is
    explicitly inconclusive.
    """
    n = len(steps)
    if n == 0:
        return "TIMEOUT"

    clearances = np.array([s.clearance for s in steps], dtype=float)
    dist_final = np.array([s.dist_final for s in steps], dtype=float)

    if np.any(clearances < -abs(collision_margin)):
        return "COLLISION"
    if any(s.reached_final for s in steps):
        return "REACHED"

    w = int(min(stall_window, n))
    tail = dist_final[-w:]
    tail_progress = float(tail[0] - tail[-1])      # > 0 means approaching
    tail_std = float(np.std(tail))
    final_dist = float(dist_final[-1])

    final_wp = (len(schedule) - 1) if schedule else 0
    on_final_leg = all(s.wp_idx >= final_wp for s in steps[-w:])

    far = final_dist > far_factor * reach_radius
    frozen = (tail_std < stall_std_eps) and (abs(tail_progress) < progress_eps)
    full_length = n >= stall_window

    return "DEADLOCK" if (far and frozen and full_length and on_final_leg) else "TIMEOUT"


# ---------------------------------------------------------------------------- #
#  The whole run
# ---------------------------------------------------------------------------- #
@dataclass
class DeviationEvent:
    """When the filter first visibly intervened. The primitive the speed sweep
    reads: activation clearance vs approach speed identifies the index family."""
    step: int
    clearance: float
    approach_speed: float


@dataclass
class HandoverState:
    """The configuration at the moment the robot switches to the FINAL goal.

    For an insertion attack this is the one state the inserted goal actually
    controls: after it, every candidate flies the same last leg to the same
    fixed G1. Statistics taken over the whole trajectory are therefore dominated
    by that shared approach and cannot tell candidates apart -- the handover can.
    Undefined for a single-leg (modification) run, where there is no switch.
    """
    step: int
    clearance: float
    phi: float
    C_d: float
    C_phi: float
    g: float
    dist_final: float


@dataclass
class RunRecord:
    label: str
    schedule: List
    filter_spec: object                    # the FilterSpec this ran under
    steps: List[StepMeasurement] = field(default_factory=list)

    # --- summary scalars (what the attacker scores on) ------------------ #
    min_C_d: float = np.inf
    min_C_phi: float = np.inf
    min_g: float = np.inf                  # over ENGAGED steps only
    max_penetration: float = 0.0
    min_clearance: float = np.inf
    final_dist: float = np.inf
    n_gave_up: int = 0                     # OBSERVED  — legal for every tier
    n_predicted_infeasible: int = 0        # ANALYTIC  — white-box only
    first_deviation: Optional[DeviationEvent] = None
    reached: bool = False
    collided: bool = False
    n_steps: int = 0

    # --- engagement extent: how LONG the filter struggled, not how deep -- #
    n_engaged: int = 0
    frac_engaged: float = 0.0
    exposure: float = 0.0                  # sum of (-g)+ over engaged steps
    # --- the handover, and the leg the attack actually happens on -------- #
    handover: Optional[HandoverState] = None
    min_clearance_final_leg: float = np.inf
    max_phi_engaged: float = -np.inf

    def is_attack_success(self) -> bool:
        """Only a confirmed unsafe outcome counts. TIMEOUT is inconclusive."""
        return self.label in ("DEADLOCK", "COLLISION")

    # -- construction --------------------------------------------------- #
    @classmethod
    def summarize(cls, steps, schedule, filter_spec, **kw) -> "RunRecord":
        label = classify_run(steps, schedule=schedule)

        def _min(attr, default=np.inf):
            vals = [getattr(s, attr) for s in steps]
            vals = [v for v in vals if v is not None and np.isfinite(v)]
            return float(min(vals)) if vals else default

        first_dev = None
        for s in steps:
            if s.trigger_safe or s.deviation > 0:
                first_dev = DeviationEvent(step=s.step, clearance=s.clearance,
                                           approach_speed=float("nan"))
                break

        # Margin statistics are meaningful only where the filter is enforcing.
        engaged = [s for s in steps if s.engaged]

        def _min_over(seq, attr, default=np.inf):
            vals = [getattr(s, attr) for s in seq
                    if getattr(s, attr) is not None and np.isfinite(getattr(s, attr))]
            return float(min(vals)) if vals else default

        # The handover: first step on the last leg. Only meaningful when there
        # IS an earlier leg (i.e. an inserted goal); a single-goal run has none.
        n_wp = len(schedule) if schedule is not None else 1
        handover = None
        final_leg = [s for s in steps if s.wp_idx >= n_wp - 1]
        if n_wp > 1 and final_leg:
            h = final_leg[0]
            handover = HandoverState(step=h.step, clearance=h.clearance, phi=h.phi,
                                     C_d=h.C_d, C_phi=h.C_phi, g=h.g,
                                     dist_final=h.dist_final)

        return cls(
            label=label,
            schedule=list(schedule) if schedule is not None else [],
            filter_spec=filter_spec,
            steps=steps,
            min_C_d=_min("C_d"),
            min_C_phi=_min_over(engaged, "C_phi"),
            min_g=_min_over(engaged, "g"),          # ENGAGED steps only
            max_penetration=max((s.penetration for s in steps), default=0.0),
            min_clearance=_min("clearance"),
            final_dist=float(steps[-1].dist_final) if steps else np.inf,
            n_gave_up=int(sum(s.gave_up for s in steps)),
            n_predicted_infeasible=int(sum(s.predicted_infeasible for s in steps)),
            first_deviation=first_dev,
            reached=any(s.reached_final for s in steps),
            collided=(label == "COLLISION"),
            n_steps=len(steps),
            n_engaged=len(engaged),
            frac_engaged=(len(engaged) / len(steps)) if steps else 0.0,
            exposure=float(sum(max(0.0, -s.g) for s in engaged
                               if np.isfinite(s.g))),
            handover=handover,
            min_clearance_final_leg=_min_over(final_leg, "clearance"),
            max_phi_engaged=float(max((s.phi for s in engaged
                                       if np.isfinite(s.phi)), default=-np.inf)),
            **kw,
        )

    def brief(self) -> str:
        return (f"{self.label:<9} steps={self.n_steps:<4} "
                f"min_C_d={self.min_C_d:.3f} min_g={self.min_g:.3f} "
                f"pen={self.max_penetration:.4f} gave_up={self.n_gave_up}")
