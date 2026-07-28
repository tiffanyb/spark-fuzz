"""
What the attacker may KNOW, SEE, and WANT.

One object — AttackerCapability — declares the tier, and everything downstream
is DERIVED from it: which fields are readable, which objective is used, and how
many surrogate filters must be rolled out. Nothing is chosen by hand, so it is
impossible to accidentally hand a black-box attacker a white-box objective.

The failure condition every tier is chasing is the same:

    g = C - demand - drift  <  0        no control in the actuator box can hold
                                        the demanded rate; the guarantee is void

They differ only in how much of g they can actually compute:

    white   knows the demand exactly            -> computes g. Surgical.
    gray    knows the demand's SHAPE, not scale -> uses C_phi; for a constant
                                                   demand the unknown scale is a
                                                   common offset, so the best
                                                   target is unchanged.
    black   knows neither                       -> uses the index-free ruler C_d
                                                   and hedges with penetration,
                                                   because a proportional demand
                                                   vanishes at the boundary and
                                                   cannot be beaten by grazing.
"""

import os
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, List, Optional

import numpy as np

from ..world.types import (FilterSpec, gray_ensemble, real_filter,
                           strict_black_ensemble, weak_black_ensemble)


# ---------------------------------------------------------------------------- #
#  The axes
# ---------------------------------------------------------------------------- #
class IndexKnowledge(Enum):
    EXACT = "exact"        # the danger-number formula and its parameters
    FAMILY = "family"      # family identified and fitted (post speed-sweep)
    UNKNOWN = "unknown"


class DemandShape(Enum):
    CONSTANT = "constant"          # d = eta        — bites even at the boundary
    PROPORTIONAL = "proportional"  # d = lambda*phi — vanishes at the boundary
    UNKNOWN = "unknown"


class Observability(Enum):
    FULL_STATE = "full"        # joint angles + velocities
    COARSE_MOTION = "coarse"   # whole-body visual; onset of dodging seen LATE


# ---------------------------------------------------------------------------- #
#  The capability
# ---------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AttackerCapability:
    """Everything the attacker knows, sees and can do. One per threat model."""

    index: IndexKnowledge
    demand_shape: DemandShape
    demand_coefficient_known: bool
    observability: Observability = Observability.COARSE_MOTION

    # Always available — stated explicitly rather than silently assumed.
    robot_model: bool = True        # the URDF is public: kinematics + limits
    scene_geometry: bool = True     # the attacker can see the room
    can_issue_goals: bool = True    # the attack surface
    can_simulate: bool = True       # owns an offline simulator
    probe_budget: Optional[int] = None   # real-machine trials allowed

    # -- what can I actually compute? ----------------------------------- #
    @property
    def can_compute_phi(self) -> bool:
        return self.index is not IndexKnowledge.UNKNOWN

    @property
    def can_compute_C_phi(self) -> bool:
        return self.can_compute_phi and self.robot_model

    @property
    def can_compute_C_d(self) -> bool:
        """The index-free retreat capacity needs only the public robot model and
        the obstacle layout — never the filter."""
        return self.robot_model and self.scene_geometry

    @property
    def can_compute_demand(self) -> bool:
        return (self.demand_shape is not DemandShape.UNKNOWN
                and self.demand_coefficient_known)

    @property
    def can_compute_g(self) -> bool:
        return self.can_compute_C_phi and self.can_compute_demand

    @property
    def needs_penetration_hedge(self) -> bool:
        """Needed when the demand does not bite at the boundary — either because
        it is proportional, or because we cannot rule that out."""
        return self.demand_shape in (DemandShape.PROPORTIONAL, DemandShape.UNKNOWN)

    @property
    def ruler(self) -> str:
        return "min_C_phi" if self.can_compute_C_phi else "min_C_d"

    @property
    def can_identify_passively(self) -> bool:
        """Full state lets the attacker harvest activation events from ordinary
        operation — identification with no injected commands at all."""
        return self.observability is Observability.FULL_STATE

    def describe(self) -> str:
        return (f"index={self.index.value} demand={self.demand_shape.value} "
                f"coeff={'yes' if self.demand_coefficient_known else 'no'} "
                f"obs={self.observability.value} -> ruler={self.ruler} "
                f"hedge={'yes' if self.needs_penetration_hedge else 'no'}")


# ---------------------------------------------------------------------------- #
#  What may be read  (the knowledge budget, enforced)
# ---------------------------------------------------------------------------- #
class KnowledgeError(AttributeError):
    """Raised when an objective reads a field its tier cannot know. This is what
    makes the soundness claim checkable instead of merely documented."""


# Behaviour anyone can observe from outside the robot.
_OBSERVABLE = frozenset({
    "label", "reached", "collided", "n_steps", "steps", "schedule",
    "max_penetration", "min_clearance", "final_dist",
    "n_gave_up",          # the robot visibly reverted to the reference control
    "first_deviation",    # onset of dodging — the speed-sweep primitive
    "is_attack_success", "brief", "filter_spec",
})


class RecordView:
    ALLOWED = _OBSERVABLE

    def __init__(self, record):
        object.__setattr__(self, "_r", record)

    def __getattr__(self, name):
        if name not in type(self).ALLOWED:
            raise KnowledgeError(
                f"{type(self).__name__} may not read '{name}' — outside this "
                f"attacker's knowledge budget")
        return getattr(object.__getattribute__(self, "_r"), name)


class WhiteView(RecordView):
    """Knows the filter and its coefficient: the exact certificate is available."""
    ALLOWED = _OBSERVABLE | {"min_g", "demand", "phi", "min_C_phi", "min_C_d",
                             "n_predicted_infeasible"}


class GrayView(RecordView):
    """Index known, demand unknown. Shared by gray and weak black-box — they
    read the same fields but differ in what they DO with them."""
    ALLOWED = _OBSERVABLE | {"min_C_phi", "min_C_d", "phi"}


class BlackView(RecordView):
    """Index unknown: only the index-free ruler survives."""
    ALLOWED = _OBSERVABLE | {"min_C_d"}


# ---------------------------------------------------------------------------- #
#  What the attacker SEES
# ---------------------------------------------------------------------------- #
class ObservabilityModel:
    def observe(self, record):
        return record


class FullState(ObservabilityModel):
    """Exact joint state; the onset of intervention is seen the instant it starts."""


class CoarseMotion(ObservabilityModel):
    """Whole-body visual. Modelled as EXACT BUT DELAYED: a deviation is only
    noticed once it grows past a visible threshold, so the measured activation
    clearance is biased inward by a roughly constant amount. That biases the
    INTERCEPT of the speed sweep but not its SLOPE — which is why the index
    family can still be identified from weak sensing."""

    def __init__(self, deviation_threshold: float = 1e-2):
        self.deviation_threshold = deviation_threshold

    def observe(self, record):
        from ..world.measure import DeviationEvent
        first = None
        for s in record.steps:
            if s.deviation > self.deviation_threshold:
                first = DeviationEvent(step=s.step, clearance=s.clearance,
                                       approach_speed=float("nan"))
                break
        record.first_deviation = first
        return record


# ---------------------------------------------------------------------------- #
#  What the attacker WANTS
# ---------------------------------------------------------------------------- #
# A confirmed break beats any near-miss, so the outcome dominates the score; the
# guidance term supplies the gradient while nothing has broken yet.
#
# A confirmed break dominates any near-miss; TIMEOUT sits in between because it
# is *inconclusive* (the robot neither arrived nor collided) yet still indicates
# more strain than a clean reach, which gives a concentrating search something to
# climb.
#
# We suspected the TIMEOUT reward was luring CEM into abundant near-stall regions
# and away from rare collisions. An ablation on seed 20 of
# G1FixedBase_D1_AG_SO_v0 (CEM, white box, budget 20) DISPROVED that: bonus 0.0
# and bonus 1.0 both produced 0/20. The actual cause was the experiment's
# batching -- with batch == budget, CEM gets a single generation and never uses
# its feedback at all. Kept configurable for future study.
_TIMEOUT_BONUS = float(os.environ.get("SIREN_TIMEOUT_BONUS", "1.0"))

LABEL_BONUS = {"DEADLOCK": 3.0, "COLLISION": 2.0,
               "TIMEOUT": _TIMEOUT_BONUS, "REACHED": 0.0}


class Objective:
    view_class = RecordView

    def score(self, record) -> float:
        v = self.view_class(record)
        base = LABEL_BONUS.get(v.label, 0.0)
        return base + self.guidance(v)

    def guidance(self, v) -> float:
        return 0.0


class CertificateObjective(Objective):
    """White-box: the demand is known, so drive the exact margin negative."""
    view_class = WhiteView

    def guidance(self, v) -> float:
        g = v.min_g
        return -float(g) if np.isfinite(g) else 0.0


class MarginProxyObjective(Objective):
    """Everyone else: the demand is not fully known, so aim at the part of the
    margin that IS computable — the authority — optionally hedged with
    penetration when the demand might vanish at the boundary."""

    def __init__(self, ruler="min_C_d", hedge=True, beta=1.0, view_class=BlackView):
        self.ruler = ruler
        self.hedge = hedge
        self.beta = beta
        self.view_class = view_class

    def guidance(self, v) -> float:
        c = getattr(v, self.ruler)
        s = -float(c) if np.isfinite(c) else 0.0
        if self.hedge:
            s += self.beta * float(v.max_penetration)
        return s


class RandomObjective(Objective):
    """The baseline: no guidance at all, only the outcome."""
    view_class = BlackView

    def guidance(self, v) -> float:
        return 0.0


# ---------------------------------------------------------------------------- #
#  Bundling
# ---------------------------------------------------------------------------- #
def view_for(cap: AttackerCapability):
    if cap.can_compute_g:
        return WhiteView
    if cap.can_compute_phi:
        return GrayView
    return BlackView


def objective_for(cap: AttackerCapability, beta: float = 1.0) -> Objective:
    if cap.can_compute_g:
        return CertificateObjective()
    return MarginProxyObjective(ruler=cap.ruler,
                                hedge=cap.needs_penetration_hedge,
                                beta=beta,
                                view_class=view_for(cap))


def specs_for(cap: AttackerCapability, d_min: float,
              index: str = "distance", real: FilterSpec = None) -> List[FilterSpec]:
    """The surrogate ensemble. White/gray get one member, so the ensemble is the
    general case rather than a black-box-only branch."""
    if cap.can_compute_demand:
        return [real or real_filter(d_min=d_min, index=index, eta=0.5)]
    if cap.demand_shape is not DemandShape.UNKNOWN:
        return gray_ensemble(cap.demand_shape.value, index=index, d_min=d_min)
    if cap.can_compute_phi:
        return weak_black_ensemble(index=index, d_min=d_min)
    return strict_black_ensemble(d_min=d_min)


@dataclass
class ThreatModel:
    name: str
    capability: AttackerCapability
    objective: Objective
    specs: List[FilterSpec]
    observability: ObservabilityModel
    aggregate: Callable = min          # WORST-CASE across the ensemble

    def describe(self) -> str:
        return (f"[{self.name}] {self.capability.describe()} "
                f"specs={len(self.specs)} aggregate={self.aggregate.__name__}")


# ---------------------------------------------------------------------------- #
#  Presets
# ---------------------------------------------------------------------------- #
PRESETS = {
    "white": AttackerCapability(
        index=IndexKnowledge.EXACT,
        demand_shape=DemandShape.CONSTANT,
        demand_coefficient_known=True,
        observability=Observability.FULL_STATE),

    "gray": AttackerCapability(
        index=IndexKnowledge.EXACT,
        demand_shape=DemandShape.CONSTANT,
        demand_coefficient_known=False),

    "gray-proportional": AttackerCapability(
        index=IndexKnowledge.EXACT,
        demand_shape=DemandShape.PROPORTIONAL,
        demand_coefficient_known=False),

    "weak-black": AttackerCapability(
        index=IndexKnowledge.FAMILY,
        demand_shape=DemandShape.UNKNOWN,
        demand_coefficient_known=False),

    "strict-black": AttackerCapability(
        index=IndexKnowledge.UNKNOWN,
        demand_shape=DemandShape.UNKNOWN,
        demand_coefficient_known=False),

    "random": AttackerCapability(
        index=IndexKnowledge.UNKNOWN,
        demand_shape=DemandShape.UNKNOWN,
        demand_coefficient_known=False),
}


def threat_model(name: str, d_min: float = 0.02, index: str = "distance",
                 real: FilterSpec = None, beta: float = 1.0,
                 observability: str = None, aggregate: Callable = min) -> ThreatModel:
    if name not in PRESETS:
        raise ValueError(f"unknown threat '{name}'; choose from {sorted(PRESETS)}")
    cap = PRESETS[name]
    if observability is not None:
        cap = replace(cap, observability=Observability(observability))

    obs = (FullState() if cap.observability is Observability.FULL_STATE
           else CoarseMotion())
    obj = RandomObjective() if name == "random" else objective_for(cap, beta=beta)

    return ThreatModel(name=name, capability=cap, objective=obj,
                       specs=specs_for(cap, d_min, index=index, real=real),
                       observability=obs, aggregate=aggregate)
