"""
The vocabulary of the World contract — what goes IN to a run.

Scene       the world we are attacking in: obstacles, start, real goal, and
            where goals may legally be placed. Fixed by the seed; frozen so it
            cannot drift between the baseline run and any candidate run (the
            whole experiment depends on that invariance).

FilterSpec  the defense we are assuming: which safety algorithm, which safety
            index, and their parameters. Swappable, because the black-box
            attacker does not know the real one and must roll out under an
            ensemble of guesses.
"""

from dataclasses import dataclass, field, replace
from typing import Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------- #
#  Scene
# ---------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Scene:
    """The fixed world for one experiment (all positions in the robot base frame
    unless the name says world)."""

    G0: np.ndarray                      # where the end-effector starts
    G1: np.ndarray                      # the legitimate goal
    base_frame: np.ndarray              # robot base -> world transform
    obstacles_world: np.ndarray         # (n, 4, 4) obstacle frames in world
    bounds: Tuple                       # ((lo,hi), (lo,hi), (lo,hi)) goal workspace
    keepout: float                      # min goal-to-obstacle-center distance
    seed: int = -1
    test_case: str = ""

    @property
    def n_obstacles(self) -> int:
        return 0 if self.obstacles_world is None else len(self.obstacles_world)

    def summary(self) -> str:
        return (f"Scene(seed={self.seed}, case={self.test_case}, "
                f"obstacles={self.n_obstacles}, keepout={self.keepout:.3f})")


# ---------------------------------------------------------------------------- #
#  FilterSpec
# ---------------------------------------------------------------------------- #
#  The demand is what the filter insists on: how fast the danger number must
#  fall. Two shapes exist across the whole value-based family:
#      constant      d = eta        (SSA, r-SSA, p-SSA)
#      proportional  d = lambda*phi (CBF, SSS and their relaxed forms)
_ALGO_DEMAND = {
    # pfm and sma are not eta/lambda filters: each takes a single gain
    # (c_pfm / c_sma). "coefficient" marks that shape so the demand ladders,
    # which scale eta or lambda, leave them alone. "none" is SPARK's own
    # ByPassSafeControl, the no-filter control arm.
    "pfm":  "coefficient",
    "sma":  "coefficient",
    "none": "coefficient",
    "ssa":  "constant",
    "rssa": "constant",
    "pssa": "constant",
    "cbf":  "proportional",
    "rcbf": "proportional",
    "sss":  "proportional",
    "rsss": "proportional",
}

_ALGO_CLASS = {
    "pfm":  "BasicPotentialFieldMethod",
    "sma":  "BasicSlidingModeAlgorithm",
    "none": "ByPassSafeControl",
    "ssa":  "BasicSafeSetAlgorithm",
    "rssa": "RelaxedSafeSetAlgorithm",
    "pssa": "ProjectedSafeSetAlgorithm",
    "cbf":  "BasicControlBarrierFunction",
    "rcbf": "RelaxedControlBarrierFunction",
    "sss":  "BasicSublevelSafeSetAlgorithm",
    "rsss": "RelaxedSublevelSafeSetAlgorithm",
}

# Filters with no slack variable give up (QP infeasible) and fall back to the
# unmodified reference control; filters with slack stay solvable but pay slack.
_HARD = {"ssa", "cbf", "sss"}


@dataclass(frozen=True)
class FilterSpec:
    """A safety-filter configuration: which algorithm, which index, what rates."""

    algo: str = "ssa"                   # key of _ALGO_CLASS
    index: str = "distance"             # "distance" (first order) | "velocity" (second order)
    d_min: float = 0.02                 # keep-out distance inside the safety index
    eta: Optional[float] = None         # constant demand
    lam: Optional[float] = None         # proportional demand
    k: Optional[float] = None           # velocity-index coefficient
    #: raw dotted-path overrides applied LAST, e.g.
    #:   {"safe_algo.safety_buffer": 0.1,
    #:    "safety_index.min_distance.environment": 0.1}
    #: This is what makes a config file able to reproduce SPARK's
    #: shipped settings exactly, including fields FilterSpec has no
    #: named slot for (safety_buffer, use_slack, phi_n, ...).
    overrides: Optional[dict] = None
    slack_weight: Optional[float] = None
    c: Optional[float] = None           # single gain for pfm (c_pfm) / sma (c_sma)
    label: str = ""

    # -- derived -------------------------------------------------------- #
    @property
    def demand_shape(self) -> str:
        return _ALGO_DEMAND[self.algo]

    @property
    def is_hard(self) -> bool:
        """Hard filters give up -> u_ref (collision). Soft ones slack (stall)."""
        return self.algo in _HARD

    @property
    def class_name(self) -> str:
        return _ALGO_CLASS[self.algo]

    @property
    def demand_value(self) -> float:
        """The demand parameter actually in force (eta or lambda)."""
        return float(self.eta if self.demand_shape == "constant" else self.lam)

    def named(self, label: str) -> "FilterSpec":
        return replace(self, label=label)

    def __str__(self) -> str:
        rate = (f"eta={self.eta}" if self.demand_shape == "constant"
                else f"lam={self.lam}")
        return f"{self.label or self.algo}({self.algo}/{self.index}, {rate}, d_min={self.d_min})"


# ---------------------------------------------------------------------------- #
#  Ensembles — what the attacker rolls out under
# ---------------------------------------------------------------------------- #
#  For ROUTING (the surrogate's only job) the filter family collapses to
#  essentially one scalar: how hard it resists approaching an obstacle. So the
#  ensemble spans resistance, and — when the index is unknown — the index family
#  too. Hard-vs-soft fallback is deliberately NOT an ensemble axis: it changes
#  the symptom at failure (collide vs stall), not the path leading there.
#
#  Resistance is expressed as a FRACTION of a reference demand, not as absolute
#  numbers. Absolute levels are a trap: they were once hard-coded at
#  {0.1, 0.5, 1.0}, and on the G1 the authority available where the filter
#  engages is about 0.064, so every member of the ensemble was a filter that
#  could never satisfy its own demand. The whole ensemble modelled broken
#  filters, and the demand guard rejected it. Scaling to the deployed rate keeps
#  the surrogates in the regime the real filter occupies.
_RESISTANCE_FRAC = {"weak": 0.25, "med": 1.0, "strong": 2.0}
DEFAULT_ETA_REF = 0.02          # sane for the G1 benchmark; override per scene


def real_filter(algo="ssa", index="distance", d_min=0.02,
                eta=None, lam=None, k=None, c=None) -> FilterSpec:
    """The white-box case: the filter we actually know is deployed."""
    return FilterSpec(algo=algo, index=index, d_min=d_min, eta=eta, lam=lam, k=k,
                      c=c, label="real")


def gray_ensemble(demand_shape="constant", index="distance", d_min=0.02,
                  eta_ref=None, lam_ref=10.0) -> list:
    """Gray-box: family known, coefficient guessed. One member — the guess."""
    eta_ref = DEFAULT_ETA_REF if eta_ref is None else eta_ref
    if demand_shape == "constant":
        return [FilterSpec(algo="ssa", index=index, d_min=d_min,
                           eta=eta_ref * _RESISTANCE_FRAC["med"], k=0.1,
                           label="gray/guess")]
    return [FilterSpec(algo="cbf", index=index, d_min=d_min,
                       lam=lam_ref, k=0.1, label="gray/guess")]


def weak_black_ensemble(index="distance", d_min=0.02, eta_ref=None) -> list:
    """Weak black-box: the index family has been identified by the speed sweep,
    so only resistance is unknown. 3 members, spanning the deployed rate."""
    eta_ref = DEFAULT_ETA_REF if eta_ref is None else eta_ref
    return [FilterSpec(algo="ssa", index=index, d_min=d_min,
                       eta=eta_ref * f, k=0.1, label=f"{index}/{name}")
            for name, f in _RESISTANCE_FRAC.items()]


def strict_black_ensemble(d_min=0.02, eta_ref=None) -> list:
    """Strict black-box: neither the index family nor the resistance is known.
    6 members = resistance {weak, med, strong} x index {distance, velocity}."""
    eta_ref = DEFAULT_ETA_REF if eta_ref is None else eta_ref
    out = []
    for index in ("distance", "velocity"):
        for name, f in _RESISTANCE_FRAC.items():
            out.append(FilterSpec(algo="ssa", index=index, d_min=d_min,
                                  eta=eta_ref * f, k=0.1,
                                  label=f"{index}/{name}"))
    return out
