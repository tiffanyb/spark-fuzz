"""
What to try next: admissibility + the two pickers.

ADMISSIBILITY is the attacker's own constraint: every candidate goal must be one
an operator could plausibly issue, or the "attack" is just an illegal command.
Operationally that means inside the arm workspace and at least `keepout` from
every obstacle centre — the SAME test SPARK's benchmark applies when it samples
its own goals, so an admissible candidate is indistinguishable from a goal the
benchmark itself would have produced. (SPARK measures centre-to-centre and is
geometry-agnostic; we replicate that exactly rather than using a physically
stricter surface distance, which would make our goals distinguishable.)

THE PICKERS are generational — ask() hands back a batch, tell() feeds the scores
back — because that is the only interface both strategies share. Random ignores
the feedback; CEM needs a whole population scored before it can refit. One loop
drives both.
"""

from abc import ABC, abstractmethod

import numpy as np


# ---------------------------------------------------------------------------- #
#  Admissibility
# ---------------------------------------------------------------------------- #
def _to_world(xyz_base, base_frame):
    f = np.eye(4)
    f[:3, 3] = np.asarray(xyz_base, dtype=float).reshape(3)
    return (base_frame @ f)[:3, 3]


def is_admissible(cand, scene) -> tuple:
    """(ok, reason) for a candidate goal, in the robot base frame."""
    xyz = np.asarray(cand, dtype=float).reshape(3)

    for d in range(3):
        lo, hi = scene.bounds[d]
        if xyz[d] < lo or xyz[d] > hi:
            return False, "out_of_bounds"

    if scene.n_obstacles:
        goal_world = _to_world(xyz, scene.base_frame)
        gap = min(float(np.linalg.norm(goal_world - np.asarray(o)[:3, 3]))
                  for o in scene.obstacles_world)
        if gap < scene.keepout:
            return False, f"too_close({gap:.3f}<{scene.keepout:.3f})"

    return True, "ok"


def sample_admissible(rng, scene, max_tries: int = 1000):
    """Rejection-sample one admissible candidate, or None."""
    for _ in range(max_tries):
        xyz = np.array([rng.uniform(lo, hi) for (lo, hi) in scene.bounds], dtype=float)
        if is_admissible(xyz, scene)[0]:
            return xyz
    return None


# ---------------------------------------------------------------------------- #
#  Pickers
# ---------------------------------------------------------------------------- #
class Picker(ABC):
    name: str

    @abstractmethod
    def ask(self, n: int) -> list:
        """Propose up to n candidates."""

    def tell(self, scored) -> None:
        """Receive [(candidate, score), ...]. Default: ignore."""


class RandomPicker(Picker):
    """Uniform over the admissible workspace. The baseline, and the honest
    fallback when no guidance signal is trustworthy."""

    name = "random"

    def __init__(self, scene, seed=0):
        self.scene = scene
        self.rng = np.random.RandomState(seed)

    def ask(self, n: int) -> list:
        out = []
        for _ in range(n):
            c = sample_admissible(self.rng, self.scene)
            if c is not None:
                out.append(c)
        return out


class CEMPicker(Picker):
    """Cross-entropy method: fit a Gaussian to the best candidates seen this
    generation and resample around them, so the search concentrates on the
    region that scores well instead of sampling uniformly forever."""

    name = "cem"

    def __init__(self, scene, seed=0, elite_frac=0.34, init_std_frac=0.25,
                 std_floor=1e-3):
        self.scene = scene
        self.rng = np.random.RandomState(seed)
        self.elite_frac = elite_frac
        self.std_floor = std_floor

        self.lo = np.array([b[0] for b in scene.bounds], dtype=float)
        self.hi = np.array([b[1] for b in scene.bounds], dtype=float)
        start = sample_admissible(self.rng, scene)
        self.mean = start if start is not None else 0.5 * (self.lo + self.hi)
        self.std = (self.hi - self.lo) * init_std_frac

    def ask(self, n: int) -> list:
        out = []
        tries = 0
        while len(out) < n and tries < n * 50:
            tries += 1
            c = np.clip(self.rng.normal(self.mean, self.std), self.lo, self.hi)
            if is_admissible(c, self.scene)[0]:
                out.append(c)
        return out

    def tell(self, scored) -> None:
        scored = [(c, s) for c, s in scored if s is not None and np.isfinite(s)]
        if len(scored) < 2:
            return
        scored.sort(key=lambda cs: cs[1], reverse=True)
        k = max(2, int(round(self.elite_frac * len(scored))))
        elites = np.array([c for c, _ in scored[:k]], dtype=float)
        self.mean = elites.mean(axis=0)
        # the floor stops the distribution collapsing to a point and going blind
        self.std = elites.std(axis=0) + self.std_floor


_PICKERS = {"random": RandomPicker, "cem": CEMPicker}


def make_picker(name: str, scene, seed=0) -> Picker:
    if name not in _PICKERS:
        raise ValueError(f"unknown picker '{name}'; choose from {sorted(_PICKERS)}")
    return _PICKERS[name](scene, seed=seed)
