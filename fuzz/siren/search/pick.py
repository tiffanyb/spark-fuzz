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

    def __init__(self, scene, seed=0, n_init=0):
        self.scene = scene
        self.rng = np.random.RandomState(seed)
        # n_init > 0 => start on the obstacle anchors, then revert to uniform.
        # Random ignores feedback, so this isolates the VALUE OF THE ANCHORS
        # themselves from any optimiser's ability to exploit them.
        self._pending = list(initial_design(
            np.random.RandomState(seed), scene, n_init,
            obstacle_anchored=True)) if n_init else []

    def ask(self, n: int) -> list:
        out = []
        if self._pending:
            out, self._pending = self._pending[:n], self._pending[n:]
        for _ in range(n - len(out)):
            c = sample_admissible(self.rng, self.scene)
            if c is not None:
                out.append(c)
        return out


def initial_design(rng, scene, n, jitter=0.05, obstacle_anchored=True):
    """Initial design anchored to the obstacles, then padded with random draws.

    The modes of the objective sit near obstacles, and obstacle locations are
    known to the attacker in every threat tier, so this is informed prior
    placement rather than a trick. It is factored out here so that every picker
    can be given the SAME first batch -- otherwise a comparison between them
    measures the seeding, not the optimiser.

    With obstacle_anchored=False the same number of evaluations is spent on
    uniform admissible draws instead. That is the control arm for "is anchoring
    the initial design to the obstacles worth anything?", a question that cannot
    be answered while every arm is anchored.
    """
    if not obstacle_anchored:
        out = []
        while len(out) < n:
            p = sample_admissible(rng, scene)
            if p is None:
                break
            out.append(p)
        return out

    lo = np.array([b[0] for b in scene.bounds], dtype=float)
    hi = np.array([b[1] for b in scene.bounds], dtype=float)
    pts = []
    obs = getattr(scene, "obstacles_world", None)
    if obs is not None and len(obs):
        inv = np.linalg.inv(scene.base_frame)
        for o in obs:
            c_base = (inv @ np.append(np.asarray(o)[:3, 3], 1.0))[:3]
            for _ in range(30):                     # jitter until admissible
                p = np.clip(c_base + rng.normal(0, jitter, 3), lo, hi)
                if is_admissible(p, scene)[0]:
                    pts.append(p)
                    break
            if len(pts) >= n:
                break
    while len(pts) < n:
        p = sample_admissible(rng, scene)
        if p is None:
            break
        pts.append(p)
    return pts[:n]


class CEMPicker(Picker):
    """Cross-entropy method: fit a Gaussian to the best candidates seen this
    generation and resample around them, so the search concentrates on the
    region that scores well instead of sampling uniformly forever."""

    name = "cem"

    def __init__(self, scene, seed=0, elite_frac=0.34, init_std_frac=0.25,
                 std_floor=1e-3, n_init=0, obstacle_anchored=True):
        self.scene = scene
        self.rng = np.random.RandomState(seed)
        self.elite_frac = elite_frac
        self.std_floor = std_floor

        self.lo = np.array([b[0] for b in scene.bounds], dtype=float)
        self.hi = np.array([b[1] for b in scene.bounds], dtype=float)
        start = sample_admissible(self.rng, scene)
        self.mean = start if start is not None else 0.5 * (self.lo + self.hi)
        self.std = (self.hi - self.lo) * init_std_frac
        # n_init > 0 => emit the shared obstacle-seeded design first, then fit
        # the Gaussian to its elites. Drawn from a DEDICATED stream keyed only on
        # `seed`, so that every picker built with the same seed gets a
        # bit-identical initial design regardless of what else it drew first.
        self._pending = list(initial_design(
            np.random.RandomState(seed), scene, n_init,
            obstacle_anchored=obstacle_anchored)) if n_init else []

    def ask(self, n: int) -> list:
        if self._pending:
            take, self._pending = self._pending[:n], self._pending[n:]
            return take
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


#: Every optimiser spends its first _N_INIT evaluations on an initial design.
#: "*_seeded" anchors that design to the obstacles; the plain name spends the
#: same budget on uniform draws. Holding the SIZE fixed across both is what makes
#: the seeded/unseeded pair a clean test of anchoring rather than of head start.
_N_INIT = 8

ARMS = ("random", "random_seeded", "cem", "cem_seeded", "bo", "bo_seeded")


def make_picker(name: str, scene, seed=0) -> Picker:
    if name not in ARMS:
        raise ValueError(f"unknown picker '{name}'; choose from {list(ARMS)}")
    anchored = name.endswith("_seeded")
    base = name[:-len("_seeded")] if anchored else name

    if base == "random":
        # uniform sampling either way; the seeded arm just starts on the anchors
        return RandomPicker(scene, seed=seed,
                            n_init=(_N_INIT if anchored else 0))
    if base == "cem":
        return CEMPicker(scene, seed=seed, n_init=_N_INIT,
                         obstacle_anchored=anchored)
    from .bo import BOPicker                   # lazy: bo.py imports from here
    return BOPicker(scene, seed=seed, n_init=_N_INIT,
                    obstacle_anchored=anchored)
