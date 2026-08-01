"""
Bayesian-optimisation picker — the alternative to CEM.

Why this is worth measuring against CEM. The search space is only 3-D (a goal
position), each evaluation costs seconds, and the budget is 20-40 points. That is
textbook Bayesian-optimisation territory: low-dimensional, expensive, sample
starved. It also fixes CEM's structural weakness directly rather than by patches
-- a single Gaussian tracks ONE mode and only ever shrinks, so CEM needs
multi-start and epsilon-greedy bolted on, whereas an acquisition function
balances exploration against exploitation by construction and models the whole
box at once.

The honest caveat: a GP assumes the objective is smooth, and ours is measurably
not. The gate is a step, outcome labels jump, IK redundancy makes nearby goals
resolve to different arm branches, and a collision truncates the run. So this is
expected to be imperfect -- which is exactly why it is measured against CEM on a
labelled scene rather than argued about.

No sklearn/botorch in this environment, so the GP is written out: RBF kernel,
Cholesky solve, length scale chosen by marginal likelihood over a small grid, and
Expected Improvement maximised by sampling admissible candidates. All of that is
cheap and adequate at 3-D.

Deliberately numpy-only, no scipy. MuJoCo and casadi/IPOPT already each bring a
libomp, which is why every entry point in this package must run under the
KMP_DUPLICATE_LIB_OK prelude documented in the README; adding scipy.linalg would
put a third OpenMP runtime into that pile for no gain. The pieces needed --
Cholesky, triangular solve, and the normal CDF/PDF -- are a few lines each.
"""

import math

import numpy as np

from .pick import Picker, initial_design, sample_admissible

_SQRT2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _norm_cdf(z):
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(z) / _SQRT2))


def _norm_pdf(z):
    return _INV_SQRT_2PI * np.exp(-0.5 * np.asarray(z) ** 2)


class GP:
    """Minimal Gaussian process with an RBF kernel and homoscedastic noise."""

    def __init__(self, length_scale=0.1, signal_var=1.0, noise_var=1e-3):
        self.length_scale = length_scale
        self.signal_var = signal_var
        self.noise_var = noise_var
        self._X = None

    def _k(self, A, B):
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return self.signal_var * np.exp(-0.5 * d2 / (self.length_scale ** 2))

    def _cho_solve(self, B):
        """Solve K x = B given the stored Cholesky factor L (K = L Lᵀ)."""
        return np.linalg.solve(self._L.T, np.linalg.solve(self._L, B))

    def fit(self, X, y):
        self._X = np.atleast_2d(X)
        self._ymean = float(np.mean(y))
        self._ystd = float(np.std(y)) or 1.0
        yz = (np.asarray(y, float) - self._ymean) / self._ystd
        K = self._k(self._X, self._X) + self.noise_var * np.eye(len(self._X))
        self._L = np.linalg.cholesky(K)
        self._alpha = self._cho_solve(yz)
        return self

    def log_marginal_likelihood(self, X, y):
        try:
            self.fit(X, y)
        except np.linalg.LinAlgError:
            return -np.inf          # not positive-definite => reject this scale
        yz = (np.asarray(y, float) - self._ymean) / self._ystd
        return float(-0.5 * yz @ self._alpha
                     - np.log(np.diag(self._L)).sum()
                     - 0.5 * len(yz) * np.log(2 * np.pi))

    def predict(self, Xs):
        Xs = np.atleast_2d(Xs)
        Ks = self._k(Xs, self._X)
        mu = Ks @ self._alpha
        v = self._cho_solve(Ks.T)
        var = self.signal_var - np.einsum("ij,ji->i", Ks, v)
        var = np.maximum(var, 1e-12)
        return mu * self._ystd + self._ymean, np.sqrt(var) * self._ystd


class BOPicker(Picker):
    """Expected-Improvement search over admissible goals.

    Initial design is obstacle-seeded rather than uniform: the modes of this
    objective are anchored to obstacles, and their locations are known in every
    threat tier (the attacker can see the room), so spending the first few
    evaluations near each obstacle is informed prior placement rather than a
    workaround. This is the same structural knowledge CEM needs multi-start to
    exploit -- held FIXED across both arms so the comparison measures the
    optimiser and not the seeding.
    """

    name = "bo"

    def __init__(self, scene, seed=0, n_init=8, n_acq_samples=2000, xi=0.01,
                 obstacle_anchored=True):
        self.scene = scene
        self.rng = np.random.RandomState(seed)
        # dedicated stream, keyed only on `seed` — see CEMPicker._pending
        self._seed_rng = np.random.RandomState(seed)
        self.obstacle_anchored = obstacle_anchored
        self.n_init = n_init
        self.n_acq_samples = n_acq_samples
        self.xi = xi                      # EI exploration offset
        self.X, self.y = [], []
        self._lo = np.array([b[0] for b in scene.bounds], float)
        self._hi = np.array([b[1] for b in scene.bounds], float)
        self._span = float(np.linalg.norm(self._hi - self._lo))

    def _fit(self):
        X = np.asarray(self.X, float)
        y = np.asarray(self.y, float)
        best, best_ll = None, -np.inf
        for ls in self._span * np.array([0.05, 0.1, 0.2, 0.4]):
            gp = GP(length_scale=ls, noise_var=1e-2)
            ll = gp.log_marginal_likelihood(X, y)
            if ll > best_ll:
                best, best_ll = gp, ll
        return best.fit(X, y) if best is not None else None

    def ask(self, n):
        if len(self.X) < self.n_init:
            return initial_design(
                self._seed_rng, self.scene, min(n, self.n_init - len(self.X)),
                obstacle_anchored=self.obstacle_anchored)

        gp = self._fit()
        if gp is None:
            return [p for p in (sample_admissible(self.rng, self.scene)
                                for _ in range(n)) if p is not None]

        cands = [c for c in (sample_admissible(self.rng, self.scene)
                             for _ in range(self.n_acq_samples)) if c is not None]
        if not cands:
            return []
        Xs = np.asarray(cands, float)
        mu, sd = gp.predict(Xs)
        best = float(np.max(self.y))
        z = (mu - best - self.xi) / np.maximum(sd, 1e-9)
        ei = (mu - best - self.xi) * _norm_cdf(z) + sd * _norm_pdf(z)

        # take the top n, spaced apart so a batch does not collapse onto one spot
        picked, order = [], np.argsort(-ei)
        min_sep = 0.02 * self._span
        for i in order:
            p = Xs[i]
            if all(np.linalg.norm(p - q) > min_sep for q in picked):
                picked.append(p)
            if len(picked) >= n:
                break
        return picked

    def tell(self, scored):
        for c, s in scored:
            if s is not None and np.isfinite(s):
                self.X.append(np.asarray(c, float))
                self.y.append(float(s))
