---
title: "Deriving $c(x)$: the control-authority ceiling of an SSA safety constraint"
author: "SPARK goal-insertion notes"
geometry: margin=1in
fontsize: 11pt
header-includes:
  - \usepackage{amsmath}
  - \usepackage{amssymb}
  - \usepackage{bm}
---

# Setup and notation

| symbol | meaning |
|---|---|
| $x\in\mathbb{R}^{n}$ | robot configuration (joint positions; the safety-index state) |
| $u\in\mathbb{R}^{m}$ | control input |
| $\mathcal{U}=\{u: |u_k|\le u_{lim,k}\}$ | actuator box ($=\prod_k[-u_{lim,k},\,u_{lim,k}]$) |
| $\dot x = f(x) + G(x)\,u$ | control-affine dynamics; $G(x)$ = actuation map |
| $\phi(x)$ | safety index (one constraint); safe set $\{\phi\le 0\}$, danger $\{\phi>0\}$ |
| $\eta>0$ | SSA margin rate: enforce $\dot\phi\le-\eta$ when $\phi\ge 0$ |
| $n$ | unit contact normal (the distance-decreasing / escape direction) |
| $J(x)$ | Jacobian of the contacting body point, $\mathbb{R}^{3\times n}$ |

We derive $c(x)$, the **maximum rate at which the actuators can drive $\phi$ down**,
and show the single-active-constraint QP is feasible **iff** $\eta \le c(x)$ (first order).

# Step 1 — the time derivative of the safety index

Differentiating $\phi$ along the dynamics and using control-affineness,

$$
\dot\phi = \nabla\phi(x)^\top \dot x
         = \nabla\phi^\top\big(f(x)+G(x)u\big)
         = \underbrace{\nabla\phi^\top f}_{=:~L_f\phi}
         + \underbrace{\nabla\phi^\top G}_{=:~L_g\phi}\;u .
$$

So $\dot\phi = L_f\phi + L_g\phi\,u$, where $L_f\phi$ is the **drift** (uncontrolled
rate) and $L_g\phi\in\mathbb{R}^{1\times m}$ is the **control gain** of the index.

For a collision index, $\phi$ depends on $x$ only through the contact-point
position $p(x)$, with Cartesian gradient equal to the normal $n$. The chain rule
$\nabla_x\phi = J(x)^\top n$ then gives

$$
\boxed{\,L_g\phi(x) = n^\top J(x)\,G(x)\,}\qquad(\in\mathbb{R}^{1\times m}).
$$

# Step 2 — feasibility of the safety constraint

The SSA QP requires a control in the box that satisfies the safety inequality:

$$
\exists\,u\in\mathcal{U}:\quad L_f\phi + L_g\phi\,u \le -\eta .
$$

Such a $u$ exists **iff the most negative achievable $\dot\phi$ already meets the bound**:

$$
\min_{u\in\mathcal{U}}\big(L_f\phi + L_g\phi\,u\big)\ \le\ -\eta
\quad\Longleftrightarrow\quad
L_f\phi + \min_{u\in\mathcal{U}} L_g\phi\,u \ \le\ -\eta .
$$

# Step 3 — minimizing a linear form over the box

$L_g\phi\,u = \sum_k (L_g\phi)_k\,u_k$ is **separable** across coordinates, so it is
minimized coordinate-wise. Pushing each $u_k$ to the box corner that makes its term
most negative,

$$
u_k^\star = -\,\mathrm{sign}\big((L_g\phi)_k\big)\,u_{lim,k}
\quad\Longrightarrow\quad
(L_g\phi)_k\,u_k^\star = -\,|(L_g\phi)_k|\,u_{lim,k},
$$

hence

$$
\min_{u\in\mathcal{U}} L_g\phi\,u
\;=\; -\sum_{k=1}^{m} u_{lim,k}\,\big|(L_g\phi)_k\big| .
$$

# Step 4 — define $c(x)$

Define the **control-authority ceiling** as the maximum achievable *decrease* rate:

$$
c(x)\;:=\;\max_{u\in\mathcal{U}}\big(-L_g\phi\,u\big)
        \;=\;-\min_{u\in\mathcal{U}} L_g\phi\,u
        \;=\;\sum_{k=1}^{m} u_{lim,k}\,\big|(L_g\phi)_k\big| .
$$

Substituting into Step 2, the feasibility condition becomes

$$
L_f\phi - c(x) \le -\eta
\quad\Longleftrightarrow\quad
\boxed{\,c(x)\ \ge\ \eta + L_f\phi\,}.
$$

**First-order / velocity control** ($f\equiv 0\Rightarrow L_f\phi=0$): this collapses to

$$
\boxed{\ \text{feasible}\iff c(x)\ge\eta\,,\qquad
c(x)=\sum_k u_{lim,k}\,\big|(L_g\phi)_k\big|
     =\big\lVert \operatorname{diag}(u_{lim})\,L_g\phi^\top\big\rVert_1 .\ }
$$

# Step 5 — closed form in kinematic terms

Inserting $L_g\phi = n^\top J(x)\,G(x)$ from Step 1:

$$
c(x)\;=\;\sum_{k=1}^{m} u_{lim,k}\,\Big|\big[\,n^\top J(x)\,G(x)\,\big]_k\Big|
      \;=\;\big\lVert \operatorname{diag}(u_{lim})\,\big(n^\top J(x)G(x)\big)^\top \big\rVert_1 .
$$

It is the **actuator-limit-weighted $\ell_1$ norm of the safety gradient in control space**.

# Step 6 — geometric reading (support function / velocity zonotope)

$c(x)$ is the **support function** of the box evaluated in the direction $-L_g\phi$:
$c(x)=h_{\mathcal{U}}(-L_g\phi)$. Equivalently, as $u$ sweeps $\mathcal{U}$ the contact
point's velocity $v=J(x)G(x)\,u$ traces a **zonotope** $\mathcal{Z}=J G\,\mathcal{U}$,
and since $\dot\phi = n^\top v$,

$$
c(x)\;=\;\max_{v\in\mathcal{Z}}\big(-n^\top v\big)\;=\;h_{\mathcal{Z}}(-n)
$$

— *how far the reachable-velocity set extends in the escape direction $-n$.*
The reachable $\dot\phi$ is exactly the interval $[-c(x),\,+c(x)]$ (symmetric box),
so $c(x)$ is its half-width.

# Step 7 — special cases

- **Ball-limited actuation** $\mathcal{U}=\{\lVert u\rVert_2\le r\}$:
  $h_{\mathcal{U}}(y)=r\lVert y\rVert_2$, so $c(x)=r\,\lVert L_g\phi\rVert_2$
  (same gradient, $\ell_2$ instead of weighted-$\ell_1$ — the norm comes from the
  limit-set geometry).
- **Per-contact reduction:** $J(x)$ has zero columns for joints *distal* to the
  contacting body, so $(L_g\phi)_k=0$ there. Only joints **proximal** to the contact
  contribute to $c(x)$ — the effective dimension is the proximal sub-chain.

# Step 8 — the degenerate case $c(x)=0$

Since $c(x)$ is a sum of non-negative terms,

$$
c(x)=0\iff L_g\phi=n^\top J(x)G(x)=\mathbf 0\iff \big(J(x)G(x)\big)^\top n=\mathbf 0
\iff n\perp \operatorname{Im}\!\big(J(x)G(x)\big).
$$

The escape direction is **orthogonal to the entire reachable-velocity subspace**:
the body can move only *tangent* to the obstacle, never separate from it. This is a
**controllability singularity of the constraint** ($\partial\dot\phi/\partial u=0$) —
not actuator saturation, not a closed-loop equilibrium. At such $x$ **no $\eta>0$ is
feasible**.

# Step 9 — extension to multiple active constraints

The scalar test holds for **one** active constraint. With several active
$i\in\mathcal A$, write $a_i:=L_g\phi_i^\top\in\mathbb R^{m}$ and
$\beta_i:=\eta_i+L_f\phi_i$. The QP is feasible iff

$$
\exists\,u\in\mathcal U:\ a_i^\top u+\beta_i\le 0\ \ \forall i
\quad\Longleftrightarrow\quad
\mu(x):=\min_{u\in\mathcal U}\ \max_{i\in\mathcal A}\big(a_i^\top u+\beta_i\big)\ \le\ 0 .
$$

$\mu(x)$ is the **collective achievable margin** — the smallest the most-violated
constraint can be driven over the box (a convex piecewise-linear minimization, i.e. an LP).

**Dual / aggregated form.** Lift $\max_i$ to a weight $y$ on the simplex
$\Delta=\{y\ge 0,\ \mathbf 1^\top y=1\}$ and swap $\min/\max$ (Sion's theorem: the payoff
$\sum_i y_i(a_i^\top u+\beta_i)$ is bilinear; $\mathcal U,\Delta$ compact convex):

$$
\mu(x)=\max_{y\in\Delta}\min_{u\in\mathcal U}\sum_i y_i\big(a_i^\top u+\beta_i\big)
      =\max_{y\in\Delta}\Big[\,y^\top\beta-\underbrace{\textstyle\sum_k u_{lim,k}\big|(\sum_i y_i a_i)_k\big|}_{=:~c_y(x)}\Big].
$$

The inner $\min$ reused Step 3. Crucially, $c_y(x)=\big\lVert\operatorname{diag}(u_{lim})\,\bar a(y)^\top\big\rVert_1$
is **the single-constraint ceiling of Step 4 applied to the aggregated normal**
$\bar a(y)=\sum_i y_i\,L_g\phi_i$. Therefore

$$
\boxed{\ \text{feasible}\iff \forall\,y\in\Delta:\ c_y(x)\ \ge\ \sum_i y_i\big(\eta_i+L_f\phi_i\big)\ .}
$$

*For every nonnegative blend of the active constraints, the aggregated control authority
must dominate the aggregated demand.* Infeasibility is certified by one witness:

$$
\exists\,y^\star\ge 0:\quad c_{y^\star}(x)\ <\ \sum_i y^\star_i\big(\eta_i+L_f\phi_i\big),
$$

the **pincer**: choose $y^\star$ so the normals partially cancel,
$\bar a(y^\star)=\sum_i y^\star_i L_g\phi_i\approx\mathbf 0$ (hence $c_{y^\star}\approx 0$), while
the demands add, $\sum_i y^\star_i(\eta_i+L_f\phi_i)>0$. In the limit this is the Farkas
certificate $\sum_i y_iL_g\phi_i=\mathbf 0,\ \sum_i y_i(\eta_i+L_f\phi_i)<0$.

**What changed vs. the single-constraint case.**

- $|\mathcal A|=1$: $\Delta=\{1\}$, $\bar a=a_1$, recovering $c(x)\ge\eta+L_f\phi$ (Step 4).
- **Per-constraint test is only necessary:** $\eta_i\le c_i(x)\ \forall i$ is the boxed
  condition at the vertices $y=e_i$; an *interior* $y^\star$ can still fail it. So no single
  scalar ceiling suffices — conflict lives in the interior of $\Delta$.
- **No closed form:** one linear form over a box decouples coordinate-wise (Step 3); a
  *max* of several couples the coordinates, so feasibility needs the LP (the maximization
  over $\Delta$). The structure is still explicit — the support function of the *aggregated*
  normal $\bar a(y)$.

# Summary

$$
\boxed{\;
\begin{gathered}
c(x)=\max_{u\in\mathcal{U}}\big(-L_g\phi\,u\big)
=\sum_{k} u_{lim,k}\,\big|[\,n^\top J(x)G(x)\,]_k\big|
=h_{\mathcal{Z}}(-n)\\[4pt]
\text{feasible}\iff c(x)\ \ge\ \eta+L_f\phi
\end{gathered}
\;}
$$

- It is the **best achievable rate of decrease of the safety index** under the
  actuator limits.
- It depends only on the **configuration** (via $J,G$) and **contact geometry**
  (via $n$) — not on velocity, for first-order dynamics.
- An attacker steers the closed loop to a boundary contact **minimizing $c(x)$**;
  $c(x)\to 0$ at singularities, so lowering $\eta$ shrinks but never closes the
  vulnerable set.
