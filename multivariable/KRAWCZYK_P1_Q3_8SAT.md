# Krawczyk Certification Plan for 8-SAT (`p=1`, `q=3`)

This note defines an exact computer-assisted theorem target and a concrete Krawczyk workflow for the `p=1`, `q=3`, `r=176.54` saddle/fixed-point system already implemented in `symmetry_reduction`.

## A. Exact theorem statement

Fix parameters `p=1`, `q=3`, `r=176.54`, and a real angle pair `(beta, gamma)`.

Let `d = 2^(2p+1) = 8`, and active indices

`I_act = {alpha in {0,...,7} : |alpha| >= 2}` (so `|I_act| = 4`).

Define `u in C^4` on active indices, with nonlinear system `g(u)=0` (defined below), and its real lift

`H(z) = [Re g(u(z)); Im g(u(z))] : R^8 -> R^8`.

Let `z0 in R^8`, `X` be an interval box centered at `z0`, and `R approx (J_H(z0))^{-1}`.

If the Krawczyk inclusion holds:

1. `K(z0, X) subset int(X)`, where
   `K(z0, X) = z0 - R H(z0) + (I - R J_H(X))(X - z0)`,
2. and `J_H(X)` is an interval enclosure of the Jacobian over `X`,

then:

- There exists `z* in X` such that `H(z*) = 0` (equivalently `g(u*)=0`).
- The zero is unique in `X`.
- `J_H(z*)` is nonsingular (equivalently the complex Jacobian `J_g(u*)` is invertible).

Important:

- A Krawczyk proof does **not require** prior proof that `rho(J_T)<1` for the fixed-point map `T`.
- `rho(J_T)<1` is a useful sufficient condition for local contraction/iteration stability, but weaker interval-Newton/Krawczyk conditions already certify existence+uniqueness.
- If desired, one can additionally certify `sup_{u in U} ||J_T(u)|| < 1` (or `rho(J_T(u))<1` via bounds) on a neighborhood `U` to obtain a formal local contraction theorem for the iteration itself.

## B. Exact root system `H(z)=0`

Use the existing `u`-space formulation (`symmetry_reduction/newton_saddle_q.py`).

### Variables

- Complex unknowns: `u = (u_alpha)_{alpha in I_act} in C^4`.
- Real interval unknowns: `z = (x_1,...,x_4,y_1,...,y_4) in R^8` with `u_j = x_j + i y_j`.

### Equations

Let:

- `A in R^(8x8)` from `build_structure_matrix(p=1, q=3)`,
- `A_act = A[I_act, :] in R^(4x8)`,
- `b_s in C^8` from `compute_b_s(p=1, betas=[beta])`,
- `coeff_alpha in C^8` from `alpha_linear_coefficients(p=1, gammas=[gamma], q=3, r=176.54)`,
- `c6_alpha = coeff_alpha^6` on active indices.

Given `u`, build `u_full in C^8` by filling active entries and zero otherwise, then:

- `w_s(u) = b_s exp((u_full A)_s) / sum_t b_t exp((u_full A)_t)`,
- `E_alpha(u) = sum_s w_s(u) A_{alpha,s}` for `alpha in I_act`.

Define `g : C^4 -> C^4` componentwise:

`g_alpha(u) = u_alpha + 6 * c6_alpha * E_alpha(u)^5`.

The root problem is `g(u)=0`.

Real-lifted system:

`H(z) = [Re g(u(z)); Im g(u(z))] in R^8`.

### Jacobian needed

Complex Jacobian (`4x4`):

`J_g(u) = I + 30 * diag(c6 * E(u)^4) * Cov_w(A_act)`,

where:

- `Cov_w(A_act) = E_w[A_act A_act^T] - E_w[A_act] E_w[A_act]^T`.

Real Jacobian (`8x8`) used by Krawczyk:

for `J_g = A + iB` (`A,B` real `4x4`),

`J_H = [[A, -B], [B, A]]`.

### Quantities to enclose rigorously

- `H(X)` and `J_H(X)` over the full interval box `X`,
- including all softmax terms `exp`, normalized sums, powers `E^5`, `E^4`,
- and a validated enclosure of `(I - R J_H(X))(X-z0)`.

## C. Concrete Krawczyk proof plan

1. **Float anchor**
   - Compute high-accuracy approximate root `u0` (Newton/homotopy in existing code).
   - Form `z0 = [Re(u0), Im(u0)]`.

2. **Approximate inverse**
   - Compute `J0 = J_H(z0)` in floating point.
   - Set `R = inv(J0)` (or verified solve preconditioner).

3. **Initial box**
   - Start with `X0 = [z0 - rad, z0 + rad]`, e.g. `rad = 1e-8` to `1e-5` componentwise.

4. **Interval evaluation**
   - Evaluate interval enclosure `J_H(Xk)`.
   - Compute
     `Kk = z0 - R H(z0) + (I - R J_H(Xk))(Xk - z0)`.

5. **Acceptance criterion**
   - Success if `Kk subset int(Xk)` (strict interior).
   - Then existence+uniqueness+Jacobian invertibility in `Xk` are certified.

6. **Adaptive refinement**
   - If inclusion fails: shrink box radius and retry.
   - If too narrow causes outward rounding issues: slightly enlarge precision and retry.

7. **Optional contraction certificate**
   - Independently bound `J_T(Xk)` for fixed-point map `T(u) = -6 c6 E(u)^5`.
   - Prove `||J_T(Xk)|| < 1` to certify local fixed-point contraction.

## D. Best first parameter point

For this repository state and current solver conventions, the easiest first certificate is:

- `beta = -pi/2`
- `gamma = 0.1361946795412982`
- `p=1`, `q=3`, `r=176.54`

Reason:

- Existing sweeps already flag this as stable for `p=1` in the high-`r` TQA family.
- Numerically, this point gives very small residual and extremely small local Jacobian radius for the fixed-point map (`rho << 1`) in current computations, making Krawczyk inclusion easiest.

Note:

- The user-suggested region near `(beta, gamma) ~ (-0.10, -0.14)` can still be attempted, but in this code snapshot it is not the easiest first proof target under the current equation conventions/normalization.

## E. Remaining gap after certification

What the computer-assisted theorem **would prove**:

- A mathematically rigorous existence+local uniqueness statement for a saddle/fixed-point root at the selected parameter point.
- Nonsingularity of the nonlinear Jacobian at that root.
- (Optionally) local contraction of the fixed-point map if separately interval-bounded.

What it **would not prove** without extra arguments:

- Global uniqueness beyond the certified box.
- Global convergence from arbitrary initialization.
- Dominance of this saddle among all saddles (requires separate objective/comparison certificate).
- Any statement uniform in `(beta, gamma)` unless a parameter-box continuation proof is added.
