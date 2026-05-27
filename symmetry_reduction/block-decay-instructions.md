# Cursor Instructions: Block-Wise Spectral Decay Analysis

## Objective
Shift the empirical eigenvalue decay analysis from the raw eigenvalue index $k$ to the Fourier block level $m$. The current global exponential fit fails at large $p$ because of the binomial multiplicity of the block levels. We need to extract the maximum eigenvalue associated with each block level $m$ and check if the decay rate with respect to $m$ remains invariant as the circuit depth $p$ scales.

## Context Files to Include
- `symmetry_reduction/eigenvalue_decay.py`
- `symmetry_reduction/eigenvalue_decay_proof.md`

## Instructions for Cursor

### Step 1: Add Block-Max Eigenvalue Extraction
In `eigenvalue_decay.py`, create a new function called `block_max_eigenvalue_analysis(p: int, gamma: float, at_saddle: bool = False)`.
1. Call the existing `eigenvalue_block_decomposition(p, gamma, at_saddle)` function to get the `block_energy` array and the sorted `eigenvalues`.
2. For each eigenvector $k$, determine its "primary" block $m$ by finding the argmax of its block energy: `primary_m = np.argmax(block_energy, axis=1)`.
3. For each block level $m \in [0, 2p+1]$, find the maximum absolute eigenvalue $|\lambda_k|$ among all eigenvectors whose `primary_m == m`. If a block has no primary eigenvectors, assign it `np.nan` or $0.0$.
4. Return a dictionary containing the array of block levels $m$ and the corresponding $\max |\lambda^{(m)}|$.

### Step 2: Implement Exponential Fitting for Block Levels
In `eigenvalue_decay.py`, add a utility to fit an exponential specifically to these block maximums: $|\lambda^{(m)}| \approx M e^{-c_m m}$. You can reuse the logic from `fit_exponential_log`, but apply it to the valid (non-zero/non-NaN) block maximums against their $m$ values instead of $k$.

### Step 3: Add a Plotting Routine for Block Decay
In `eigenvalue_decay.py`, create a new plotting function `plot_block_max_decay(p_values=(2, 3, 4, 5), gamma_values=(0.3, 1.0, np.pi), at_saddle=False, out_dir=None)`.
1. Create a 2x2 subplot grid (or similar, matching the style of `plot_decay_and_c_prime`).
2. For a few selected $(p, \gamma)$ pairs, plot $\log(\max |\lambda^{(m)}|)$ vs $m$.
3. Overlay the exponential fit $M e^{-c_m m}$.
4. In the final subplot, plot the new block-decay rate $c_m(p)$ vs $p$ for different $\gamma$ values to see if the decay rate stabilizes (flattens out at a non-zero constant) across depths.
5. Save the figure as `eigenvalue_block_max_decay_{suffix}.png` in the `out_dir`.

### Step 4: Update the Proof Documentation
In `eigenvalue_decay_proof.md`, add a new sub-section under "3. Bonami–Beckner and hypercontractivity" titled "Block-Wise Spectral Bounding". 
1. Explain that instead of bounding the global index $k$, we bound the spectral norm of the $m$-th block: $\max |\lambda^{(m)}| \le \mathcal{O}(\rho^m)$.
2. Note that the determinant ratio bound must be modified to sum over blocks $m$, accounting for the binomial multiplicity $\binom{2p+1}{m}$ at each step.