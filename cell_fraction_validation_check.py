"""
Quick sanity check after cell fraction fixes (from validation report).

Run after Fix 1–2: verifies dominant saddle at several γ̃ and reports Re Φ.
"""

import numpy as np
from qaoa_core import all_saddles

betas = [-np.pi/4, -np.pi/3, -np.pi/2, -2*np.pi/3, -3*np.pi/4]
gammas = [10*np.pi, 50*np.pi, 100*np.pi]

print("Dominant saddle check (highest Re Φ) at selected γ̃:")
for beta in betas:
    for gt in gammas:
        hw = max(15, 3 * np.sqrt(gt))
        saddles = all_saddles(gt, beta, grid_size=50, half_width=hw)
        if saddles:
            th = saddles[0][0]
            re_phi = saddles[0][1].real
            print(f"  β={beta/np.pi:.2f}π, γ̃={gt/np.pi:.0f}π: "
                  f"dominant θ*={th.real:.4f}{th.imag:+.4f}j, ReΦ={re_phi:.4f}, "
                  f"#saddles={len(saddles)}")
        else:
            print(f"  β={beta/np.pi:.2f}π, γ̃={gt/np.pi:.0f}π: no saddles found")
print("Done.")
