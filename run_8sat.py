import numpy as np
from symmetry_reduction.saddle_hessian_research.direction2_block_structure import (
    run_block_and_truncation, plot_block_heatmaps,
)

# Set depth
p = 1  

# BM24 optimal parameters for 8-SAT at p=1
betas  = np.array([-np.pi / 2], dtype=float)  
gammas = np.array([0.85], dtype=float) 

gamma_label = float(gammas.mean())

# Run the generalized k-SAT regime (q=3 means k=8)
res = run_block_and_truncation(
    p=p,
    gamma=gamma_label,
    betas=betas,
    gammas=gammas,
    q=3,          
)

plot_block_heatmaps(res, gamma_label)
print("Done! Check the figures folder for your 8-SAT heatmaps.")
