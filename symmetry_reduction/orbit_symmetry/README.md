# Orbit symmetry reduction (S_p × S_p × Z_2)

Implements the structural analysis from the cursor instructions:

- **Task 1**: Orbit classification (a, b, c), reduced Hessian, `symmetry_analysis(p, gamma, saddle=False)`, `run_symmetry_sweep(...)`
- **Task 2**: Complement vs normal truncation: `complement_truncation_analysis`, `run_complement_sweep(...)`
- **Task 4**: Symmetry commutation check: `check_symmetry_commutation(H_log, p)`
- **Task 5**: Diagonal orbits (a = c): `diagonal_orbit_analysis(p, gamma)`

## Run

From the repo root (e.g. `/Users/b/Quandela`):

```bash
# Symmetry sweep at y=0
python -c "
from symmetry_reduction.orbit_symmetry import run_symmetry_sweep
import numpy as np
run_symmetry_sweep(p_values=(2,3,4,5), gamma_values=(0.3, 1.0, np.pi, 2*np.pi))
"

# Symmetry sweep at saddle
python -c "
from symmetry_reduction.orbit_symmetry import run_symmetry_sweep
import numpy as np
run_symmetry_sweep(p_values=(2,3,4,5), gamma_values=(0.3, 1.0, np.pi, 2*np.pi), saddle=True)
"

# Complement sweep
python -c "
from symmetry_reduction.orbit_symmetry import run_complement_sweep
import numpy as np
run_complement_sweep(p_values=(3, 4, 5), gamma_values=(0.3, 1.0, np.pi, 2*np.pi))
"
```
