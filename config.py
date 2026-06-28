"""
Configuration file for superconducting concurrence calculation.
Contains all physical parameters and numerical settings.
"""

import numpy as np

# ─── Physical Parameters ──────────────────────────────────────────────────────
FIXED_PARAMS = {
    'V_SO'   : 0.01,   # Spin-orbit coupling strength
    'Delta_t': 0.1,    # Triplet pairing parameter
    'n'      : 1.875,  # Filling factor
    'Nk'     : 600,    # k-space grid size (Nk x Nk)
    't'      : 0.1,    # Hopping amplitude
}

# ─── Spatial Parameters ───────────────────────────────────────────────────────
R_FIXED     = 0.2   # Fixed distance for Green's function evaluation
THETA_FIXED = 0.0   # Fixed angle  for Green's function evaluation

# ─── U-range Settings ─────────────────────────────────────────────────────────
U_NEGATIVE = np.linspace(-3.0,  -0.099, 40)   # Negative-U region
U_NEAR_ZERO = np.linspace(-0.1,   0.1,  41)   # Dense sampling near zero
U_POSITIVE  = np.linspace( 0.1001, 4.0, 50)   # Positive-U region
U_VALS      = np.concatenate([U_NEGATIVE, U_NEAR_ZERO, U_POSITIVE])

# ─── Solver Tolerances ────────────────────────────────────────────────────────
SOLVER_CONFIG = {
    'ftol'         : 1e-8,
    'xtol'         : 1e-8,
    'gtol'         : 1e-8,
    'max_nfev'     : 5000,
    'residual_tol' : 1e-5,
    'cost_tol'     : 1e-8,
}

# ─── Continuation Settings ────────────────────────────────────────────────────
CONTINUATION_CONFIG = {
    'steps'        : 10,
    'max_recursion': 2,
}

# ─── Smoothing / Derivative Settings ─────────────────────────────────────────
SMOOTHING_WINDOW  = 3
DERIVATIVE_WINDOW = 5