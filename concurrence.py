"""
Concurrence calculation from the Green's functions.
"""

import numpy as np
from greens_functions import compute_greens_functions


def compute_concurrence(G, F, G0):
    """
    Compute the concurrence from the Green's function values.

    Parameters
    ----------
    G  : complex  – Normal Green's function at finite r.
    F  : complex  – Anomalous Green's function at finite r.
    G0 : complex  – Normal Green's function at r=0 (used as normalisation).

    Returns
    -------
    C : float
        Concurrence value in [0, 1], or np.nan if calculation is invalid.
    """
    G0_real = G0.imag

    if abs(G0_real) < 1e-12:
        print(f"Warning: G0.imag is very small ({G0_real}). Returning NaN.")
        return np.nan

    g = G.imag / G0_real
    f = F.imag / G0_real

    p_numerator   = f ** 2 + g ** 2
    p_denominator = 2 + f ** 2 - g ** 2

    if abs(p_denominator) < 1e-12:
        print(f"Warning: Denominator very small ({p_denominator}). Returning NaN.")
        return np.nan

    p = p_numerator / p_denominator
    C = max(0.0, (3 * p - 1) / 2)
    return C


def evaluate_concurrence_from_solution(u_ks, v_ks, KX, KY, r, theta):
    """
    Convenience wrapper: computes G, F, G0, then returns concurrence.

    Parameters
    ----------
    u_ks, v_ks : ndarray – coherence factors.
    KX, KY     : ndarray – k-space meshgrid.
    r, theta   : float   – real-space coordinates.

    Returns
    -------
    C  : float   – Concurrence.
    G  : complex – Normal Green's function.
    F  : complex – Anomalous Green's function.
    G0 : complex – Normal Green's function at r=0.
    """
    G, F = compute_greens_functions(u_ks, v_ks, KX, KY, r, theta)
    G0   = compute_greens_functions(u_ks, v_ks, KX, KY, 0.0, 0.0)[0]
    C    = compute_concurrence(G, F, G0)
    return C, G, F, G0