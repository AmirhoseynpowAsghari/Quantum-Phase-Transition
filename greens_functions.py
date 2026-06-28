"""
k-space Green's function computation using Simpson integration.
"""

import numpy as np
from scipy.integrate import simpson


def compute_greens_functions(u_ks, v_ks, KX, KY, r, theta):
    """
    Compute the normal (G) and anomalous (F) Green's functions
    by integrating over the full Brillouin zone.

    Parameters
    ----------
    u_ks : ndarray, shape (Nk, Nk, 2)
        Particle coherence factors for spin bands s=0,1.
    v_ks : ndarray, shape (Nk, Nk, 2)
        Hole coherence factors for spin bands s=0,1.
    KX, KY : ndarray, shape (Nk, Nk)
        Meshgrid of k-points.
    r : float
        Real-space distance.
    theta : float
        Real-space angle (radians).

    Returns
    -------
    G : complex
        Normal Green's function at (r, theta).
    F : complex
        Anomalous Green's function at (r, theta).
    """
    kx = KX[0, :]
    ky = KY[:, 0]

    rx    = r * np.cos(theta)
    ry    = r * np.sin(theta)
    phase = np.exp(1j * (KX * rx + KY * ry))

    G_sum = np.zeros(KX.shape, dtype=complex)
    F_sum = np.zeros(KX.shape, dtype=complex)

    for s in range(2):
        G_sum += -1j * u_ks[..., s] * np.conj(u_ks[..., s]) * phase
        F_sum +=  1j * np.conj(v_ks[..., s]) * np.conj(u_ks[..., s]) * np.conj(phase)

    # Two-dimensional Simpson integration
    Gx = simpson(G_sum, x=kx, axis=1)
    Fx = simpson(F_sum, x=kx, axis=1)
    G  = simpson(Gx, x=ky) / (4 * np.pi ** 2)
    F  = simpson(Fx, x=ky) / (4 * np.pi ** 2)

    return G, F