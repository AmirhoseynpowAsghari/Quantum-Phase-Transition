"""
Self-consistent gap-equation solver for the superconducting order parameters.
"""

import numpy as np
from scipy.optimize import minimize, least_squares, basinhopping
from config import FIXED_PARAMS, SOLVER_CONFIG


def _build_kspace(V_SO, Nk):
    """Build k-space arrays and dispersion helpers."""
    t  = FIXED_PARAMS['t']
    kx = np.linspace(-np.pi, np.pi, Nk)
    ky = np.linspace(-np.pi, np.pi, Nk)
    KX, KY = np.meshgrid(kx, ky)

    sqrt_sin = np.sqrt(np.sin(KX) ** 2 + np.sin(KY) ** 2)
    cos_sum  = np.cos(KX) + np.cos(KY)

    epsilon_ks = np.stack([
        -2 * t * cos_sum - 2 * V_SO * sqrt_sin,
        -2 * t * cos_sum + 2 * V_SO * sqrt_sin,
    ], axis=-1)

    s_k = 0.5 * (np.cos(KX) + np.cos(KY))
    return KX, KY, epsilon_ks, s_k, sqrt_sin


def _build_initial_guess(U, epsilon_ks):
    """Generate an initial guess for [Delta0, DeltaS, mu]."""
    if U < 0:
        delta0_guess = 0.1
        deltaS_guess = 0.05 * np.sign(U)
        mu_guess     = np.mean(epsilon_ks) - 0.1 * abs(U)
    else:
        sgn          = int(np.sign(U)) if U != 0 else 1
        delta0_guess = sgn * 0.1
        deltaS_guess = sgn * 0.05
        mu_guess     = np.mean(epsilon_ks)
    return [delta0_guess, deltaS_guess, mu_guess]


def _make_residuals(U, n, Nk, Delta_t, epsilon_ks, s_k, sqrt_sin):
    """
    Return residuals(x) and residuals_array(x) closures.
    x = [Delta0, DeltaS, mu]
    """
    t           = FIXED_PARAMS['t']
    norm_factor = 4 * Nk ** 2
    final_sum_DS_holder = [None]   # mutable container so closure can write

    def residuals(x):
        Delta0, DeltaS, mu = x

        Delta_ks    = Delta0 - (DeltaS / (4 * t)) * epsilon_ks
        E_ks_sq     = Delta_ks ** 2 + (epsilon_ks - mu) ** 2
        E_ks        = np.sqrt(np.maximum(E_ks_sq, 1e-12))
        inv_E_ks    = np.where(E_ks > 1e-12, 1.0 / E_ks, 0.0)

        # Filling equation
        term_n = (epsilon_ks - mu) * inv_E_ks
        eq_n   = 1.0 - (1.0 / (2 * Nk ** 2)) * np.sum(term_n) - n

        # Gap equations
        sum_D0 = 0.0
        sum_DS = 0.0
        for s in range(2):
            s_prime = 1 if s == 0 else -1
            weight  = (U
                       + 8 * Delta_t * s_k
                       + 4 * (Delta_t / t) * V_SO_holder[0] * s_prime * sqrt_sin)
            term_gap = Delta_ks[..., s] * inv_E_ks[..., s]
            sum_D0  += np.sum(weight * term_gap)
            sum_DS  += np.sum(term_gap)

        eq_D0 = Delta0 + sum_D0 / norm_factor
        eq_DS = DeltaS + 8 * Delta_t * sum_DS / norm_factor

        final_sum_DS_holder[0] = sum_DS
        return [eq_D0, eq_DS, eq_n]

    def residuals_array(x):
        return np.array(residuals(x), dtype=float)

    def cost_function(x):
        return float(np.sum(np.array(residuals(x)) ** 2))

    return residuals, residuals_array, cost_function, final_sum_DS_holder

# Module-level holder used inside closure (set by compute_self_consistent_improved)
V_SO_holder = [None]


def _try_least_squares(residuals_array, x0, verbose):
    """Attempt solution with scipy.optimize.least_squares."""
    cfg = SOLVER_CONFIG
    for method in ('lm', 'trf', 'dogbox'):
        try:
            if verbose:
                print(f"  least_squares / {method} ...")
            res = least_squares(
                residuals_array, x0, method=method,
                ftol=cfg['ftol'], xtol=cfg['xtol'], gtol=cfg['gtol'],
                max_nfev=cfg['max_nfev'], verbose=0,
            )
            if res.success and np.max(np.abs(res.fun)) < cfg['residual_tol']:
                if verbose:
                    print(f"  ✓ converged  cost={res.cost:.2e}  fun={res.fun}")
                return res.x, res
        except Exception as exc:
            if verbose:
                print(f"  least_squares/{method} error: {exc}")
    return None, None


def _try_minimize(cost_function, x0, U, verbose):
    """Attempt solution with scipy.optimize.minimize."""
    cfg     = SOLVER_CONFIG
    methods = (['Powell', 'Nelder-Mead', 'BFGS', 'CG']
               if U < 0 else
               ['Nelder-Mead', 'Powell', 'BFGS', 'CG'])

    for method in methods:
        try:
            if verbose:
                print(f"  minimize / {method} ...")
            res = minimize(cost_function, x0, method=method,
                           tol=1e-7,
                           options={'maxiter': 5000, 'disp': False})
            if res.success and res.fun < cfg['cost_tol']:
                if verbose:
                    print(f"  ✓ converged  cost={res.fun:.2e}")
                return res.x, res
        except Exception as exc:
            if verbose:
                print(f"  minimize/{method} error: {exc}")
    return None, None


def _try_basinhopping(cost_function, x0, U, verbose):
    """Attempt global optimisation with basin-hopping."""
    cfg = SOLVER_CONFIG
    if U < 0:
        minimizer_kwargs = {'method': 'Powell', 'options': {'ftol': 1e-7}}
        T, stepsize = 1.0, 0.7
    else:
        minimizer_kwargs = {'method': 'Nelder-Mead', 'options': {'fatol': 1e-7}}
        T, stepsize = 0.5, 0.5

    try:
        if verbose:
            print("  basin-hopping ...")
        res = basinhopping(cost_function, x0, niter=50,
                           T=T, stepsize=stepsize,
                           minimizer_kwargs=minimizer_kwargs,
                           disp=verbose)
        if res.fun < cfg['cost_tol']:
            if verbose:
                print(f"  ✓ converged  cost={res.fun:.2e}")
            return res.x, res
    except Exception as exc:
        if verbose:
            print(f"  basin-hopping error: {exc}")
    return None, None


def _build_uv(sol, epsilon_ks):
    """Compute u_ks and v_ks from the converged solution vector."""
    t = FIXED_PARAMS['t']
    Delta0, DeltaS, mu = sol
    Delta_ks = Delta0 - (DeltaS / (4 * t)) * epsilon_ks
    E_ks_sq  = Delta_ks ** 2 + (epsilon_ks - mu) ** 2
    E_ks     = np.sqrt(np.maximum(E_ks_sq, 1e-24))
    E_safe   = np.maximum(E_ks, 1e-12)
    ratio    = (epsilon_ks - mu) / E_safe
    u_ks     = np.sqrt(np.maximum(0.5 * (1 + ratio), 0))
    v_ks     = np.sqrt(np.maximum(0.5 * (1 - ratio), 0))
    return u_ks, v_ks


def compute_self_consistent_improved(V_SO, U, n, Nk, Delta_t,
                                     initial_guess=None, verbose=False):
    """
    Solve the self-consistent BdG gap equations.

    Parameters
    ----------
    V_SO          : float – Spin-orbit coupling.
    U             : float – Interaction parameter (positive or negative).
    n             : float – Target filling.
    Nk            : int   – k-grid size.
    Delta_t       : float – Triplet pairing amplitude.
    initial_guess : list  – [Delta0, DeltaS, mu] starting point (optional).
    verbose       : bool  – Print solver progress.

    Returns
    -------
    u_ks, v_ks : ndarray or None
    KX, KY     : ndarray or None
    sol        : list [Delta0, DeltaS, mu] or None
    final_sum_DS : float or None
    """
    # Store V_SO in module-level holder so the closure can read it
    V_SO_holder[0] = V_SO

    KX, KY, epsilon_ks, s_k, sqrt_sin = _build_kspace(V_SO, Nk)

    x0 = initial_guess if initial_guess is not None else _build_initial_guess(U, epsilon_ks)
    if verbose:
        print(f"Initial guess: {x0}")

    residuals, residuals_array, cost_function, sum_DS_holder = _make_residuals(
        U, n, Nk, Delta_t, epsilon_ks, s_k, sqrt_sin
    )

    # ── Solver cascade ────────────────────────────────────────────────────────
    sol, result_obj = _try_least_squares(residuals_array, x0, verbose)
    if sol is None:
        sol, result_obj = _try_minimize(cost_function, x0, U, verbose)
    if sol is None:
        sol, result_obj = _try_basinhopping(cost_function, x0, U, verbose)

    if sol is None:
        if verbose:
            print(f"All solvers failed for V_SO={V_SO}, U={U}.")
        return None, None, None, None, None, None

    # ── Residual quality check ────────────────────────────────────────────────
    final_res = residuals_array(sol)
    if np.max(np.abs(final_res)) > SOLVER_CONFIG['residual_tol']:
        if verbose:
            print(f"Warning: large residuals {final_res} for U={U}.")

    # ── Build coherence factors ───────────────────────────────────────────────
    u_ks, v_ks = _build_uv(sol, epsilon_ks)

    norm_check = u_ks ** 2 + v_ks ** 2
    if not np.allclose(norm_check, 1.0, atol=1e-4) and verbose:
        print(f"Warning: normalization error max={np.max(np.abs(norm_check-1)):.2e}")

    # Trigger one final residuals call to populate sum_DS_holder
    residuals(sol)
    final_sum_DS = sum_DS_holder[0]

    if verbose:
        D0, DS, mu = sol
        print(f"Solution: Delta0={D0:.4f}  DeltaS={DS:.4f}  mu={mu:.4f}")

    return u_ks, v_ks, KX, KY, sol, final_sum_DS