"""
High-level analysis loop: sweeps U, calls solvers, computes concurrence.
"""

import numpy as np
import time

from config import FIXED_PARAMS, U_VALS, R_FIXED, THETA_FIXED
from self_consistent import compute_self_consistent_improved
from continuation import solve_with_continuation
from concurrence import evaluate_concurrence_from_solution


def run_U_sweep(U_vals=None, fixed_params=None,
                r=None, theta=None, verbose=True):
    """
    Sweep over U values, solve the self-consistent equations at each point,
    and compute the concurrence.

    Parameters
    ----------
    U_vals       : array-like – U values to sweep (defaults to config.U_VALS).
    fixed_params : dict       – Physical parameters   (defaults to config.FIXED_PARAMS).
    r, theta     : float      – Real-space coordinates (defaults to config values).
    verbose      : bool       – Print progress.

    Returns
    -------
    U_vals   : ndarray  – U values used.
    C_vals   : ndarray  – Concurrence at each U.
    metadata : dict     – Extra info (elapsed time, success count, …).
    """
    if U_vals       is None: U_vals       = U_VALS
    if fixed_params is None: fixed_params = FIXED_PARAMS
    if r            is None: r            = R_FIXED
    if theta        is None: theta        = THETA_FIXED

    V_SO    = fixed_params['V_SO']
    Delta_t = fixed_params['Delta_t']
    n       = fixed_params['n']
    Nk      = fixed_params['Nk']

    U_vals  = np.asarray(U_vals)
    C_vals  = np.full(len(U_vals), np.nan)

    # Separate solution caches for positive / negative U
    pos_solutions   = {}
    neg_solutions   = {}
    last_sol        = None
    last_U          = None
    successful      = 0
    start_time      = time.time()

    for i, u_val in enumerate(U_vals):
        if verbose:
            print(f"\n─── U = {u_val:.6f}  [{i+1}/{len(U_vals)}] ───")

        # ── Choose initial guess ───────────────────────────────────────────
        guess = _pick_initial_guess(u_val, pos_solutions, neg_solutions,
                                    last_sol, last_U, verbose)

        # ── Direct solve ──────────────────────────────────────────────────
        u_ks, v_ks, KX, KY, sol, _ = compute_self_consistent_improved(
            V_SO, u_val, n, Nk, Delta_t,
            initial_guess=guess, verbose=verbose,
        )

        # ── Continuation fallback ─────────────────────────────────────────
        if sol is None:
            if verbose:
                print("Direct solve failed – trying continuation …")
            start_sol, start_u = _pick_continuation_start(
                u_val, pos_solutions, neg_solutions, last_sol, last_U
            )
            if start_sol is not None:
                u_ks, v_ks, KX, KY, sol, _ = solve_with_continuation(
                    u_val, V_SO, n, Nk, Delta_t,
                    initial_guess_sol=start_sol,
                    start_U=start_u,
                    verbose=verbose,
                )

        # ── Store & compute concurrence ───────────────────────────────────
        if sol is not None:
            successful += 1
            last_sol    = sol
            last_U      = u_val
            _store_solution(u_val, sol, pos_solutions, neg_solutions)

            try:
                C, *_ = evaluate_concurrence_from_solution(
                    u_ks, v_ks, KX, KY, r, theta
                )
                C_vals[i] = C
                if verbose:
                    print(f"Concurrence C = {C:.6f}")
            except Exception as exc:
                if verbose:
                    print(f"Concurrence error: {exc}")
        else:
            if verbose:
                print(f"Failed to converge for U={u_val:.6f}")

    elapsed = time.time() - start_time
    metadata = {
        'elapsed_time'    : elapsed,
        'successful_points': successful,
        'total_points'    : len(U_vals),
    }
    if verbose:
        _print_summary(metadata)

    return U_vals, C_vals, metadata


# ── Internal helpers ───────────────────────────────────────────────────────────

def _pick_initial_guess(u_val, pos_sol, neg_sol, last_sol, last_U, verbose):
    cache = pos_sol if u_val >= 0 else neg_sol
    if cache:
        solved_Us = np.array(list(cache.keys()))
        closest_U = solved_Us[np.argmin(np.abs(solved_Us - u_val))]
        if verbose:
            print(f"Warm start from U={closest_U:.6f}")
        return cache[closest_U]
    if last_sol is not None and last_U is not None and abs(u_val - last_U) < 2.0:
        if verbose:
            print(f"Warm start from last U={last_U:.6f}")
        return last_sol
    return None


def _pick_continuation_start(u_val, pos_sol, neg_sol, last_sol, last_U):
    cache = pos_sol if u_val >= 0 else neg_sol
    if cache:
        solved_Us = np.array(list(cache.keys()))
        closest_U = solved_Us[np.argmin(np.abs(solved_Us - u_val))]
        return cache[closest_U], closest_U
    if last_sol is not None and last_U is not None and abs(u_val - last_U) < 5.0:
        return last_sol, last_U
    return None, None


def _store_solution(u_val, sol, pos_sol, neg_sol):
    if u_val >= 0:
        pos_sol[u_val] = sol
    else:
        neg_sol[u_val] = sol


def _print_summary(meta):
    print("\n═══ Summary ═══")
    print(f"Successful points : {meta['successful_points']} / {meta['total_points']}"
          f"  ({meta['successful_points']/meta['total_points']*100:.1f} %)")
    print(f"Elapsed time      : {meta['elapsed_time']:.2f} s")