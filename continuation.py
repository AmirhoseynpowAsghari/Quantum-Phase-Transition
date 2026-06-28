"""
Parameter-continuation solver: walks from a known solution
to a target U value in small steps to maintain convergence.
"""

import numpy as np
from self_consistent import compute_self_consistent_improved
from config import CONTINUATION_CONFIG


def _build_U_path(start_U, U_target, n_steps):
    """
    Build a path from start_U to U_target with denser sampling
    near zero where convergence is hardest.
    """
    same_sign = (start_U * U_target >= 0)

    if same_sign or start_U == 0 or U_target == 0:
        return np.linspace(start_U, U_target, n_steps + 1)[1:]

    # Sign change – cluster more points near zero
    third = n_steps // 3 + 1
    if start_U < U_target:   # negative → positive
        seg1 = np.linspace(start_U, -0.01, third)[1:]
        seg2 = np.linspace(-0.01,   0.01, third)[1:]
        seg3 = np.linspace( 0.01, U_target, third)[1:]
    else:                     # positive → negative
        seg1 = np.linspace(start_U,  0.01, third)[1:]
        seg2 = np.linspace( 0.01,  -0.01, third)[1:]
        seg3 = np.linspace(-0.01, U_target, third)[1:]

    return np.concatenate([seg1, seg2, seg3])


def solve_with_continuation(U_target, V_SO, n, Nk, Delta_t,
                             initial_guess_sol=None,
                             start_U=None,
                             verbose=False,
                             continuation_steps=None,
                             max_recursion=None):
    """
    Walk from a known solution at *start_U* to *U_target* using
    parameter continuation.

    Parameters
    ----------
    U_target          : float – Desired interaction value.
    V_SO              : float – Spin-orbit coupling.
    n                 : float – Filling.
    Nk                : int   – k-grid size.
    Delta_t           : float – Triplet pairing amplitude.
    initial_guess_sol : list  – Starting [Delta0, DeltaS, mu] (optional).
    start_U           : float – U value matching *initial_guess_sol*.
    verbose           : bool  – Print progress.
    continuation_steps: int   – Number of intermediate steps.
    max_recursion     : int   – How many times steps can be doubled on failure.

    Returns
    -------
    Same 6-tuple as compute_self_consistent_improved.
    """
    cfg             = CONTINUATION_CONFIG
    n_steps         = continuation_steps if continuation_steps is not None else cfg['steps']
    max_rec         = max_recursion       if max_recursion       is not None else cfg['max_recursion']

    # ── Find a bootstrap solution if none provided ────────────────────────────
    if initial_guess_sol is None or start_U is None:
        bootstrap_Us = ([-0.1, -0.01, -0.5, -1.0]
                        if U_target < 0 else
                        [ 0.0,  0.01,  0.5,  1.0])

        for boot_U in bootstrap_Us:
            if verbose:
                print(f"Continuation bootstrap: trying U={boot_U} ...")
            u_ks, v_ks, KX, KY, sol, sum_DS = compute_self_consistent_improved(
                V_SO, boot_U, n, Nk, Delta_t, verbose=verbose
            )
            if sol is not None:
                start_U           = boot_U
                initial_guess_sol = sol
                if verbose:
                    print(f"Bootstrap succeeded at U={boot_U}.")
                break
        else:
            if verbose:
                print("Continuation: all bootstrap points failed.")
            return None, None, None, None, None, None

    if verbose:
        print(f"Continuation: {start_U:.6f} → {U_target:.6f}  ({n_steps} steps)")

    U_path        = _build_U_path(start_U, U_target, n_steps)
    current_guess = initial_guess_sol
    current_U     = start_U
    last_result   = (None,) * 6

    for step_idx, U_step in enumerate(U_path):
        if verbose:
            print(f"  step {step_idx+1}/{len(U_path)}: U={current_U:.6f} → {U_step:.6f}")

        u_ks, v_ks, KX, KY, sol, sum_DS = compute_self_consistent_improved(
            V_SO, U_step, n, Nk, Delta_t,
            initial_guess=current_guess, verbose=False,
        )

        if sol is None:
            if verbose:
                print(f"  ✗ failed at U={U_step:.6f} (step {step_idx+1})")

            # Try with finer sub-steps (recursive)
            if max_rec > 0 and step_idx > 0:
                if verbose:
                    print(f"  Refining with {n_steps*2} steps ...")
                return solve_with_continuation(
                    U_target, V_SO, n, Nk, Delta_t,
                    initial_guess_sol=current_guess,
                    start_U=current_U,
                    verbose=verbose,
                    continuation_steps=n_steps * 2,
                    max_recursion=max_rec - 1,
                )
            else:
                if verbose:
                    print("  Max recursion reached. Aborting.")
                return None, None, None, None, None, None

        current_guess = sol
        current_U     = U_step
        last_result   = (u_ks, v_ks, KX, KY, sol, sum_DS)

    if verbose:
        print(f"Continuation succeeded at U={U_target}.")
    return last_result