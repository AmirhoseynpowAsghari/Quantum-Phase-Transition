"""
Plotting routines for concurrence and its derivative vs U.
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import compute_numerical_derivative


def plot_concurrence(U_vals, C_vals, fixed_params, save_path=None):
    """
    Plot concurrence C(U) and its numerical derivative dC/dU.

    Parameters
    ----------
    U_vals       : array-like – U values.
    C_vals       : array-like – Concurrence values (NaN for failed points).
    fixed_params : dict       – Used for axis titles.
    save_path    : str | None – If given, figure is saved here.
    """
    U_vals     = np.asarray(U_vals)
    C_vals     = np.asarray(C_vals, dtype=float)
    dC_dU_vals = compute_numerical_derivative(U_vals, C_vals, window_size=5)

    V_SO = fixed_params.get('V_SO', '?')
    n    = fixed_params.get('n',    '?')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left panel: C(U) ──────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(U_vals, C_vals, 'b-', linewidth=1.5, label='C(U)')
    ax.scatter(U_vals, C_vals, color='blue', s=12, zorder=5)
    ax.set_xlabel('U', fontsize=12)
    ax.set_ylabel('Concurrence', fontsize=12)
    ax.set_title(f'Concurrence vs U\n'
                 f'$V_{{SO}}={V_SO}$,  $n={n}$', fontsize=11)
    ax.grid(True, alpha=0.4)
    ax.legend()

    # ── Right panel: dC/dU ────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(U_vals, dC_dU_vals, 'r-', linewidth=1.5, label='dC/dU')
    ax.scatter(U_vals, dC_dU_vals, color='red', s=12, zorder=5)
    ax.set_xlabel('U', fontsize=12)
    ax.set_ylabel('dC/dU', fontsize=12)
    ax.set_title('Derivative of Concurrence vs U', fontsize=11)
    ax.grid(True, alpha=0.4)
    ax.legend()

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")

    plt.show()
    return fig, axes