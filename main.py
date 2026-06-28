"""
Entry point – orchestrates the full calculation and plotting pipeline.

Usage
-----
    python main.py
"""

from config import FIXED_PARAMS, U_VALS, R_FIXED, THETA_FIXED
from analysis import run_U_sweep
from plotting import plot_concurrence


def main():
    print("=" * 60)
    print("  Superconducting Concurrence  –  U sweep")
    print("=" * 60)

    # ── Run the sweep ─────────────────────────────────────────────────────────
    U_vals, C_vals, metadata = run_U_sweep(
        U_vals       = U_VALS,
        fixed_params = FIXED_PARAMS,
        r            = R_FIXED,
        theta        = THETA_FIXED,
        verbose      = True,
    )

    # ── Plot results ──────────────────────────────────────────────────────────
    plot_concurrence(
        U_vals, C_vals,
        fixed_params = FIXED_PARAMS,
        save_path    = "concurrence_vs_U.png",
    )


if __name__ == "__main__":
    main()