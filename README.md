# Quantum Phase Transition

A modular Python codebase for studying concurrence and phase-transition-like behavior in a spin-orbit-coupled superconducting system. The project solves self-consistent Bogoliubov–de Gennes (BdG)-style gap equations over a two-dimensional square-lattice momentum grid, evaluates real-space Green's functions, and plots the concurrence as a function of the interaction parameter `U`.

## Features

- Sweeps the interaction strength `U` across negative, near-zero, and positive regimes.
- Solves coupled self-consistency equations for superconducting order parameters and chemical potential.
- Computes normal and anomalous Green's functions using two-dimensional Simpson integration over the Brillouin zone.
- Evaluates quantum concurrence from the Green's functions.
- Uses warm starts and continuation to improve convergence across difficult parameter regions.
- Produces a plot of concurrence `C(U)` and its numerical derivative `dC/dU`.

## Repository Structure

```text
Quantum-Phase-Transition/
├── main.py               # Entry point for the full sweep and plotting workflow
├── config.py             # Physical parameters, U sweep settings, and solver options
├── analysis.py           # High-level U-sweep loop
├── self_consistent.py    # Self-consistent gap-equation solver
├── continuation.py       # Parameter-continuation fallback solver
├── greens_functions.py   # Brillouin-zone Green's function integration
├── concurrence.py        # Concurrence calculation from Green's functions
├── plotting.py           # Plotting routines for C(U) and dC/dU
├── utils.py              # Smoothing and numerical differentiation utilities
└── README.md
```

## Requirements

This project uses Python 3 and the following packages:

```text
numpy
scipy
matplotlib
```

Recommended versions:

```text
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
```

## Installation

Clone the repository:

```bash
git clone https://github.com/AmirhoseynpowAsghari/Quantum-Phase-Transition.git
cd Quantum-Phase-Transition
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install numpy scipy matplotlib
```

Optional: create a `requirements.txt` file for reproducible installs:

```bash
pip freeze > requirements.txt
```

## Quick Start

Run the full calculation:

```bash
python main.py
```

The script will:

1. Load the physical and numerical parameters from `config.py`.
2. Sweep the interaction parameter `U` using the arrays defined in `config.py`.
3. Solve the self-consistent equations at each point.
4. Compute the concurrence for the configured real-space separation.
5. Plot `C(U)` and `dC/dU`.
6. Save the figure as:

```text
concurrence_vs_U.png
```

## Configuration

Most settings are controlled in `config.py`.

### Physical Parameters

```python
FIXED_PARAMS = {
    "V_SO": 0.01,    # Spin-orbit coupling strength
    "Delta_t": 0.1, # Triplet pairing parameter
    "n": 1.875,     # Filling factor
    "Nk": 600,      # k-space grid size: Nk x Nk
    "t": 0.1,       # Hopping amplitude
}
```

### Real-Space Probe Point

```python
R_FIXED = 0.2
THETA_FIXED = 0.0
```

### Interaction Sweep

```python
U_NEGATIVE = np.linspace(-3.0, -0.099, 40)
U_NEAR_ZERO = np.linspace(-0.1, 0.1, 41)
U_POSITIVE = np.linspace(0.1001, 4.0, 50)
U_VALS = np.concatenate([U_NEGATIVE, U_NEAR_ZERO, U_POSITIVE])
```

### Solver Controls

```python
SOLVER_CONFIG = {
    "ftol": 1e-8,
    "xtol": 1e-8,
    "gtol": 1e-8,
    "max_nfev": 5000,
    "residual_tol": 1e-5,
    "cost_tol": 1e-8,
}
```

### Continuation Controls

```python
CONTINUATION_CONFIG = {
    "steps": 10,
    "max_recursion": 2,
}
```

For a faster test run, reduce `Nk` and use fewer `U` points. For example, set `Nk` to `50` and shorten the `U_NEGATIVE`, `U_NEAR_ZERO`, and `U_POSITIVE` arrays.

## How the Workflow Runs

```text
main.py
│
├── run_U_sweep(...)                         # analysis.py
│   │
│   ├── choose an initial guess or warm start
│   ├── compute_self_consistent_improved(...) # self_consistent.py
│   │   ├── least_squares solver attempts
│   │   ├── minimize solver attempts
│   │   └── basinhopping fallback
│   │
│   ├── solve_with_continuation(...)          # continuation.py, if direct solve fails
│   └── evaluate_concurrence_from_solution(...) # concurrence.py
│       └── compute_greens_functions(...)     # greens_functions.py
│
└── plot_concurrence(...)                     # plotting.py
    └── compute_numerical_derivative(...)     # utils.py
```

## Module Overview

### `main.py`

Entry point for the project. It runs the configured `U` sweep, computes concurrence values, and saves the final plot.

### `config.py`

Defines physical parameters, the real-space evaluation point, the `U` sweep arrays, solver tolerances, continuation settings, and smoothing options.

### `analysis.py`

Contains `run_U_sweep`, which coordinates the calculation across all `U` values. It stores converged solutions and reuses them as warm starts for nearby points.

### `self_consistent.py`

Builds the momentum-space grid and solves the self-consistent equations for:

```text
Delta0, DeltaS, mu
```

The solver tries multiple numerical strategies:

1. `scipy.optimize.least_squares`
2. `scipy.optimize.minimize`
3. `scipy.optimize.basinhopping`

### `continuation.py`

Provides a fallback strategy when a direct solve fails. It walks from a known solution to the target `U` value through smaller intermediate steps.

### `greens_functions.py`

Computes the normal Green's function `G` and anomalous Green's function `F` by integrating over the Brillouin zone with Simpson integration.

### `concurrence.py`

Computes concurrence using normalized imaginary parts of the Green's functions.

### `plotting.py`

Generates a two-panel Matplotlib figure showing:

- concurrence versus `U`
- numerical derivative of concurrence versus `U`

### `utils.py`

Provides helper functions for smoothing data and computing finite-difference derivatives.

## Output

Running:

```bash
python main.py
```

creates a plot file:

```text
concurrence_vs_U.png
```

The figure contains two panels:

```text
C(U)       : concurrence as a function of interaction strength
 dC/dU     : numerical derivative, useful for identifying sharp changes
```

The terminal also prints solver progress, convergence information, concurrence values, and a final summary.

## Performance Notes

The runtime is strongly affected by `Nk`, because the code uses an `Nk x Nk` momentum grid. The default value `Nk = 600` can be computationally expensive. For debugging or development, start with a smaller grid such as:

```python
FIXED_PARAMS["Nk"] = 50
```

After confirming the workflow works, increase `Nk` for more accurate results.

## Troubleshooting

### `ModuleNotFoundError: No module named 'numpy'`

Install the required packages:

```bash
pip install numpy scipy matplotlib
```

### The script runs slowly

Reduce `Nk` in `config.py` and/or reduce the number of sampled `U` values.

### Some `U` values fail to converge

The code already attempts warm starts and continuation. If failures persist, try:

- reducing the spacing between nearby `U` values
- increasing `CONTINUATION_CONFIG["steps"]`
- increasing solver iteration limits in `SOLVER_CONFIG`
- using a smaller `Nk` while debugging

## Saving Raw Data

To save the sweep results as a CSV file, add the following to `main.py` after `run_U_sweep(...)` returns:

```python
import numpy as np

np.savetxt(
    "concurrence_results.csv",
    np.column_stack([U_vals, C_vals]),
    delimiter=",",
    header="U,Concurrence",
    comments="",
)
```

## Suggested `requirements.txt`

A simple `requirements.txt` for this project would be:

```text
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
```

## Citation

If you use this code in academic work, cite the relevant literature for BdG theory, superconducting Green's functions, and concurrence. You may also acknowledge this repository:

```bibtex
@misc{quantum_phase_transition,
  author = {AmirhoseynpowAsghari},
  title = {Quantum Phase Transition},
  year = {2026},
  url = {https://github.com/AmirhoseynpowAsghari/Quantum-Phase-Transition}
}
```

## License

No license file is currently included in the repository. Add a `LICENSE` file before distributing or reusing the code publicly.
