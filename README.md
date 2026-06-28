# README.md

```markdown
# Superconducting Concurrence Calculator

A modular Python toolkit for computing quantum **concurrence** in a
spin-orbit-coupled superconductor as a function of the interaction
parameter **U**.  
The code solves the self-consistent Bogoliubov–de Gennes (BdG) gap
equations on a 2-D square-lattice k-space grid and evaluates the
real-space Green's functions that enter the concurrence formula.

---

## Table of Contents

- [Physics Background](#physics-background)
- [Project Structure](#project-structure)
- [Module Descriptions](#module-descriptions)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [How It Works](#how-it-works)
- [Solver Strategy](#solver-strategy)
- [Output](#output)
- [Extending the Code](#extending-the-code)
- [Dependencies](#dependencies)
- [License](#license)

---

## Physics Background

### Model

The system is a 2-D square-lattice superconductor with

| Symbol    | Meaning                                      |
|-----------|----------------------------------------------|
| `t`       | Nearest-neighbour hopping amplitude          |
| `V_SO`    | Rashba spin-orbit coupling strength          |
| `U`       | On-site interaction (attractive `U<0`, repulsive `U>0`) |
| `Delta_t` | Triplet pairing amplitude                    |
| `n`       | Average electron filling per site            |

The single-particle dispersion for spin band *s* = ±1 is

```
ε_k^s = -2t(cos kx + cos ky) ∓ 2 V_SO √(sin²kx + sin²ky)
```

### BdG Gap Equations

Three coupled self-consistency equations are solved simultaneously:

```
Δ₀ + Σ_k  w_k  Δ_k / E_k  = 0          (singlet gap)
ΔS + 8Δt  Σ_k  Δ_k / E_k  = 0          (triplet gap)
n  = 1 - (1/2Nk²) Σ_k (ε_k - μ)/E_k   (filling)
```

where `E_k = √(Δ_k² + (ε_k - μ)²)` is the quasiparticle energy.

### Green's Functions

The normal **G** and anomalous **F** Green's functions are obtained
by a 2-D Simpson integration over the Brillouin zone:

```
G(r,θ) = -(i/4π²) ∫∫ u_k u_k* e^{ik·r} d²k
F(r,θ) =  (i/4π²) ∫∫ v_k* u_k* e^{-ik·r} d²k
```

### Concurrence

The entanglement concurrence between two sites separated by **r** is

```
p = (f² + g²) / (2 + f² - g²)
C = max(0,  (3p - 1) / 2)
```

where `g = Im G / Im G₀` and `f = Im F / Im G₀`.

---

## Project Structure

```
superconducting_concurrence/
│
├── main.py               ← Entry point
├── config.py             ← All parameters and numerical settings
├── greens_functions.py   ← BZ-integrated Green's functions
├── concurrence.py        ← Concurrence from G and F
├── self_consistent.py    ← BdG gap-equation solver (multi-method cascade)
├── continuation.py       ← Parameter-continuation wrapper
├── analysis.py           ← High-level U-sweep loop
├── plotting.py           ← Matplotlib figures
├── utils.py              ← Smoothing & numerical differentiation
│
├── requirements.txt
└── README.md
```

---

## Module Descriptions

### `config.py`
Central place for **every** tuneable parameter.  
Edit this file to change physics or numerics — no other file needs touching.

| Section | Key variables |
|---------|--------------|
| Physical | `V_SO`, `Delta_t`, `n`, `Nk`, `t` |
| Spatial | `R_FIXED`, `THETA_FIXED` |
| U range | `U_NEGATIVE`, `U_NEAR_ZERO`, `U_POSITIVE`, `U_VALS` |
| Solver tolerances | `SOLVER_CONFIG` dict |
| Continuation | `CONTINUATION_CONFIG` dict |
| Post-processing | `SMOOTHING_WINDOW`, `DERIVATIVE_WINDOW` |

---

### `greens_functions.py`
```
compute_greens_functions(u_ks, v_ks, KX, KY, r, theta)
    → G (complex), F (complex)
```
Performs a **2-D Simpson integration** of the spectral weights over
the full Brillouin zone for a given real-space separation `(r, θ)`.

---

### `concurrence.py`
```
compute_concurrence(G, F, G0)          → C (float)
evaluate_concurrence_from_solution(…)  → C, G, F, G0
```
`evaluate_concurrence_from_solution` is a convenience wrapper that
calls `compute_greens_functions` internally, so the caller only needs
`u_ks`, `v_ks`, `KX`, `KY`.

---

### `self_consistent.py`
Core solver.  Exposes one public function:
```
compute_self_consistent_improved(V_SO, U, n, Nk, Delta_t,
                                 initial_guess=None, verbose=False)
    → u_ks, v_ks, KX, KY, sol, final_sum_DS
```
Internally uses a **three-stage solver cascade**
(see [Solver Strategy](#solver-strategy)).

---

### `continuation.py`
```
solve_with_continuation(U_target, V_SO, n, Nk, Delta_t, …)
    → u_ks, v_ks, KX, KY, sol, final_sum_DS
```
Walks from a known solution at `start_U` to `U_target` in small steps,
passing each converged solution as the initial guess for the next step.
Automatically refines the step size (up to `max_recursion` times) on failure.

---

### `analysis.py`
```
run_U_sweep(U_vals, fixed_params, r, theta, verbose)
    → U_vals (ndarray), C_vals (ndarray), metadata (dict)
```
Orchestrates the full sweep:
1. Picks warm-start guesses from previously solved points.  
2. Calls `compute_self_consistent_improved` (direct solve).  
3. Falls back to `solve_with_continuation` if direct solve fails.  
4. Calls `evaluate_concurrence_from_solution` and records `C`.

---

### `plotting.py`
```
plot_concurrence(U_vals, C_vals, fixed_params, save_path=None)
    → fig, axes
```
Produces a two-panel figure:
- **Left** — Concurrence *C(U)*  
- **Right** — Numerical derivative *dC/dU*

Optionally saves to disk (`save_path`).

---

### `utils.py`
```
smooth_data(x_vals, y_vals, window_size=3)          → ndarray
compute_numerical_derivative(x_vals, y_vals, …)     → ndarray
```
`smooth_data` linearly interpolates across NaN gaps then applies a
rolling mean.  
`compute_numerical_derivative` uses centred finite differences on the
smoothed data; boundary points use one-sided differences.

---

## Installation

### 1 — Clone the repository

```bash
git clone https://github.com/your-username/superconducting-concurrence.git
cd superconducting-concurrence
```

### 2 — Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
```

---

## Quick Start

```bash
python main.py
```

This will:
1. Read all parameters from `config.py`.  
2. Sweep U from −3 to +4 (131 points by default).  
3. Print solver progress to the terminal.  
4. Display a two-panel Matplotlib figure.  
5. Save the figure as `concurrence_vs_U.png`.

---

## Configuration

Open `config.py` and adjust any of the following:

```python
# ── Physical Parameters ───────────────────────────────────────
FIXED_PARAMS = {
    'V_SO'   : 0.01,   # Spin-orbit coupling
    'Delta_t': 0.1,    # Triplet pairing amplitude
    'n'      : 1.875,  # Filling factor  (0 < n < 2)
    'Nk'     : 600,    # k-grid size  (larger → slower but more accurate)
    't'      : 0.1,    # Hopping amplitude
}

# ── Real-space probe point ────────────────────────────────────
R_FIXED     = 0.2    # distance |r|
THETA_FIXED = 0.0    # angle    θ  (radians)

# ── U sweep ──────────────────────────────────────────────────
U_NEGATIVE  = np.linspace(-3.0,  -0.099, 40)
U_NEAR_ZERO = np.linspace(-0.1,   0.1,  41)
U_POSITIVE  = np.linspace( 0.1001, 4.0, 50)
```

> **Tip — fast test run**: set `'Nk': 50` and reduce the U arrays to
> ~10 points each. A full `Nk=600` sweep takes several hours on a
> single CPU core.

---

## How It Works

```
main.py
  │
  ├─ run_U_sweep()                      [analysis.py]
  │     │
  │     ├─ for each U:
  │     │     │
  │     │     ├─ compute_self_consistent_improved()   [self_consistent.py]
  │     │     │     ├─ _build_kspace()
  │     │     │     ├─ _build_initial_guess()
  │     │     │     ├─ _try_least_squares()
  │     │     │     ├─ _try_minimize()
  │     │     │     └─ _try_basinhopping()
  │     │     │
  │     │     ├─ (on failure) solve_with_continuation()  [continuation.py]
  │     │     │     └─ compute_self_consistent_improved() × N steps
  │     │     │
  │     │     └─ evaluate_concurrence_from_solution()  [concurrence.py]
  │     │           ├─ compute_greens_functions()      [greens_functions.py]
  │     │           └─ compute_concurrence()
  │     │
  │     └─ returns U_vals, C_vals, metadata
  │
  └─ plot_concurrence()                 [plotting.py]
        └─ compute_numerical_derivative()  [utils.py]
              └─ smooth_data()
```

---

## Solver Strategy

The self-consistent equations are nonlinear and can be hard to solve,
especially near phase boundaries or for negative U.  
Three solvers are tried **in cascade**; the first one to satisfy the
residual tolerance wins.

| Priority | Solver | Methods tried | Notes |
|----------|--------|---------------|-------|
| 1 | `scipy.optimize.least_squares` | `lm`, `trf`, `dogbox` | Fastest; works well away from singularities |
| 2 | `scipy.optimize.minimize` | `Powell`, `Nelder-Mead`, `BFGS`, `CG` | Robust to non-smooth landscapes |
| 3 | `scipy.optimize.basinhopping` | wraps Powell / Nelder-Mead | Global search; used as last resort |

If **all direct solvers** fail the code falls back to
**parameter continuation**: it starts from the nearest already-solved
point and walks toward the target U in small steps, using each
converged solution as the warm start for the next step.  
If a step fails, the step count is doubled (up to `max_recursion` times).

---

## Output

### Terminal

```
════════════════════════════════════════════════════
  Superconducting Concurrence  –  U sweep
════════════════════════════════════════════════════

─── U = -3.000000  [1/131] ───
  Initial guess: [0.1, -0.05, -0.312]
  least_squares / lm ...
  ✓ converged  cost=3.21e-17  fun=[…]
  Concurrence C = 0.034521

...

═══ Summary ═══
Successful points : 128 / 131  (97.7 %)
Elapsed time      : 4823.41 s
```

### Figure

A PNG file `concurrence_vs_U.png` with two panels:

```
┌────────────────────────┬───────────────────────────┐
│   Concurrence C(U)     │   Derivative  dC/dU        │
│                        │                            │
│  0.8 ┤    ╭──╮         │  4 ┤       │               │
│  0.4 ┤ ╭──╯  ╰──       │  0 ┤───────┼───────        │
│  0.0 ┤─╯               │ -4 ┤       │               │
│      └────────── U      │    └──────────── U         │
└────────────────────────┴───────────────────────────┘
```

---

## Extending the Code

### Change the lattice geometry
Edit `_build_kspace()` in `self_consistent.py` — replace the square-lattice
dispersion `ε_k` with your own.

### Add temperature dependence
Replace the zero-temperature coherence factors in `_build_uv()` with
Fermi–Dirac-weighted versions.

### Sweep a different parameter (e.g. V_SO)
In `analysis.py`, copy `run_U_sweep` and replace the U loop with a
`V_SO` loop, passing `U` as a fixed argument to
`compute_self_consistent_improved`.

### Save raw data
Add after the sweep in `main.py`:

```python
import numpy as np
np.savetxt("results.csv",
           np.column_stack([U_vals, C_vals]),
           header="U,Concurrence",
           delimiter=",")
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥ 1.24 | Array maths, k-space grids |
| `scipy` | ≥ 1.10 | Optimisation, Simpson integration, interpolation |
| `matplotlib` | ≥ 3.7 | Plotting |

All are available on PyPI and installable via `pip`.

---

## License

This project is released under the **MIT License**.  
See [`LICENSE`](LICENSE) for the full text.

---

## Citation

If you use this code in published work, please cite the relevant
BdG / concurrence literature and acknowledge this repository:

```
@misc{superconducting_concurrence,
  author = {Your Name},
  title  = {Superconducting Concurrence Calculator},
  year   = {2025},
  url    = {https://github.com/your-username/superconducting-concurrence}
}
```
```
