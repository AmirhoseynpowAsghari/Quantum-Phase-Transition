"""
Numerical utilities: smoothing and numerical differentiation.
"""

import numpy as np
from scipy.interpolate import interp1d


def smooth_data(x_vals, y_vals, window_size=3):
    """
    Smooth *y_vals* over *x_vals* using a rolling mean after linear
    interpolation across NaN gaps.

    Parameters
    ----------
    x_vals      : array-like – Independent axis.
    y_vals      : array-like – Data (may contain NaNs).
    window_size : int        – Rolling-mean half-width = window_size // 2.

    Returns
    -------
    smoothed : ndarray – Smoothed data; NaNs preserved where input was NaN.
    """
    y_vals      = np.array(y_vals, dtype=float)
    valid       = ~np.isnan(y_vals)

    if np.sum(valid) <= 1:
        return y_vals

    x_v, y_v = x_vals[valid], y_vals[valid]

    try:
        interp    = interp1d(x_v, y_v, kind='linear',
                             bounds_error=False, fill_value='extrapolate')
        y_interp  = interp(x_vals)
    except ValueError as exc:
        print(f"Smoothing interpolation failed: {exc}. Returning original.")
        return y_vals

    half      = window_size // 2
    smoothed  = np.copy(y_interp)

    for i in range(len(y_interp)):
        window = y_interp[max(0, i - half): min(len(y_interp), i + half + 1)]
        valid_w = ~np.isnan(window)
        smoothed[i] = np.nanmean(window[valid_w]) if np.any(valid_w) else np.nan

    smoothed[~valid] = np.nan
    return smoothed


def compute_numerical_derivative(x_vals, y_vals, window_size=5):
    """
    Centred finite-difference derivative on smoothed data.

    Parameters
    ----------
    x_vals      : array-like
    y_vals      : array-like (may contain NaNs)
    window_size : int – Passed to smooth_data.

    Returns
    -------
    deriv : ndarray – dY/dX; NaN where not computable.
    """
    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)

    if np.sum(~np.isnan(y_vals)) <= 2:
        return np.full_like(y_vals, np.nan)

    ys    = smooth_data(x_vals, y_vals, window_size=window_size)
    deriv = np.full_like(ys, np.nan)

    # Interior points – centred difference
    for i in range(1, len(x_vals) - 1):
        if not any(np.isnan([ys[i-1], ys[i], ys[i+1]])):
            dx = x_vals[i+1] - x_vals[i-1]
            if abs(dx) > 1e-12:
                deriv[i] = (ys[i+1] - ys[i-1]) / dx

    # Boundary points – one-sided difference
    for i, (a, b) in enumerate([(0, 1), (-1, -2)]):
        if not any(np.isnan([ys[a], ys[b]])):
            dx = x_vals[a] - x_vals[b]
            if abs(dx) > 1e-12:
                deriv[a] = (ys[a] - ys[b]) / dx

    return deriv