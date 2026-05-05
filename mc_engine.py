"""
Monte Carlo GBM simulation engine.

Exact log-Euler discretisation under risk-neutral measure:
    S(t+dt) = S(t) * exp( (r - q - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z )

Variance reduction: antithetic variates (halves effective draw count).
"""

import numpy as np


def simulate_gbm(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    q: float = 0.0,
    antithetic: bool = True,
    seed: int | None = None,
) -> tuple[np.ndarray, float]:
    """
    Simulate GBM paths under the risk-neutral measure.

    Parameters
    ----------
    S0        : initial spot
    r         : continuously-compounded risk-free rate
    sigma     : annualised volatility
    T         : total horizon in years
    n_steps   : number of time steps
    n_paths   : number of paths  (must be even when antithetic=True)
    q         : continuous dividend / repo yield
    antithetic: pair each Z with -Z to reduce variance
    seed      : RNG seed for reproducibility

    Returns
    -------
    paths : ndarray, shape (n_paths, n_steps + 1)
    dt    : float, size of each time step
    """
    rng = np.random.default_rng(seed)
    dt  = T / n_steps

    drift   = (r - q - 0.5 * sigma ** 2) * dt
    diffuse = sigma * np.sqrt(dt)

    if antithetic:
        half   = n_paths // 2
        Z_half = rng.standard_normal((half, n_steps))
        Z      = np.concatenate([Z_half, -Z_half], axis=0)
    else:
        Z = rng.standard_normal((n_paths, n_steps))

    log_inc   = drift + diffuse * Z                                 # (n_paths, n_steps)
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_inc, axis=1)], axis=1
    )                                                               # (n_paths, n_steps+1)

    return S0 * np.exp(log_paths), dt


def path_stats(paths: np.ndarray, dt: float) -> dict:
    """Return basic statistics over the simulated paths."""
    n_paths, n_steps_1 = paths.shape
    T     = (n_steps_1 - 1) * dt
    times = np.linspace(0, T, n_steps_1)
    return {
        "mean_path":   paths.mean(axis=0),
        "p5_path":     np.percentile(paths, 5,  axis=0),
        "p25_path":    np.percentile(paths, 25, axis=0),
        "p75_path":    np.percentile(paths, 75, axis=0),
        "p95_path":    np.percentile(paths, 95, axis=0),
        "times":       times,
        "final_mean":  float(paths[:, -1].mean()),
        "final_std":   float(paths[:, -1].std()),
    }
