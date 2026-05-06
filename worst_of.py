"""
Worst-of Autocall pricer.

Multi-underlying GBM under risk-neutral measure with Cholesky correlation.
Recall and PDI barriers are based on the WORST PERFORMER across all underlyings:
    perf_i(t) = S_i(t) / S_i(0)
    worst(t)  = min_i perf_i(t)

Autocall: worst(t_obs) >= B_ac  → called, pays N*(1 + coupon*t)
PDI     : min_t worst(t) < B_pdi → downside at maturity
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WorstOfParams:
    # Underlyings
    spots:    list[float]           # S0 for each asset
    vols:     list[float]           # annualised vol for each asset
    corr:     list[list[float]]     # correlation matrix (n x n)
    q:        list[float]           # dividend yields

    # Market
    r:        float = 0.05

    # Product
    notional:    float = 100.0
    T:           float = 3.0
    obs_freq:    float = 1.0
    coupon_pa:   float = 0.08
    barrier_ac:  float = 1.00       # % of initial level
    barrier_pdi: float = 0.60

    # MC
    n_paths:    int   = 50_000
    n_steps_py: int   = 252         # steps per year
    antithetic: bool  = True
    seed:       Optional[int] = 42

    @property
    def n_assets(self) -> int:
        return len(self.spots)

    @property
    def total_steps(self) -> int:
        return round(self.n_steps_py * self.T)

    @property
    def obs_dates(self) -> np.ndarray:
        n = round(self.T / self.obs_freq)
        return np.array([self.obs_freq * (i + 1) for i in range(n)])

    @property
    def corr_matrix(self) -> np.ndarray:
        return np.array(self.corr)


def simulate_worst_of(p: WorstOfParams) -> tuple[np.ndarray, float]:
    """
    Simulate worst-of paths via multi-asset GBM + Cholesky.

    Returns
    -------
    worst_paths : ndarray shape (n_paths, total_steps+1)
                  worst performer across assets at each time step
    dt          : float
    """
    rng    = np.random.default_rng(p.seed)
    dt     = p.T / p.total_steps
    n      = p.n_assets
    n_p    = p.n_paths
    n_s    = p.total_steps

    # Cholesky: correlate the Brownian increments
    L = np.linalg.cholesky(p.corr_matrix)   # (n, n)

    # Draw standard normals: (n_paths, n_steps, n_assets)
    if p.antithetic:
        half   = n_p // 2
        Z_half = rng.standard_normal((half, n_s, n))
        Z      = np.concatenate([Z_half, -Z_half], axis=0)
    else:
        Z      = rng.standard_normal((n_p, n_s, n))

    # Correlate: Z_corr[path, step, :] = L @ Z[path, step, :]
    Z_corr = Z @ L.T                        # (n_paths, n_steps, n_assets)

    # Build log-returns for each asset
    spots  = np.array(p.spots)
    vols   = np.array(p.vols)
    qs     = np.array(p.q)

    drifts = (p.r - qs - 0.5 * vols ** 2) * dt           # (n,)
    diff   = vols * np.sqrt(dt)                            # (n,)

    # log_inc[path, step, asset]
    log_inc = drifts[None, None, :] + diff[None, None, :] * Z_corr

    # Cumulative → prices normalised by S0 (= performance)
    log_cum  = np.concatenate(
        [np.zeros((n_p, 1, n)), np.cumsum(log_inc, axis=1)], axis=1
    )                                                       # (n_paths, n_steps+1, n_assets)
    perfs    = np.exp(log_cum)                              # perf_i = S_i(t)/S_i(0)

    # Worst performer at each time step
    worst_paths = perfs.min(axis=2)                         # (n_paths, n_steps+1)

    return worst_paths, dt


def price_worst_of(p: WorstOfParams) -> dict:
    """
    Price a worst-of autocall on simulated paths.

    Returns a results dict with price, std_error, probabilities, payoffs.
    """
    worst_paths, dt = simulate_worst_of(p)

    n_paths  = worst_paths.shape[0]
    n_steps  = worst_paths.shape[1] - 1
    obs_idx  = np.clip(
        np.round(p.obs_dates / dt).astype(int), 0, n_steps
    )

    payoffs    = np.full(n_paths, np.nan)
    called     = np.zeros(n_paths, dtype=bool)
    call_date  = np.full(n_paths, np.nan)
    prob_by_obs = {}

    # Observation dates
    for obs_t, idx in zip(p.obs_dates, obs_idx):
        worst_obs  = worst_paths[:, idx]
        ac_trigger = (~called) & (worst_obs >= p.barrier_ac)

        disc = np.exp(-p.r * obs_t)
        recall_pmt = p.notional * (1.0 + p.coupon_pa * obs_t)
        payoffs    = np.where(ac_trigger & np.isnan(payoffs),
                              disc * recall_pmt, payoffs)
        call_date  = np.where(ac_trigger & ~called, obs_t, call_date)
        called     = called | ac_trigger
        prob_by_obs[obs_t] = float(ac_trigger.mean())

    # PDI: running minimum of worst performer
    path_min      = worst_paths[:, 1:].min(axis=1)
    pdi_triggered = path_min < p.barrier_pdi

    # Maturity payoff
    worst_T  = worst_paths[:, -1]
    disc_T   = np.exp(-p.r * p.T)
    mat_pmt  = np.where(pdi_triggered,
                         p.notional * worst_T,     # downside on worst
                         p.notional)               # protected
    payoffs  = np.where(np.isnan(payoffs), disc_T * mat_pmt, payoffs)

    price    = float(payoffs.mean())
    std_err  = float(payoffs.std(ddof=1) / np.sqrt(n_paths))
    pdi_mask = ~called & pdi_triggered
    ok_mask  = ~called & ~pdi_triggered

    return {
        "price":          price,
        "std_error":      std_err,
        "ci_lo":          price - 1.96 * std_err,
        "ci_hi":          price + 1.96 * std_err,
        "payoffs":        payoffs,
        "prob_called":    float(called.mean()),
        "prob_pdi":       float(pdi_mask.mean()),
        "prob_mat_ok":    float(ok_mask.mean()),
        "avg_call_date":  float(np.nanmean(call_date)) if called.any() else float("nan"),
        "prob_by_obs":    prob_by_obs,
        "called_mask":    called,
        "pdi_mask":       pdi_mask,
        "worst_paths":    worst_paths,
        "dt":             dt,
    }
