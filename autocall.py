"""
Multi-barrier Autocall structured product pricer.

Supported structures
--------------------
1. Standard Autocall (Knock-Out)
   - Recall barrier B_ac: if S(t_i) >= B_ac*S0 on obs date i → called,
     pays notional * (1 + coupon_pa * t_i)
   - PDI barrier B_pdi: if path min < B_pdi*S0 → downside participation
     at maturity (N * S_T/S0); otherwise capital protected

2. Phoenix Autocall (with coupon barrier)
   - Same recall logic
   - Coupon barrier B_cpn: pays periodic coupon N*coupon_pa*obs_freq
     even if not recalled, as long as S(t_i) >= B_cpn*S0
   - Optional memory coupon: accumulates unpaid coupons

Risk-neutral pricing via Monte Carlo on pre-simulated GBM paths.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AutocallParams:
    """All product and market parameters in one place."""
    # Market
    S0:          float = 100.0
    r:           float = 0.05
    sigma:       float = 0.20
    q:           float = 0.02

    # Product
    notional:    float = 100.0
    T:           float = 3.0
    obs_freq:    float = 1.0    # years between observation dates
    coupon_pa:   float = 0.08   # annual coupon rate

    # Barriers (as fractions of S0)
    barrier_ac:  float = 1.00   # autocall / recall barrier
    barrier_pdi: float = 0.60   # put down-and-in barrier
    barrier_cpn: Optional[float] = None  # coupon barrier (None = standard)
    memory:      bool  = False  # Phoenix memory coupon

    # MC
    n_paths:     int   = 50_000
    n_steps:     int   = 252    # daily steps per year (× T total)
    antithetic:  bool  = True
    seed:        Optional[int] = 42

    @property
    def obs_dates(self) -> np.ndarray:
        n = round(self.T / self.obs_freq)
        return np.array([self.obs_freq * (i + 1) for i in range(n)])

    @property
    def total_steps(self) -> int:
        return round(self.n_steps * self.T)


@dataclass
class PricingResult:
    price:      float
    std_error:  float
    ci_lo:      float
    ci_hi:      float
    payoffs:    np.ndarray          # shape (n_paths,)

    # Scenario probabilities
    prob_called:    float
    prob_pdi:       float
    prob_mat_ok:    float

    # Call profile
    avg_call_date:      float
    prob_by_obs_date:   dict        # {obs_date: probability}
    called_mask:        np.ndarray  # bool (n_paths,)
    pdi_mask:           np.ndarray  # bool (n_paths,) — not called + PDI hit
    mat_ok_mask:        np.ndarray  # bool (n_paths,) — survived to mat, no PDI

    def __str__(self) -> str:
        lines = [
            f"  Fair Value       : {self.price:.4f}",
            f"  Std Error        : {self.std_error:.4f}",
            f"  95% CI           : [{self.ci_lo:.4f}, {self.ci_hi:.4f}]",
            f"  P(Called)        : {self.prob_called*100:.2f}%",
            f"  P(PDI at mat.)   : {self.prob_pdi*100:.2f}%",
            f"  P(Mat. protected): {self.prob_mat_ok*100:.2f}%",
        ]
        if not np.isnan(self.avg_call_date):
            lines.append(f"  Avg call date    : {self.avg_call_date:.4f}y")
        for t, p in self.prob_by_obs_date.items():
            lines.append(f"    P(call at T={t:.2f}y) : {p*100:.2f}%")
        return "\n".join(lines)


def price_autocall(p: AutocallParams, paths: np.ndarray, dt: float) -> PricingResult:
    """
    Price the autocall on pre-simulated paths.

    Parameters
    ----------
    p     : AutocallParams
    paths : ndarray, shape (n_paths, n_steps+1)
    dt    : time step size

    Returns
    -------
    PricingResult
    """
    n_paths  = paths.shape[0]
    n_steps  = paths.shape[1] - 1
    obs_idx  = np.clip(
        np.round(p.obs_dates / dt).astype(int), 0, n_steps
    )

    payoffs        = np.full(n_paths, np.nan)
    called         = np.zeros(n_paths, dtype=bool)
    call_date      = np.full(n_paths, np.nan)
    pending_cpn    = np.zeros(n_paths)   # memory coupon accumulator
    prob_by_obs    = {}

    # ── Walk through observation dates ───────────────────────────────────────
    for obs_t, idx in zip(p.obs_dates, obs_idx):
        S_obs       = paths[:, idx]
        active      = ~called

        # Phoenix: intermediate coupon payment
        if p.barrier_cpn is not None:
            cpn_eligible = active & (S_obs >= p.barrier_cpn * p.S0)
            coupon_amt   = p.notional * p.coupon_pa * p.obs_freq
            if p.memory:
                pending_cpn += coupon_amt          # everyone accrues
                # will be paid at recall or maturity
            else:
                # Cash coupon paid now — we'll discount it and add to payoff
                disc_cpn = np.exp(-p.r * obs_t) * coupon_amt
                payoffs  = np.where(
                    cpn_eligible & np.isnan(payoffs),
                    disc_cpn,
                    np.where(cpn_eligible & ~np.isnan(payoffs),
                             payoffs + disc_cpn, payoffs),
                )

        # Autocall trigger
        ac_trigger = active & (S_obs >= p.barrier_ac * p.S0)

        if p.barrier_cpn is None:
            # Standard: coupon = coupon_pa * obs_t (accrual)
            recall_amt = p.notional * (1.0 + p.coupon_pa * obs_t)
        else:
            if p.memory:
                recall_amt = p.notional + pending_cpn  # array
            else:
                recall_amt = p.notional * (1.0 + p.coupon_pa * obs_t)

        disc       = np.exp(-p.r * obs_t)
        new_payoff = disc * recall_amt

        payoffs    = np.where(ac_trigger & np.isnan(payoffs),
                              new_payoff, payoffs)
        call_date  = np.where(ac_trigger & ~called, obs_t, call_date)
        called     = called | ac_trigger

        prob_by_obs[obs_t] = float(ac_trigger.mean())

    # ── Maturity payoff ───────────────────────────────────────────────────────
    path_min      = paths[:, 1:].min(axis=1)            # skip t=0
    pdi_triggered = path_min < p.barrier_pdi * p.S0

    S_T    = paths[:, -1]
    disc_T = np.exp(-p.r * p.T)

    if p.barrier_cpn is not None and p.memory:
        mat_cpn = pending_cpn
    else:
        mat_cpn = 0.0

    mat_payoff = np.where(
        pdi_triggered,
        p.notional * (S_T / p.S0) + mat_cpn,   # downside
        p.notional + mat_cpn,                    # protected
    )
    payoffs = np.where(np.isnan(payoffs), disc_T * mat_payoff, payoffs)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    price    = float(payoffs.mean())
    std_err  = float(payoffs.std(ddof=1) / np.sqrt(n_paths))
    ci_lo    = price - 1.96 * std_err
    ci_hi    = price + 1.96 * std_err

    pdi_mask    = ~called & pdi_triggered
    mat_ok_mask = ~called & ~pdi_triggered

    return PricingResult(
        price            = price,
        std_error        = std_err,
        ci_lo            = ci_lo,
        ci_hi            = ci_hi,
        payoffs          = payoffs,
        prob_called      = float(called.mean()),
        prob_pdi         = float(pdi_mask.mean()),
        prob_mat_ok      = float(mat_ok_mask.mean()),
        avg_call_date    = float(np.nanmean(call_date)) if called.any() else float('nan'),
        prob_by_obs_date = prob_by_obs,
        called_mask      = called,
        pdi_mask         = pdi_mask,
        mat_ok_mask      = mat_ok_mask,
    )


def run_pricing(p: AutocallParams) -> tuple[PricingResult, np.ndarray, float]:
    """
    Convenience wrapper: simulate + price in one call.

    Returns (result, paths, dt).
    """
    from mc_engine import simulate_gbm
    paths, dt = simulate_gbm(
        S0        = p.S0,
        r         = p.r,
        sigma     = p.sigma,
        T         = p.T,
        n_steps   = p.total_steps,
        n_paths   = p.n_paths,
        q         = p.q,
        antithetic= p.antithetic,
        seed      = p.seed,
    )
    result = price_autocall(p, paths, dt)
    return result, paths, dt
