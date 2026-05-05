"""
Greeks via bump-and-reprice.

Same seed is used for base and bumped simulations so the Monte Carlo noise
cancels out in the finite-difference — the variance in the Greek estimate
comes only from the bump, not from path differences.

Greek       Formula
---------   -------------------------------------------------------
Delta       (V(S0+dS) - V(S0-dS)) / (2*dS)          central diff
Gamma       (V(S0+dS) - 2*V(S0) + V(S0-dS)) / dS^2
Vega        (V(sigma+dv) - V(sigma-dv)) / (2*dv)
Theta       -(V(T-dT) - V(T)) / dT                   forward diff
Rho         (V(r+dr) - V(r-dr)) / (2*dr)
"""

from __future__ import annotations
import numpy as np
from dataclasses import replace
from autocall import AutocallParams, run_pricing


# Default bump sizes
_BUMPS = dict(
    dS    = 1.0,     # spot bump in absolute units
    dVol  = 0.01,    # 1 vol point
    dT    = 1/365,   # 1 calendar day
    dR    = 0.0001,  # 1 basis point
)


def _price(p: AutocallParams) -> float:
    result, _, _ = run_pricing(p)
    return result.price


def _bump_spot(p: AutocallParams, new_S0: float) -> AutocallParams:
    """
    Return a new AutocallParams with a different starting spot, but keeping
    all barrier LEVELS fixed in absolute terms (as they would be post-issuance).
    """
    B_ac  = p.barrier_ac  * p.S0       # absolute recall level
    B_pdi = p.barrier_pdi * p.S0       # absolute PDI level
    B_cpn = p.barrier_cpn * p.S0 if p.barrier_cpn is not None else None
    return replace(
        p,
        S0          = new_S0,
        barrier_ac  = B_ac  / new_S0,
        barrier_pdi = B_pdi / new_S0,
        barrier_cpn = B_cpn / new_S0 if B_cpn is not None else None,
    )


def delta(p: AutocallParams, dS: float | None = None) -> float:
    """dV/dS — central difference, barrier levels held fixed."""
    h = dS or _BUMPS["dS"]
    v_up   = _price(_bump_spot(p, p.S0 + h))
    v_down = _price(_bump_spot(p, p.S0 - h))
    return (v_up - v_down) / (2 * h)


def gamma(p: AutocallParams, dS: float | None = None) -> float:
    """d2V/dS2 — central difference, barrier levels held fixed."""
    h    = dS or _BUMPS["dS"]
    v0   = _price(p)
    v_up = _price(_bump_spot(p, p.S0 + h))
    v_dn = _price(_bump_spot(p, p.S0 - h))
    return (v_up - 2 * v0 + v_dn) / h ** 2


def vega(p: AutocallParams, dVol: float | None = None) -> float:
    """dV/dsigma per 1 vol point — central difference."""
    h    = dVol or _BUMPS["dVol"]
    v_up = _price(replace(p, sigma=p.sigma + h))
    v_dn = _price(replace(p, sigma=max(p.sigma - h, 1e-4)))
    return (v_up - v_dn) / (2 * h)


def theta(p: AutocallParams, dT: float | None = None) -> float:
    """dV/dt per calendar day — forward difference."""
    h  = dT or _BUMPS["dT"]
    v0 = _price(p)
    vT = _price(replace(p, T=max(p.T - h, 1e-4)))
    return -(vT - v0) / h


def rho(p: AutocallParams, dR: float | None = None) -> float:
    """dV/dr per 1 bp — central difference."""
    h    = dR or _BUMPS["dR"]
    v_up = _price(replace(p, r=p.r + h))
    v_dn = _price(replace(p, r=p.r - h))
    return (v_up - v_dn) / (2 * h)


def all_greeks(
    p: AutocallParams,
    dS:   float | None = None,
    dVol: float | None = None,
    dT:   float | None = None,
    dR:   float | None = None,
) -> dict:
    """
    Compute all first-order Greeks.

    Returns a dict with keys: Delta, Gamma, Vega, Theta, Rho, plus the
    bump sizes used.
    """
    dS_  = dS   or _BUMPS["dS"]
    dV_  = dVol or _BUMPS["dVol"]
    dT_  = dT   or _BUMPS["dT"]
    dR_  = dR   or _BUMPS["dR"]

    # Pre-compute base + spot bumps together to reuse v0 for gamma
    v0   = _price(p)
    v_su = _price(_bump_spot(p, p.S0 + dS_))
    v_sd = _price(_bump_spot(p, p.S0 - dS_))
    v_vu = _price(replace(p, sigma=p.sigma + dV_))
    v_vd = _price(replace(p, sigma=max(p.sigma - dV_, 1e-4)))
    v_Td = _price(replace(p, T=max(p.T - dT_, 1e-4)))
    v_ru = _price(replace(p, r=p.r + dR_))
    v_rd = _price(replace(p, r=p.r - dR_))

    return {
        "Delta": (v_su - v_sd) / (2 * dS_),
        "Gamma": (v_su - 2 * v0 + v_sd) / dS_ ** 2,
        "Vega":  (v_vu - v_vd) / (2 * dV_),
        "Theta": -(v_Td - v0) / dT_,
        "Rho":   (v_ru - v_rd) / (2 * dR_),
        "_bumps": {"dS": dS_, "dVol": dV_, "dT": dT_, "dR": dR_},
    }


def greeks_profile(
    p: AutocallParams,
    param: str,
    values: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Compute all Greeks across a range of a single parameter.

    Parameters
    ----------
    param  : one of 'S0', 'sigma', 'T', 'r', 'barrier_pdi', 'barrier_ac'
    values : array of parameter values to sweep

    Returns
    -------
    dict with keys 'values', 'price', 'Delta', 'Gamma', 'Vega', 'Theta', 'Rho'
    """
    prices, deltas, gammas, vegas, thetas, rhos = [], [], [], [], [], []

    for v in values:
        p_bump = replace(p, **{param: float(v)})
        g = all_greeks(p_bump)
        prices.append(_price(p_bump))
        deltas.append(g["Delta"])
        gammas.append(g["Gamma"])
        vegas.append(g["Vega"])
        thetas.append(g["Theta"])
        rhos.append(g["Rho"])

    return {
        "values": values,
        "Price":  np.array(prices),
        "Delta":  np.array(deltas),
        "Gamma":  np.array(gammas),
        "Vega":   np.array(vegas),
        "Theta":  np.array(thetas),
        "Rho":    np.array(rhos),
    }


def convergence_analysis(
    p: AutocallParams,
    path_counts: list[int] | None = None,
) -> dict:
    """
    Price the autocall at increasing n_paths to visualise MC convergence.

    Returns arrays: n_paths, prices, std_errors, ci_lo, ci_hi.
    """
    if path_counts is None:
        path_counts = [500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000]

    prices, stds, lo, hi = [], [], [], []
    for n in path_counts:
        result, _, _ = run_pricing(replace(p, n_paths=n))
        prices.append(result.price)
        stds.append(result.std_error)
        lo.append(result.ci_lo)
        hi.append(result.ci_hi)

    return {
        "n_paths":    np.array(path_counts),
        "prices":     np.array(prices),
        "std_errors": np.array(stds),
        "ci_lo":      np.array(lo),
        "ci_hi":      np.array(hi),
    }
