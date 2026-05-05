import sys
sys.stdout.reconfigure(encoding="utf-8")

"""
Monte Carlo Autocall Pricer -- CLI Demo
========================================
Standard 3-year annual autocall + Phoenix variant.
Prints pricing results, Greeks, convergence table.
"""

import numpy as np
import time

from autocall import AutocallParams, run_pricing
from greeks import all_greeks, convergence_analysis

SEP  = "=" * 64
sep  = "-" * 64

# ============================================================================
# 1. Standard Autocall (annual obs, no coupon barrier)
# ============================================================================
print(SEP)
print("  MONTE CARLO AUTOCALL PRICER")
print(SEP)

p = AutocallParams(
    S0          = 100.0,
    r           = 0.05,
    sigma       = 0.20,
    q           = 0.02,
    notional    = 100.0,
    T           = 3.0,
    obs_freq    = 1.0,        # annual observations
    coupon_pa   = 0.08,       # 8% p.a.
    barrier_ac  = 1.00,       # recall at 100% of S0
    barrier_pdi = 0.60,       # PDI at 60% of S0
    barrier_cpn = None,       # standard autocall (no coupon barrier)
    n_paths     = 100_000,
    n_steps     = int(252 * 3),
    antithetic  = True,
    seed        = 42,
)

print(f"\n{sep}")
print("  1 . STANDARD AUTOCALL  (3Y, annual, 8% coupon, B_ac=100%, B_pdi=60%)")
print(sep)
print(f"  S0={p.S0}  r={p.r*100:.1f}%  sigma={p.sigma*100:.0f}%  "
      f"q={p.q*100:.1f}%  N={int(p.n_paths):,} paths")
print()

t0 = time.perf_counter()
result, paths, dt = run_pricing(p)
elapsed = time.perf_counter() - t0

print(result)
print(f"\n  Computed in {elapsed:.2f}s  ({int(p.n_paths):,} paths x {p.total_steps:,} steps)")

# ============================================================================
# 2. Phoenix Autocall (quarterly, coupon barrier, memory)
# ============================================================================
print(f"\n{sep}")
print("  2 . PHOENIX AUTOCALL  (3Y, quarterly, 10% p.a., B_cpn=70%, memory)")
print(sep)

p_phx = AutocallParams(
    S0          = 100.0,
    r           = 0.05,
    sigma       = 0.20,
    q           = 0.02,
    notional    = 100.0,
    T           = 3.0,
    obs_freq    = 0.25,       # quarterly
    coupon_pa   = 0.10,
    barrier_ac  = 1.00,
    barrier_pdi = 0.60,
    barrier_cpn = 0.70,       # coupon paid if S >= 70% S0
    memory      = True,       # accumulate missed coupons
    n_paths     = 100_000,
    n_steps     = int(252 * 3),
    antithetic  = True,
    seed        = 42,
)

t0 = time.perf_counter()
result_phx, _, _ = run_pricing(p_phx)
elapsed_phx = time.perf_counter() - t0

print(result_phx)
print(f"\n  Computed in {elapsed_phx:.2f}s")

# ============================================================================
# 3. Greeks (bump-and-reprice)
# ============================================================================
print(f"\n{sep}")
print("  3 . GREEKS (bump-and-reprice, standard autocall)")
print(sep)
print("  Computing ... (7 additional MC runs)")

t0 = time.perf_counter()
g = all_greeks(p)
t_greek = time.perf_counter() - t0

print(f"\n  {'Greek':<10} {'Value':>14}  {'Interpretation'}")
print("  " + "-" * 55)
interp = {
    "Delta": "dV/dS               -- +/-1 spot move",
    "Gamma": "d2V/dS2             -- curvature",
    "Vega":  "dV/dsigma per 1 vol point",
    "Theta": "dV/dt per day       -- time decay",
    "Rho":   "dV/dr per 1 bp      -- rate sensitivity",
}
for name, desc in interp.items():
    print(f"  {name:<10} {g[name]:>14.6f}  {desc}")
print(f"\n  Computed in {t_greek:.2f}s")

# ============================================================================
# 4. Convergence
# ============================================================================
print(f"\n{sep}")
print("  4 . MC CONVERGENCE  (antithetic variates)")
print(sep)

conv = convergence_analysis(p, path_counts=[1_000, 2_000, 5_000,
                                             10_000, 25_000, 50_000, 100_000])

print(f"\n  {'N paths':>10}  {'Price':>10}  {'Std Err':>10}  {'95% CI width':>14}")
print("  " + "-" * 50)
for n, pr, se, lo, hi in zip(conv["n_paths"], conv["prices"],
                               conv["std_errors"], conv["ci_lo"], conv["ci_hi"]):
    print(f"  {n:>10,}  {pr:>10.4f}  {se:>10.4f}  {(hi-lo):>14.4f}")

# ============================================================================
# 5. Sensitivity: price vs spot (delta profile)
# ============================================================================
print(f"\n{sep}")
print("  5 . PRICE vs SPOT  (holding all else fixed)")
print(sep)
print(f"\n  {'Spot S0':>10}  {'Price':>10}  {'vs par':>10}")
print("  " + "-" * 34)

from dataclasses import replace
from greeks import _bump_spot
spots = [70, 80, 90, 95, 100, 105, 110, 120, 130]
for s in spots:
    p_s = replace(_bump_spot(p, float(s)), n_paths=20_000, seed=42)
    r2, _, _ = run_pricing(p_s)
    flag = " <-- current" if s == 100 else ""
    print(f"  {s:>10.0f}  {r2.price:>10.4f}  {r2.price - p.notional:>+10.4f}{flag}")

print(f"\n{SEP}\n")
