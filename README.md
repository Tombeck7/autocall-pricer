# Monte Carlo Autocall Pricer

A fully vectorised Monte Carlo pricer for multi-barrier Autocall structured notes, built from scratch in Python.

## Features

- **GBM simulation** under risk-neutral measure — exact log-Euler discretisation
- **Multi-barrier Autocall** — recall barrier + Put Down-and-In (PDI) barrier
- **Phoenix variant** — coupon barrier with optional memory coupon
- **Variance reduction** — antithetic variates (pair Z / −Z)
- **Greeks via bump-and-reprice** — Delta, Gamma, Vega, Theta, Rho (central differences, same seed)
- **Convergence analysis** — price & std error vs N paths
- **Interactive Streamlit UI** — paths visualisation, payoff distribution, Greeks, CDF

## Product Structure

```
Observation dates: t1, t2, ..., tn

  S(ti) >= B_ac * S0  → recalled, pays N * (1 + coupon * ti)
  S(ti) <  B_ac * S0  → continues

At maturity T (if never recalled):
  min(S) <  B_pdi * S0  → pays N * S(T)/S0   [full downside]
  min(S) >= B_pdi * S0  → pays N              [capital protected]
```

## Stack

`Python` · `NumPy` · `SciPy` · `Streamlit` · `Plotly`

## Quick Start

```bash
pip install -r requirements.txt

# Interactive UI
streamlit run app.py

# CLI demo (pricing + Greeks + convergence)
python main.py
```

## Project Structure

```
├── mc_engine.py    # GBM simulation + antithetic variates
├── autocall.py     # AutocallParams dataclass + price_autocall()
├── greeks.py       # Bump-and-reprice: all_greeks(), convergence_analysis()
├── main.py         # CLI demo
├── app.py          # Streamlit interactive dashboard
└── requirements.txt
```

## Usage Example

```python
from autocall import AutocallParams, run_pricing

p = AutocallParams(
    S0=100, r=0.05, sigma=0.20, q=0.02,
    T=3.0, obs_freq=1.0, coupon_pa=0.08,
    barrier_ac=1.00, barrier_pdi=0.60,
    n_paths=100_000, antithetic=True, seed=42,
)

result, paths, dt = run_pricing(p)
print(result)
# Fair Value: 97.xx | P(Called): ~65% | P(PDI): ~10%
```
