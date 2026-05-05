import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dataclasses import replace

from mc_engine import simulate_gbm, path_stats
from autocall import AutocallParams, price_autocall, run_pricing
from greeks import all_greeks, convergence_analysis

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Autocall MC Pricer",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
div[data-testid="stMetricValue"] { font-size: 1.35rem; font-weight: 700; }
div[data-testid="stMetricDelta"] { font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Parameters")

st.sidebar.subheader("Market")
S0    = st.sidebar.number_input("Spot  S0",      1.0, 500.0, 100.0, 1.0)
r     = st.sidebar.slider("Risk-free  r (%)",    0.0, 15.0, 5.0, 0.1) / 100
sigma = st.sidebar.slider("Volatility  σ (%)",   5.0, 80.0, 20.0, 0.5) / 100
q     = st.sidebar.slider("Dividend  q (%)",     0.0, 10.0, 2.0, 0.1) / 100

st.sidebar.subheader("Product")
T           = st.sidebar.slider("Maturity  T (years)", 1.0, 5.0, 3.0, 0.5)
obs_freq    = st.sidebar.selectbox("Observations", [0.25, 0.5, 1.0],
                                    format_func=lambda x: {0.25:"Quarterly",0.5:"Semi-annual",1.0:"Annual"}[x])
coupon_pa   = st.sidebar.slider("Coupon p.a. (%)",  0.0, 30.0,  8.0, 0.5) / 100
barrier_ac  = st.sidebar.slider("Recall barrier (%S0)",  80, 120, 100) / 100
barrier_pdi = st.sidebar.slider("PDI barrier (%S0)",     40,  90,  60) / 100
phoenix_on  = st.sidebar.checkbox("Phoenix (coupon barrier)", value=False)
barrier_cpn = None
if phoenix_on:
    barrier_cpn = st.sidebar.slider("Coupon barrier (%S0)", 50, 100, 70) / 100
    memory_cpn  = st.sidebar.checkbox("Memory coupon", value=True)
else:
    memory_cpn = False

st.sidebar.subheader("Monte Carlo")
n_paths    = st.sidebar.select_slider("Paths", [5_000,10_000,25_000,50_000,100_000], 50_000)
n_steps_py = st.sidebar.select_slider("Steps/year", [52, 126, 252], 252)
seed       = st.sidebar.number_input("Seed", 0, 9999, 42, 1)
antithetic = st.sidebar.checkbox("Antithetic variates", value=True)

# ── Build params ──────────────────────────────────────────────────────────────
p = AutocallParams(
    S0=S0, r=r, sigma=sigma, q=q,
    notional=100.0, T=T, obs_freq=obs_freq,
    coupon_pa=coupon_pa,
    barrier_ac=barrier_ac, barrier_pdi=barrier_pdi,
    barrier_cpn=barrier_cpn, memory=memory_cpn,
    n_paths=n_paths, n_steps=n_steps_py,
    antithetic=antithetic, seed=seed,
)

# ── Cached simulation + pricing ───────────────────────────────────────────────
@st.cache_data(show_spinner="Running Monte Carlo simulation...")
def cached_run(p_hash):
    return run_pricing(AutocallParams(**p_hash))

# Use a dict hash of params for caching
import dataclasses
p_dict = dataclasses.asdict(p)
result, paths, dt = cached_run(tuple(sorted(p_dict.items())))

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🏗️ Structures",
    "📊 Pricing",
    "📈 Paths",
    "🔢 Greeks",
    "🎯 Convergence",
    "📉 Payoff Distribution",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Pricing
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 — Structures
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.header("Structured Product Catalogue")
    st.caption("Price and compare the two main autocall structures side by side.")

    # ── Parameters shared by both ─────────────────────────────────────────────
    st.subheader("Common market parameters")
    mc1, mc2, mc3, mc4 = st.columns(4)
    cS0    = mc1.number_input("Spot S0",         1.0, 500.0, float(S0),    1.0, key="cs0")
    cSigma = mc2.slider("Volatility σ (%)",       5.0, 80.0,  sigma*100,   0.5, key="csv") / 100
    cR     = mc3.slider("Risk-free r (%)",         0.0, 15.0,  r*100,      0.1, key="cr")  / 100
    cQ     = mc4.slider("Dividend q (%)",          0.0, 10.0,  q*100,      0.1, key="cq")  / 100

    st.divider()

    col_std, col_phx = st.columns(2)

    # ── Standard Autocall ─────────────────────────────────────────────────────
    with col_std:
        st.markdown("### 📘 Standard Autocall")
        st.markdown("""
| Feature | Value |
|---------|-------|
| Maturity | configurable |
| Observations | Annual |
| Recall barrier | 100 % S0 |
| Coupon | paid only at recall |
| PDI barrier | 60 % S0 |
| Capital at mat. | protected unless PDI |
        """)

        sT    = st.slider("Maturity (Y)",    1.0, 5.0, 3.0, 0.5, key="sT")
        sC    = st.slider("Annual coupon (%)", 1.0, 25.0, 8.0, 0.5, key="sC") / 100
        sBac  = st.slider("Recall barrier (%S0)", 80, 120, 100, key="sBac") / 100
        sBpdi = st.slider("PDI barrier (%S0)",    40, 90,   60, key="sBpdi") / 100
        sN    = st.select_slider("Paths", [10_000,25_000,50_000,100_000], 25_000, key="sN")

        p_std = AutocallParams(
            S0=cS0, r=cR, sigma=cSigma, q=cQ,
            notional=100.0, T=sT, obs_freq=1.0,
            coupon_pa=sC, barrier_ac=sBac, barrier_pdi=sBpdi,
            barrier_cpn=None, memory=False,
            n_paths=sN, n_steps=int(252*sT), antithetic=True, seed=42,
        )

        if st.button("Price Standard Autocall", type="primary", key="btn_std"):
            with st.spinner("Running MC..."):
                @st.cache_data
                def price_std(pk): return run_pricing(AutocallParams(**dict(pk)))
                r_std, _, _ = price_std(tuple(sorted(dataclasses.asdict(p_std).items())))

            st.metric("Fair Value",  f"{r_std.price:.4f}", f"{r_std.price-100:.2f} vs par")
            st.metric("Std Error",   f"{r_std.std_error:.4f}")
            c1s, c2s, c3s = st.columns(3)
            c1s.metric("P(Called)",   f"{r_std.prob_called*100:.1f}%")
            c2s.metric("P(PDI)",      f"{r_std.prob_pdi*100:.1f}%")
            c3s.metric("P(Mat. OK)",  f"{r_std.prob_mat_ok*100:.1f}%")
            if not np.isnan(r_std.avg_call_date):
                st.caption(f"Avg call date: **{r_std.avg_call_date:.2f}y**  |  "
                           f"95% CI: [{r_std.ci_lo:.3f}, {r_std.ci_hi:.3f}]")

            # Call prob bar chart
            fig_s = go.Figure(go.Bar(
                x=[f"T={t:.1f}y" for t in r_std.prob_by_obs_date],
                y=[v*100 for v in r_std.prob_by_obs_date.values()],
                marker_color="#4CAF50",
            ))
            fig_s.update_layout(title="P(called) by obs date", template="plotly_dark",
                                height=250, margin=dict(t=35,b=20))
            st.plotly_chart(fig_s, use_container_width=True)

    # ── Phoenix Autocall ──────────────────────────────────────────────────────
    with col_phx:
        st.markdown("### 🌸 Phoenix Autocall")
        st.markdown("""
| Feature | Value |
|---------|-------|
| Maturity | configurable |
| Observations | Quarterly |
| Recall barrier | 100 % S0 |
| Coupon barrier | pays coupon even if not recalled |
| PDI barrier | 60 % S0 |
| Memory coupon | optional |
        """)

        pT    = st.slider("Maturity (Y)",     1.0, 5.0, 3.0, 0.5, key="pT")
        pC    = st.slider("Annual coupon (%)", 1.0, 30.0, 10.0, 0.5, key="pC") / 100
        pBac  = st.slider("Recall barrier (%S0)",  80, 120, 100, key="pBac")  / 100
        pBpdi = st.slider("PDI barrier (%S0)",     40, 90,   60, key="pBpdi") / 100
        pBcpn = st.slider("Coupon barrier (%S0)",  50, 100,  70, key="pBcpn") / 100
        pMem  = st.checkbox("Memory coupon", value=True, key="pmem")
        pN    = st.select_slider("Paths", [10_000,25_000,50_000,100_000], 25_000, key="pN")

        p_phx = AutocallParams(
            S0=cS0, r=cR, sigma=cSigma, q=cQ,
            notional=100.0, T=pT, obs_freq=0.25,
            coupon_pa=pC, barrier_ac=pBac, barrier_pdi=pBpdi,
            barrier_cpn=pBcpn, memory=pMem,
            n_paths=pN, n_steps=int(252*pT), antithetic=True, seed=42,
        )

        if st.button("Price Phoenix Autocall", type="primary", key="btn_phx"):
            with st.spinner("Running MC..."):
                @st.cache_data
                def price_phx(pk): return run_pricing(AutocallParams(**dict(pk)))
                r_phx, _, _ = price_phx(tuple(sorted(dataclasses.asdict(p_phx).items())))

            st.metric("Fair Value",  f"{r_phx.price:.4f}", f"{r_phx.price-100:.2f} vs par")
            st.metric("Std Error",   f"{r_phx.std_error:.4f}")
            c1p, c2p, c3p = st.columns(3)
            c1p.metric("P(Called)",   f"{r_phx.prob_called*100:.1f}%")
            c2p.metric("P(PDI)",      f"{r_phx.prob_pdi*100:.1f}%")
            c3p.metric("P(Mat. OK)",  f"{r_phx.prob_mat_ok*100:.1f}%")
            if not np.isnan(r_phx.avg_call_date):
                st.caption(f"Avg call date: **{r_phx.avg_call_date:.2f}y**  |  "
                           f"95% CI: [{r_phx.ci_lo:.3f}, {r_phx.ci_hi:.3f}]")

            # Call prob bar chart
            fig_p = go.Figure(go.Bar(
                x=[f"T={t:.2f}y" for t in r_phx.prob_by_obs_date],
                y=[v*100 for v in r_phx.prob_by_obs_date.values()],
                marker_color="#E91E63",
            ))
            fig_p.update_layout(title="P(called) by obs date", template="plotly_dark",
                                height=250, margin=dict(t=35,b=20))
            st.plotly_chart(fig_p, use_container_width=True)

    st.divider()

    # ── Maturity payoff diagram ───────────────────────────────────────────────
    st.subheader("Payoff at Maturity (if never recalled)")
    S_range_pf = np.linspace(cS0 * 0.30, cS0 * 1.50, 400)
    B_pdi_lvl  = sBpdi * cS0

    payoff_ok  = np.full(len(S_range_pf), 100.0)
    payoff_pdi = 100.0 * S_range_pf / cS0

    fig_pf = go.Figure()
    fig_pf.add_trace(go.Scatter(
        x=S_range_pf, y=np.where(S_range_pf >= B_pdi_lvl, payoff_ok, payoff_pdi),
        name="Maturity payoff", line=dict(color="#4C9BE8", width=3),
    ))
    fig_pf.add_vline(x=B_pdi_lvl, line_dash="dash", line_color="tomato",
                     annotation_text=f"PDI = {sBpdi*100:.0f}%")
    fig_pf.add_vline(x=cS0, line_dash="dot", line_color="white",
                     annotation_text=f"S0 = {cS0:.0f}")
    fig_pf.add_hline(y=100, line_dash="dot", line_color="gray")
    fig_pf.update_layout(
        xaxis_title="S at maturity", yaxis_title="Payoff (% notional)",
        template="plotly_dark", height=320,
    )
    st.plotly_chart(fig_pf, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("Autocall Fair Value")

    # KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Fair Value",  f"{result.price:.4f}",
              delta=f"{result.price - 100:.2f} vs par")
    c2.metric("Std Error",   f"{result.std_error:.4f}")
    c3.metric("95% CI",      f"[{result.ci_lo:.3f}, {result.ci_hi:.3f}]")
    c4.metric("CI Width",    f"{result.ci_hi - result.ci_lo:.4f}")
    c5.metric("Paths",       f"{n_paths:,}")

    st.divider()

    # Scenario probabilities
    st.subheader("Scenario Breakdown")
    ca, cb, cc = st.columns(3)
    ca.metric("P(Called early)",     f"{result.prob_called*100:.2f}%",
              help="Autocall triggered on an observation date")
    cb.metric("P(PDI at maturity)",  f"{result.prob_pdi*100:.2f}%",
              help="Path hit PDI barrier AND product reached maturity")
    cc.metric("P(Mat. protected)",   f"{result.prob_mat_ok*100:.2f}%",
              help="Survived to maturity without PDI trigger")

    if not np.isnan(result.avg_call_date):
        st.caption(f"Average call date (conditional on being called): **{result.avg_call_date:.3f}y**")

    # Call probability by observation date
    obs_t   = list(result.prob_by_obs_date.keys())
    obs_p   = list(result.prob_by_obs_date.values())
    cum_p   = np.cumsum(obs_p)

    fig_bar = make_subplots(specs=[[{"secondary_y": True}]])
    fig_bar.add_trace(go.Bar(
        x=[f"T={t:.2f}y" for t in obs_t], y=[v * 100 for v in obs_p],
        name="P(call at obs)", marker_color="#4C9BE8",
    ), secondary_y=False)
    fig_bar.add_trace(go.Scatter(
        x=[f"T={t:.2f}y" for t in obs_t], y=cum_p * 100,
        name="Cumulative P(call)", mode="lines+markers",
        line=dict(color="orange", width=2), marker=dict(size=7),
    ), secondary_y=True)
    fig_bar.update_layout(
        title="Call Probability by Observation Date",
        yaxis_title="P(call at date) (%)",
        yaxis2_title="Cumulative P(call) (%)",
        template="plotly_dark", height=360,
        legend=dict(orientation="h", y=1.08),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Barrier diagram
    st.subheader("Product Structure")
    fig_struct = go.Figure()
    t_line = np.linspace(0, T, 300)
    fig_struct.add_hrect(y0=barrier_ac * S0, y1=barrier_ac * S0 * 1.5,
                          fillcolor="rgba(0,200,100,0.08)", line_width=0,
                          annotation_text="Recall zone", annotation_position="top right")
    fig_struct.add_hrect(y0=0, y1=barrier_pdi * S0,
                          fillcolor="rgba(255,80,80,0.08)", line_width=0,
                          annotation_text="PDI zone", annotation_position="bottom right")
    fig_struct.add_hline(y=barrier_ac * S0, line_dash="dash", line_color="limegreen",
                          annotation_text=f"Recall {barrier_ac*100:.0f}%", line_width=2)
    fig_struct.add_hline(y=barrier_pdi * S0, line_dash="dash", line_color="tomato",
                          annotation_text=f"PDI {barrier_pdi*100:.0f}%", line_width=2)
    fig_struct.add_hline(y=S0, line_color="white", line_width=1, line_dash="dot",
                          annotation_text=f"S0={S0}")
    if barrier_cpn is not None:
        fig_struct.add_hline(y=barrier_cpn * S0, line_dash="dot", line_color="gold",
                              annotation_text=f"Coupon barrier {barrier_cpn*100:.0f}%", line_width=1.5)
    for ot in p.obs_dates:
        fig_struct.add_vline(x=ot, line_color="gray", line_width=0.8, line_dash="dot")
    fig_struct.update_layout(
        xaxis_title="Time (years)", yaxis_title="Underlying level",
        template="plotly_dark", height=320,
    )
    st.plotly_chart(fig_struct, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Paths
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("Simulated GBM Paths")

    n_show = st.slider("Number of paths to display", 20, 500, 100)
    show_fans = st.checkbox("Show percentile fan", value=True)

    # Time axis
    n_total_steps = paths.shape[1] - 1
    times = np.linspace(0, T, n_total_steps + 1)

    fig_paths = go.Figure()

    # Percentile fan
    if show_fans:
        ps = path_stats(paths, dt)
        for lo, hi, alpha, name in [
            ("p5_path",  "p95_path", 0.10, "5-95%"),
            ("p25_path", "p75_path", 0.20, "25-75%"),
        ]:
            fig_paths.add_trace(go.Scatter(
                x=np.concatenate([times, times[::-1]]),
                y=np.concatenate([ps[hi], ps[lo][::-1]]),
                fill='toself', fillcolor=f'rgba(100,150,255,{alpha})',
                line=dict(width=0), name=name, showlegend=True,
            ))
        fig_paths.add_trace(go.Scatter(
            x=times, y=ps["mean_path"], name="Mean path",
            line=dict(color="white", width=2),
        ))

    # Individual paths — colour by outcome
    sample_idx = np.random.default_rng(0).choice(len(paths), min(n_show, len(paths)), replace=False)
    for i in sample_idx:
        if result.called_mask[i]:
            col, grp = "rgba(0,200,100,0.35)", "Called"
        elif result.pdi_mask[i]:
            col, grp = "rgba(255,80,80,0.35)", "PDI"
        else:
            col, grp = "rgba(100,160,255,0.35)", "Mat. OK"
        fig_paths.add_trace(go.Scatter(
            x=times, y=paths[i],
            line=dict(color=col, width=0.7),
            showlegend=False, hoverinfo="skip",
        ))

    # Barrier lines
    fig_paths.add_hline(y=barrier_ac * S0, line_dash="dash", line_color="limegreen",
                          line_width=2, annotation_text=f"Recall {barrier_ac*100:.0f}%")
    fig_paths.add_hline(y=barrier_pdi * S0, line_dash="dash", line_color="tomato",
                          line_width=2, annotation_text=f"PDI {barrier_pdi*100:.0f}%")
    if barrier_cpn is not None:
        fig_paths.add_hline(y=barrier_cpn * S0, line_dash="dot", line_color="gold",
                              line_width=1.5, annotation_text=f"Cpn {barrier_cpn*100:.0f}%")
    for ot in p.obs_dates:
        fig_paths.add_vline(x=ot, line_color="gray", line_width=0.8, line_dash="dot")

    # Legend proxies
    for col, name in [("limegreen","Called"),("tomato","PDI hit"),("rgba(100,160,255,0.8)","Mat. OK")]:
        fig_paths.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                        line=dict(color=col, width=2), name=name))

    fig_paths.update_layout(
        xaxis_title="Time (years)", yaxis_title="Underlying S(t)",
        title=f"{n_show} sample paths  |  green=called, red=PDI, blue=mat OK",
        template="plotly_dark", height=520,
        legend=dict(orientation="h", y=1.05),
    )
    st.plotly_chart(fig_paths, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Greeks
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("Greeks — Bump-and-Reprice")

    st.caption("Each Greek requires 2 extra full MC runs (central difference). "
               "Gamma requires 3. Total: 7 additional simulations.")

    compute_g = st.button("Compute Greeks", type="primary")

    if compute_g:
        with st.spinner("Running 7 MC simulations for bump-and-reprice..."):
            @st.cache_data
            def cached_greeks(p_key):
                return all_greeks(AutocallParams(**dict(p_key)))
            g = cached_greeks(tuple(sorted(p_dict.items())))

        g1, g2, g3, g4, g5 = st.columns(5)
        g1.metric("Delta",  f"{g['Delta']:.4f}",  help="dV/dS, bump = 1.0")
        g2.metric("Gamma",  f"{g['Gamma']:.6f}",  help="d2V/dS2, bump = 1.0")
        g3.metric("Vega",   f"{g['Vega']:.4f}",   help="dV/dsigma per 1 vol pt")
        g4.metric("Theta",  f"{g['Theta']:.4f}",  help="dV/dt per calendar day")
        g5.metric("Rho",    f"{g['Rho']:.6f}",    help="dV/dr per 1 bp")

        st.divider()
        st.subheader("Sensitivity: Price vs Spot")

        @st.cache_data
        def spot_profile(p_key, spots):
            base = AutocallParams(**dict(p_key))
            prices_, deltas_ = [], []
            for s in spots:
                r2, _, _ = run_pricing(replace(base, S0=float(s), n_paths=20_000))
                prices_.append(r2.price)
                d2 = (run_pricing(replace(base, S0=float(s)+1, n_paths=20_000))[0].price
                    - run_pricing(replace(base, S0=float(s)-1, n_paths=20_000))[0].price) / 2
                deltas_.append(d2)
            return np.array(prices_), np.array(deltas_)

        spots_range = np.arange(max(S0 * 0.5, 50), S0 * 1.5 + 1, S0 * 0.05)
        with st.spinner("Computing price profile..."):
            prices_s, deltas_s = spot_profile(tuple(sorted(p_dict.items())),
                                               tuple(spots_range.tolist()))

        fig_g = make_subplots(rows=1, cols=2,
                               subplot_titles=["Price vs Spot", "Delta vs Spot"])
        fig_g.add_trace(go.Scatter(x=spots_range, y=prices_s, name="Price",
                                    line=dict(color="#4C9BE8", width=2)), row=1, col=1)
        fig_g.add_trace(go.Scatter(x=spots_range, y=deltas_s, name="Delta",
                                    line=dict(color="#F4A261", width=2)), row=1, col=2)
        for fig_col in [1, 2]:
            fig_g.add_vline(x=S0, line_dash="dash", line_color="tomato", row=1, col=fig_col)
            fig_g.add_hline(y=barrier_ac*S0 if fig_col == 1 else 0,
                             line_dash="dot", line_color="limegreen", row=1, col=fig_col)
        fig_g.update_layout(template="plotly_dark", height=380,
                             showlegend=False, margin=dict(t=40))
        st.plotly_chart(fig_g, use_container_width=True)

    else:
        st.info("Click **Compute Greeks** to run bump-and-reprice (takes ~10-30s depending on path count).")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Convergence
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.header("MC Convergence Analysis")
    st.caption("Price estimate and 95% confidence interval vs number of paths. "
               "Antithetic variates halve the effective variance.")

    run_conv = st.button("Run Convergence Analysis", type="primary")
    if run_conv:
        counts = [500, 1_000, 2_000, 5_000, 10_000, 25_000, 50_000, 100_000]

        @st.cache_data
        def cached_conv(p_key, counts_key):
            return convergence_analysis(AutocallParams(**dict(p_key)), list(counts_key))

        with st.spinner("Running convergence analysis..."):
            conv = cached_conv(tuple(sorted(p_dict.items())), tuple(counts))

        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(
            x=conv["n_paths"], y=conv["prices"],
            mode="lines+markers", name="MC Price",
            line=dict(color="#4C9BE8", width=2), marker=dict(size=7),
        ))
        fig_conv.add_trace(go.Scatter(
            x=np.concatenate([conv["n_paths"], conv["n_paths"][::-1]]),
            y=np.concatenate([conv["ci_hi"], conv["ci_lo"][::-1]]),
            fill="toself", fillcolor="rgba(76,155,232,0.15)",
            line=dict(width=0), name="95% CI",
        ))
        fig_conv.add_hline(y=result.price, line_dash="dash", line_color="orange",
                            annotation_text=f"Full price={result.price:.4f}")
        fig_conv.update_layout(
            xaxis_title="Number of Paths", yaxis_title="Price",
            xaxis_type="log", template="plotly_dark", height=400,
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig_conv, use_container_width=True)

        # Std error decay
        fig_se = go.Figure()
        fig_se.add_trace(go.Scatter(
            x=conv["n_paths"], y=conv["std_errors"],
            mode="lines+markers", name="Std Error",
            line=dict(color="tomato", width=2), marker=dict(size=7),
        ))
        theory_se = conv["std_errors"][0] * np.sqrt(counts[0] / conv["n_paths"])
        fig_se.add_trace(go.Scatter(
            x=conv["n_paths"], y=theory_se,
            mode="lines", name="1/sqrt(N) theory",
            line=dict(color="gray", dash="dash"),
        ))
        fig_se.update_layout(
            xaxis_title="N paths", yaxis_title="Std Error",
            xaxis_type="log", yaxis_type="log",
            template="plotly_dark", height=320,
            legend=dict(orientation="h", y=1.05),
        )
        st.caption("Standard error decay (log-log) — should follow 1/sqrt(N)")
        st.plotly_chart(fig_se, use_container_width=True)

    else:
        st.info("Click **Run Convergence Analysis** to sweep path counts (takes ~30s).")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Payoff Distribution
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.header("Discounted Payoff Distribution")

    fig_hist = go.Figure()

    # Per-scenario histograms
    scenario_data = [
        (result.called_mask,   "Called early",  "#4CAF50"),
        (result.pdi_mask,      "PDI hit",       "#F44336"),
        (result.mat_ok_mask,   "Mat. protected","#2196F3"),
    ]
    for mask, name, color in scenario_data:
        if mask.any():
            fig_hist.add_trace(go.Histogram(
                x=result.payoffs[mask], name=name,
                marker_color=color, opacity=0.65,
                nbinsx=60, histnorm="probability density",
            ))

    fig_hist.add_vline(x=result.price, line_dash="dash", line_color="white",
                        line_width=2, annotation_text=f"FV={result.price:.4f}",
                        annotation_position="top right")
    fig_hist.add_vline(x=100.0, line_dash="dot", line_color="gold",
                        annotation_text="Par=100", annotation_position="top left")
    fig_hist.update_layout(
        barmode="overlay",
        xaxis_title="Discounted Payoff",
        yaxis_title="Density",
        title="Distribution of Discounted Payoffs by Scenario",
        template="plotly_dark", height=450,
        legend=dict(orientation="h", y=1.05),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Summary stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean payoff",   f"{result.payoffs.mean():.4f}")
    c2.metric("Std payoff",    f"{result.payoffs.std():.4f}")
    c3.metric("5th pctile",    f"{np.percentile(result.payoffs, 5):.4f}")
    c4.metric("95th pctile",   f"{np.percentile(result.payoffs, 95):.4f}")

    # CDF
    sorted_pf = np.sort(result.payoffs)
    cdf       = np.arange(1, len(sorted_pf) + 1) / len(sorted_pf)

    fig_cdf = go.Figure()
    fig_cdf.add_trace(go.Scatter(x=sorted_pf, y=cdf * 100, mode="lines",
                                  name="CDF", line=dict(color="#4C9BE8", width=2)))
    fig_cdf.add_vline(x=result.price, line_dash="dash", line_color="white",
                       annotation_text="Fair Value")
    fig_cdf.add_hline(y=50, line_dash="dot", line_color="gray")
    fig_cdf.update_layout(
        xaxis_title="Payoff", yaxis_title="CDF (%)",
        title="Cumulative Distribution of Payoffs",
        template="plotly_dark", height=320,
    )
    st.plotly_chart(fig_cdf, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Monte Carlo Autocall Pricer | GBM · Antithetic Variates · Bump-and-Reprice Greeks")
