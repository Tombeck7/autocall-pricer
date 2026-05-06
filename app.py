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
from live_data import fetch, risk_free_rate
from pdf_export import generate_autocall_pdf

# ── Design tokens ─────────────────────────────────────────────────────────────
C = dict(
    primary="#6366f1", success="#22c55e", danger="#ef4444",
    warning="#f59e0b", info="#38bdf8",   purple="#a855f7",
    text="#e2e8f0",    muted="#94a3b8",
    grid="rgba(255,255,255,0.06)", line="rgba(255,255,255,0.12)",
    bg="rgba(0,0,0,0)",
)

def sf(fig, title="", height=420):
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=C["text"]), x=0.01),
        template="plotly_dark", paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
        font=dict(family="Inter, sans-serif", size=11, color=C["text"]),
        margin=dict(t=45 if title else 20, b=36, l=52, r=16),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0,
                    orientation="h", y=1.06, font=dict(size=10)),
        hoverlabel=dict(bgcolor="#1e293b", bordercolor=C["line"],
                        font=dict(size=11, color=C["text"])),
    )
    fig.update_xaxes(gridcolor=C["grid"], linecolor=C["line"], zeroline=False)
    fig.update_yaxes(gridcolor=C["grid"], linecolor=C["line"], zeroline=False)
    return fig

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Autocall MC Pricer",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

div[data-testid="stMetricValue"] { font-size: 1.35rem; font-weight: 700; }
div[data-testid="stMetricLabel"] { font-size: 0.75rem; color: #94a3b8; font-weight: 500;
    letter-spacing:.04em; text-transform:uppercase; }
div[data-testid="stMetricDelta"] { font-size: 0.8rem; }
.live-banner {
    background: linear-gradient(90deg,#0f2027,#203a43,#2c5364);
    border: 1px solid #38bdf8; border-radius:10px;
    padding:12px 18px; margin-bottom:16px;
}
.ticker-pill { background:#6366f1; color:white; padding:3px 10px;
    border-radius:20px; font-weight:700; font-size:.85rem; }
.price-big { font-size:1.6rem; font-weight:700; color:#e2e8f0; }
.chg-pos { color:#22c55e; font-weight:600; }
.chg-neg { color:#ef4444; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in dict(S0=100.0, r=5.0, sigma=20.0, q=2.0, live=None).items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("📐 Autocall MC Pricer")
st.sidebar.divider()

st.sidebar.subheader("🔴 Live Market Data")
col_t, col_b = st.sidebar.columns([3, 1])
ticker_input = col_t.text_input("Ticker", value="SPY", label_visibility="collapsed",
                                 placeholder="SPY, AAPL...")
if col_b.button("Load", width='stretch', key="load_ac"):
    with st.spinner(f"Fetching {ticker_input.upper()}..."):
        data = fetch(ticker_input.strip().upper())
    if data:
        st.session_state.live  = data
        st.session_state.S0    = round(data["price"], 2)
        st.session_state.sigma = round(data["sigma_3m"] * 100, 1)
        try:
            st.session_state.r = round(risk_free_rate() * 100, 2)
        except Exception:
            pass
        st.sidebar.success(f"{data['ticker']}  {data['price']:.2f}  ({data['change_pct']:+.2f}%)")
    else:
        st.sidebar.error("Ticker not found.")

live = st.session_state.live
if live:
    chg_class = "chg-pos" if live["change"] >= 0 else "chg-neg"
    chg_sign  = "+" if live["change"] >= 0 else ""
    beta_str = f" &nbsp; Beta <b style='color:#e2e8f0'>{live['beta']:.2f}</b>" if live.get('beta') else ""
    hv_str   = f"HV 1M <b style='color:#e2e8f0'>{live['sigma_1m']*100:.1f}%</b> &nbsp; HV 3M <b style='color:#e2e8f0'>{live['sigma_3m']*100:.1f}%</b>{beta_str}"
    st.markdown(
        f"<div class='live-banner'><span class='ticker-pill'>{live['ticker']}</span>&nbsp;"
        f"<span class='price-big'>{live['price']:.2f} {live['currency']}</span>&nbsp;"
        f"<span class='{chg_class}'>{chg_sign}{live['change_pct']:.2f}%</span>&nbsp;&nbsp;"
        f"<span style='color:#94a3b8;font-size:.85rem'>{hv_str}</span></div>",
        unsafe_allow_html=True,
    )

st.sidebar.divider()
st.sidebar.subheader("⚙️ Market")
S0    = st.sidebar.number_input("Spot S0",         1.0, 10000.0, float(st.session_state.S0), 1.0)
r     = st.sidebar.slider("Risk-free r (%)",       0.0, 15.0, float(st.session_state.r),    0.1) / 100
sigma = st.sidebar.slider("Volatility σ (%)",      5.0, 80.0,  float(st.session_state.sigma), 0.5) / 100
q     = st.sidebar.slider("Dividend q (%)",        0.0, 10.0,  2.0, 0.1) / 100

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
n_paths    = st.sidebar.select_slider("Paths", [5_000,10_000,25_000,50_000,100_000], 10_000)
n_steps_py = st.sidebar.select_slider("Steps/year", [52, 126, 252], 52)
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
    return run_pricing(AutocallParams(**dict(p_hash)))

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
    "🎲 Worst-of Autocall",
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
            fig_s.update_layout(title="P(called) by obs date", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color="#e2e8f0"),
                                height=250, margin=dict(t=35,b=20))
            st.plotly_chart(fig_s, width='stretch')

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
            fig_p.update_layout(title="P(called) by obs date", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color="#e2e8f0"),
                                height=250, margin=dict(t=35,b=20))
            st.plotly_chart(fig_p, width='stretch')

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
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color="#e2e8f0"), height=320,
    )
    st.plotly_chart(fig_pf, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    if result is None:
        st.info("Click **Run Pricing** in the sidebar to run the Monte Carlo simulation.")
    st.header("Autocall Fair Value")

    # -- Live market data panel ------------------------------------------
    if live:
        st.divider()
        st.subheader(f"📡 {live['ticker']} — Live Market Data")
        r1 = st.columns(6)
        r1[0].metric('Price',    f"{live['price']:.2f} {live['currency']}", f"{live['change_pct']:+.2f}%")
        r1[1].metric('Day Hi/Lo', f"{live.get('high_day', live['price']):.2f} / {live.get('low_day', live['price']):.2f}")
        r1[2].metric('52W Hi/Lo', f"{live.get('high_52w',0):.2f} / {live.get('low_52w',0):.2f}")
        r1[3].metric('HV 1M',    f"{live['sigma_1m']*100:.1f}%", help='Historical vol 1 month')
        r1[4].metric('HV 3M',    f"{live['sigma_3m']*100:.1f}%", help='Historical vol 3 months')
        r1[5].metric('HV 1Y',    f"{live['sigma_1y']*100:.1f}%", help='Historical vol 1 year')

        r2 = st.columns(6)
        if live.get('iv_atm_near'):
            r2[0].metric('IV ATM (near)', f"{live['iv_atm_near']*100:.1f}%",
                         help=f"Expiry: {live['iv_chain'].get('exp_near','')}")
        if live.get('iv_atm_far'):
            r2[1].metric('IV ATM (far)',  f"{live['iv_atm_far']*100:.1f}%",
                         help=f"Expiry: {live['iv_chain'].get('exp_far','')}")
        if live.get('beta'):    r2[2].metric('Beta',       f"{live['beta']:.2f}")
        if live.get('pe_ratio'): r2[3].metric('P/E',       f"{live['pe_ratio']:.1f}x")
        if live.get('div_yield'): r2[4].metric('Div yield', f"{live['div_yield']*100:.2f}%")
        if live.get('market_cap_str'): r2[5].metric('Mkt cap', live['market_cap_str'])

        info_parts = []
        if live.get('name') != live['ticker']: info_parts.append(f"**{live['name']}**")
        if live.get('sector'):   info_parts.append(live['sector'])
        if live.get('industry'): info_parts.append(live['industry'])
        if live.get('target_price'):   info_parts.append(f"Target: {live['target_price']:.2f}")
        if live.get('analyst_rating'): info_parts.append(f"Analyst: {live['analyst_rating'].upper()}")
        if info_parts: st.caption('  ·  '.join(info_parts))

        # Real IV smile
        smile_near = live['iv_chain'].get('smile_near')
        if smile_near is not None and len(smile_near) > 3:
            st.subheader('📉 Real IV Smile (option chain)')
            fig_iv_ac = go.Figure()
            fig_iv_ac.add_trace(go.Scatter(
                x=smile_near['strike'], y=smile_near['mid_iv']*100,
                mode='lines+markers', name=f"IV {live['iv_chain'].get('exp_near','near')}",
                line=dict(color='#6366f1', width=2), marker=dict(size=5),
            ))
            smile_far = live['iv_chain'].get('smile_far')
            if smile_far is not None and len(smile_far) > 3:
                fig_iv_ac.add_trace(go.Scatter(
                    x=smile_far['strike'], y=smile_far['mid_iv']*100,
                    mode='lines+markers', name=f"IV {live['iv_chain'].get('exp_far','far')}",
                    line=dict(color='#f59e0b', width=2, dash='dot'), marker=dict(size=5),
                ))
            fig_iv_ac.add_vline(x=live['price'], line_dash='dash', line_color='#ef4444',
                                annotation_text='Spot')
            fig_iv_ac.update_layout(
                template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)', height=300,
                font=dict(family='Inter, sans-serif', color='#e2e8f0'),
                xaxis_title='Strike', yaxis_title='IV (%)',
                margin=dict(t=20,b=40,l=52,r=16),
                legend=dict(orientation='h', y=1.05, bgcolor='rgba(0,0,0,0)'),
            )
            fig_iv_ac.update_xaxes(gridcolor='rgba(255,255,255,0.06)')
            fig_iv_ac.update_yaxes(gridcolor='rgba(255,255,255,0.06)')
            st.plotly_chart(fig_iv_ac, width='stretch')


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
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color="#e2e8f0"), height=360,
        legend=dict(orientation="h", y=1.08),
    )
    st.plotly_chart(fig_bar, width='stretch')

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
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color="#e2e8f0"), height=320,
    )
    st.plotly_chart(fig_struct, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Paths
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    if result is None:
        st.info("Click **Run Pricing** in the sidebar first.")
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
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color="#e2e8f0"), height=520,
        legend=dict(orientation="h", y=1.05),
    )
    st.plotly_chart(fig_paths, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Greeks
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    if result is None:
        st.info("Click **Run Pricing** in the sidebar first.")
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
        fig_g.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color="#e2e8f0"), height=380,
                             showlegend=False, margin=dict(t=40))
        st.plotly_chart(fig_g, width='stretch')

    else:
        st.info("Click **Compute Greeks** to run bump-and-reprice (takes ~10-30s depending on path count).")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Convergence
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    if result is None:
        st.info("Click **Run Pricing** in the sidebar first.")
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
            xaxis_type="log", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color="#e2e8f0"), height=400,
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig_conv, width='stretch')

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
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color="#e2e8f0"), height=320,
            legend=dict(orientation="h", y=1.05),
        )
        st.caption("Standard error decay (log-log) — should follow 1/sqrt(N)")
        st.plotly_chart(fig_se, width='stretch')

    else:
        st.info("Click **Run Convergence Analysis** to sweep path counts (takes ~30s).")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Payoff Distribution
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    if result is None:
        st.info("Click **Run Pricing** in the sidebar first.")
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
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color="#e2e8f0"), height=450,
        legend=dict(orientation="h", y=1.05),
    )
    st.plotly_chart(fig_hist, width='stretch')

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
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color="#e2e8f0"), height=320,
    )
    st.plotly_chart(fig_cdf, width='stretch')

# ── Footer ────────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — Worst-of Autocall
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — Worst-of Autocall
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    import pandas as _pd_wo
    import json as _js_wo
    from worst_of import WorstOfParams, price_worst_of

    st.header("Worst-of Autocall Pricer")
    st.caption("Multi-underlying GBM with Cholesky correlation. Recall and PDI on the worst performer.")

    st.subheader("Underlyings")
    n_wo = st.radio("Number of assets", [2, 3], horizontal=True, key="wo_n")

    wo_spots, wo_vols, wo_qs = [], [], []
    for i, col in enumerate(st.columns(n_wo)):
        col.markdown(f"**Asset {i+1}**")
        wo_spots.append(col.number_input(f"Spot {i+1}", 1.0, 10000.0, float(S0), 1.0, key=f"wos{i}"))
        wo_vols.append( col.slider(f"Vol {i+1} (%)", 5.0, 80.0, 20.0+i*3, 0.5, key=f"wov{i}") / 100)
        wo_qs.append(   col.slider(f"Div {i+1} (%)", 0.0, 10.0, 2.0,      0.1, key=f"woq{i}") / 100)

    st.subheader("Correlation Matrix")
    corr_wo = np.eye(n_wo).tolist()
    n_pairs = n_wo * (n_wo - 1) // 2
    corr_cols = st.columns(n_pairs)
    ci = 0
    for a in range(n_wo):
        for b in range(a + 1, n_wo):
            rho = corr_cols[ci].slider(f"rho({a+1},{b+1})", -0.99, 0.99, 0.60, 0.01, key=f"wo_r{a}{b}")
            corr_wo[a][b] = rho
            corr_wo[b][a] = rho
            ci += 1

    lbl = [f"A{i+1}" for i in range(n_wo)]
    st.dataframe(_pd_wo.DataFrame(corr_wo, index=lbl, columns=lbl).style.format("{:.2f}"),
                 width='stretch')

    st.subheader("Product")
    c1, c2, c3, c4, c5 = st.columns(5)
    wo_T    = c1.slider("Maturity (Y)",  1.0, 5.0, 3.0, 0.5, key="wo_T")
    wo_freq = c2.selectbox("Obs.", [0.25, 0.5, 1.0],
                            format_func=lambda x: {0.25:"Quarterly",0.5:"Semi-annual",1.0:"Annual"}[x],
                            key="wo_freq")
    wo_cpn  = c3.slider("Coupon (%)", 1.0, 25.0, 8.0, 0.5, key="wo_cpn") / 100
    wo_bac  = c4.slider("Recall (%)", 80, 120, 100, key="wo_bac") / 100
    wo_bpdi = c5.slider("PDI (%)",    40,  90,  60, key="wo_bpdi") / 100

    wm1, wm2 = st.columns(2)
    wo_npaths = wm1.select_slider("Paths", [10_000, 25_000, 50_000], 25_000, key="wo_paths")
    wo_seed   = int(wm2.number_input("Seed", 0, 9999, 42, 1, key="wo_seed"))

    if st.button("Price Worst-of", type="primary", key="btn_wo"):
        try:
            np.linalg.cholesky(np.array(corr_wo))
        except np.linalg.LinAlgError:
            st.error("Correlation matrix not positive definite — reduce correlations.")
            st.stop()

        @st.cache_data
        def _price_wo(spots, vols, corr_str, qs, r_, T_, freq_, cpn_, bac_, bpdi_, np_, seed_):
            return price_worst_of(WorstOfParams(
                spots=list(spots), vols=list(vols),
                corr=_js_wo.loads(corr_str), q=list(qs),
                r=r_, notional=100.0, T=T_, obs_freq=freq_,
                coupon_pa=cpn_, barrier_ac=bac_, barrier_pdi=bpdi_,
                n_paths=np_, n_steps_py=252, antithetic=True, seed=seed_,
            ))

        with st.spinner(f"Running {wo_npaths:,} paths x {n_wo} assets..."):
            res = _price_wo(
                tuple(wo_spots), tuple(wo_vols), _js_wo.dumps(corr_wo), tuple(wo_qs),
                r, wo_T, wo_freq, wo_cpn, wo_bac, wo_bpdi, wo_npaths, wo_seed,
            )

        # KPIs
        k1, k2, k3 = st.columns(3)
        k1.metric("Fair Value", f"{res['price']:.4f}", f"{res['price']-100:.2f} vs par")
        k2.metric("Std Error",  f"{res['std_error']:.4f}")
        k3.metric("95% CI",     f"[{res['ci_lo']:.3f}, {res['ci_hi']:.3f}]")
        p1, p2, p3 = st.columns(3)
        p1.metric("P(Called)",  f"{res['prob_called']*100:.1f}%")
        p2.metric("P(PDI)",     f"{res['prob_pdi']*100:.1f}%")
        p3.metric("P(Mat. OK)", f"{res['prob_mat_ok']*100:.1f}%")
        if not np.isnan(res["avg_call_date"]):
            st.caption(f"Avg call date: **{res['avg_call_date']:.2f}y**")

        # Call prob bar chart
        fig_b = go.Figure(go.Bar(
            x=[f"T={t:.2f}y" for t in res["prob_by_obs"]],
            y=[v*100 for v in res["prob_by_obs"].values()],
            marker_color="#6366f1",
        ))
        fig_b.update_layout(title="P(called) by obs date", xaxis_title="Date",
                             yaxis_title="Prob (%)", template="plotly_dark",
                             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                             height=260, font=dict(family="Inter, sans-serif", color="#e2e8f0"),
                             margin=dict(t=40,b=30,l=40,r=10))
        st.plotly_chart(fig_b, width='stretch')

        # Path fan
        st.subheader("Worst Performer Paths")
        wop  = res["worst_paths"]
        tw   = np.linspace(0, wo_T, wop.shape[1])
        n_sh = min(150, wo_npaths)
        sidx = np.random.default_rng(0).choice(wo_npaths, n_sh, replace=False)

        fig_p = go.Figure()
        for lo_p, hi_p, al in [(5,95,0.10),(25,75,0.20)]:
            fig_p.add_trace(go.Scatter(
                x=np.concatenate([tw, tw[::-1]]),
                y=np.concatenate([np.percentile(wop,hi_p,axis=0)*100,
                                   np.percentile(wop,lo_p,axis=0)[::-1]*100]),
                fill="toself", fillcolor=f"rgba(99,102,241,{al})",
                line=dict(width=0), showlegend=False))
        for i in sidx:
            cl = "#22c55e" if res["called_mask"][i] else ("#ef4444" if res["pdi_mask"][i] else "#38bdf8")
            fig_p.add_trace(go.Scatter(x=tw, y=wop[i]*100,
                line=dict(color=cl, width=0.6), showlegend=False, hoverinfo="skip"))
        fig_p.add_hline(y=wo_bac*100, line_dash="dash", line_color="#22c55e", line_width=2,
                        annotation_text=f"Recall {wo_bac*100:.0f}%")
        fig_p.add_hline(y=wo_bpdi*100, line_dash="dash", line_color="#ef4444", line_width=2,
                        annotation_text=f"PDI {wo_bpdi*100:.0f}%")
        for ot in np.arange(wo_freq, wo_T+0.01, wo_freq):
            fig_p.add_vline(x=ot, line_color="rgba(255,255,255,0.12)",
                             line_width=0.8, line_dash="dot")
        for cl_l, nm in [("#22c55e","Called"),("#ef4444","PDI"),("#38bdf8","Mat.OK")]:
            fig_p.add_trace(go.Scatter(x=[None],y=[None],mode="lines",
                line=dict(color=cl_l,width=2),name=nm))
        fig_p.update_layout(title=f"Worst of {n_wo} assets (%)",
                             xaxis_title="Time (years)", yaxis_title="Worst perf (%)",
                             template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                             plot_bgcolor="rgba(0,0,0,0)", height=460,
                             font=dict(family="Inter, sans-serif", color="#e2e8f0"),
                             legend=dict(orientation="h",y=1.05,bgcolor="rgba(0,0,0,0)"),
                             margin=dict(t=40,b=40,l=52,r=16))
        fig_p.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
        fig_p.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
        st.plotly_chart(fig_p, width='stretch')

        # Payoff histogram
        fig_h = go.Figure()
        for msk, nm, cl in [
            (res["called_mask"],                             "Called",  "#22c55e"),
            (res["pdi_mask"],                               "PDI",     "#ef4444"),
            (~res["called_mask"] & ~res["pdi_mask"],        "Mat. OK", "#38bdf8"),
        ]:
            if msk.any():
                fig_h.add_trace(go.Histogram(x=res["payoffs"][msk], nbinsx=50,
                    name=nm, marker_color=cl, opacity=0.65, histnorm="probability density"))
        fig_h.add_vline(x=res["price"], line_dash="dash", line_color="white",
                        annotation_text=f"FV={res['price']:.2f}")
        fig_h.update_layout(barmode="overlay", title="Payoff Distribution",
                             xaxis_title="Discounted Payoff", yaxis_title="Density",
                             template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                             plot_bgcolor="rgba(0,0,0,0)", height=300,
                             font=dict(family="Inter, sans-serif", color="#e2e8f0"),
                             legend=dict(orientation="h",y=1.05,bgcolor="rgba(0,0,0,0)"),
                             margin=dict(t=40,b=30,l=52,r=16))
        st.plotly_chart(fig_h, width='stretch')

    else:
        st.info("Configure assets and click **Price Worst-of**.")


st.divider()
st.caption("Monte Carlo Autocall Pricer | GBM · Antithetic Variates · Bump-and-Reprice Greeks")
