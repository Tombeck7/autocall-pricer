
# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — Worst-of Autocall
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.header("🎲 Worst-of Autocall Pricer")
    st.caption("Multi-underlying GBM with Cholesky correlation. Recall and PDI triggered on the worst performer.")

    from worst_of import WorstOfParams, price_worst_of
    import json as _json

    st.subheader("Underlyings")
    n_assets_wo = st.radio("Number of underlyings", [2, 3], horizontal=True, key="wo_n")

    wo_cols  = st.columns(n_assets_wo)
    wo_spots, wo_vols, wo_qs = [], [], []
    for i, col in enumerate(wo_cols):
        col.markdown(f"**Asset {i+1}**")
        wo_spots.append(col.number_input(f"Spot S{i+1}", 1.0, 10000.0, float(S0), 1.0, key=f"wo_s{i}"))
        wo_vols.append( col.slider(f"Vol σ{i+1} (%)", 5.0, 80.0, 20.0 + i*3, 0.5, key=f"wo_v{i}") / 100)
        wo_qs.append(   col.slider(f"Div q{i+1} (%)", 0.0, 10.0, 2.0,        0.1, key=f"wo_q{i}") / 100)

    st.subheader("Correlation Matrix")
    corr_inputs = st.columns(n_assets_wo * (n_assets_wo - 1) // 2)
    corr_mat_wo = np.eye(n_assets_wo).tolist()
    idx_c = 0
    for i in range(n_assets_wo):
        for j in range(i + 1, n_assets_wo):
            rho = corr_inputs[idx_c].slider(f"ρ({i+1},{j+1})", -0.99, 0.99, 0.60, 0.01, key=f"wo_rho{i}{j}")
            corr_mat_wo[i][j] = rho
            corr_mat_wo[j][i] = rho
            idx_c += 1

    import pandas as _pd2
    _labels = [f"A{i+1}" for i in range(n_assets_wo)]
    st.dataframe(
        _pd2.DataFrame(corr_mat_wo, index=_labels, columns=_labels).style.format("{:.2f}"),
        use_container_width=True,
    )

    st.subheader("Product Parameters")
    wc1, wc2, wc3, wc4, wc5 = st.columns(5)
    wo_T    = wc1.slider("Maturity (Y)",     1.0, 5.0,  3.0, 0.5, key="wo_T")
    wo_freq = wc2.selectbox("Observations", [0.25, 0.5, 1.0],
                             format_func=lambda x: {0.25:"Quarterly",0.5:"Semi-annual",1.0:"Annual"}[x],
                             key="wo_freq")
    wo_cpn  = wc3.slider("Coupon p.a. (%)",  1.0, 25.0, 8.0, 0.5, key="wo_cpn") / 100
    wo_bac  = wc4.slider("Recall barrier (%)", 80, 120, 100, key="wo_bac") / 100
    wo_bpdi = wc5.slider("PDI barrier (%)",    40,  90,  60, key="wo_bpdi") / 100

    wm1, wm2 = st.columns(2)
    wo_paths_n = wm1.select_slider("MC paths", [10_000, 25_000, 50_000, 100_000], 25_000, key="wo_paths")
    wo_seed_n  = wm2.number_input("Seed", 0, 9999, 42, 1, key="wo_seed")

    if st.button("Price Worst-of Autocall", type="primary", key="btn_wo"):
        try:
            np.linalg.cholesky(np.array(corr_mat_wo))
        except np.linalg.LinAlgError:
            st.error("Correlation matrix is not positive definite — reduce correlations.")
            st.stop()

        wp = WorstOfParams(
            spots=wo_spots, vols=wo_vols, corr=corr_mat_wo, q=wo_qs,
            r=r, notional=100.0, T=wo_T, obs_freq=wo_freq,
            coupon_pa=wo_cpn, barrier_ac=wo_bac, barrier_pdi=wo_bpdi,
            n_paths=wo_paths_n, n_steps_py=252, antithetic=True, seed=int(wo_seed_n),
        )

        with st.spinner(f"MC simulation: {wo_paths_n:,} paths × {n_assets_wo} assets..."):
            @st.cache_data
            def _run_wo(k):
                d = dict(k)
                return price_worst_of(WorstOfParams(
                    spots=list(eval(d["spots"])), vols=list(eval(d["vols"])),
                    corr=_json.loads(d["corr"]), q=list(eval(d["q"])),
                    r=float(d["r"]), notional=100.0,
                    T=float(d["T"]), obs_freq=float(d["obs_freq"]),
                    coupon_pa=float(d["coupon_pa"]),
                    barrier_ac=float(d["barrier_ac"]),
                    barrier_pdi=float(d["barrier_pdi"]),
                    n_paths=int(d["n_paths"]), n_steps_py=252,
                    antithetic=True, seed=int(d["seed"]),
                ))
            ck = tuple(sorted({
                "spots": str(wo_spots), "vols": str(wo_vols),
                "corr": _json.dumps(corr_mat_wo), "q": str(wo_qs),
                "r": r, "T": wo_T, "obs_freq": wo_freq,
                "coupon_pa": wo_cpn, "barrier_ac": wo_bac,
                "barrier_pdi": wo_bpdi, "n_paths": wo_paths_n, "seed": int(wo_seed_n),
            }.items()))
            res_wo = _run_wo(ck)

        # KPIs
        k1, k2, k3 = st.columns(3)
        k1.metric("Fair Value",  f"{res_wo['price']:.4f}", f"{res_wo['price']-100:.2f} vs par")
        k2.metric("Std Error",   f"{res_wo['std_error']:.4f}")
        k3.metric("95% CI",      f"[{res_wo['ci_lo']:.3f}, {res_wo['ci_hi']:.3f}]")
        p1, p2, p3 = st.columns(3)
        p1.metric("P(Called)",  f"{res_wo['prob_called']*100:.1f}%")
        p2.metric("P(PDI)",     f"{res_wo['prob_pdi']*100:.1f}%")
        p3.metric("P(Mat. OK)", f"{res_wo['prob_mat_ok']*100:.1f}%")
        if not np.isnan(res_wo["avg_call_date"]):
            st.caption(f"Avg call date: **{res_wo['avg_call_date']:.2f}y**")

        # Call prob bars
        fig_wo_b = go.Figure(go.Bar(
            x=[f"T={t:.2f}y" for t in res_wo["prob_by_obs"]],
            y=[v*100 for v in res_wo["prob_by_obs"].values()],
            marker_color="#6366f1",
        ))
        fig_wo_b.update_layout(
            title="P(called) by obs date", xaxis_title="Date", yaxis_title="Prob (%)",
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=260, font=dict(family="Inter, sans-serif", color="#e2e8f0"),
            margin=dict(t=40, b=30, l=40, r=10),
        )
        st.plotly_chart(fig_wo_b, use_container_width=True)

        # Worst-of path fan
        st.subheader("Worst Performer Paths")
        wop   = res_wo["worst_paths"]
        tw    = np.linspace(0, wo_T, wop.shape[1])
        n_s   = min(150, wo_paths_n)
        sidx  = np.random.default_rng(0).choice(wo_paths_n, n_s, replace=False)

        fig_wop = go.Figure()
        for lo_p, hi_p, alph in [(5, 95, 0.10), (25, 75, 0.20)]:
            fig_wop.add_trace(go.Scatter(
                x=np.concatenate([tw, tw[::-1]]),
                y=np.concatenate([np.percentile(wop, hi_p, axis=0)*100,
                                   np.percentile(wop, lo_p, axis=0)[::-1]*100]),
                fill="toself", fillcolor=f"rgba(99,102,241,{alph})",
                line=dict(width=0), showlegend=False,
            ))
        for i in sidx:
            cl = "#22c55e" if res_wo["called_mask"][i] else ("#ef4444" if res_wo["pdi_mask"][i] else "#38bdf8")
            fig_wop.add_trace(go.Scatter(x=tw, y=wop[i]*100,
                line=dict(color=cl, width=0.6), showlegend=False, hoverinfo="skip"))
        fig_wop.add_hline(y=wo_bac*100, line_dash="dash", line_color="#22c55e", line_width=2,
                           annotation_text=f"Recall {wo_bac*100:.0f}%")
        fig_wop.add_hline(y=wo_bpdi*100, line_dash="dash", line_color="#ef4444", line_width=2,
                           annotation_text=f"PDI {wo_bpdi*100:.0f}%")
        for ot in wp.obs_dates:
            fig_wop.add_vline(x=ot, line_color="rgba(255,255,255,0.12)", line_width=0.8, line_dash="dot")
        for cl_l, nm in [("#22c55e","Called"), ("#ef4444","PDI"), ("#38bdf8","Mat.OK")]:
            fig_wop.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                line=dict(color=cl_l, width=2), name=nm))
        fig_wop.update_layout(
            title=f"Worst of {n_assets_wo} assets (% of initial)",
            xaxis_title="Time (years)", yaxis_title="Worst performer (%)",
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=480, font=dict(family="Inter, sans-serif", color="#e2e8f0"),
            legend=dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=40, b=40, l=52, r=16),
        )
        fig_wop.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
        fig_wop.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
        st.plotly_chart(fig_wop, use_container_width=True)

        # Payoff distribution
        fig_woh = go.Figure()
        for msk, nm, cl in [
            (res_wo["called_mask"], "Called",  "#22c55e"),
            (res_wo["pdi_mask"],    "PDI",     "#ef4444"),
            (~res_wo["called_mask"] & ~res_wo["pdi_mask"], "Mat. OK", "#38bdf8"),
        ]:
            if msk.any():
                fig_woh.add_trace(go.Histogram(x=res_wo["payoffs"][msk], nbinsx=50,
                    name=nm, marker_color=cl, opacity=0.65, histnorm="probability density"))
        fig_woh.add_vline(x=res_wo["price"], line_dash="dash", line_color="white",
                           annotation_text=f"FV={res_wo['price']:.2f}")
        fig_woh.update_layout(
            barmode="overlay", title="Discounted Payoff Distribution",
            xaxis_title="Payoff", yaxis_title="Density",
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=300, font=dict(family="Inter, sans-serif", color="#e2e8f0"),
            legend=dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=40, b=30, l=52, r=16),
        )
        st.plotly_chart(fig_woh, use_container_width=True)

    else:
        st.info("Configure assets and click **Price Worst-of Autocall**.")
