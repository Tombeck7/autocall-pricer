
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
                 use_container_width=True)

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
        st.plotly_chart(fig_b, use_container_width=True)

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
        st.plotly_chart(fig_p, use_container_width=True)

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
        st.plotly_chart(fig_h, use_container_width=True)

    else:
        st.info("Configure assets and click **Price Worst-of**.")
