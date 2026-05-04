"""
pages/4_Model_Risk.py
----------------------
Model Risk dashboard:
  · Barrier options: Local vol MC vs flat BS as barrier varies
  · Asian options: Heston MC vs flat BS across strikes
  · Vol premium surface for Asians
  · Model risk waterfall
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from options_lib import (EuropeanOption, MarketData, OptionType,
                          BlackScholes, Heston, HestonParams, MonteCarlo)
from options_lib.instruments.barrier import BarrierOption, BarrierType
from options_lib.instruments.asian import AsianOption, AverageType
from options_lib.market_data.vol_surface import SVIParams, VolSurface
from options_lib.market_data.local_vol import build_local_vol_surface
from options_lib.models.local_vol_mc import LocalVolBarrierPricer
from options_lib.models.heston_asian_mc import HestonAsianPricer

PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(17,19,24,1)',
    font=dict(family='DM Mono', color='#e8eaf0', size=11),
    xaxis=dict(gridcolor='#1e2530', linecolor='#1e2530', zerolinecolor='#1e2530'),
    yaxis=dict(gridcolor='#1e2530', linecolor='#1e2530', zerolinecolor='#1e2530'),
    margin=dict(l=40, r=20, t=50, b=40),
)

st.markdown("""
<h1 style="font-family:'Syne',sans-serif;font-size:32px;font-weight:800;
           letter-spacing:-0.02em;margin-bottom:4px;">Model Risk</h1>
<p style="font-family:'DM Mono',monospace;font-size:12px;color:#5a6278;margin-bottom:28px;">
    Same vanilla calibration · Different dynamics · Different exotic prices
</p>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Common parameters")
    S     = st.number_input("Spot", value=100.0, step=1.0)
    K     = st.number_input("Strike", value=100.0, step=1.0)
    T     = st.number_input("Expiry (yr)", value=1.0, step=0.25)
    r     = st.number_input("Rate", value=0.05, step=0.005, format="%.3f")
    sigma = st.number_input("ATM vol σ", value=0.20, step=0.01, min_value=0.01)

    st.markdown("---")
    st.markdown("### Heston params")
    v0    = st.number_input("v₀", value=sigma**2, format="%.4f", min_value=0.0001)
    kappa = st.number_input("κ", value=2.0, step=0.1)
    v_bar = st.number_input("v̄", value=sigma**2, format="%.4f", min_value=0.0001)
    xi    = st.number_input("ξ", value=0.30, step=0.05, min_value=0.01)
    rho_h = st.number_input("ρ (Heston)", value=-0.70, step=0.05, min_value=-0.99, max_value=0.99)

    st.markdown("---")
    st.markdown("### Surface skew")
    rho_svi = st.slider("SVI skew ρ", -0.95, -0.10, -0.55, step=0.05)
    n_paths = st.select_slider("MC paths", [5_000, 10_000, 20_000, 50_000], value=10_000)

    st.markdown("---")
    run_btn = st.button("⟳  Run Model Risk Analysis", type="primary")

mkt   = MarketData(spot=S, rate=r)
otype = OptionType.CALL

# Build a local vol surface with negative skew
@st.cache_data(ttl=600)
def build_lv(S, r, sigma, rho_svi):
    slices = {
        '6M': SVIParams(sigma**2*0.90, 0.12, rho_svi*0.90, 0.0, 0.10, 0.5),
        '1Y': SVIParams(sigma**2,      0.15, rho_svi,       0.0, 0.13, 1.0),
        '2Y': SVIParams(sigma**2*1.10, 0.18, min(rho_svi*1.05, -0.01), 0.0, 0.17, 2.0),
    }
    fwd = {d: S * np.exp(r * s.expiry) for d, s in slices.items()}
    surf = VolSurface(slices, fwd, S, r, 'DEMO')
    lv   = build_local_vol_surface(surf, n_S=40, n_T=20)
    atm_iv = float(surf.svi_slices['1Y'].implied_vol(np.array([0.0]))[0])
    return lv, surf, atm_iv

lv_surf, vol_surf, atm_iv = build_lv(S, r, sigma, rho_svi)
heston_params = HestonParams(v0=v0, kappa=kappa, v_bar=v_bar, xi=xi, rho=rho_h)

tabs = st.tabs(["⚡ Barrier Model Risk", "◎ Asian Vol Premium", "△ Price Comparison"])


# ── Tab 1: Barrier model risk ──────────────────────────────────────────────────
with tabs[0]:
    st.markdown("""
    <div style="background:#111318;border:1px solid #1e2530;border-radius:8px;padding:14px;
                margin-bottom:20px;font-family:'DM Mono',monospace;font-size:12px;color:#5a6278;">
    <span style="color:#00d4b8;font-weight:500;">Key insight:</span>
    Local vol and flat BS are both calibrated to the same vanilla surface.
    Yet they price down-and-out calls differently because as spot falls toward the barrier,
    local vol assigns higher vol (negative skew → lower spot = higher vol),
    increasing the knockout probability. Flat BS misses this entirely.
    </div>
    """, unsafe_allow_html=True)

    b_col1, b_col2 = st.columns([1.5, 1])
    with b_col1:
        barriers_pct = np.array([0.65, 0.70, 0.75, 0.80, 0.85, 0.88, 0.92])
        barriers     = barriers_pct * S

    with b_col2:
        barrier_type_sel = st.selectbox("Barrier type", ["Down-and-Out Call", "Down-and-In Call"])
        n_steps_b = st.select_slider("Time steps", [20, 52, 100, 252], value=52)

    btype = BarrierType.DOWN_AND_OUT if "Out" in barrier_type_sel else BarrierType.DOWN_AND_IN

    if run_btn:
        pricer = LocalVolBarrierPricer(
            lv_surface=lv_surf, bs_sigma=atm_iv,
            n_paths=n_paths, n_steps=n_steps_b,
            antithetic=True, seed=42,
        )

        with st.spinner(f"Running local vol MC on {len(barriers)} barriers..."):
            lv_prices, bs_prices, model_risks = [], [], []
            progress = st.progress(0)

            for idx, H in enumerate(barriers):
                inst_b = BarrierOption(K, T, otype, float(H), btype)
                res    = pricer.price(inst_b, mkt)
                lv_prices.append(res.lv_price)
                bs_prices.append(res.bs_price)
                model_risks.append(res.model_risk)
                progress.progress((idx + 1) / len(barriers))

            progress.empty()

        # Price comparison
        fig_b = go.Figure()
        fig_b.add_trace(go.Scatter(
            x=barriers_pct * 100, y=lv_prices,
            name="Local Vol MC",
            mode='lines+markers',
            line=dict(color='#00d4b8', width=2.5),
            marker=dict(size=8, color='#00d4b8'),
        ))
        fig_b.add_trace(go.Scatter(
            x=barriers_pct * 100, y=bs_prices,
            name="Flat BS MC",
            mode='lines+markers',
            line=dict(color='#5a6278', width=2.5, dash='dash'),
            marker=dict(size=8, color='#5a6278'),
        ))
        fig_b.add_vline(x=100, line_color='#f59e0b', line_dash='dot', line_width=1.5,
                        annotation_text="ATM", annotation_font_color='#f59e0b')
        fig_b.update_layout(
            **PLOTLY_LAYOUT,
            title=f"{barrier_type_sel} Price: Local Vol vs Flat BS",
            xaxis_title="Barrier (% of spot)", yaxis_title="Option Price",
            height=320,
            legend=dict(font=dict(size=11), bgcolor='rgba(0,0,0,0)'),
        )
        st.plotly_chart(fig_b, use_container_width=True)

        # Model risk curve
        fig_mr = go.Figure()
        fig_mr.add_trace(go.Scatter(
            x=barriers_pct * 100, y=model_risks,
            mode='lines+markers+text',
            line=dict(color='#ef4444', width=2.5),
            marker=dict(size=8, color='#ef4444'),
            fill='tozeroy',
            fillcolor='rgba(239,68,68,0.10)',
            text=[f"{v:.3f}" for v in model_risks],
            textposition='top center',
            textfont=dict(size=10, color='#e8eaf0'),
        ))
        fig_mr.update_layout(
            **PLOTLY_LAYOUT,
            title="Model Risk |LV − BS|  (same vanilla surface, different dynamics)",
            xaxis_title="Barrier (% of spot)", yaxis_title="|Model Risk| ($)",
            height=280,
        )
        st.plotly_chart(fig_mr, use_container_width=True)

        # Risk table
        import pandas as pd
        df = pd.DataFrame({
            "Barrier (% spot)": [f"{b*100:.0f}%" for b in barriers_pct],
            "Local Vol price": [f"{p:.4f}" for p in lv_prices],
            "Flat BS price":  [f"{p:.4f}" for p in bs_prices],
            "Model risk ($)": [f"{r:.4f}" for r in model_risks],
            "Model risk (%)": [f"{r/max(b,0.001)*100:.1f}%" for r, b in zip(model_risks, bs_prices)],
        })
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Configure parameters and click **Run Model Risk Analysis** in the sidebar.")


# ── Tab 2: Asian vol premium ───────────────────────────────────────────────────
with tabs[1]:
    st.markdown("""
    <div style="background:#111318;border:1px solid #1e2530;border-radius:8px;padding:14px;
                margin-bottom:20px;font-family:'DM Mono',monospace;font-size:12px;color:#5a6278;">
    <span style="color:#7c6af7;font-weight:500;">Key insight:</span>
    Under Heston, the arithmetic average has a different distribution than under flat BS.
    Stochastic vol introduces fat tails and negative skew into the average — the
    vol premium (Heston − BS) is positive and larger for longer expiries and OTM options.
    </div>
    """, unsafe_allow_html=True)

    n_obs = st.slider("Observation frequency", 4, 252, 52,
                       help="Number of averaging dates (52 = weekly)")

    strikes_asian = np.array([85., 90., 95., 100., 105., 110., 115.])

    if run_btn:
        pricer_asian = HestonAsianPricer(
            params=heston_params,
            n_paths=n_paths,
            n_steps=n_obs,
            antithetic=True,
            control_variate=True,
            seed=42,
        )

        heston_prices_a, bs_prices_a, premiums_a = [], [], []
        with st.spinner("Running Heston Asian MC..."):
            prog = st.progress(0)
            for idx, Ka in enumerate(strikes_asian):
                asian = AsianOption(Ka, T, otype, AverageType.ARITHMETIC, n_obs)
                res_a = pricer_asian.price(asian, mkt)
                heston_prices_a.append(res_a.price)
                bs_prices_a.append(res_a.bs_price)
                premiums_a.append(res_a.vol_premium)
                prog.progress((idx + 1) / len(strikes_asian))
            prog.empty()

        fig_asian = go.Figure()
        fig_asian.add_trace(go.Scatter(
            x=strikes_asian, y=heston_prices_a,
            name="Heston MC",
            mode='lines+markers',
            line=dict(color='#7c6af7', width=2.5),
            marker=dict(size=8),
        ))
        fig_asian.add_trace(go.Scatter(
            x=strikes_asian, y=bs_prices_a,
            name="Flat BS MC",
            mode='lines+markers',
            line=dict(color='#5a6278', width=2.5, dash='dash'),
            marker=dict(size=8),
        ))
        fig_asian.add_vline(x=S, line_color='#f59e0b', line_dash='dot', line_width=1.5,
                            annotation_text=f"S={S:.0f}", annotation_font_color='#f59e0b')
        fig_asian.update_layout(
            **PLOTLY_LAYOUT,
            title=f"Arithmetic Asian Call Price: Heston vs BS  (n={n_obs} obs)",
            xaxis_title="Strike (K)", yaxis_title="Price",
            height=320,
            legend=dict(font=dict(size=11), bgcolor='rgba(0,0,0,0)'),
        )
        st.plotly_chart(fig_asian, use_container_width=True)

        # Vol premium by strike
        prem_pct = [p / max(b, 0.001) * 100 for p, b in zip(premiums_a, bs_prices_a)]
        colors_p = ['#00d4b8' if p >= 0 else '#ef4444' for p in premiums_a]

        pr1, pr2 = st.columns(2)
        with pr1:
            fig_prem = go.Figure(go.Bar(
                x=strikes_asian, y=premiums_a,
                marker_color=colors_p,
                text=[f"{v:+.4f}" for v in premiums_a],
                textposition='outside', textfont=dict(size=10, color='#e8eaf0'),
            ))
            fig_prem.update_layout(**PLOTLY_LAYOUT, title="Stochastic Vol Premium ($ absolute)",
                                    xaxis_title="Strike", height=260)
            st.plotly_chart(fig_prem, use_container_width=True)

        with pr2:
            fig_pct = go.Figure(go.Bar(
                x=strikes_asian, y=prem_pct,
                marker_color=colors_p,
                text=[f"{v:+.1f}%" for v in prem_pct],
                textposition='outside', textfont=dict(size=10, color='#e8eaf0'),
            ))
            fig_pct.update_layout(**PLOTLY_LAYOUT, title="Stochastic Vol Premium (% relative)",
                                    xaxis_title="Strike", yaxis_title="%", height=260)
            st.plotly_chart(fig_pct, use_container_width=True)
    else:
        st.info("Click **Run Model Risk Analysis** to compute.")


# ── Tab 3: Side-by-side price comparison ──────────────────────────────────────
with tabs[2]:
    st.markdown("### European option: BS vs Heston  (same σ, different dynamics)")

    spots = np.linspace(S * 0.7, S * 1.3, 80)
    bs_model = BlackScholes(sigma=sigma)
    h_model  = Heston(heston_params)

    bs_prices_e = [bs_model.price(EuropeanOption(K, T, otype), MarketData(s, r)) for s in spots]
    h_prices_e  = [h_model.price(EuropeanOption(K, T, otype), MarketData(s, r)) for s in spots]
    diffs       = [h - b for h, b in zip(h_prices_e, bs_prices_e)]

    fig_comp = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              row_heights=[0.65, 0.35],
                              vertical_spacing=0.06)

    fig_comp.add_trace(go.Scatter(x=spots, y=bs_prices_e, name="BS",
                                   line=dict(color='#5a6278', width=2)), row=1, col=1)
    fig_comp.add_trace(go.Scatter(x=spots, y=h_prices_e, name="Heston",
                                   line=dict(color='#7c6af7', width=2.5)), row=1, col=1)
    fig_comp.add_vline(x=S, line_color='#00d4b8', line_dash='dash', line_width=1.5,
                        row=1, col=1)
    fig_comp.add_vline(x=K, line_color='#f59e0b', line_dash='dash', line_width=1.5,
                        row=1, col=1)

    fig_comp.add_trace(go.Scatter(x=spots, y=diffs, name="Heston−BS",
                                   line=dict(color='#00d4b8', width=2),
                                   fill='tozeroy',
                                   fillcolor='rgba(0,212,184,0.10)'), row=2, col=1)
    fig_comp.add_hline(y=0, line_color='#5a6278', line_dash='dot', row=2, col=1)

    fig_comp.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(17,19,24,1)',
        font=dict(family='DM Mono', color='#e8eaf0', size=11),
        height=440, margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(font=dict(size=11), bgcolor='rgba(0,0,0,0)'),
    )
    for r_idx in [1, 2]:
        fig_comp.update_xaxes(gridcolor='#1e2530', linecolor='#1e2530', row=r_idx, col=1)
        fig_comp.update_yaxes(gridcolor='#1e2530', linecolor='#1e2530', row=r_idx, col=1)

    fig_comp.update_yaxes(title_text="Price", row=1, col=1)
    fig_comp.update_yaxes(title_text="Difference", row=2, col=1)
    fig_comp.update_xaxes(title_text="Spot", row=2, col=1)
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("""
    <div style="font-family:'DM Mono',monospace;font-size:11px;color:#5a6278;
                background:#111318;border:1px solid #1e2530;border-radius:6px;padding:12px;">
    The difference Heston − BS is the <span style="color:#00d4b8">smile effect</span>:
    negative skew (ρ &lt; 0) makes ITM calls cheaper and OTM calls more expensive
    than flat-vol BS. Both models agree at-the-money by construction when v₀ = v̄ = σ².
    </div>
    """, unsafe_allow_html=True)
