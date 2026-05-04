"""
pages/5_PnL_Attribution.py
---------------------------
P&L Attribution dashboard:
  · Taylor decomposition: Delta, Gamma, Vega, Theta, Vanna, Volga
  · Interactive scenario: input dS and dσ, see attributed P&L
  · Historical backtest over a simulated GBM path
  · Cumulative P&L with Greek decomposition
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from options_lib import BlackScholes, EuropeanOption, MarketData, OptionType
from options_lib.risk import PnLAttributor, summarise_backtest

PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(17,19,24,1)',
    font=dict(family='DM Mono', color='#e8eaf0', size=11),
    xaxis=dict(gridcolor='#1e2530', linecolor='#1e2530', zerolinecolor='#1e2530'),
    yaxis=dict(gridcolor='#1e2530', linecolor='#1e2530', zerolinecolor='#1e2530'),
    margin=dict(l=40, r=20, t=50, b=40),
)

GREEK_COLORS = {
    'Delta'   : '#00d4b8',
    'Gamma'   : '#7c6af7',
    'Vega'    : '#f59e0b',
    'Theta'   : '#ef4444',
    'Vanna'   : '#10b981',
    'Volga'   : '#3b82f6',
    'Unexplained': '#5a6278',
}

st.markdown("""
<h1 style="font-family:'Syne',sans-serif;font-size:32px;font-weight:800;
           letter-spacing:-0.02em;margin-bottom:4px;">P&amp;L Attribution</h1>
<p style="font-family:'DM Mono',monospace;font-size:12px;color:#5a6278;margin-bottom:28px;">
    dV ≈ Δ·dS + ½Γ·dS² + V·dσ + Θ·dt + Vanna·dS·dσ + ½Volga·dσ²
</p>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Option parameters")
    S        = st.number_input("Spot S₀", value=100.0, step=1.0)
    K        = st.number_input("Strike K", value=100.0, step=1.0)
    T        = st.number_input("Expiry (yr)", value=1.0, step=0.25, min_value=0.05)
    r        = st.number_input("Rate", value=0.05, step=0.005, format="%.3f")
    sigma    = st.number_input("Vol σ₀", value=0.20, step=0.01, min_value=0.01)
    opt_type = st.selectbox("Type", ["Call", "Put"])

    st.markdown("---")
    st.markdown("### Single-step scenario")
    dS_pct   = st.slider("Spot move dS (%)", -15.0, 15.0, 2.0, step=0.5)
    dSigma   = st.slider("Vol move dσ", -0.10, 0.10, 0.01, step=0.005, format="%.3f")
    dt_days  = st.slider("Time decay dt (days)", 0, 30, 1, step=1)

    st.markdown("---")
    st.markdown("### Backtest")
    n_days   = st.slider("Simulation days", 10, 252, 63)
    vol_real = st.slider("Realised vol", 0.10, 0.60, 0.22, step=0.01,
                          help="Realised vol for the GBM path (may differ from implied)")
    bt_seed  = st.number_input("Random seed", value=42, step=1)


# ── Setup ──────────────────────────────────────────────────────────────────────
otype  = OptionType.CALL if opt_type == "Call" else OptionType.PUT
mkt    = MarketData(spot=S, rate=r)
model  = BlackScholes(sigma=sigma)
inst   = EuropeanOption(K, T, otype)
attr   = PnLAttributor(model, inst, mkt, sigma_t0=sigma)

dS     = S * dS_pct / 100
dt     = dt_days / 365

tabs = st.tabs(["◑ Single-step", "◐ Backtest"])


# ── Tab 1: Single-step attribution ────────────────────────────────────────────
with tabs[0]:
    mkt_new = MarketData(spot=S + dS, rate=r)
    result  = attr.explain(mkt_new, sigma_t1=sigma + dSigma, dt=dt)

    # Headline metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Actual P&L",  f"{result.actual_pnl:+.4f}")
    m2.metric("Explained",   f"{result.explained_pnl:+.4f}")
    m3.metric("Unexplained", f"{result.unexplained:+.4f}",
              delta=f"{'OK' if abs(result.unexplained) < 0.01 else 'Large'}")
    m4.metric("Expl. ratio", f"{result.explanation_ratio:.1%}")

    st.markdown("---")

    # Waterfall
    components = {
        'Delta':  result.delta_pnl,
        'Gamma':  result.gamma_pnl,
        'Vega':   result.vega_pnl,
        'Theta':  result.theta_pnl,
        'Vanna':  result.vanna_pnl,
        'Volga':  result.volga_pnl,
        'Unexplained': result.unexplained,
    }
    labels   = list(components.keys())
    values   = list(components.values())
    measures = ['relative'] * (len(labels) - 1) + ['relative']

    colors_bar = [GREEK_COLORS[k] for k in labels]

    fig_wf = go.Figure(go.Waterfall(
        orientation='v',
        measure=['relative'] * len(labels),
        x=labels, y=values,
        connector=dict(line=dict(color='#1e2530', width=1.5)),
        increasing=dict(marker=dict(color='#00d4b8', line=dict(color='#00d4b8'))),
        decreasing=dict(marker=dict(color='#ef4444', line=dict(color='#ef4444'))),
        text=[f"{v:+.4f}" for v in values],
        textposition='outside',
        textfont=dict(color='#e8eaf0', size=11, family='DM Mono'),
    ))

    # Add actual P&L line
    fig_wf.add_hline(y=result.actual_pnl, line_color='#f59e0b',
                      line_dash='dash', line_width=2,
                      annotation_text=f"Actual: {result.actual_pnl:+.4f}",
                      annotation_font_color='#f59e0b')

    fig_wf.update_layout(
        **PLOTLY_LAYOUT,
        title=f"P&L Attribution  dS={dS_pct:+.1f}%  dσ={dSigma:+.3f}  dt={dt_days}d",
        yaxis_title="$ P&L", height=360,
    )
    st.plotly_chart(fig_wf, use_container_width=True)

    # Pie chart of absolute contributions
    abs_vals = [abs(v) for v in values]
    total_abs = sum(abs_vals)
    if total_abs > 0:
        pcts = [v / total_abs * 100 for v in abs_vals]
        fig_pie = go.Figure(go.Pie(
            labels=labels, values=abs_vals,
            marker=dict(colors=[GREEK_COLORS[k] for k in labels],
                         line=dict(color='#0a0b0d', width=1.5)),
            textinfo='label+percent',
            textfont=dict(family='DM Mono', size=11, color='#e8eaf0'),
            hole=0.45,
        ))
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='DM Mono', color='#e8eaf0'),
            showlegend=False,
            height=300,
            margin=dict(l=10, r=10, t=30, b=10),
            title=dict(text="P&L composition (absolute)", font=dict(size=13)),
            annotations=[dict(text=f"Total<br>{result.actual_pnl:+.3f}",
                              x=0.5, y=0.5, showarrow=False,
                              font=dict(size=13, color='#e8eaf0', family='DM Mono'))],
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Greek detail table
    with st.expander("Greeks at t=0"):
        g = attr.greeks_0
        import pandas as pd
        greek_df = pd.DataFrame([{
            "Greek": "Δ Delta",  "Value": f"{g.delta:.4f}", "dS": f"{dS:+.2f}", "P&L": f"{result.delta_pnl:+.4f}",
        }, {
            "Greek": "Γ Gamma",  "Value": f"{g.gamma:.4f}", "dS²/2": f"{0.5*dS**2:.4f}", "P&L": f"{result.gamma_pnl:+.4f}",
        }, {
            "Greek": "V Vega",   "Value": f"{g.vega:.2f}",  "dσ": f"{dSigma:+.4f}", "P&L": f"{result.vega_pnl:+.4f}",
        }, {
            "Greek": "Θ Theta",  "Value": f"{g.theta:.4f}", "dt (days)": f"{dt_days}", "P&L": f"{result.theta_pnl:+.4f}",
        }, {
            "Greek": "Vanna",    "Value": f"{g.vanna:.4f}", "dS·dσ": f"{dS*dSigma:+.6f}", "P&L": f"{result.vanna_pnl:+.4f}",
        }, {
            "Greek": "Volga",    "Value": f"{g.volga:.4f}", "dσ²/2": f"{0.5*dSigma**2:+.6f}", "P&L": f"{result.volga_pnl:+.4f}",
        }])
        st.dataframe(greek_df, use_container_width=True, hide_index=True)


# ── Tab 2: Backtest attribution ────────────────────────────────────────────────
with tabs[1]:
    np.random.seed(int(bt_seed))
    dt_bk  = 1 / 252
    drifts  = (r - 0.5 * vol_real**2) * dt_bk
    shocks  = vol_real * np.sqrt(dt_bk) * np.random.standard_normal(n_days)
    log_ret = drifts + shocks
    spot_path  = S * np.exp(np.cumsum(log_ret))
    spot_path  = np.insert(spot_path, 0, S)
    sigma_path = sigma + 0.02 * np.cumsum(np.random.standard_normal(n_days + 1))
    sigma_path = np.clip(sigma_path, 0.05, 1.5)

    with st.spinner("Running backtest attribution..."):
        results_bt = attr.backtest(spot_path, sigma_path, dt=dt_bk)
        summary    = summarise_backtest(results_bt)

    days = list(range(len(results_bt)))

    # Extract component series
    delta_series  = [r.delta_pnl  for r in results_bt]
    gamma_series  = [r.gamma_pnl  for r in results_bt]
    vega_series   = [r.vega_pnl   for r in results_bt]
    theta_series  = [r.theta_pnl  for r in results_bt]
    actual_series = [r.actual_pnl for r in results_bt]
    unexplained_s = [r.unexplained for r in results_bt]

    cum_actual = np.cumsum(actual_series)
    cum_delta  = np.cumsum(delta_series)
    cum_gamma  = np.cumsum(gamma_series)
    cum_vega   = np.cumsum(vega_series)
    cum_theta  = np.cumsum(theta_series)

    # Summary metrics
    sm1, sm2, sm3, sm4 = st.columns(4)
    sm1.metric("Total actual P&L", f"{summary['actual_pnl']:+.4f}")
    sm2.metric("Total explained",  f"{summary['explained_pnl']:+.4f}")
    sm3.metric("Avg daily theta",  f"{summary['avg_theta_pnl']:.4f}")
    sm4.metric("Explanation ratio", f"{summary['explanation_ratio']:.1%}")

    st.markdown("---")

    # Cumulative P&L stacked area
    fig_bt = go.Figure()
    for name, data, color in [
        ("Theta",  cum_theta,  '#ef4444'),
        ("Gamma",  cum_gamma,  '#7c6af7'),
        ("Vega",   cum_vega,   '#f59e0b'),
        ("Delta",  cum_delta,  '#00d4b8'),
    ]:
        fig_bt.add_trace(go.Scatter(
            x=days, y=data, name=name,
            mode='lines', line=dict(color=color, width=1.8),
            stackgroup='explained',
            fillcolor=color.replace(')', ',0.25)').replace('rgb', 'rgba') if color.startswith('rgb') else color,
        ))

    fig_bt.add_trace(go.Scatter(
        x=days, y=cum_actual, name="Actual P&L",
        mode='lines', line=dict(color='#ffffff', width=2.5, dash='dot'),
    ))
    fig_bt.update_layout(
        **PLOTLY_LAYOUT,
        title="Cumulative P&L — Greek Attribution",
        xaxis_title="Trading day", yaxis_title="Cumulative P&L ($)",
        height=360,
        legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0)'),
    )
    st.plotly_chart(fig_bt, use_container_width=True)

    # Daily P&L bars + actual line
    fig_daily = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               row_heights=[0.7, 0.3], vertical_spacing=0.05)
    for name, data, color in [
        ("Delta", delta_series, '#00d4b8'),
        ("Gamma", gamma_series, '#7c6af7'),
        ("Vega",  vega_series,  '#f59e0b'),
        ("Theta", theta_series, '#ef4444'),
    ]:
        fig_daily.add_trace(go.Bar(x=days, y=data, name=name,
                                    marker_color=color, opacity=0.8), row=1, col=1)
    fig_daily.add_trace(go.Scatter(x=days, y=actual_series, name="Actual",
                                    mode='lines', line=dict(color='#fff', width=2),
                                    showlegend=True), row=1, col=1)

    # Spot path
    fig_daily.add_trace(go.Scatter(
        x=days, y=spot_path[1:], name="Spot",
        mode='lines', line=dict(color='#5a6278', width=1.5),
        showlegend=True,
    ), row=2, col=1)
    fig_daily.add_hline(y=K, line_color='#f59e0b', line_dash='dot', line_width=1,
                         row=2, col=1)

    fig_daily.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(17,19,24,1)',
        font=dict(family='DM Mono', color='#e8eaf0', size=11),
        barmode='stack',
        height=420, margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0)'),
    )
    for rw in [1, 2]:
        fig_daily.update_xaxes(gridcolor='#1e2530', linecolor='#1e2530', row=rw, col=1)
        fig_daily.update_yaxes(gridcolor='#1e2530', linecolor='#1e2530', row=rw, col=1)
    fig_daily.update_yaxes(title_text="Daily P&L", row=1, col=1)
    fig_daily.update_yaxes(title_text="Spot", row=2, col=1)
    fig_daily.update_xaxes(title_text="Trading day", row=2, col=1)
    st.plotly_chart(fig_daily, use_container_width=True)

    # Unexplained P&L distribution
    with st.expander("Unexplained P&L distribution (model quality check)"):
        fig_unex = go.Figure()
        fig_unex.add_trace(go.Histogram(
            x=unexplained_s, nbinsx=25,
            marker_color='#5a6278', marker_line_color='#0a0b0d', marker_line_width=0.5,
            name='Unexplained',
        ))
        fig_unex.add_vline(x=0, line_color='#00d4b8', line_dash='dash', line_width=2)
        fig_unex.update_layout(
            **PLOTLY_LAYOUT,
            title="Daily Unexplained P&L  (should be tightly around 0)",
            xaxis_title="Unexplained P&L", yaxis_title="Count", height=240,
        )
        st.plotly_chart(fig_unex, use_container_width=True)

        avg_u  = np.mean(unexplained_s)
        std_u  = np.std(unexplained_s)
        max_u  = max(abs(u) for u in unexplained_s)
        uc1, uc2, uc3 = st.columns(3)
        uc1.metric("Mean unexplained", f"{avg_u:+.5f}")
        uc2.metric("Std unexplained",  f"{std_u:.5f}")
        uc3.metric("Max |unexplained|", f"{max_u:.5f}")