"""
pages/3_Vega_Matrix.py
-----------------------
Vega Matrix dashboard:
  · Full K × T pillar sensitivity heatmap
  · Row sums (vega by expiry)
  · Column sums (vega by strike)
  · Portfolio aggregation
  · Scenario P&L estimation
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from options_lib import EuropeanOption, MarketData, OptionType, BlackScholes
from options_lib.market_data.vol_surface import SVIParams, VolSurface
from options_lib.risk import SurfaceGreekEngine, VegaMatrix, VolSurfaceScenario, scenario_pnl

PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(17,19,24,1)',
    font=dict(family='DM Mono', color='#e8eaf0', size=11),
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(gridcolor='#1e2530', linecolor='#1e2530'),
    yaxis=dict(gridcolor='#1e2530', linecolor='#1e2530'),
)

st.markdown("""
<h1 style="font-family:'Syne',sans-serif;font-size:32px;font-weight:800;
           letter-spacing:-0.02em;margin-bottom:4px;">Vega Matrix</h1>
<p style="font-family:'DM Mono',monospace;font-size:12px;color:#5a6278;margin-bottom:28px;">
    Pillar-by-pillar vol surface sensitivity · Term structure · Strike bucketing
</p>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Instrument")
    S        = st.number_input("Spot", value=100.0, step=1.0)
    K_inst   = st.number_input("Strike", value=100.0, step=1.0)
    T_inst   = st.number_input("Expiry (yr)", value=1.0, step=0.25, min_value=0.05)
    r        = st.number_input("Rate", value=0.05, step=0.005, format="%.3f")
    sigma    = st.number_input("σ (for BS ref)", value=0.20, step=0.01, min_value=0.01)
    opt_type = st.selectbox("Type", ["Call", "Put"])
    bump_bp  = st.number_input("Bump size (bp)", value=1.0, step=0.5, min_value=0.1)

    st.markdown("---")
    st.markdown("### Vol surface (SVI)")
    rho_s  = st.slider("Skew ρ", -0.99, 0.99, -0.55, step=0.05)
    xi_s   = st.slider("Vol-of-vol ξ (b param)", 0.05, 0.50, 0.14, step=0.01)

    st.markdown("---")
    st.markdown("### Scenario P&L")
    par_shift  = st.slider("Parallel shift (%)", -5.0, 5.0, 0.0, step=0.5) / 100
    skew_shift = st.slider("Skew shift (Δρ)", -0.30, 0.30, 0.0, step=0.02)
    curv_shift = st.slider("Curvature shift (Δσ_svi)", -0.20, 0.20, 0.0, step=0.01)


# ── Build surface ──────────────────────────────────────────────────────────────
otype = OptionType.CALL if opt_type == "Call" else OptionType.PUT
mkt   = MarketData(spot=S, rate=r)

# 4-expiry surface for richness
a_level = sigma**2 * 0.9
svi_slices = {
    '3M': SVIParams(a=a_level*0.85, b=xi_s*0.85, rho=rho_s*0.85, m=0.0, sigma=0.08, expiry=0.25),
    '6M': SVIParams(a=a_level*0.92, b=xi_s*0.92, rho=rho_s*0.92, m=0.0, sigma=0.10, expiry=0.50),
    '1Y': SVIParams(a=a_level,      b=xi_s,       rho=rho_s,       m=0.0, sigma=0.13, expiry=1.00),
    '2Y': SVIParams(a=a_level*1.10, b=xi_s*1.08,  rho=rho_s*1.05, m=0.0, sigma=0.17, expiry=2.00),
}
fwd = {d: S * np.exp(r * s.expiry) for d, s in svi_slices.items()}
surface = VolSurface(svi_slices=svi_slices, forwards=fwd, spot=S, rate=r, ticker='DEMO')
engine  = SurfaceGreekEngine(surface, mkt)

# Pillar strikes grid
pillar_K = np.array([0.75, 0.85, 0.92, 1.00, 1.08, 1.15, 1.25]) * S

# Instrument
inst = EuropeanOption(K_inst, T_inst, otype)

with st.spinner("Computing vega matrix..."):
    vm = engine.vega_matrix(inst, pillar_strikes=pillar_K, bump_bp=bump_bp)
    sg = engine.compute(inst)

# ── Summary metrics ────────────────────────────────────────────────────────────
bs_price = BlackScholes(sigma=sigma).price(inst, mkt)
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Option price",    f"{vm.base_price:.4f}")
m2.metric("Total vega",      f"{vm.total_vega:+.5f}", delta=f"{bump_bp:.0f}bp parallel")
m3.metric("Parallel vega",   f"{sg.vega_parallel:+.4f}", delta="per 1%")
m4.metric("Skew sensitivity",f"{sg.skew_sensitivity:+.4f}", delta="per Δρ")
m5.metric("Curvature sens.", f"{sg.curvature_sensitivity:+.4f}", delta="per Δσ_svi")

st.markdown("---")

# ── Main vega matrix heatmap ───────────────────────────────────────────────────
st.markdown("### Pillar Vega Matrix  *($ per 1bp IV move at each pillar)*")

k_labels   = [f"{k:.0f}" for k in pillar_K]
exp_labels = vm.expiry_dates

# Annotated heatmap
annotations = []
for i in range(len(exp_labels)):
    for j in range(len(k_labels)):
        val = vm.matrix[i, j]
        if not np.isnan(val):
            annotations.append(dict(
                x=j, y=i,
                text=f"{val:.5f}",
                font=dict(size=9, color='#e8eaf0' if abs(val) < vm.matrix.max() * 0.6 else '#000'),
                showarrow=False,
            ))

fig_vm = go.Figure(go.Heatmap(
    z=vm.matrix,
    x=k_labels,
    y=exp_labels,
    colorscale=[
        [0.0,  '#0f172a'],
        [0.3,  '#1e3a5f'],
        [0.6,  '#0f6e56'],
        [0.8,  '#00d4b8'],
        [1.0,  '#f59e0b'],
    ],
    colorbar=dict(
        title=dict(text=f"$ / {bump_bp:.0f}bp", font=dict(size=11, color='#5a6278')),
        tickfont=dict(size=10, color='#e8eaf0'),
        thickness=14, len=0.7,
    ),
    xgap=2, ygap=2,
    hovertemplate='K=%{x}<br>Expiry=%{y}<br>Vega=%{z:.6f}<extra></extra>',
))

fig_vm.update_layout(
    **PLOTLY_LAYOUT,
    title=f"Vega Matrix — {opt_type} K={K_inst:.0f} T={T_inst:.2f}yr",
    xaxis_title="Strike (K)",
    yaxis_title="Expiry",
    annotations=annotations,
    height=320,
)

# Highlight the option's own expiry row
opt_exp_label = None
for d, svi in surface.svi_slices.items():
    if abs(svi.expiry - T_inst) < 0.15:
        opt_exp_label = d
        break

if opt_exp_label and opt_exp_label in exp_labels:
    idx_row = exp_labels.index(opt_exp_label)
    fig_vm.add_shape(type="rect",
                     x0=-0.5, x1=len(k_labels) - 0.5,
                     y0=idx_row - 0.5, y1=idx_row + 0.5,
                     line=dict(color='#7c6af7', width=2),
                     fillcolor='rgba(124,106,247,0.05)')

st.plotly_chart(fig_vm, use_container_width=True)

# ── Expiry + strike bar charts ─────────────────────────────────────────────────
bar1, bar2 = st.columns(2)

with bar1:
    st.markdown("**Vega by expiry** *(row sums)*")
    ev = vm.expiry_vegas
    fig_ev = go.Figure(go.Bar(
        x=exp_labels, y=ev,
        marker_color=['#00d4b8' if v >= 0 else '#ef4444' for v in ev],
        text=[f"{v:.5f}" for v in ev],
        textposition='outside',
        textfont=dict(size=10, color='#e8eaf0'),
    ))
    fig_ev.update_layout(**PLOTLY_LAYOUT, height=260, yaxis_title=f"$ / {bump_bp:.0f}bp")
    st.plotly_chart(fig_ev, use_container_width=True)

with bar2:
    st.markdown("**Vega by strike** *(column sums)*")
    sv = vm.strike_vegas
    fig_sv = go.Figure(go.Bar(
        x=k_labels, y=sv,
        marker_color=['#7c6af7' if v >= 0 else '#f59e0b' for v in sv],
        text=[f"{v:.5f}" for v in sv],
        textposition='outside',
        textfont=dict(size=10, color='#e8eaf0'),
    ))
    fig_sv.update_layout(**PLOTLY_LAYOUT, height=260,
                          xaxis_title="Strike", yaxis_title=f"$ / {bump_bp:.0f}bp")
    st.plotly_chart(fig_sv, use_container_width=True)

st.markdown("---")

# ── Per-expiry vega bucketing ──────────────────────────────────────────────────
st.markdown("### Term Structure of Vega Exposure")

vega_by_exp    = sg.vega_by_expiry
ts_dv01        = sg.term_structure_dv01
dates_sorted   = sorted(vega_by_exp.keys())
vega_vals      = [vega_by_exp[d] for d in dates_sorted]
dv01_vals      = [ts_dv01.get(d, 0) for d in dates_sorted]

fig_ts = make_subplots(specs=[[{"secondary_y": True}]])
fig_ts.add_trace(go.Bar(x=dates_sorted, y=vega_vals, name="Vega (per 1%)",
                         marker_color=['#00d4b8' if v >= 0 else '#ef4444' for v in vega_vals]), secondary_y=False)
fig_ts.add_trace(go.Scatter(x=dates_sorted, y=dv01_vals, name="DV01 (per 1bp)",
                             mode='lines+markers',
                             line=dict(color='#f59e0b', width=2.5),
                             marker=dict(size=8)), secondary_y=True)
fig_ts.update_layout(
    **PLOTLY_LAYOUT,
    title="Vega Term Structure — SVI-bucketed",
    height=300,
    legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0)'),
)
fig_ts.update_yaxes(title_text="$ / 1% vol shift", secondary_y=False, gridcolor='#1e2530')
fig_ts.update_yaxes(title_text="$ / 1bp vol shift", secondary_y=True, gridcolor='rgba(0,0,0,0)')
st.plotly_chart(fig_ts, use_container_width=True)

st.markdown("---")

# ── Scenario P&L ──────────────────────────────────────────────────────────────
st.markdown("### Scenario P&L Estimate  *(first-order approximation)*")

scenario = VolSurfaceScenario(
    parallel_shift  = par_shift,
    skew_shift      = skew_shift,
    curvature_shift = curv_shift,
)
result = scenario_pnl(engine, inst, scenario)

sc1, sc2, sc3, sc4, sc5 = st.columns(5)
sc1.metric("Base price",      f"{result['base_price']:.4f}")
sc2.metric("Parallel P&L",   f"{result['pnl_parallel']:+.4f}",
           delta=f"{par_shift*100:+.1f}% shift")
sc3.metric("Skew P&L",       f"{result['pnl_skew']:+.4f}",
           delta=f"Δρ={skew_shift:+.2f}")
sc4.metric("Curvature P&L",  f"{result['pnl_curvature']:+.4f}",
           delta=f"Δσ={curv_shift:+.2f}")
sc5.metric("Total P&L",      f"{result['total_pnl']:+.4f}",
           delta=f"{result['total_pnl']/result['base_price']*100:+.2f}%")

# Waterfall chart
labels   = ["Base", "Parallel", "Skew", "Curvature", "Total"]
values   = [0, result['pnl_parallel'], result['pnl_skew'], result['pnl_curvature'], 0]
measures = ["absolute", "relative", "relative", "relative", "total"]
colors_wf = ['#5a6278', '#00d4b8' if result['pnl_parallel'] >= 0 else '#ef4444',
             '#7c6af7' if result['pnl_skew'] >= 0 else '#f59e0b',
             '#10b981' if result['pnl_curvature'] >= 0 else '#ef4444',
             '#00d4b8' if result['total_pnl'] >= 0 else '#ef4444']

fig_wf = go.Figure(go.Waterfall(
    name="P&L",
    orientation="v",
    measure=measures,
    x=labels,
    y=values,
    connector=dict(line=dict(color='#1e2530', width=1.5)),
    increasing=dict(marker=dict(color='#00d4b8')),
    decreasing=dict(marker=dict(color='#ef4444')),
    totals=dict(marker=dict(color='#7c6af7')),
    text=[f"{v:+.4f}" if v != 0 else "" for v in values],
    textposition="outside",
    textfont=dict(color='#e8eaf0', size=11),
))
fig_wf.update_layout(**PLOTLY_LAYOUT, title="P&L Waterfall", height=300,
                      yaxis_title="$ P&L")
st.plotly_chart(fig_wf, use_container_width=True)