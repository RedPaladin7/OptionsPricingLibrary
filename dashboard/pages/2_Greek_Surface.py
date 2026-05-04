"""
pages/2_Greek_Surface.py
-------------------------
Greek surface dashboard:
  · Delta, Gamma, Vega, Theta, Vanna, Volga, Charm, Rho
  · Heatmaps across (strike × expiry) grid
  · 3D Greek surface for selected Greek
  · BS PDE verification
  · Model: BS (analytical) or Heston (bump-and-reprice via dynamically calibrated SVI)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from options_lib import BlackScholes, EuropeanOption, MarketData, OptionType, Heston, HestonParams
from options_lib.risk import GreekEngine, GreekSurface
from options_lib.market_data.option_chain import fetch_option_chain
from options_lib.market_data.vol_surface import calibrate_vol_surface, SVIParams, VolSurface

PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(17,19,24,1)',
    font=dict(family='DM Mono', color='#e8eaf0', size=11),
    margin=dict(l=40, r=20, t=50, b=40),
)

GREEK_COLORMAPS = {
    'delta'  : [[0, '#1e3a5f'], [0.5, '#00d4b8'], [1, '#f59e0b']],
    'gamma'  : [[0, '#0f172a'], [0.5, '#7c6af7'], [1, '#f59e0b']],
    'vega'   : [[0, '#0f172a'], [0.5, '#10b981'], [1, '#00d4b8']],
    'theta'  : [[0, '#f59e0b'], [0.5, '#ef4444'], [1, '#0f172a']],
    'vanna'  : [[0, '#ef4444'], [0.5, '#111318'], [1, '#00d4b8']],
    'volga'  : [[0, '#0f172a'], [0.5, '#7c6af7'], [1, '#f59e0b']],
    'charm'  : [[0, '#f59e0b'], [0.5, '#111318'], [1, '#7c6af7']],
    'rho'    : [[0, '#0f172a'], [0.5, '#3b82f6'], [1, '#00d4b8']],
}

st.markdown("""
<h1 style="font-family:'Syne',sans-serif;font-size:32px;font-weight:800;
           letter-spacing:-0.02em;margin-bottom:4px;">Greek Surface</h1>
<p style="font-family:'DM Mono',monospace;font-size:12px;color:#5a6278;margin-bottom:28px;">
    Δ  Γ  V  Θ  Vanna  Volga  Charm  ρ  across all strikes &amp; expiries
</p>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Market Data Source")
    data_mode = st.radio("Source", ["Synthetic (Demo)", "Live (yfinance)"])
    
    if data_mode == "Live (yfinance)":
        ticker = st.text_input("Ticker", value="SPY")
        live_r = st.number_input("Risk-free rate", value=0.05, step=0.005)
        live_q = st.number_input("Dividend yield", value=0.013, step=0.001)
        fetch_btn = st.button("⟳ Fetch Data & Calibrate")
    else:
        syn_spot = st.number_input("Spot", value=100.0, step=1.0)
        syn_r    = st.number_input("Rate", value=0.05, step=0.005, format="%.3f")
        syn_q    = st.number_input("Div yield", value=0.0, step=0.005, format="%.3f")
        syn_vol  = st.number_input("ATM Volatility σ", value=0.20, step=0.01)
        syn_skew = st.slider("Vol Skew (ρ)", min_value=-0.9, max_value=0.1, value=-0.5)

    st.markdown("---")
    st.markdown("### Grid parameters")
    opt_type  = st.selectbox("Option type", ["Call", "Put"])
    k_lo      = st.number_input("Min strike (% spot)", value=70.0) / 100
    k_hi      = st.number_input("Max strike (% spot)", value=130.0) / 100
    n_strikes = st.slider("N strikes", 5, 30, 15)
    t_lo      = st.number_input("Min expiry (yr)", value=0.1, step=0.05, min_value=0.05)
    t_hi      = st.number_input("Max expiry (yr)", value=2.0, step=0.25)
    n_exp     = st.slider("N expiries", 3, 15, 8)

    st.markdown("---")
    model_choice = st.selectbox("Model Engine", ["Black-Scholes (analytical)", "Heston (Surface Calibrated)"])
    selected_greek_3d = st.selectbox("Greek for 3D view", ["delta", "gamma", "vega", "theta", "vanna", "volga"])

# ── Data Pipelines ────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def get_live_surface(ticker, r, q):
    chain = fetch_option_chain(ticker, rate=r, div_yield=q, n_expiries=4)
    surf = calibrate_vol_surface(chain, verbose=False)
    return surf

def get_synthetic_surface(S, r, q, sigma, skew):
    slices = {
        '3M': SVIParams(a=sigma**2*0.9, b=0.12, rho=skew*0.9, m=0.0, sigma=0.10, expiry=0.25),
        '6M': SVIParams(a=sigma**2,     b=0.14, rho=skew,     m=0.0, sigma=0.12, expiry=0.50),
        '1Y': SVIParams(a=sigma**2*1.1, b=0.16, rho=skew*1.1, m=0.0, sigma=0.15, expiry=1.00),
    }
    forwards = {d: S * np.exp((r-q) * s.expiry) for d, s in slices.items()}
    return VolSurface(svi_slices=slices, forwards=forwards, spot=S, rate=r, ticker='DEMO')

@st.cache_data(show_spinner=False)
def calibrate_heston(_surface, S, r, q):
    initial = HestonParams(v0=0.04, kappa=2.0, v_bar=0.04, xi=0.3, rho=-0.5)
    mkt_strikes, mkt_expiries, mkt_ivs = [], [], []
    for T_val in _surface.expiries:
        for K_val in np.linspace(S * 0.8, S * 1.2, 7):
            mkt_strikes.append(K_val)
            mkt_expiries.append(T_val)
            mkt_ivs.append(_surface.implied_vol(K_val, T_val))
    calibrated = Heston(initial).calibrate(
        np.array(mkt_strikes), np.array(mkt_expiries), np.array(mkt_ivs), MarketData(S, r, q), initial, verbose=False
    )
    return calibrated.params

# Setup global market variables
if data_mode == "Live (yfinance)":
    if 'fetch_btn' in locals() and fetch_btn:
        with st.spinner("Fetching data..."):
            surface = get_live_surface(ticker, live_r, live_q)
            st.session_state['live_greek_data'] = (surface, surface.spot, live_r, live_q)
    elif 'live_greek_data' in st.session_state:
        surface, S, r, q = st.session_state['live_greek_data']
    else:
        st.info("Click 'Fetch Data & Calibrate' in the sidebar.")
        st.stop()
else:
    S, r, q, sigma = syn_spot, syn_r, syn_q, syn_vol
    surface = get_synthetic_surface(S, r, q, sigma, syn_skew)

# Get ATM vol for the BS reference engine
sigma_ref = float(surface.svi_slices[list(surface.svi_slices.keys())[0]].implied_vol(np.array([0.0]))[0])

otype = OptionType.CALL if opt_type == "Call" else OptionType.PUT
mkt   = MarketData(spot=S, rate=r, div_yield=q)

if model_choice == "Heston (Surface Calibrated)":
    with st.spinner("Calibrating Heston..."):
        heston_params = calibrate_heston(surface, S, r, q)
    model = Heston(heston_params)
    st.sidebar.success(f"**Calibrated Params:**\nv₀:{heston_params.v0:.3f} | κ:{heston_params.kappa:.2f}\nρ:{heston_params.rho:.2f}")
else:
    model  = BlackScholes(sigma=sigma_ref)

# ── Compute Greek surface ──────────────────────────────────────────────────────
strikes  = np.linspace(S * k_lo, S * k_hi, n_strikes)
expiries = np.linspace(t_lo, t_hi, n_exp)

with st.spinner(f"Computing {model_choice} Greeks across grid..."):
    greek_surf = GreekSurface(model, mkt, strikes, expiries, otype)
    greek_surf.compute()

# ── ATM cross-section metrics ──────────────────────────────────────────────────
atm_idx = np.argmin(np.abs(strikes - S))
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("ATM Delta",  f"{greek_surf.delta_surface[n_exp//2, atm_idx]:.4f}")
m2.metric("ATM Gamma",  f"{greek_surf.gamma_surface[n_exp//2, atm_idx]:.4f}")
m3.metric("ATM Vega",   f"{greek_surf.vega_surface[n_exp//2, atm_idx]:.2f}")
m4.metric("ATM Theta",  f"{greek_surf.theta_surface[n_exp//2, atm_idx]:.4f}")
m5.metric("ATM Vanna",  f"{greek_surf.vanna_surface[n_exp//2, atm_idx]:.4f}")
m6.metric("ATM Volga",  f"{greek_surf.volga_surface[n_exp//2, atm_idx]:.4f}")

st.markdown("---")

# ── Helper: heatmap trace ──────────────────────────────────────────────────────
def make_heatmap(z_data, title, cmap_key):
    cmap = GREEK_COLORMAPS.get(cmap_key, 'RdBu_r')
    return go.Heatmap(
        x=np.round(strikes, 1), y=[f"{t:.2f}yr" for t in expiries], z=z_data,
        colorscale=cmap, colorbar=dict(thickness=8, len=0.8, tickfont=dict(size=9, color='#e8eaf0'), title=dict(text=title, font=dict(size=10, color='#5a6278'))),
        hovertemplate='K=%{x}<br>T=%{y}<br>Value=%{z:.4f}<extra></extra>', xgap=0.5, ygap=0.5,
    )

# ── 2×4 Heatmap grid ──────────────────────────────────────────────────────────
st.markdown("### Greek Heatmaps  —  Strike (x) × Expiry (y)")
fig_grid = make_subplots(rows=2, cols=4, subplot_titles=["Δ Delta", "Γ Gamma", "V Vega", "Θ Theta", "Vanna", "Volga", "Charm", "ρ Rho"], horizontal_spacing=0.06, vertical_spacing=0.12)

charm_surf = np.zeros((n_exp, n_strikes))
rho_surf   = np.zeros((n_exp, n_strikes))
bs_for_cr  = BlackScholes(sigma=sigma_ref)
for i, T_v in enumerate(expiries):
    for j, K_v in enumerate(strikes):
        inst = EuropeanOption(float(K_v), float(T_v), otype)
        charm_surf[i, j] = bs_for_cr.charm(inst, mkt)
        rho_surf[i, j]   = bs_for_cr.rho(inst, mkt)

greek_data = [
    (greek_surf.delta_surface, 'delta', 1, 1), (greek_surf.gamma_surface, 'gamma', 1, 2),
    (greek_surf.vega_surface, 'vega', 1, 3), (greek_surf.theta_surface, 'theta', 1, 4),
    (greek_surf.vanna_surface, 'vanna', 2, 1), (greek_surf.volga_surface, 'volga', 2, 2),
    (charm_surf, 'charm', 2, 3), (rho_surf, 'rho', 2, 4),
]

for z_data, cmap_key, row, col in greek_data:
    fig_grid.add_trace(make_heatmap(z_data, cmap_key, cmap_key), row=row, col=col)
    fig_grid.update_xaxes(tickfont=dict(size=8), row=row, col=col)
    fig_grid.update_yaxes(tickfont=dict(size=8), row=row, col=col)

fig_grid.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(17,19,24,1)', font=dict(family='DM Mono', color='#e8eaf0', size=10), height=560, margin=dict(l=30, r=20, t=50, b=20))
st.plotly_chart(fig_grid, use_container_width=True)

st.markdown("---")

# ── 3D Greek surface ───────────────────────────────────────────────────────────
st.markdown(f"### 3D — {selected_greek_3d.title()} Surface")
z_map = {'delta': greek_surf.delta_surface, 'gamma': greek_surf.gamma_surface, 'vega': greek_surf.vega_surface, 'theta': greek_surf.theta_surface, 'vanna': greek_surf.vanna_surface, 'volga': greek_surf.volga_surface}
fig_3d = go.Figure(data=[go.Surface(x=np.round(strikes, 1), y=expiries, z=z_map[selected_greek_3d], colorscale=GREEK_COLORMAPS.get(selected_greek_3d, 'RdBu_r'))])
fig_3d.update_layout(paper_bgcolor='rgba(0,0,0,0)', scene=dict(bgcolor='rgba(10,11,13,1)', xaxis=dict(title='Strike', gridcolor='#1e2530'), yaxis=dict(title='Expiry', gridcolor='#1e2530'), zaxis=dict(title=selected_greek_3d.title(), gridcolor='#1e2530')), height=460, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig_3d, use_container_width=True)
